/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Right-eye buffer twins (VKBufferTwins): sparse and GPU-only buffers get a twin in the mirror of their memory at
// the same offset, which the right eye's copies, dispatches and passes use; CPU-visible buffers stay single.

#include "Layer.h"
#include "../S3DAPI/GlobalData.h"
#include <deque>

namespace wiz3d { namespace vk {

VkBuffer BufferTwin(DeviceData* d, VkBuffer b)
{
    auto it = d->buffers.find(b);
    return it != d->buffers.end() ? it->second.twin : VK_NULL_HANDLE;
}

VkBufferView BufferViewTwin(DeviceData* d, VkBufferView v)
{
    auto it = d->bufferViews.find(v);
    if (it == d->bufferViews.end()) return v;
    if (it->second.twin) return it->second.twin;
    const VkBuffer twin = BufferTwin(d, it->second.ci.buffer);
    if (!twin) return v;
    VkBufferViewCreateInfo ci = it->second.ci;
    ci.buffer = twin;
    if (d->CreateBufferView(d->device, &ci, nullptr, &it->second.twin) != VK_SUCCESS)
    {
        it->second.twin = VK_NULL_HANDLE;
        return v;
    }
    return it->second.twin;
}

// Same create info as the game's, so memory requirements match and the twin
// can sit at the game buffer's offset in the mirror.
static VkBuffer CreateTwin(DeviceData* d, const BufferInfo& info)
{
    VkBufferCreateInfo ci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ci.flags = info.flags;
    ci.size = info.size;
    ci.usage = info.usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer twin = VK_NULL_HANDLE;
    if (d->CreateBuffer(d->device, &ci, nullptr, &twin) != VK_SUCCESS) return VK_NULL_HANDLE;
    return twin;
}

void TwinOnCreateBuffer(DeviceData* d, VkBuffer buf)
{
    if (!Vk().VKBufferTwins || !d->stereoEnabled) return;
    BufferInfo info;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->buffers.find(buf);
        if (it == d->buffers.end() || !(it->second.flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT)) return;
        info = it->second;
    }
    const VkBuffer twin = CreateTwin(d, info);
    std::lock_guard<std::shared_mutex> g(d->lock);
    auto it = d->buffers.find(buf);
    if (!twin || it == d->buffers.end()) { if (twin) d->DestroyBuffer(d->device, twin, nullptr); ++d->statTwinFailures; return; }
    it->second.twin = twin;
    ++d->statTwins;
    Log("[twin] sparse buffer %p: twin %p (%llu MB virtual)\n", (void*)buf, (void*)twin, (unsigned long long)(info.size >> 20));
}

void TwinOnBindBuffer(DeviceData* d, VkBuffer buf)
{
    if (!Vk().VKBufferTwins || !d->stereoEnabled) return;
    std::lock_guard<std::shared_mutex> g(d->lock);
    auto it = d->buffers.find(buf);
    if (it == d->buffers.end() || it->second.twin || (it->second.flags & VK_BUFFER_CREATE_SPARSE_BINDING_BIT)) return;
    auto mit = d->memory.find(it->second.memory);
    if (mit == d->memory.end()) return;
    const VkMemoryPropertyFlags props = d->memProps.memoryTypes[mit->second.typeIndex].propertyFlags;
    if (!(props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) || (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) return;
    const VkBuffer twin = CreateTwin(d, it->second);
    if (!twin) { ++d->statTwinFailures; return; }
    VkMemoryRequirements req;
    d->GetBufferMemoryRequirements(d->device, twin, &req);
    const VkDeviceMemory mirror = EnsureMirror(d, mit->second);
    const bool fits = mirror && (req.memoryTypeBits & (1u << mit->second.typeIndex)) &&
                      it->second.offset % req.alignment == 0 && it->second.offset + req.size <= mit->second.size;
    if (!fits || d->BindBufferMemory(d->device, twin, mirror, it->second.offset) != VK_SUCCESS)
    {
        d->DestroyBuffer(d->device, twin, nullptr);
        if (d->statTwinFailures++ < 16)
            Log("[twin] buffer %p (%llu KB) not twinned: mirror=%p fits=%d\n", (void*)buf, (unsigned long long)(it->second.size >> 10), (void*)mirror, (int)fits);
        return;
    }
    it->second.twin = twin;
    ++d->statTwins;
    d->twinBytes += it->second.size;
}

void TwinOnDestroyBuffer(DeviceData* d, VkBuffer twin)
{
    d->DestroyBuffer(d->device, twin, nullptr);
    std::lock_guard<std::shared_mutex> g(d->lock);
    if (d->statTwins) --d->statTwins;
}

// The game's binds again on each twin, memory swapped for its mirror; an
// unbind (null memory) unbinds the twin too. Same call, same semaphores.
static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueBindSparse(VkQueue queue, uint32_t count, const VkBindSparseInfo* pInfo, VkFence fence)
{
    DeviceData* d = FindDevice(queue);
    if (!Vk().VKBufferTwins || !d->stereoEnabled || !count) { const auto ql = FgQueueLock(); return d->QueueBindSparse(queue, count, pInfo, fence); }
    std::vector<VkBindSparseInfo> infos(pInfo, pInfo + count);
    std::vector<std::vector<VkSparseBufferMemoryBindInfo>> bufInfos(count);
    std::deque<std::vector<VkSparseMemoryBind>> binds;   // stable addresses for pBinds
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (uint32_t i = 0; i < count; ++i)
        {
            const VkBindSparseInfo& in = pInfo[i];
            if (!in.bufferBindCount) continue;
            auto& out = bufInfos[i];
            out.assign(in.pBufferBinds, in.pBufferBinds + in.bufferBindCount);
            for (uint32_t j = 0; j < in.bufferBindCount; ++j)
            {
                const VkSparseBufferMemoryBindInfo& bb = in.pBufferBinds[j];
                const VkBuffer twin = BufferTwin(d, bb.buffer);
                if (!twin) continue;
                binds.emplace_back(bb.pBinds, bb.pBinds + bb.bindCount);
                bool ok = true;
                for (VkSparseMemoryBind& b : binds.back())
                {
                    if (!b.memory) continue;
                    auto mit = d->memory.find(b.memory);
                    const VkDeviceMemory m = mit != d->memory.end() ? EnsureMirror(d, mit->second) : VK_NULL_HANDLE;
                    if (!m) { ok = false; break; }
                    d->twinBytes += b.size;
                    b.memory = m;
                }
                if (!ok)
                {
                    binds.pop_back(); ++d->statTwinFailures; continue;
                }
                out.push_back({ twin, (uint32_t)binds.back().size(), binds.back().data() });
                ++d->statSparseBindsMirrored;
            }
            infos[i].bufferBindCount = (uint32_t)out.size();
            infos[i].pBufferBinds = out.data();
        }
    }
    const auto ql = FgQueueLock();
    return d->QueueBindSparse(queue, count, infos.data(), fence);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateBufferView(VkDevice device, const VkBufferViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBufferView* pView)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->CreateBufferView(device, pCreateInfo, pAllocator, pView);
    if (r == VK_SUCCESS && Vk().VKBufferTwins)
    {
        DeviceData::BufferViewInfo vi = { *pCreateInfo, VK_NULL_HANDLE };
        vi.ci.pNext = nullptr;
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->bufferViews[*pView] = vi;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyBufferView(VkDevice device, VkBufferView view, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    VkBufferView twin = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->bufferViews.find(view);
        if (it != d->bufferViews.end()) { twin = it->second.twin; d->bufferViews.erase(it); }
    }
    if (twin) d->DestroyBufferView(device, twin, nullptr);
    d->DestroyBufferView(device, view, pAllocator);
}

// A 64 MB ring of small slots, seeded by the caller from the game's bytes.
VkBuffer ScratchSlot(DeviceData* d, VkDeviceSize n, VkDeviceSize* off)
{
    const VkDeviceSize kSize = 64ull << 20;
    VkBuffer& ring = d->scratchBuffer;
    VkDeviceMemory& ringMem = d->scratchMemory;
    VkDeviceSize& head = d->scratchHead;
    if (n > kSize / 2) return VK_NULL_HANDLE;
    if (!ring)
    {
        std::lock_guard<std::mutex> g(d->shadowLock);
        if (!ring)
        {
            VkBufferCreateInfo ci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            ci.size = kSize;
            ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            VkBuffer b = VK_NULL_HANDLE;
            if (d->CreateBuffer(d->device, &ci, nullptr, &b) != VK_SUCCESS) return VK_NULL_HANDLE;
            VkMemoryRequirements req;
            d->GetBufferMemoryRequirements(d->device, b, &req);
            VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UINT32_MAX);
            VkDeviceMemory m = VK_NULL_HANDLE;
            if (mai.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->device, &mai, nullptr, &m) != VK_SUCCESS ||
                d->BindBufferMemory(d->device, b, m, 0) != VK_SUCCESS)
            {
                if (m) d->FreeMemory(d->device, m, nullptr);
                d->DestroyBuffer(d->device, b, nullptr);
                return VK_NULL_HANDLE;
            }
            ringMem = m;
            ring = b;
        }
    }
    const VkDeviceSize a = d->props.limits.minStorageBufferOffsetAlignment;
    const VkDeviceSize align = a > 256 ? a : 256;
    std::lock_guard<std::shared_mutex> g(d->lock);
    VkDeviceSize o = (head + align - 1) / align * align;
    // At a few KB a frame the GPU finished with a slot long before it is reused; the seed barrier
    // (ALL_COMMANDS -> TRANSFER) every slot gets orders it after its last reader too.
    if (o + n > kSize) o = 0;
    head = o + n;
    *off = o;
    ++d->statScratchSlots;
    return ring;
}

void TwinsDestroy(DeviceData* d)
{
    for (auto& kv : d->bufferViews)
        if (kv.second.twin) d->DestroyBufferView(d->device, kv.second.twin, nullptr);
    d->bufferViews.clear();
    if (d->scratchBuffer) d->DestroyBuffer(d->device, d->scratchBuffer, nullptr);
    if (d->scratchMemory) d->FreeMemory(d->device, d->scratchMemory, nullptr);
    d->scratchBuffer = VK_NULL_HANDLE;
    d->scratchMemory = VK_NULL_HANDLE;
}

#define WIZ3D_TWIN_HOOK(name) { "vk" #name, (PFN_vkVoidFunction)&Hook_##name }
const HookEntry g_twinHooks[] = {
    WIZ3D_TWIN_HOOK(QueueBindSparse),
    WIZ3D_TWIN_HOOK(CreateBufferView), WIZ3D_TWIN_HOOK(DestroyBufferView),
    { nullptr, nullptr },
};

}} // namespace wiz3d::vk
