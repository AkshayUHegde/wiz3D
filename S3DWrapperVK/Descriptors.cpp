/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Descriptors: layouts padded so the stereo block sits at kStereoSet; sets naming a doubled resource get a shadow set
// with the siblings, which the right eye binds (DX11's BindEye swap, done once at write time instead of per draw).

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include <algorithm>
#include <string.h>

namespace wiz3d { namespace vk {

// Layouts
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDescriptorSetLayout(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorSetLayout* pSetLayout)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->CreateDescriptorSetLayout(device, pCreateInfo, pAllocator, pSetLayout);
    if (r != VK_SUCCESS) return r;
    DescriptorSetLayoutInfo info;
    info.bindings.assign(pCreateInfo->pBindings, pCreateInfo->pBindings + pCreateInfo->bindingCount);
    for (auto& b : info.bindings) b.pImmutableSamplers = nullptr;
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->setLayouts[*pSetLayout] = info;
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyDescriptorSetLayout(VkDevice device, VkDescriptorSetLayout layout, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->setLayouts.erase(layout);
    }
    d->DestroyDescriptorSetLayout(device, layout, pAllocator);
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreatePipelineLayout(
    VkDevice device, const VkPipelineLayoutCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkPipelineLayout* pPipelineLayout)
{
    DeviceData* d = FindDevice(device);
    PipelineLayoutInfo info;
    info.gameSetCount = pCreateInfo->setLayoutCount;
    VkResult r;
    if (d->stereoEnabled && pCreateInfo->setLayoutCount <= kStereoSet &&
        d->props.limits.maxBoundDescriptorSets > kStereoSet)
    {
        std::vector<VkDescriptorSetLayout> layouts(pCreateInfo->pSetLayouts, pCreateInfo->pSetLayouts + pCreateInfo->setLayoutCount);
        while (layouts.size() < kStereoSet) layouts.push_back(d->emptySetLayout);
        layouts.push_back(d->stereoSetLayout);
        VkPipelineLayoutCreateInfo ci = *pCreateInfo;
        ci.setLayoutCount = (uint32_t)layouts.size();
        ci.pSetLayouts = layouts.data();
        r = d->CreatePipelineLayout(device, &ci, pAllocator, pPipelineLayout);
        info.patched = (r == VK_SUCCESS);
    }
    else
    {
        r = d->CreatePipelineLayout(device, pCreateInfo, pAllocator, pPipelineLayout);
        if (r == VK_SUCCESS && d->stereoEnabled)
            Log("[layout] %p not patched: %u sets (max %u)\n", (void*)*pPipelineLayout,
                pCreateInfo->setLayoutCount, d->props.limits.maxBoundDescriptorSets);
    }
    if (r != VK_SUCCESS) return r;
    std::lock_guard<std::shared_mutex> g(d->lock);
    for (uint32_t i = 0; i < pCreateInfo->setLayoutCount; ++i)
    {
        uint32_t n = 0;
        auto lit = d->setLayouts.find(pCreateInfo->pSetLayouts[i]);
        if (lit != d->setLayouts.end())
            for (const auto& b : lit->second.bindings)
                if (b.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || b.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
                    n += b.descriptorCount;
        info.dynCounts.push_back(n);
    }
    d->pipelineLayouts[*pPipelineLayout] = info;
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyPipelineLayout(VkDevice device, VkPipelineLayout layout, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->pipelineLayouts.erase(layout);
        d->objGen.fetch_add(1, std::memory_order_release);
    }
    d->DestroyPipelineLayout(device, layout, pAllocator);
}

// Shadow pools
static VkDescriptorPool NewShadowPool(DeviceData* d)
{
    const uint32_t kSets = 4096;
    VkDescriptorPoolSize sizes[] = {
        { VK_DESCRIPTOR_TYPE_SAMPLER, 4096 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16384 },
        { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 16384 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4096 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 2048 },
        { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 2048 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8192 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8192 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 4096 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 2048 },
        { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1024 },
    };
    VkDescriptorPoolCreateInfo ci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.maxSets = kSets;
    ci.poolSizeCount = _countof(sizes);
    ci.pPoolSizes = sizes;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    if (d->CreateDescriptorPool(d->device, &ci, nullptr, &pool) != VK_SUCCESS) return VK_NULL_HANDLE;
    d->shadowPools.push_back(pool);
    return pool;
}

void ShadowPoolsDestroy(DeviceData* d)
{
    for (VkDescriptorPool p : d->shadowPools) d->DestroyDescriptorPool(d->device, p, nullptr);
    d->shadowPools.clear();
}

// Allocates n shadow sets in one call, a new pool taking what the current one cannot; entries stay null on failure.
static void AllocateShadows(DeviceData* d, uint32_t n, const VkDescriptorSetLayout* layouts, VkDescriptorSet* out)
{
    std::lock_guard<std::mutex> g(d->shadowLock);
    for (uint32_t done = 0; done < n; )
    {
        VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        ai.descriptorSetCount = n - done;
        ai.pSetLayouts = layouts + done;
        ai.descriptorPool = d->shadowPools.empty() ? NewShadowPool(d) : d->shadowPools.back();
        if (!ai.descriptorPool) return;
        if (d->AllocateDescriptorSets(d->device, &ai, out + done) == VK_SUCCESS) return;
        // The pool ran out part-way: a fresh one takes the remainder, and if
        // even that refuses the batch, fall back to one at a time.
        ai.descriptorPool = NewShadowPool(d);
        if (!ai.descriptorPool) return;
        if (d->AllocateDescriptorSets(d->device, &ai, out + done) == VK_SUCCESS) return;
        ai.descriptorSetCount = 1;
        VkResult r = d->AllocateDescriptorSets(d->device, &ai, out + done);
        if (r != VK_SUCCESS) { Log("[dset] shadow alloc failed %d\n", r); return; }
        ++done;
    }
}

// Copies `set`'s written bindings into `shadow`, so bindings written before the shadow existed are kept
// (wideBindings: all of them; copying an unwritten one is undefined, and later writes overwrite what matters).
static void SeedCopies(VkDescriptorSet set, VkDescriptorSet shadow,
                       const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                       uint64_t writtenMask, bool wideBindings,
                       std::vector<VkCopyDescriptorSet>& out)
{
    for (const auto& b : bindings)
    {
        if (!b.descriptorCount) continue;
        // Unwritten bindings hold nothing worth carrying over.
        if (!wideBindings && b.binding < 64 && !(writtenMask & (1ull << b.binding))) continue;
        VkCopyDescriptorSet c = { VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET };
        c.srcSet = set; c.srcBinding = b.binding;
        c.dstSet = shadow; c.dstBinding = b.binding;
        c.descriptorCount = b.descriptorCount;
        out.push_back(c);
    }
}

// One shadow, for the copy path. Lock may be held by the caller.
static VkDescriptorSet CreateShadow(DeviceData* d, VkDescriptorSet set, VkDescriptorSetLayout layout,
                                    const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                                    uint64_t writtenMask, bool wideBindings)
{
    VkDescriptorSet shadow = VK_NULL_HANDLE;
    AllocateShadows(d, 1, &layout, &shadow);
    if (!shadow) return VK_NULL_HANDLE;
    std::vector<VkCopyDescriptorSet> copies;
    SeedCopies(set, shadow, bindings, writtenMask, wideBindings, copies);
    if (!copies.empty()) d->UpdateDescriptorSets(d->device, 0, nullptr, (uint32_t)copies.size(), copies.data());
    ++d->statShadowSets;
    return shadow;
}

// Lock held: the layout's bindings, for CreateShadow.
static std::vector<VkDescriptorSetLayoutBinding> LayoutBindings(DeviceData* d, VkDescriptorSetLayout layout)
{
    auto lit = d->setLayouts.find(layout);
    return lit == d->setLayouts.end() ? std::vector<VkDescriptorSetLayoutBinding>() : lit->second.bindings;
}

VkDescriptorSet ShadowSetFor(DeviceData* d, VkDescriptorSet set)
{
    const DescriptorSetInfo* info = d->sets.Find(set);
    return info && info->stereoAny ? info->shadow : VK_NULL_HANDLE;
}

// Lock NOT held. Sets written once before their key was learned still hold the game's buffer: they move to the patched
// sibling. forced: claim and re-point keys (d->repointKeys) move even when `apply` is off.
unsigned RepointLearnedSets(DeviceData* d, bool apply, unsigned* forced)
{
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorBufferInfo> infos;
    unsigned stale = 0;
    if (forced) *forced = 0;
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->sets.ForEach([&](VkDescriptorSet, DescriptorSetInfo& info)
    {
        if (!info.shadow) return;
        for (UboRef& r : info.ubos)
        {
            VkBuffer sib = VK_NULL_HANDLE, cs;
            const RightSource src = ConstantsRightSource(d, info.layout, r.binding, r.buffer, &sib);
            if (src == RightSource::Game) continue;
            // claimMirror: a claim key's cached binding moves to the claim sibling, whatever it referenced before,
            // and a binding left on the claim sibling by a dropped key moves back to the shared one.
            const bool claimKey = src == RightSource::ClaimSibling;
            const bool unclaim = !claimKey && Traits().claimMirror && r.shadowBuffer && BufferClaimSibling(d, r.buffer, &cs) && r.shadowBuffer == cs;
            if (claimKey || unclaim) { if (r.shadowBuffer == sib) continue; }
            else if (r.shadowBuffer) continue;
            ++stale;
            const bool force = !apply && (claimKey || unclaim || (!d->repointKeys.empty() && d->repointKeys.count(LearnKey(info.layout, r.binding)) != 0));
            if (!apply && !force) continue;
            if (force && forced) ++*forced;
            VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w.dstSet = info.shadow;
            w.dstBinding = r.binding;
            w.dstArrayElement = r.element;
            w.descriptorCount = 1;
            w.descriptorType = r.uniform ? (r.dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
                                         : (r.dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
            writes.push_back(w);
            infos.push_back({ sib, r.offset, r.range });
            r.shadowBuffer = sib; info.stereoAny = true;
        }
    });
    if (writes.empty()) return stale;
    for (size_t i = 0; i < writes.size(); ++i) writes[i].pBufferInfo = &infos[i];
    d->UpdateDescriptorSets(d->device, (uint32_t)writes.size(), writes.data(), 0, nullptr);
    return stale;
}

// Pools and sets

// The game's pool, mirrored: same sizes, flags and pNext, so any batch its
// pool accepts ours accepts too and shadows come from the matching pool.
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDescriptorPool(
    VkDevice device, const VkDescriptorPoolCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorPool* pPool)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->CreateDescriptorPool(device, pCreateInfo, pAllocator, pPool);
    if (r != VK_SUCCESS || !d->stereoEnabled) return r;
    PoolInfo info;
    if (d->CreateDescriptorPool(device, pCreateInfo, nullptr, &info.mirror) != VK_SUCCESS)
    {
        info.mirror = VK_NULL_HANDLE;
        Log("[dset] mirror pool failed (maxSets %u); its sets fall back to lazy shadows\n", pCreateInfo->maxSets);
    }
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->pools[*pPool] = std::move(info);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_AllocateDescriptorSets(
    VkDevice device, const VkDescriptorSetAllocateInfo* pAllocateInfo, VkDescriptorSet* pDescriptorSets)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->AllocateDescriptorSets(device, pAllocateInfo, pDescriptorSets);
    if (r != VK_SUCCESS) return r;
    const uint32_t n = pAllocateInfo->descriptorSetCount;

    VkDescriptorPool mirror = VK_NULL_HANDLE;
    {
        std::shared_lock<std::shared_mutex> rd(d->lock, std::defer_lock);
        std::unique_lock<std::shared_mutex> ex(d->lock, std::defer_lock);
        if (Traits().sharedCollect) rd.lock(); else ex.lock();   // a map read only
        auto pit = d->pools.find(pAllocateInfo->descriptorPool);
        if (pit != d->pools.end()) mirror = pit->second.mirror;
    }
    // One shadow allocation per game call rather than one per set on the first
    // stereo write, which cost ~3850 allocations and ~30k seed copies a frame.
    std::vector<VkDescriptorSet> shadows;
    if (mirror)
    {
        shadows.resize(n, VK_NULL_HANDLE);
        VkDescriptorSetAllocateInfo ai = *pAllocateInfo;
        ai.descriptorPool = mirror;
        if (d->AllocateDescriptorSets(device, &ai, shadows.data()) != VK_SUCCESS) shadows.clear();
    }

    // Nodes are built unlocked and land under their shard's lock alone; the device lock stays shared, so descriptor
    // writes never queue behind an allocation.
    static thread_local SetTable::Map s_infos;
    static thread_local std::unordered_set<VkDescriptorSet> s_poolSets;
    static thread_local std::vector<SetTable::Node> s_dead;
    s_infos.clear(); s_poolSets.clear();
    unsigned shadowed = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        DescriptorSetInfo info;
        info.layout = pAllocateInfo->pSetLayouts[i];
        info.pool = pAllocateInfo->descriptorPool;
        if (i < shadows.size() && shadows[i]) { info.shadow = shadows[i]; info.shadowPool = mirror; ++shadowed; }
        s_infos[pDescriptorSets[i]] = info;
        s_poolSets.insert(pDescriptorSets[i]);
    }
    const long long held0 = PerfNow();
    while (!s_infos.empty()) d->sets.Insert(s_infos.extract(s_infos.begin()), s_dead);
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto pit = d->pools.find(pAllocateInfo->descriptorPool);
        std::lock_guard<std::mutex> pl(d->poolSetsLock);
        d->statShadowSets += shadowed;
        if (pit != d->pools.end()) pit->second.sets.merge(s_poolSets);
    }
    d->perf.allocHeld.fetch_add(PerfNow() - held0, std::memory_order_relaxed);
    s_dead.clear();
    return r;
}

// Lock held.
static void ReleaseShadow(DeviceData* d, DescriptorSetInfo& info)
{
    if (!info.shadow) return;
    if (info.shadowPool)   // from the pool mirror: one call, no scan
    {
        d->FreeDescriptorSets(d->device, info.shadowPool, 1, &info.shadow);
        info.shadow = VK_NULL_HANDLE;
        return;
    }
    // Legacy lazy shadow: our pools all have the free bit, but not which one.
    std::lock_guard<std::mutex> g(d->shadowLock);
    for (VkDescriptorPool p : d->shadowPools)
        if (d->FreeDescriptorSets(d->device, p, 1, &info.shadow) == VK_SUCCESS) break;
    info.shadow = VK_NULL_HANDLE;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_FreeDescriptorSets(
    VkDevice device, VkDescriptorPool pool, uint32_t count, const VkDescriptorSet* pSets)
{
    DeviceData* d = FindDevice(device);
    {
        PerfScope pf(d->perf.descFree);
        static thread_local std::vector<VkDescriptorSet> mirrored;
        // Nodes leave the map under the lock but are destroyed after it, with the mirror's free.
        static thread_local std::vector<SetTable::Node> dead;
        static thread_local std::vector<std::unordered_set<VkDescriptorSet>::node_type> deadPoolSets;
        mirrored.clear();
        VkDescriptorPool mirror = VK_NULL_HANDLE;
        {
            std::shared_lock<std::shared_mutex> g(d->lock);
            const long long held0 = PerfNow();
            auto pit = d->pools.find(pool);
            for (uint32_t i = 0; i < count; ++i)
            {
                DescriptorSetInfo* info = d->sets.Find(pSets[i]);
                if (!info) continue;
                // These sets share one game pool, so one mirror frees them together.
                if (info->shadow && info->shadowPool)
                {
                    mirror = info->shadowPool;
                    mirrored.push_back(info->shadow);
                    info->shadow = VK_NULL_HANDLE;
                }
                else ReleaseShadow(d, *info);
                d->sets.Extract(pSets[i], dead);
            }
            if (pit != d->pools.end())
            {
                std::lock_guard<std::mutex> pl(d->poolSetsLock);
                for (uint32_t i = 0; i < count; ++i) deadPoolSets.push_back(pit->second.sets.extract(pSets[i]));
            }
            d->perf.freeHeld.fetch_add(PerfNow() - held0, std::memory_order_relaxed);
        }
        dead.clear();
        deadPoolSets.clear();
        if (!mirrored.empty()) d->FreeDescriptorSets(device, mirror, (uint32_t)mirrored.size(), mirrored.data());
    }
    return d->FreeDescriptorSets(device, pool, count, pSets);
}

// Device lock held (shared at least). Drops every set of a game pool into `dead` / `deadSets`, freed by the caller
// after unlocking; mirror-owned shadows go with the mirror, which the caller resets or destroys whole.
using SetNode = SetTable::Node;
static void ForgetPool(DeviceData* d, VkDescriptorPool pool, std::vector<SetNode>& dead, std::unordered_set<VkDescriptorSet>& deadSets)
{
    auto pit = d->pools.find(pool);
    if (pit == d->pools.end())
    {
        d->sets.ExtractIf([&](DescriptorSetInfo& i) { if (i.pool != pool) return false; ReleaseShadow(d, i); return true; }, dead);
        return;
    }
    {
        std::lock_guard<std::mutex> pl(d->poolSetsLock);
        deadSets.swap(pit->second.sets);
    }
    for (VkDescriptorSet s : deadSets)
    {
        DescriptorSetInfo* info = d->sets.Find(s);
        if (!info) continue;
        if (!info->shadowPool) ReleaseShadow(d, *info);
        d->sets.Extract(s, dead);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_ResetDescriptorPool(VkDevice device, VkDescriptorPool pool, VkDescriptorPoolResetFlags flags)
{
    DeviceData* d = FindDevice(device);
    {
        PerfScope pf(d->perf.descFree);
        VkDescriptorPool mirror = VK_NULL_HANDLE;
        static thread_local std::vector<SetNode> dead;
        static thread_local std::unordered_set<VkDescriptorSet> deadSets;
        {
            std::shared_lock<std::shared_mutex> g(d->lock);
            const long long held0 = PerfNow();
            ForgetPool(d, pool, dead, deadSets);
            auto pit = d->pools.find(pool);
            if (pit != d->pools.end()) mirror = pit->second.mirror;
            d->perf.freeHeld.fetch_add(PerfNow() - held0, std::memory_order_relaxed);
        }
        dead.clear();
        deadSets.clear();
        // One reset frees every shadow the pool held.
        if (mirror) d->ResetDescriptorPool(device, mirror, flags);
    }
    return d->ResetDescriptorPool(device, pool, flags);
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyDescriptorPool(VkDevice device, VkDescriptorPool pool, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    VkDescriptorPool mirror = VK_NULL_HANDLE;
    std::vector<SetNode> dead;
    std::unordered_set<VkDescriptorSet> deadSets;
    {
        PerfScope pf(d->perf.descFree);
        std::lock_guard<std::shared_mutex> g(d->lock);
        ForgetPool(d, pool, dead, deadSets);
        auto pit = d->pools.find(pool);
        if (pit != d->pools.end()) { mirror = pit->second.mirror; d->pools.erase(pit); }
    }
    if (mirror) d->DestroyDescriptorPool(device, mirror, nullptr);
    d->DestroyDescriptorPool(device, pool, pAllocator);
}

// Updates
static bool IsImageDescriptor(VkDescriptorType t)
{
    return t == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || t == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
           t == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE || t == VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
}

static bool IsUboDescriptor(VkDescriptorType t)
{
    return t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || t == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
}

// Dynamic storage buffers take dynamic offsets in binding order too, so they are recorded to keep the uniform ones'
// offsets right; plain storage buffers feed the dispatch-replay write gate.
static bool IsTrackedBuffer(VkDescriptorType t)
{
    return IsUboDescriptor(t) || t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ||
           t == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
}

// Lock held. Remembers a uniform-buffer descriptor for the constants path.
static void RecordUbo(DescriptorSetInfo& info, const VkWriteDescriptorSet& w, uint32_t k, VkBuffer shadowBuffer)
{
    UboRef ref;
    ref.binding = w.dstBinding;
    ref.element = w.dstArrayElement + k;
    ref.dynamic = w.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC || w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
    ref.uniform = IsUboDescriptor(w.descriptorType);
    ref.buffer = w.pBufferInfo[k].buffer;
    ref.offset = w.pBufferInfo[k].offset;
    ref.range = w.pBufferInfo[k].range;
    ref.shadowBuffer = shadowBuffer;
    auto pos = std::lower_bound(info.ubos.begin(), info.ubos.end(), ref,
        [](const UboRef& a, const UboRef& b) { return a.binding != b.binding ? a.binding < b.binding : a.element < b.element; });
    if (pos != info.ubos.end() && pos->binding == ref.binding && pos->element == ref.element) *pos = ref;
    else info.ubos.insert(pos, ref);
}

// Applies writes to their sets' shadows with sibling views and buffers. Lock NOT held: one shared-locked lookup pass,
// then shadows are created and written unlocked, so descriptor-writing threads do not wait on the command hooks.
static void MirrorWrites(DeviceData* d, uint32_t count, const VkWriteDescriptorSet* writes)
{
    struct Pending
    {
        DescriptorSetInfo* info;
        VkDescriptorSet set;
        VkDescriptorSetLayout layout;
        std::vector<VkDescriptorSetLayoutBinding> bindings;   // only when a shadow must be created
        uint64_t writtenMask;
        bool     wideBindings;
    };
    struct Slot { uint32_t base; uint8_t kind; };   // 1 = image arena, 2 = buffer arena
    // Thread-local arenas, cleared but never freed: no heap allocation per substituted write.
    static thread_local std::vector<Pending> create;
    static thread_local std::vector<VkWriteDescriptorSet> shadowWrites;
    static thread_local std::vector<DescriptorSetInfo*> shadowInfos;   // parallel to shadowWrites
    static thread_local std::vector<VkDescriptorImageInfo> imgArena;
    static thread_local std::vector<VkDescriptorBufferInfo> bufArena;
    static thread_local std::vector<Slot> slots;
    create.clear(); shadowWrites.clear(); shadowInfos.clear();
    imgArena.clear(); bufArena.clear(); slots.clear();
    {
        PerfScope pl(d->perf.descLookup);
        const long long wait0 = PerfNow();
        // Shared: this pass only READS the maps, and the per-set state it
        // mutates belongs to a set no other thread may legally be writing.
        std::shared_lock<std::shared_mutex> g(d->lock);
        const long long held0 = PerfNow();
        d->perf.descWaitW.fetch_add(held0 - wait0, std::memory_order_relaxed);
        d->statDescWrites.fetch_add(count, std::memory_order_relaxed);
        d->statDescCalls.fetch_add(1, std::memory_order_relaxed);
        // A batch writes the same set repeatedly, and d->sets is thousands of
        // entries: remembering the last one skips most of the lookups.
        VkDescriptorSet lastSet = VK_NULL_HANDLE;
        DescriptorSetInfo* lastInfo = nullptr;
        for (uint32_t i = 0; i < count; ++i)
        {
            const VkWriteDescriptorSet& w = writes[i];
            if (w.dstSet != lastSet)
            {
                lastInfo = d->sets.Find(w.dstSet);
                lastSet = w.dstSet;
            }
            if (!lastInfo) continue;
            DescriptorSetInfo& info = *lastInfo;

            bool stereo = false;
            const bool ubo = IsUboDescriptor(w.descriptorType) && w.pBufferInfo;
            const bool tracked = IsTrackedBuffer(w.descriptorType) && w.pBufferInfo;
            if (IsImageDescriptor(w.descriptorType) && w.pImageInfo)
            {
                for (uint32_t k = 0; k < w.descriptorCount && !stereo; ++k)
                    stereo = ViewSibling(d, w.pImageInfo[k].imageView, nullptr);
                // A doubled image whose view has no right-eye view: the shadow (or the game's set) keeps the left one.
                if (gInfo.CollectDebugInformation)
                    for (uint32_t k = 0; k < w.descriptorCount; ++k)
                        if (!ViewSibling(d, w.pImageInfo[k].imageView, nullptr) && ViewOfDoubledImage(d, w.pImageInfo[k].imageView))
                            d->statLeftInRight.fetch_add(1, std::memory_order_relaxed);
            }
            // A buffer range is stereo when its (layout, binding) holds a camera block and the buffer has a sibling;
            // other ranges of it keep the game's copy. `tracked`: a camera block may sit in a storage buffer.
            bool uboStereo = false;
            if (tracked && ConstantsRangeIsStereo(d, info.layout, w.dstBinding))
                for (uint32_t k = 0; k < w.descriptorCount && !uboStereo; ++k)
                    uboStereo = BufferSibling(d, w.pBufferInfo[k].buffer, nullptr);
            if (stereo) info.stereoImages = true;
            // Storage image with a sibling = a real per-eye write target; the
            // dispatch-replay gate keys on this, not on sampled inputs.
            if (stereo && w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) info.stereoStorage = true;
            // VKShadowSetTwins: a buffer with a twin goes into the shadow as its twin, as IssuePushes does for pushes.
            bool twinStereo = false;
            // A from-left draw's set keeps the left's buffers, as ReissueReplayBinds does for push descriptors.
            const bool leftBuffers = d->leftBufferSetLayouts.count(info.layout) != 0;
            if (Vk().VKShadowSetTwins && Vk().VKBufferTwins && !uboStereo && tracked && !leftBuffers)
                for (uint32_t k = 0; k < w.descriptorCount && !twinStereo; ++k)
                    twinStereo = BufferTwin(d, w.pBufferInfo[k].buffer) != VK_NULL_HANDLE;
            stereo = stereo || uboStereo || twinStereo;
            if (w.dstBinding < 64) info.writtenMask |= 1ull << w.dstBinding;
            else info.wideBindings = true;

            if (stereo) info.stereoAny = true;
            bool willHaveShadow = info.shadow != VK_NULL_HANDLE;
            if (!willHaveShadow)
            {
                // Fallback only: a pool whose mirror could not be created.
                for (const Pending& p : create) if (p.set == w.dstSet) { willHaveShadow = true; break; }
                if (!willHaveShadow && stereo)
                {
                    create.push_back({ &info, w.dstSet, info.layout, LayoutBindings(d, info.layout), info.writtenMask, info.wideBindings });
                    willHaveShadow = true;
                }
            }
            // A key the constants path has given up on (deadKeys) is no longer recorded.
            if (tracked && !d->deadKeys.count(LearnKey(info.layout, w.dstBinding)))
                for (uint32_t k = 0; k < w.descriptorCount; ++k) RecordUbo(info, w, k, VK_NULL_HANDLE);
            // Its old ref must not keep claiming that the shadow reads our sibling.
            else if (tracked)
                for (UboRef& r : info.ubos)
                    if (r.binding == w.dstBinding && r.element >= w.dstArrayElement && r.element < w.dstArrayElement + w.descriptorCount)
                        r.shadowBuffer = VK_NULL_HANDLE;
            if (!willHaveShadow) continue;             // plain set, nothing to mirror yet

            // The game's pNext stays: an acceleration structure or inline uniform block write carries its data there.
            VkWriteDescriptorSet sw = w;
            sw.dstSet = info.shadow;                   // null only while a fallback shadow is pending
            Slot slot = { 0, 0 };
            if (stereo && IsImageDescriptor(w.descriptorType))
            {
                slot.kind = 1; slot.base = (uint32_t)imgArena.size();
                imgArena.insert(imgArena.end(), w.pImageInfo, w.pImageInfo + w.descriptorCount);
                for (size_t j = slot.base; j < imgArena.size(); ++j)
                {
                    // ... and its 2D images not shaped like the screen (per-instance data), as IssuePushes does.
                    if (leftBuffers)
                    {
                        auto vit = d->views.find(imgArena[j].imageView);
                        auto iit = vit != d->views.end() ? d->images.find(vit->second.image) : d->images.end();
                        if (iit != d->images.end() && iit->second.ci.imageType == VK_IMAGE_TYPE_2D &&
                            OffAspect(iit->second.ci.extent.width, iit->second.ci.extent.height, d->bbSize.width ? d->bbSize : VkExtent2D{ 16, 9 }))
                            continue;
                    }
                    VkImageView sib;
                    if (ViewSibling(d, imgArena[j].imageView, &sib)) imgArena[j].imageView = sib;
                }
            }
            else if (uboStereo)
            {
                slot.kind = 2; slot.base = (uint32_t)bufArena.size();
                bufArena.insert(bufArena.end(), w.pBufferInfo, w.pBufferInfo + w.descriptorCount);
                for (uint32_t k = 0; k < w.descriptorCount; ++k)
                {
                    VkBuffer sib;
                    if (ConstantsRightSource(d, info.layout, w.dstBinding, w.pBufferInfo[k].buffer, &sib) != RightSource::Game)
                    {
                        bufArena[slot.base + k].buffer = sib;
                        for (UboRef& r : info.ubos)
                            if (r.binding == w.dstBinding && r.element == w.dstArrayElement + k) r.shadowBuffer = sib;
                    }
                }
            }
            else if (twinStereo)
            {
                slot.kind = 2; slot.base = (uint32_t)bufArena.size();
                bufArena.insert(bufArena.end(), w.pBufferInfo, w.pBufferInfo + w.descriptorCount);
                for (uint32_t k = 0; k < w.descriptorCount; ++k)
                    if (const VkBuffer t = BufferTwin(d, w.pBufferInfo[k].buffer)) bufArena[slot.base + k].buffer = t;
            }
            shadowWrites.push_back(sw);
            shadowInfos.push_back(&info);
            slots.push_back(slot);
        }
        d->perf.descHeld.fetch_add(PerfNow() - held0, std::memory_order_relaxed);
    }
    // Resolved after every append, so arena growth cannot dangle a pointer.
    for (size_t i = 0; i < shadowWrites.size(); ++i)
    {
        if (slots[i].kind == 1) shadowWrites[i].pImageInfo = imgArena.data() + slots[i].base;
        else if (slots[i].kind == 2) shadowWrites[i].pBufferInfo = bufArena.data() + slots[i].base;
    }

    // Shadows first (one allocation and one seed update for the whole batch),
    // then publish them, then the writes.
    if (!create.empty())
    {
        PerfScope pc(d->perf.descCreate);
        std::vector<VkDescriptorSetLayout> layouts;
        std::vector<VkDescriptorSet> shadows(create.size(), VK_NULL_HANDLE);
        layouts.reserve(create.size());
        for (const Pending& p : create) layouts.push_back(p.layout);
        AllocateShadows(d, (uint32_t)create.size(), layouts.data(), shadows.data());

        std::vector<VkCopyDescriptorSet> seed;
        for (size_t i = 0; i < create.size(); ++i)
            if (shadows[i]) SeedCopies(create[i].set, shadows[i], create[i].bindings, create[i].writtenMask, create[i].wideBindings, seed);
        if (!seed.empty()) d->UpdateDescriptorSets(d->device, 0, nullptr, (uint32_t)seed.size(), seed.data());

        std::lock_guard<std::shared_mutex> g(d->lock);
        for (size_t i = 0; i < create.size(); ++i)
        {
            create[i].info->shadow = shadows[i];
            if (shadows[i]) ++d->statShadowSets;
        }
    }
    PerfScope pw(d->perf.descWrite);
    // Only the fallback path leaves a write without its shadow set.
    if (!create.empty())
        for (size_t i = 0; i < shadowWrites.size(); ++i)
            if (!shadowWrites[i].dstSet) shadowWrites[i].dstSet = shadowInfos[i]->shadow;
    size_t n = 0;
    for (size_t i = 0; i < shadowWrites.size(); ++i)
        if (shadowWrites[i].dstSet) shadowWrites[n++] = shadowWrites[i];
    if (n) d->UpdateDescriptorSets(d->device, (uint32_t)n, shadowWrites.data(), 0, nullptr);
}

static void MirrorCopies(DeviceData* d, uint32_t count, const VkCopyDescriptorSet* copies)
{
    std::vector<VkCopyDescriptorSet> shadowCopies;
    for (uint32_t i = 0; i < count; ++i)
    {
        const VkCopyDescriptorSet& c = copies[i];
        DescriptorSetInfo* src = d->sets.Find(c.srcSet);
        DescriptorSetInfo* dst = d->sets.Find(c.dstSet);
        if (!dst) continue;
        const VkDescriptorSet srcShadow = src ? src->shadow : VK_NULL_HANDLE;
        if (c.dstBinding < 64) dst->writtenMask |= 1ull << c.dstBinding;
        else dst->wideBindings = true;
        // Copying from a stereo set makes the destination stereo, which is what
        // creating its shadow used to signal.
        if (src && src->stereoAny) dst->stereoAny = true;
        if (!dst->shadow)
        {
            if (!srcShadow) continue;
            dst->shadow = CreateShadow(d, c.dstSet, dst->layout, LayoutBindings(d, dst->layout), dst->writtenMask, dst->wideBindings);
            if (!dst->shadow) continue;
        }
        VkCopyDescriptorSet sc = c;
        sc.pNext = nullptr;
        sc.srcSet = srcShadow ? srcShadow : c.srcSet;
        sc.dstSet = dst->shadow;
        shadowCopies.push_back(sc);
    }
    if (!shadowCopies.empty())
        d->UpdateDescriptorSets(d->device, 0, nullptr, (uint32_t)shadowCopies.size(), shadowCopies.data());
}

static VKAPI_ATTR void VKAPI_CALL Hook_UpdateDescriptorSets(
    VkDevice device, uint32_t writeCount, const VkWriteDescriptorSet* pWrites, uint32_t copyCount, const VkCopyDescriptorSet* pCopies)
{
    DeviceData* d = FindDevice(device);
    d->UpdateDescriptorSets(device, writeCount, pWrites, copyCount, pCopies);
    if (!d->stereoEnabled) return;
    PerfScope pt(d->perf.descriptors);
    for (uint32_t i = 0; i < writeCount; ++i)
        if (pWrites[i].descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) { ++d->statInlineBlocks; d->statInlineBytes += pWrites[i].descriptorCount; }
    if (writeCount) MirrorWrites(d, writeCount, pWrites);
    if (copyCount)
    {
        const long long t0 = PerfNow();
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->perf.descWait.fetch_add(PerfNow() - t0, std::memory_order_relaxed);
        MirrorCopies(d, copyCount, pCopies);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateDescriptorUpdateTemplate(
    VkDevice device, const VkDescriptorUpdateTemplateCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDescriptorUpdateTemplate* pTemplate)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->CreateDescriptorUpdateTemplate(device, pCreateInfo, pAllocator, pTemplate);
    if (r != VK_SUCCESS) return r;
    DescriptorUpdateTemplateInfo info;
    info.entries.assign(pCreateInfo->pDescriptorUpdateEntries, pCreateInfo->pDescriptorUpdateEntries + pCreateInfo->descriptorUpdateEntryCount);
    info.layout = pCreateInfo->descriptorSetLayout;
    info.bindPoint = pCreateInfo->pipelineBindPoint;
    info.pipelineLayout = pCreateInfo->pipelineLayout;
    info.set = pCreateInfo->set;
    info.pushDescriptors = pCreateInfo->templateType == VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS;
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->templates[*pTemplate] = info;
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyDescriptorUpdateTemplate(VkDevice device, VkDescriptorUpdateTemplate t, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->templates.erase(t);
    }
    d->DestroyDescriptorUpdateTemplate(device, t, pAllocator);
}

// Templates describe raw memory; expand to writes so the shadow path is shared.
static VKAPI_ATTR void VKAPI_CALL Hook_UpdateDescriptorSetWithTemplate(
    VkDevice device, VkDescriptorSet set, VkDescriptorUpdateTemplate t, const void* pData)
{
    DeviceData* d = FindDevice(device);
    d->UpdateDescriptorSetWithTemplate(device, set, t, pData);
    if (!d->stereoEnabled) return;
    PerfScope pt(d->perf.descriptors);
    DescriptorUpdateTemplateInfo tmpl;
    {
        const long long t0 = PerfNow();
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->perf.descWait.fetch_add(PerfNow() - t0, std::memory_order_relaxed);
        ++d->statTemplateCalls;
        auto it = d->templates.find(t);
        if (it == d->templates.end()) return;
        tmpl = it->second;
    }
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<std::vector<VkDescriptorImageInfo>> imgs;
    std::vector<std::vector<VkDescriptorBufferInfo>> bufs;
    std::vector<std::vector<VkBufferView>> texels;
    imgs.reserve(tmpl.entries.size());
    bufs.reserve(tmpl.entries.size());
    texels.reserve(tmpl.entries.size());
    for (const auto& e : tmpl.entries)
    {
        VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = set;
        w.dstBinding = e.dstBinding;
        w.dstArrayElement = e.dstArrayElement;
        w.descriptorCount = e.descriptorCount;
        w.descriptorType = e.descriptorType;
        const char* base = (const char*)pData + e.offset;
        if (IsImageDescriptor(e.descriptorType) || e.descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER)
        {
            imgs.emplace_back();
            for (uint32_t k = 0; k < e.descriptorCount; ++k)
                imgs.back().push_back(*(const VkDescriptorImageInfo*)(base + k * e.stride));
            w.pImageInfo = imgs.back().data();
        }
        else if (e.descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || e.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER)
        {
            texels.emplace_back();
            for (uint32_t k = 0; k < e.descriptorCount; ++k)
                texels.back().push_back(*(const VkBufferView*)(base + k * e.stride));
            w.pTexelBufferView = texels.back().data();
        }
        else
        {
            bufs.emplace_back();
            for (uint32_t k = 0; k < e.descriptorCount; ++k)
                bufs.back().push_back(*(const VkDescriptorBufferInfo*)(base + k * e.stride));
            w.pBufferInfo = bufs.back().data();
        }
        writes.push_back(w);
    }
    MirrorWrites(d, (uint32_t)writes.size(), writes.data());
}

const HookEntry g_descriptorHooks[] = {
    WIZ3D_HOOK(CreateDescriptorSetLayout), WIZ3D_HOOK(DestroyDescriptorSetLayout),
    WIZ3D_HOOK(CreatePipelineLayout), WIZ3D_HOOK(DestroyPipelineLayout),
    WIZ3D_HOOK(AllocateDescriptorSets), WIZ3D_HOOK(FreeDescriptorSets),
    WIZ3D_HOOK(CreateDescriptorPool), WIZ3D_HOOK(ResetDescriptorPool), WIZ3D_HOOK(DestroyDescriptorPool),
    WIZ3D_HOOK(UpdateDescriptorSets),
    WIZ3D_HOOK(CreateDescriptorUpdateTemplate), WIZ3D_HOOK(DestroyDescriptorUpdateTemplate),
    WIZ3D_HOOK(UpdateDescriptorSetWithTemplate),
    { nullptr, nullptr },
};

}} // namespace wiz3d::vk
