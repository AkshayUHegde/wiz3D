/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The stereo parameter sets the patched shaders read at kStereoSet: left { sep 0, conv }, right { sep, conv }, plus
// the variants below; host-coherent, rewritten at present when separation or convergence change.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include "../S3DAPI/KeyboardHook.h"
#include "../S3DAPI/ReadData.h"

namespace wiz3d { namespace vk {

uint32_t FindMemoryType(DeviceData* d, uint32_t typeBits, VkMemoryPropertyFlags want, uint32_t preferred)
{
    if (preferred < d->memProps.memoryTypeCount && (typeBits & (1u << preferred)) &&
        (d->memProps.memoryTypes[preferred].propertyFlags & want) == want)
        return preferred;
    for (uint32_t i = 0; i < d->memProps.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (d->memProps.memoryTypes[i].propertyFlags & want) == want)
            return i;
    return UINT32_MAX;
}

bool StereoParamsInit(DeviceData* d)
{
    // Binding 1: the camera-window hit counter, present only when the game lets fragments write.
    const bool counters = Vk().VKFsCameraWindows && d->fragmentAtomics;
    VkDescriptorSetLayoutBinding b[2] = {};
    b[0].binding = 0;
    b[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    b[0].descriptorCount = 1;
    b[0].stageFlags = VK_SHADER_STAGE_ALL;
    b[1].binding = 1;
    b[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b[1].descriptorCount = 1;
    b[1].stageFlags = VK_SHADER_STAGE_ALL;   // the sprite VS reads the camera's row 3 the light shaders store
    VkDescriptorSetLayoutCreateInfo lci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = counters ? 2 : 1;
    lci.pBindings = b;
    if (d->CreateDescriptorSetLayout(d->device, &lci, nullptr, &d->stereoSetLayout) != VK_SUCCESS) return false;
    lci.bindingCount = 0;
    if (d->CreateDescriptorSetLayout(d->device, &lci, nullptr, &d->emptySetLayout) != VK_SUCCESS) return false;

    // Third set: the right eye with no vertex shift, for pinned draws that still reconstruct (fsSep kept: they read the
    // shifted depth). widthSets: a pair of right-eye sets per viewport width after those three (StereoSetForWidth).
    const uint32_t setCount = Traits().widthSets ? 3 + 2 * DeviceData::kWidthSets : 3;
    VkDescriptorPoolSize ps[2] = { { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, setCount }, { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, setCount } };
    VkDescriptorPoolCreateInfo pci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = setCount;
    pci.poolSizeCount = counters ? 2 : 1;
    pci.pPoolSizes = ps;
    if (d->CreateDescriptorPool(d->device, &pci, nullptr, &d->stereoPool) != VK_SUCCESS) return false;

    std::vector<VkDescriptorSetLayout> layouts(setCount, d->stereoSetLayout);
    VkDescriptorSetAllocateInfo ai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    ai.descriptorPool = d->stereoPool;
    ai.descriptorSetCount = setCount;
    ai.pSetLayouts = layouts.data();
    std::vector<VkDescriptorSet> sets(setCount);
    if (d->AllocateDescriptorSets(d->device, &ai, sets.data()) != VK_SUCCESS) return false;
    d->stereoSetLeft = sets[0];
    d->stereoSetRight = sets[1];
    d->stereoSetRightNoShift = sets[2];
    for (uint32_t i = 0; i + 3 < setCount; i += 2)
    {
        d->stereoWidthRight[i / 2] = sets[3 + i];
        d->stereoWidthNoShift[i / 2] = sets[4 + i];
    }

    d->stereoStride = d->props.limits.minUniformBufferOffsetAlignment;
    if (d->stereoStride < 32) d->stereoStride = 32;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = d->stereoStride * setCount;
    bci.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d->CreateBuffer(d->device, &bci, nullptr, &d->stereoBuffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d->GetBufferMemoryRequirements(d->device, d->stereoBuffer, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX) return false;
    if (d->AllocateMemory(d->device, &mai, nullptr, &d->stereoMemory) != VK_SUCCESS) return false;
    if (d->BindBufferMemory(d->device, d->stereoBuffer, d->stereoMemory, 0) != VK_SUCCESS) return false;
    void* p = nullptr;
    if (d->MapMemory(d->device, d->stereoMemory, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) return false;
    d->stereoMapped = (float*)p;
    memset(p, 0, (size_t)bci.size);

    std::vector<VkDescriptorBufferInfo> bi(setCount);
    std::vector<VkWriteDescriptorSet> w(setCount);
    for (uint32_t i = 0; i < setCount; ++i)
    {
        bi[i] = { d->stereoBuffer, d->stereoStride * i, 32 };
        w[i] = {};
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = sets[i];
        w[i].dstBinding = 0;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    d->UpdateDescriptorSets(d->device, setCount, w.data(), 0, nullptr);
    if (counters)
    {
        VkBufferCreateInfo cci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        cci.size = 64;
        cci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (d->CreateBuffer(d->device, &cci, nullptr, &d->camCounterBuffer) != VK_SUCCESS) return false;
        VkMemoryRequirements creq;
        d->GetBufferMemoryRequirements(d->device, d->camCounterBuffer, &creq);
        VkMemoryAllocateInfo cai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        cai.allocationSize = creq.size;
        cai.memoryTypeIndex = FindMemoryType(d, creq.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, UINT32_MAX);
        if (cai.memoryTypeIndex == UINT32_MAX) return false;
        if (d->AllocateMemory(d->device, &cai, nullptr, &d->camCounterMemory) != VK_SUCCESS) return false;
        if (d->BindBufferMemory(d->device, d->camCounterBuffer, d->camCounterMemory, 0) != VK_SUCCESS) return false;
        void* cp = nullptr;
        if (d->MapMemory(d->device, d->camCounterMemory, 0, VK_WHOLE_SIZE, 0, &cp) != VK_SUCCESS) return false;
        d->camCounters = (uint32_t*)cp;
        memset(cp, 0, 64);
        const VkDescriptorBufferInfo cbi = { d->camCounterBuffer, 0, 64 };
        std::vector<VkWriteDescriptorSet> cw(setCount);
        for (uint32_t i = 0; i < setCount; ++i)
        {
            cw[i] = {};
            cw[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            cw[i].dstSet = sets[i];
            cw[i].dstBinding = 1;
            cw[i].descriptorCount = 1;
            cw[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            cw[i].pBufferInfo = &cbi;
        }
        d->UpdateDescriptorSets(d->device, setCount, cw.data(), 0, nullptr);
    }
    d->lastSep = -1.f;
    StereoParamsUpdate(d);
    Log("StereoParams: set layout=%p sets L=%p R=%p stride=%llu maxBoundSets=%u\n",
        (void*)d->stereoSetLayout, (void*)d->stereoSetLeft, (void*)d->stereoSetRight,
        (unsigned long long)d->stereoStride, d->props.limits.maxBoundDescriptorSets);
    return true;
}

// A right-eye set whose member 5 holds half this viewport's width. Past kWidthSets widths the plain set stays,
// and those shaders keep the left's screen position.
VkDescriptorSet StereoSetForWidth(DeviceData* d, VkDescriptorSet set, uint32_t width)
{
    if (!width || !d->stereoWidthRight[0]) return set;
    const bool right = set == d->stereoSetRight;
    if (!right && set != d->stereoSetRightNoShift) return set;
    std::lock_guard<std::mutex> g(d->stereoWidthLock);
    uint32_t i = 0;
    while (i < d->stereoWidthCount && d->stereoWidths[i] != width) ++i;
    if (i == DeviceData::kWidthSets)
    {
        static std::atomic<int> s_full{4};
        if (s_full.fetch_sub(1) > 0) Log("[rsxpos] no set left for viewport width %u: its positions stay at the left's\n", width);
        return set;
    }
    if (i == d->stereoWidthCount)
    {
        d->stereoWidths[d->stereoWidthCount++] = width;
        const float half = width * 0.5f;
        memcpy((char*)d->stereoMapped + d->stereoStride * (3 + 2 * i) + 5 * sizeof(float), &half, sizeof(half));
        memcpy((char*)d->stereoMapped + d->stereoStride * (4 + 2 * i) + 5 * sizeof(float), &half, sizeof(half));
        Log("[rsxpos] viewport width %u takes right-eye sets %u\n", width, i);
    }
    return right ? d->stereoWidthRight[i] : d->stereoWidthNoShift[i];
}

void StereoParamsDestroy(DeviceData* d)
{
    for (DeviceData::RcbStash& e : d->rcbStashes)
    {
        for (int k = 0; k < DeviceData::RcbStash::kSlots; ++k)
        {
            if (e.img[k]) d->DestroyImage(d->device, e.img[k], nullptr);
            if (e.mem[k]) d->FreeMemory(d->device, e.mem[k], nullptr);
        }
        if (e.mid) d->DestroyImage(d->device, e.mid, nullptr);
        if (e.midMem) d->FreeMemory(d->device, e.midMem, nullptr);
        if (e.low) d->DestroyImage(d->device, e.low, nullptr);
        if (e.lowMem) d->FreeMemory(d->device, e.lowMem, nullptr);
        if (e.probeMap) d->UnmapMemory(d->device, e.probeMem);
        if (e.probeBuf) d->DestroyBuffer(d->device, e.probeBuf, nullptr);
        if (e.probeMem) d->FreeMemory(d->device, e.probeMem, nullptr);
    }
    d->rcbStashes.clear();
    if (d->camCounters) { d->UnmapMemory(d->device, d->camCounterMemory); d->camCounters = nullptr; }
    if (d->camCounterBuffer) d->DestroyBuffer(d->device, d->camCounterBuffer, nullptr);
    if (d->camCounterMemory) d->FreeMemory(d->device, d->camCounterMemory, nullptr);
    d->camCounterBuffer = VK_NULL_HANDLE; d->camCounterMemory = VK_NULL_HANDLE;
    if (d->stereoMapped) { d->UnmapMemory(d->device, d->stereoMemory); d->stereoMapped = nullptr; }
    if (d->stereoBuffer) d->DestroyBuffer(d->device, d->stereoBuffer, nullptr);
    if (d->stereoMemory) d->FreeMemory(d->device, d->stereoMemory, nullptr);
    if (d->stereoPool) d->DestroyDescriptorPool(d->device, d->stereoPool, nullptr);
    if (d->stereoSetLayout) d->DestroyDescriptorSetLayout(d->device, d->stereoSetLayout, nullptr);
    if (d->emptySetLayout) d->DestroyDescriptorSetLayout(d->device, d->emptySetLayout, nullptr);
    d->stereoBuffer = VK_NULL_HANDLE; d->stereoMemory = VK_NULL_HANDLE; d->stereoPool = VK_NULL_HANDLE;
    d->stereoSetLayout = VK_NULL_HANDLE; d->emptySetLayout = VK_NULL_HANDLE;
}

// The DX11 shift CB's two numbers (Context11Proxy::EnsureStereoShiftCBs): sep in clip units per unit w, and conv as
// the w of zero shift. sep is scaled by the camera's _11 once one is found, by VKSeparationScale until then.
void StereoParamsUpdate(DeviceData* d)
{
    if (!d->stereoMapped) return;
    // One snapshot of what the hotkey thread writes, for this present.
    gKbdHook.m_Access.Enter();
    DataInput now = gInfo.Input;
    const float eyeShift = wiz3D_GetEffectiveEyeShift();
    gKbdHook.m_Access.Leave();
    const CameraPreset* p = now.GetActivePreset();
    // DX11's StereoConvergence(): 1/One_div_ZPS at either sign, and 0 at 0, where the eye shift is 0 as well.
    const float conv = (p && p->One_div_ZPS != 0.f) ? 1.f / p->One_div_ZPS : 0.f;
    // The shaders apply the forward shift per draw and the constants path corrects the inverses: both need the same
    // clip-space offset, so sep carries the projection's _11 (DX11's camXScale).
    const float p11 = d->cameraP11.load(std::memory_order_relaxed);
    const float scale = (p11 > 0.f) ? p11 : Vk().VKSeparationScale;
    const float sep = now.StereoActive ? eyeShift * scale : 0.f;
    // World-unit eye offset (the constants path's eyeSep): patched fragment shaders add eye*right to
    // camera-relative positions. None where sep is none.
    const float eye = (now.StereoActive && p && conv != 0.f) ? p->StereoBase * gInfo.SeparationScale * Vk().VKSeparationScale : 0.f;
    // DX11's PersistSettings: a changed input is written to UserProfile.xml five seconds after its last change, so a
    // game that is killed or crashes keeps it too.
    static DataInput s_saved = now;
    static ULONGLONG s_writeAt = 0;
    const ULONGLONG tick = GetTickCount64();
    if (s_saved.IsChanged(now))
    {
        s_saved = now;
        s_writeAt = tick + 5000;
    }
    else if (s_writeAt && tick >= s_writeAt)
    {
        s_writeAt = 0;
        WriteInputData(&s_saved);
        Log("StereoParams: saved StereoBase=%.6f One_div_ZPS=%.6f to UserProfile.xml\n", p ? p->StereoBase : 0.f, p ? p->One_div_ZPS : 0.f);
    }
    if (sep == d->lastSep && conv == d->lastConv && eye == d->lastEye) return;
    d->lastSep = sep;
    d->lastConv = conv;
    d->lastEye = eye;
    // VKNearClamp: everything nearer than wMin (a fraction of conv) gets wMin's parallax, so a first-person weapon
    // centimetres from the eye stops diverging while the world is untouched. A convergence behind the eye has none.
    const float wMin = (Vk().VKNearClamp > 0.f && conv > 0.f) ? Vk().VKNearClamp * conv : 0.f;
    float* left = d->stereoMapped;
    float* right = (float*)((char*)d->stereoMapped + d->stereoStride);
    float* rightNoShift = (float*)((char*)d->stereoMapped + d->stereoStride * 2);
    // [4] fsSep: the shift the depth being read was drawn with, which pinned full-screen quads still read.
    left[0] = 0.f;  left[1] = conv;  left[2] = wMin;  left[3] = 0.f; left[4] = 0.f;
    right[0] = sep; right[1] = conv; right[2] = wMin; right[3] = eye; right[4] = sep;
    rightNoShift[0] = 0.f; rightNoShift[1] = conv; rightNoShift[2] = wMin; rightNoShift[3] = eye; rightNoShift[4] = sep;
    for (uint32_t i = 0; d->stereoWidthRight[0] && i < DeviceData::kWidthSets; ++i)
    {
        memcpy((char*)d->stereoMapped + d->stereoStride * (3 + 2 * i), right, 5 * sizeof(float));
        memcpy((char*)d->stereoMapped + d->stereoStride * (4 + 2 * i), rightNoShift, 5 * sizeof(float));
    }
    Log("StereoParams: sep=%.6f conv=%.4f wMin=%.4f eye=%.4f (eyeShift=%.6f scale=%.3f active=%d)\n",
        sep, conv, wMin, eye, eyeShift, Vk().VKSeparationScale, (int)now.StereoActive);
}

}} // namespace wiz3d::vk
