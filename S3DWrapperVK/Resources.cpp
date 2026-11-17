/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Images that look like scene targets get a right-eye sibling with its own memory, views and framebuffers, as
// Texture2D11Proxy does on DX11; the rules are StereoHeuristic.h's, applied to VkImageCreateInfo.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"

namespace wiz3d { namespace vk {

static bool IsOneComponent32(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_R32_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT:
        return true;
    default:
        return false;
    }
}

// StereoHeuristic.h's IsSquareSize / IsLessThanBB, same knobs.
static bool IsSquareSize(uint32_t w, uint32_t h, BOOL bigInStereo, const VkExtent2D& bb)
{
    if (w != h) return false;
    if (bb.width)
    {
        if (bb.width == bb.height && w == bb.width) return false;
        if (bigInStereo && w >= bb.width && h >= bb.height) return false;
    }
    return true;
}

// A shadow atlas is a depth target whose shape is unrelated to the screen, whatever the resolution (StereoHeuristic.h).
static bool IsShadowAtlasShape(uint32_t w, uint32_t h, const VkExtent2D& bb)
{
    if (!bb.width || !bb.height || !h) return false;
    if (w == bb.width && h == bb.height) return false;
    const float aRT = (float)w / (float)h;
    const float aBB = (float)bb.width / (float)bb.height;
    const float ratio = (aRT > aBB) ? aRT / aBB : aBB / aRT;
    // 1.5 separates a square page from any widescreen backbuffer (16:9 is 1.78) and keeps scene-shaped targets inside.
    return ratio >= 1.5f;
}

static bool IsLessThanBB(uint32_t w, uint32_t h, const VkExtent2D& bb)
{
    if (!bb.width) return false;
    if (w == h) return false;
    if (w + 5 >= bb.width || h + 5 >= bb.height) return false;
    const float aRT = (float)w / h, aBB = (float)bb.width / bb.height;
    return fabsf(aRT - aBB) >= 0.01f;
}

// An aspect off the backbuffer's by more than 0.03 of it: not a scene target (shadow atlases, light maps, probes).
bool OffAspect(uint32_t w, uint32_t h, const VkExtent2D& bb)
{
    if (!bb.width || !bb.height || !h) return false;
    const float a = (float)bb.width / (float)bb.height;
    return fabsf((float)w / (float)h - a) > 0.03f * a;
}

static bool ShouldDoubleImage(const VkImageCreateInfo& ci, const VkExtent2D& bb, const char** reason)
{
    *reason = "";
    // VKDoubleFroxelVolumes: a froxel volume (16+ slices, at the backbuffer's aspect or 16:9, since the window need not
    // match the render) that the right eye fills and integrates in its own frustum.
    if (Vk().VKDoubleFroxelVolumes && ci.imageType == VK_IMAGE_TYPE_3D && (ci.usage & VK_IMAGE_USAGE_STORAGE_BIT) &&
        ci.extent.depth >= 16 && ci.mipLevels == 1 &&
        (!OffAspect(ci.extent.width, ci.extent.height, VkExtent2D{ 16, 9 }) || (bb.width && !OffAspect(ci.extent.width, ci.extent.height, bb))))
        return true;
    if (ci.imageType != VK_IMAGE_TYPE_2D) { *reason = "not2D"; return false; }
    // Image shapes a family decides (FamilyVK.h).
    bool familyDoubled = false;
    if (FamiliesImageRule(ci, bb, &familyDoubled, reason)) return familyDoubled;
    // Layered attachments are doubled (sibling image and view copy the full create-info); cubes are refused below.
    if (ci.arrayLayers != 1 && !Vk().VKDoubleArrayImages) { *reason = "array"; return false; }
    if (ci.flags & (VK_IMAGE_CREATE_SPARSE_BINDING_BIT | VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT)) { *reason = "sparseOrCube"; return false; }
    const uint32_t w = ci.extent.width, h = ci.extent.height;
    const bool depth = (ci.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
    const bool color = (ci.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) != 0;
    const bool storage = (ci.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0;
    // VKDoubleTransferImages: presenter frames (transfer-only, or also sampled when uncompressed and wide), with no
    // aspect test, as the frame's size need not follow the window's.
    const VkImageUsageFlags xfer = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const bool blockCompressed = ci.format >= VK_FORMAT_BC1_RGB_UNORM_BLOCK && ci.format <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK;
    if (Vk().VKDoubleTransferImages && ci.mipLevels == 1 && ci.arrayLayers == 1 &&
        (ci.usage == xfer || (ci.usage == (xfer | VK_IMAGE_USAGE_SAMPLED_BIT) && !blockCompressed && w >= 1280)))
        return true;
    // Usage patterns a family decides (FamilyVK.h).
    if (FamiliesImageUsageRule(ci, bb, &familyDoubled, reason)) return familyDoubled;
    if (!depth && !color && !storage) { *reason = "noTargetUsage"; return false; }
    if (gInfo.RenderTargetCreationMode == 0) { *reason = "mode0"; return false; }
    if (gInfo.RenderTargetCreationMode == 1) return true;
    if (w == 1 || h == 1) { *reason = "1pixel"; return false; }
    if (depth)
    {
        // Depth a family keeps shared: cascades fitted once for the left camera need one atlas for both eyes.
        if (FamiliesSharedDepth(ci, bb)) { *reason = "offAspectDepth"; return false; }
        // Shadow atlases: light-space constants never shift, so a sibling would be identical (StereoHeuristic.h);
        // VKShadowAtlasByShape picks them by shape, otherwise by an area of 4 backbuffers or more.
        if (Vk().VKShadowAtlasByShape
                ? IsShadowAtlasShape(w, h, bb)
                : (bb.width && (uint64_t)w * h >= 4ull * bb.width * bb.height))
        { *reason = "shadowAtlas"; return false; }
        if (gInfo.MonoDepthStencilTextures) { *reason = "monoDSFlag"; return false; }
        if (gInfo.CreateSquareDSInMono && IsSquareSize(w, h, gInfo.CreateBigSquareDSInStereo, bb)) { *reason = "squareDS"; return false; }
        if (gInfo.CreateDSThatLessThanBBInMono && IsLessThanBB(w, h, bb)) { *reason = "lessThanBB"; return false; }
        return true;
    }
    if (color)
    {
        // Colour targets a family decides (FamilyVK.h).
        if (FamiliesColourTargetRule(ci, bb, &familyDoubled, reason)) return familyDoubled;
        // Size and aspect alone do not mark scratch targets: small fixed-size targets can be part of a scene chain.
        if (IsOneComponent32(ci.format) && gInfo.CreateOneComponentRTInMono) { *reason = "oneComponentFmt"; return false; }
        if (gInfo.MonoRenderTargetTextures) { *reason = "monoRTFlag"; return false; }
        if (gInfo.CreateSquareRTInMono && IsSquareSize(w, h, gInfo.CreateBigSquareRTInStereo, bb)) { *reason = "squareRT"; return false; }
        if (gInfo.CreateRTThatLessThanBBInMono && IsLessThanBB(w, h, bb)) { *reason = "lessThanBB"; return false; }
        // VKUiTargetMaxPixels: an off-aspect target this small is a per-label UI target, composited back unshifted
        // (aspect alone cannot tell UI from scene buffers, so the size cap decides).
        if (Vk().VKUiTargetMaxPixels > 0 && (uint64_t)w * h <= (uint64_t)Vk().VKUiTargetMaxPixels
                && IsLessThanBB(w, h, bb))
        { *reason = "uiTarget"; return false; }
        return true;
    }
    // Storage-only images (compute outputs): doubled when scene-sized, so a replayed dispatch can write the right eye;
    // VKDoubleStorageImages=0 for games that replay no dispatch.
    if (!Vk().VKDoubleStorageImages) { *reason = "storageDisabled"; return false; }
    if (!bb.width) { *reason = "storageNoBB"; return false; }
    if (w * 2 < bb.width || h * 2 < bb.height) { *reason = "storageSmall"; return false; }
    // Size cannot tell a live dynamic-resolution rung from a spare one: they are fractions of a percent apart.
    return true;
}

bool ImageIsStereo(DeviceData* d, VkImage img, VkImage* sibling)
{
    auto it = d->images.find(img);
    if (it == d->images.end() || !it->second.sibling) return false;
    if (sibling) *sibling = it->second.sibling;
    return true;
}

bool ViewSibling(DeviceData* d, VkImageView view, VkImageView* sibling)
{
    auto it = d->views.find(view);
    if (it == d->views.end() || !it->second.sibling) return false;
    if (sibling) *sibling = it->second.sibling;
    return true;
}

bool ViewOfDoubledImage(DeviceData* d, VkImageView view)
{
    auto it = d->views.find(view);
    if (it == d->views.end()) return false;
    auto iit = d->images.find(it->second.image);
    return iit != d->images.end() && iit->second.sibling;
}

static const char* UsageString(VkImageUsageFlags u, char* buf, size_t n)
{
    buf[0] = 0;
    if (u & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)         strcat_s(buf, n, "color ");
    if (u & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT) strcat_s(buf, n, "depth ");
    if (u & VK_IMAGE_USAGE_SAMPLED_BIT)                  strcat_s(buf, n, "sampled ");
    if (u & VK_IMAGE_USAGE_STORAGE_BIT)                  strcat_s(buf, n, "storage ");
    if (u & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)             strcat_s(buf, n, "tsrc ");
    if (u & VK_IMAGE_USAGE_TRANSFER_DST_BIT)             strcat_s(buf, n, "tdst ");
    if (u & VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT)         strcat_s(buf, n, "input ");
    return buf;
}

// Memory
static VKAPI_ATTR VkResult VKAPI_CALL Hook_AllocateMemory(
    VkDevice device, const VkMemoryAllocateInfo* pAllocateInfo, const VkAllocationCallbacks* pAllocator, VkDeviceMemory* pMemory)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->AllocateMemory(device, pAllocateInfo, pAllocator, pMemory);
    if (r == VK_SUCCESS)
    {
        MemoryInfo mi;
        mi.size = pAllocateInfo->allocationSize;
        mi.typeIndex = pAllocateInfo->memoryTypeIndex;
        for (auto p = (const VkBaseInStructure*)pAllocateInfo->pNext; p; p = p->pNext)
        {
            if (p->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO ||
                p->sType == VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO_NV ||
                p->sType == VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR)
                mi.dedicated = true;
            if (p->sType == VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO)
                mi.allocFlags = ((const VkMemoryAllocateFlagsInfo*)p)->flags & ~VK_MEMORY_ALLOCATE_DEVICE_MASK_BIT;
        }
        d->perf.vramGame.fetch_add((long long)mi.size, std::memory_order_relaxed);
        std::lock_guard<std::shared_mutex> g(d->lock);
        ++d->statGameAllocs;
        if (mi.dedicated) ++d->statDedicatedAllocs;
        d->statGameAllocBytes += mi.size;
        d->memory[*pMemory] = mi;
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_FreeMemory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    VkDeviceMemory mirror = VK_NULL_HANDLE;
    bool ourMapping = false;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->memory.find(memory);
        if (it != d->memory.end())
        {
            mirror = it->second.mirror;
            ourMapping = it->second.layerMapped;
            d->perf.vramGame.fetch_sub((long long)it->second.size, std::memory_order_relaxed);
            if (mirror) d->perf.vramAdded.fetch_sub((long long)it->second.size, std::memory_order_relaxed);
            d->memory.erase(it);
        }
    }
    if (ourMapping) d->UnmapMemory(device, memory);   // the game never unmapped what it never mapped
    if (mirror) d->FreeMemory(device, mirror, pAllocator);
    d->FreeMemory(device, memory, pAllocator);
}

// Images
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateImage(
    VkDevice device, const VkImageCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImage* pImage)
{
    DeviceData* d = FindDevice(device);
    Breadcrumb("game CreateImage %ux%u fmt=%d usage=0x%x", pCreateInfo->extent.width, pCreateInfo->extent.height,
               (int)pCreateInfo->format, (unsigned)pCreateInfo->usage);
    // Frame generation's depth and motion copies: doubled, and readable for the squeeze at the present.
    const bool fgInput = d->stereoEnabled && FgInputImage(*pCreateInfo);
    VkImageCreateInfo fgCi;
    if (fgInput) { fgCi = *pCreateInfo; fgCi.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT; pCreateInfo = &fgCi; }
    VkResult r = d->CreateImage(device, pCreateInfo, pAllocator, pImage);
    if (r != VK_SUCCESS) return r;

    // The sibling gets the game's pNext chain verbatim: anything tying the image to external or special memory would
    // make it invalid on our own memory, so the first chains are logged.
    {
        static std::atomic<int> s_chain{24};
        if (pCreateInfo->pNext && s_chain.fetch_sub(1, std::memory_order_relaxed) > 0)
        {
            char types[160] = {}; size_t len = 0;
            for (auto p = (const VkBaseInStructure*)pCreateInfo->pNext; p; p = p->pNext)
                len += (size_t)snprintf(types + len, sizeof(types) > len ? sizeof(types) - len : 0, " %d", (int)p->sType);
            Log("[img] pNext on %ux%u fmt=%d flags=0x%x tiling=%d:%s\n", pCreateInfo->extent.width,
                pCreateInfo->extent.height, (int)pCreateInfo->format, (unsigned)pCreateInfo->flags,
                (int)pCreateInfo->tiling, types);
        }
    }
    ImageInfo info;
    info.ci = *pCreateInfo;
    info.ci.pNext = nullptr;
    for (auto p = (const VkBaseInStructure*)pCreateInfo->pNext; p; p = p->pNext)
        if (p->sType == VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_IMAGE_CREATE_INFO_NV &&
            ((const VkDedicatedAllocationImageCreateInfoNV*)p)->dedicatedAllocation)
            info.nvDedicated = true;

    const char* reason = "";
    const bool dbl = d->stereoEnabled && (fgInput || ShouldDoubleImage(*pCreateInfo, d->bbSize, &reason));
    if (dbl)
    {
        // Same description, same pNext chain (the NV dedicated flag must match the allocation made at bind time).
        VkImage sib = VK_NULL_HANDLE;
        // VKSiblingBudget: a per-game cap on siblings (0 = unlimited), for drivers that fail past a sibling count.
        static std::atomic<int> s_budget{0};
        if (Vk().VKSiblingBudget > 0 &&
            s_budget.fetch_add(1, std::memory_order_relaxed) >= Vk().VKSiblingBudget)
        {
            static bool once = false;
            if (!once) { once = true; Log("[img] sibling budget (%d) reached; further images stay mono\n", Vk().VKSiblingBudget); }
            reason = "budget";
            info.monoWhy = reason;
            std::lock_guard<std::shared_mutex> g0(d->lock);
            d->images[*pImage] = info;
            return r;
        }
        Breadcrumb("sibling CreateImage %ux%u fmt=%d", pCreateInfo->extent.width, pCreateInfo->extent.height, (int)pCreateInfo->format);
        if (d->CreateImage(device, pCreateInfo, pAllocator, &sib) == VK_SUCCESS)
        {
            info.sibling = sib;
            ++d->statImagesDoubled;
        }
        else reason = "siblingCreateFailed";
    }
    // Each row is a synchronous write under the log lock: doubled images are all logged, mono ones capped, since
    // streamed textures number in the thousands.
    static std::atomic<int> s_monoLog{2000};
    if (info.sibling || s_monoLog.fetch_sub(1, std::memory_order_relaxed) > 0)
    {
        char usage[96];
        Log("[img] %p %ux%u fmt=%d mips=%u samples=%u usage=[%s] -> %s%s\n",
            (void*)*pImage, pCreateInfo->extent.width, pCreateInfo->extent.height, pCreateInfo->format,
            pCreateInfo->mipLevels, pCreateInfo->samples, UsageString(pCreateInfo->usage, usage, sizeof(usage)),
            info.sibling ? "STEREO" : "mono", info.sibling ? "" : (reason[0] ? reason : ""));
    }
    if (!info.sibling) info.monoWhy = reason;
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->images[*pImage] = info;
    return r;
}

// Siblings outlive their game image by one more present than the swapchain can have in flight,
// so nothing frees an image the GPU is still reading.
void DrainRetiredImages(DeviceData* d)
{
    const unsigned kRetireFrames = d->swapImages ? d->swapImages + 1 : 4;
    std::vector<RetiredImage> due;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (d->retiredImages.empty()) return;
        size_t keep = 0;
        for (size_t i = 0; i < d->retiredImages.size(); ++i)
        {
            if (d->frames - d->retiredImages[i].frame >= kRetireFrames) due.push_back(d->retiredImages[i]);
            else d->retiredImages[keep++] = d->retiredImages[i];
        }
        d->retiredImages.resize(keep);
    }
    for (const RetiredImage& ri : due)
    {
        d->DestroyImage(d->device, ri.image, nullptr);
        if (ri.memory) d->FreeMemory(d->device, ri.memory, nullptr);
    }
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyImage(VkDevice device, VkImage image, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    VkImage sib = VK_NULL_HANDLE;
    VkDeviceMemory sibMem = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->images.find(image);
        if (it != d->images.end())
        {
            sib = it->second.sibling;
            if (!it->second.siblingInMirror)
            {
                sibMem = it->second.siblingMem;
                d->perf.vramAdded.fetch_sub((long long)it->second.siblingMemSize, std::memory_order_relaxed);
            }
            else
            {
                auto mt = d->memory.find(it->second.gameMem);
                if (mt != d->memory.end())
                {
                    auto& v = mt->second.stereoRanges;
                    for (size_t k = 0; k < v.size(); ++k) if (v[k].image == image) { v.erase(v.begin() + k); break; }
                }
            }
            d->images.erase(it);
        }
    }
    // Deferred, not immediate: our replay referenced this sibling and the game's
    // own retirement says nothing about when that work finishes.
    if (sib)
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->retiredImages.push_back({ sib, sibMem, d->frames });
    }
    else if (sibMem) d->FreeMemory(device, sibMem, pAllocator);
    d->DestroyImage(device, image, pAllocator);
}

// The right eye's copy of a game allocation: same size and type (and, for buffer twins, allocate
// flags), so whatever the game binds at an offset can be twinned at the same offset.
VkDeviceMemory EnsureMirror(DeviceData* d, MemoryInfo& mi)
{
    if (mi.mirror) return mi.mirror;
    VkMemoryAllocateFlagsInfo fl = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO };
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = mi.size;
    mai.memoryTypeIndex = mi.typeIndex;
    if (mi.allocFlags && Vk().VKBufferTwins) { fl.flags = mi.allocFlags; mai.pNext = &fl; }
    if (d->AllocateMemory(d->device, &mai, nullptr, &mi.mirror) != VK_SUCCESS)
    {
        mi.mirror = VK_NULL_HANDLE;
        Log("[img] mirror alloc of %llu bytes failed\n", (unsigned long long)mi.size);
        return VK_NULL_HANDLE;
    }
    ++d->statMirrors;
    d->perf.vramAdded.fetch_add((long long)mi.size, std::memory_order_relaxed);
    return mi.mirror;
}

// Binds the sibling into a mirror of the game's allocation at the same offset (aliasing-safe), or into its own
// allocation when it must; the memory type follows the game's when the sibling allows it.
static VkResult BindSibling(DeviceData* d, VkImage image, VkDeviceMemory gameMem, VkDeviceSize gameOff)
{
    ImageInfo* info;
    MemoryInfo* mi = nullptr;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->images.find(image);
        if (it == d->images.end() || !it->second.sibling || it->second.siblingMem) return VK_SUCCESS;
        info = &it->second;
        auto mt = d->memory.find(gameMem);
        if (mt != d->memory.end()) mi = &mt->second;
    }
    Breadcrumb("sibling bind %p fmt=%d off=%llu", (void*)image, (int)info->ci.format, (unsigned long long)gameOff);
    VkMemoryRequirements req;
    d->GetImageMemoryRequirements(d->device, info->sibling, &req);
    const uint32_t preferred = mi ? mi->typeIndex : UINT32_MAX;

    // An image that requires a dedicated allocation (depth/stencil, whose compression metadata the driver keeps
    // alongside) must not be bound into a mirror.
    bool needsOwn = false;
    if (d->GetImageMemoryRequirements2)
    {
        VkMemoryDedicatedRequirements ded = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS };
        VkMemoryRequirements2 req2 = { VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &ded };
        VkImageMemoryRequirementsInfo2 ii = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2 };
        ii.image = info->sibling;
        d->GetImageMemoryRequirements2(d->device, &ii, &req2);
        // Only the mandatory flag: drivers set "prefers" on nearly every render target.
        needsOwn = ded.requiresDedicatedAllocation != 0;
        req = req2.memoryRequirements;
    }

    // A game-dedicated allocation may still hold many images, so it is mirrored too; a driver that refuses the bind
    // falls through to a private allocation below.
    const bool mayAlias = !needsOwn && (!info->nvDedicated || d->dedicatedImageAliasing);
    {
        // Which of the four conditions sends a sibling to its own allocation.
        static std::atomic<int> s_log{8};
        if (s_log.load(std::memory_order_relaxed) > 0 &&
            !(mi && mayAlias && (req.memoryTypeBits & (1u << mi->typeIndex))) &&
            s_log.fetch_sub(1, std::memory_order_relaxed) > 0)
            Log("[img] dedicated path: mi=%d gameDedicated=%d nvDedicated=%d aliasingExt=%d typeBitsOk=%d (gameType=%u bits=0x%x) size=%llu\n",
                mi ? 1 : 0, mi ? (int)mi->dedicated : -1, (int)info->nvDedicated, (int)d->dedicatedImageAliasing,
                mi ? (int)((req.memoryTypeBits & (1u << mi->typeIndex)) != 0) : -1,
                mi ? mi->typeIndex : 0u, req.memoryTypeBits, (unsigned long long)req.size);
    }
    if (mi && mayAlias && (req.memoryTypeBits & (1u << mi->typeIndex)))
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (EnsureMirror(d, *mi))
        {
            VkResult r = d->BindImageMemory(d->device, info->sibling, mi->mirror, gameOff);
            if (r == VK_SUCCESS)
            {
                // Counting overlaps is a linear scan under the device lock (quadratic over a packed allocation):
                // only the first ten aliases are counted.
                if (d->statAliases < 10)
                {
                    for (const auto& rg : mi->stereoRanges)
                        if (gameOff < rg.off + rg.size && rg.off < gameOff + req.size)
                        {
                            ++d->statAliases;
                            Log("[alias] image %p at %llu+%llu overlaps %p at %llu+%llu in %p\n", (void*)image,
                                (unsigned long long)gameOff, (unsigned long long)req.size, (void*)rg.image,
                                (unsigned long long)rg.off, (unsigned long long)rg.size, (void*)gameMem);
                            break;
                        }
                    mi->stereoRanges.push_back({ gameOff, req.size, image });
                }
                info->siblingMem = mi->mirror;
                info->siblingInMirror = true;
                info->gameMem = gameMem;
                info->gameOff = gameOff;
                return VK_SUCCESS;
            }
            Log("[img] sibling bind into mirror failed %d for %p\n", r, (void*)image);
        }
    }

    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, preferred);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, 0, preferred);
    VkDedicatedAllocationMemoryAllocateInfoNV nv = { VK_STRUCTURE_TYPE_DEDICATED_ALLOCATION_MEMORY_ALLOCATE_INFO_NV };
    VkMemoryDedicatedAllocateInfo khr = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    if (info->nvDedicated) { nv.image = info->sibling; mai.pNext = &nv; }
    else { khr.image = info->sibling; mai.pNext = &khr; }
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkResult r = d->AllocateMemory(d->device, &mai, nullptr, &mem);
    if (r != VK_SUCCESS) { Log("[img] sibling alloc failed %d for %p\n", r, (void*)image); return r; }
    r = d->BindImageMemory(d->device, info->sibling, mem, 0);
    if (r != VK_SUCCESS) { Log("[img] sibling bind failed %d for %p\n", r, (void*)image); d->FreeMemory(d->device, mem, nullptr); return r; }
    d->perf.vramAdded.fetch_add((long long)req.size, std::memory_order_relaxed);
    std::lock_guard<std::shared_mutex> g(d->lock);
    ++d->statSibAllocs;
    d->statSibAllocBytes += req.size;
    info->siblingMem = mem;
    info->siblingMemSize = req.size;
    info->siblingInMirror = false;
    return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_BindImageMemory(VkDevice device, VkImage image, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->BindImageMemory(device, image, memory, memoryOffset);
    if (r == VK_SUCCESS)
    {
        {   // more than one image in an allocation means the game aliases it
            std::lock_guard<std::shared_mutex> g(d->lock);
            auto mt = d->memory.find(memory);
            if (mt != d->memory.end()) ++mt->second.imageBinds;
        }
        BindSibling(d, image, memory, memoryOffset);
    }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_BindImageMemory2(VkDevice device, uint32_t bindInfoCount, const VkBindImageMemoryInfo* pBindInfos)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->BindImageMemory2(device, bindInfoCount, pBindInfos);
    if (r == VK_SUCCESS)
    {
        {
            std::lock_guard<std::shared_mutex> g(d->lock);
            for (uint32_t i = 0; i < bindInfoCount; ++i)
            {
                auto mt = d->memory.find(pBindInfos[i].memory);
                if (mt != d->memory.end()) ++mt->second.imageBinds;
            }
        }
        for (uint32_t i = 0; i < bindInfoCount; ++i) BindSibling(d, pBindInfos[i].image, pBindInfos[i].memory, pBindInfos[i].memoryOffset);
    }
    return r;
}

// Views
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateImageView(
    VkDevice device, const VkImageViewCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkImageView* pView)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->CreateImageView(device, pCreateInfo, pAllocator, pView);
    if (r != VK_SUCCESS) return r;
    ImageViewInfo vi;
    vi.image = pCreateInfo->image;
    vi.range = pCreateInfo->subresourceRange;
    VkImage sib = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        ImageIsStereo(d, pCreateInfo->image, &sib);
        auto it = d->images.find(pCreateInfo->image);
        if (it != d->images.end())
        {
            if (vi.range.levelCount == VK_REMAINING_MIP_LEVELS) vi.range.levelCount = it->second.ci.mipLevels - vi.range.baseMipLevel;
            if (vi.range.layerCount == VK_REMAINING_ARRAY_LAYERS) vi.range.layerCount = it->second.ci.arrayLayers - vi.range.baseArrayLayer;
        }
    }
    if (sib)
    {
        VkImageViewCreateInfo ci = *pCreateInfo;
        ci.image = sib;
        if (d->CreateImageView(device, &ci, pAllocator, &vi.sibling) != VK_SUCCESS)
        {
            vi.sibling = VK_NULL_HANDLE;
            Log("[img] sibling view failed for image %p\n", (void*)pCreateInfo->image);
        }
    }
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->views[*pView] = vi;
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyImageView(VkDevice device, VkImageView view, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    VkImageView sib = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->views.find(view);
        if (it != d->views.end()) { sib = it->second.sibling; d->views.erase(it); }
    }
    if (sib) d->DestroyImageView(device, sib, pAllocator);
    d->DestroyImageView(device, view, pAllocator);
}

// Framebuffers
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateFramebuffer(
    VkDevice device, const VkFramebufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkFramebuffer* pFramebuffer)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->CreateFramebuffer(device, pCreateInfo, pAllocator, pFramebuffer);
    if (r != VK_SUCCESS) return r;
    FramebufferInfo fi;
    fi.renderPass = pCreateInfo->renderPass;
    fi.width = pCreateInfo->width; fi.height = pCreateInfo->height; fi.attachments = pCreateInfo->attachmentCount;
    if (!(pCreateInfo->flags & VK_FRAMEBUFFER_CREATE_IMAGELESS_BIT) && pCreateInfo->attachmentCount)
    {
        std::vector<VkImageView> att(pCreateInfo->pAttachments, pCreateInfo->pAttachments + pCreateInfo->attachmentCount);
        fi.views = att;
        bool any = false;
        uint32_t missing = 0;
        {
            std::lock_guard<std::shared_mutex> g(d->lock);
            for (auto& v : att)
            {
                VkImageView s;
                auto vit = d->views.find(v);
                if (ViewSibling(d, v, &s))
                {
                    v = s;
                    any = true;
                    if (vit != d->views.end())
                    {
                        fi.doubled.push_back(vit->second.image);
                        auto dit = d->images.find(vit->second.image);
                        if (dit != d->images.end() && dit->second.sibling)
                        {
                            const VkImageCreateInfo& ici = dit->second.ci;
                            VkImageSubresourceRange range = vit->second.range;
                            range.levelCount = 1;   // an attachment view has one level
                            if (range.layerCount == VK_REMAINING_ARRAY_LAYERS) range.layerCount = ici.arrayLayers - range.baseArrayLayer;
                            fi.mixed.push_back({ (uint32_t)(&v - att.data()), vit->second.image, dit->second.sibling, range, ici.extent.width, ici.extent.height });
                        }
                    }
                    continue;
                }
                // Depth a family keeps shared keeps the game's view without making the pass mono, so its colour
                // siblings are still written.
                auto iit = vit != d->views.end() ? d->images.find(vit->second.image) : d->images.end();
                if (iit == d->images.end() || !FamiliesSharedDepth(iit->second.ci, d->bbSize)) ++missing;
            }
        }
        // Every attachment or none: a mixed framebuffer keeps the game's own view
        // for whatever we refused, so the replay would draw into it twice.
        if (any && missing && Vk().VKMixedFramebufferMono)
        {
            Log("[fb] mixed framebuffer %p: %u of %u attachments undoubled, pass stays mono\n",
                (void*)*pFramebuffer, missing, (unsigned)att.size());
            fi.doubled.clear();
            any = false;
        }
        else fi.mixed.clear();
        if (any)
        {
            VkFramebufferCreateInfo ci = *pCreateInfo;
            ci.pAttachments = att.data();
            if (d->CreateFramebuffer(device, &ci, pAllocator, &fi.sibling) != VK_SUCCESS)
            {
                fi.sibling = VK_NULL_HANDLE;
                Log("[fb] sibling framebuffer failed for %p\n", (void*)*pFramebuffer);
            }
        }
    }
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->framebuffers[*pFramebuffer] = fi;
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyFramebuffer(VkDevice device, VkFramebuffer fb, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    VkFramebuffer sib = VK_NULL_HANDLE;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->framebuffers.find(fb);
        if (it != d->framebuffers.end()) { sib = it->second.sibling; d->framebuffers.erase(it); }
    }
    if (sib) d->DestroyFramebuffer(device, sib, pAllocator);
    d->DestroyFramebuffer(device, fb, pAllocator);
}

// Each render pass's attachments: what follows a classic pass copies them out of their final layouts.
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateRenderPass(VkDevice device, const VkRenderPassCreateInfo* pCreateInfo,
                                                            const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass)
{
    DeviceData* d = FindDevice(device);
    const VkResult r = d->CreateRenderPass(device, pCreateInfo, pAllocator, pRenderPass);
    if (r != VK_SUCCESS) return r;
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->renderPassAtts[*pRenderPass].assign(pCreateInfo->pAttachments, pCreateInfo->pAttachments + pCreateInfo->attachmentCount);
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateRenderPass2(VkDevice device, const VkRenderPassCreateInfo2* pCreateInfo,
                                                             const VkAllocationCallbacks* pAllocator, VkRenderPass* pRenderPass)
{
    DeviceData* d = FindDevice(device);
    const VkResult r = d->CreateRenderPass2(device, pCreateInfo, pAllocator, pRenderPass);
    if (r != VK_SUCCESS) return r;
    std::vector<VkAttachmentDescription> att(pCreateInfo->attachmentCount);
    for (uint32_t i = 0; i < pCreateInfo->attachmentCount; ++i)
    {
        const VkAttachmentDescription2& a = pCreateInfo->pAttachments[i];
        att[i] = { a.flags, a.format, a.samples, a.loadOp, a.storeOp, a.stencilLoadOp, a.stencilStoreOp, a.initialLayout, a.finalLayout };
    }
    std::lock_guard<std::shared_mutex> g(d->lock);
    d->renderPassAtts[*pRenderPass] = att;
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyRenderPass(VkDevice device, VkRenderPass pass, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->renderPassAtts.erase(pass);
    }
    d->DestroyRenderPass(device, pass, pAllocator);
}

// Swapchain images: siblings for the presentable images so the final pass has a right-eye target;
// the composite blits both into the real one.
bool RegisterSwapchainImages(DeviceData* d, VkSwapchainKHR sc)
{
    SwapchainInfo* info;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->swapchains.find(sc);
        if (it == d->swapchains.end()) return false;
        info = &it->second;
    }
    uint32_t n = 0;
    if (d->GetSwapchainImagesKHR(d->device, sc, &n, nullptr) != VK_SUCCESS || !n) return false;
    std::vector<VkImage> imgs(n);
    if (d->GetSwapchainImagesKHR(d->device, sc, &n, imgs.data()) != VK_SUCCESS) return false;

    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = info->ci.imageFormat;
    ci.extent = { info->ci.imageExtent.width, info->ci.imageExtent.height, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = info->ci.imageUsage | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    std::lock_guard<std::shared_mutex> g(d->lock);
    info->images = imgs;
    for (VkImage img : imgs)
    {
        if (d->images.count(img)) continue;
        ImageInfo ii;
        ii.ci = ci;
        ii.swapchain = true;
        if (d->stereoEnabled && d->CreateImage(d->device, &ci, nullptr, &ii.sibling) == VK_SUCCESS)
        {
            VkMemoryRequirements req;
            d->GetImageMemoryRequirements(d->device, ii.sibling, &req);
            VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
            mai.allocationSize = req.size;
            mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UINT32_MAX);
            if (d->AllocateMemory(d->device, &mai, nullptr, &ii.siblingMem) != VK_SUCCESS ||
                d->BindImageMemory(d->device, ii.sibling, ii.siblingMem, 0) != VK_SUCCESS)
            {
                Log("[img] swapchain sibling memory failed\n");
                d->DestroyImage(d->device, ii.sibling, nullptr);
                ii.sibling = VK_NULL_HANDLE;
            }
            else { ii.siblingMemSize = req.size; d->perf.vramAdded.fetch_add((long long)req.size, std::memory_order_relaxed); }
        }
        d->images[img] = ii;
        Log("[img] swapchain image %p -> %s\n", (void*)img, ii.sibling ? "STEREO" : "mono");
    }
    return true;
}

const HookEntry g_resourceHooks[] = {
    WIZ3D_HOOK(AllocateMemory), WIZ3D_HOOK(FreeMemory),
    WIZ3D_HOOK(CreateImage), WIZ3D_HOOK(DestroyImage),
    WIZ3D_HOOK(BindImageMemory), WIZ3D_HOOK(BindImageMemory2),
    WIZ3D_HOOK(CreateImageView), WIZ3D_HOOK(DestroyImageView),
    WIZ3D_HOOK(CreateFramebuffer), WIZ3D_HOOK(DestroyFramebuffer),
    WIZ3D_HOOK(CreateRenderPass), WIZ3D_HOOK(CreateRenderPass2), WIZ3D_HOOK(DestroyRenderPass),
    { nullptr, nullptr },
};

}} // namespace wiz3d::vk
