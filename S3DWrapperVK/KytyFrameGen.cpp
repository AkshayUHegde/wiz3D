/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// KytyPS5 (Demon's Souls) with DLSS-G over Streamline, which sits above the layer: at its present the fake swapchain
// image becomes two squeezed views (depth and motion too), so one generated image carries both eyes (KYTY_FRAMEGEN).

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include <MinHook.h>
#include <atomic>
#include <mutex>
#include <stdlib.h>
#include <string.h>
#include <vector>

namespace wiz3d { namespace vk {

thread_local int t_foreignCalls = 0;

// DLSS-G submits from Streamline's own thread under its lock, the compose from inside its present hook without it:
// while DLSS-G runs, every queue call the layer makes takes this lock, so no two threads share a queue.
static std::mutex s_queueLock;

namespace {

// Streamline 2.14's layouts (sl_struct.h, sl_core_types.h), as S3DWrapper12/Streamline12.h declares them.
struct SlStructType { uint32_t a; uint16_t b, c; uint8_t d[8]; };
struct SlBase { SlBase* next; SlStructType structType; size_t structVersion; };
struct SlResource : SlBase
{
    uint8_t  type;
    void*    native;
    void*    memory;
    void*    view;
    uint32_t state, width, height, nativeFormat, mipLevels, arrayLayers;
    uint64_t gpuVirtualAddress;
    uint32_t flags, usage;
    uint16_t internalFlags, reserved;
};
struct SlExtent { uint32_t top, left, width, height; };
struct SlResourceTag : SlBase { SlResource* resource; uint32_t type; uint32_t lifecycle; SlExtent extent; };
struct SlViewport : SlBase { uint32_t value; };

const uint32_t kTagDepth = 0, kTagMotion = 1, kTagHudless = 2, kTagUi = 23, kTagBackbuffer = 53;
const uint32_t kFgViewport = 0;         // KytyPS5's frame-gen.cpp: DLSS-G's viewport; DLSS super resolution has 1
const size_t   kMvecScaleOffset = 360;  // sl::Constants 2.x: cameraViewToClip @32, jitter @352, mvecScale @360
const size_t   kDlssgModeOffset = 32;   // sl::DLSSGOptions: mode @32, numFramesToGenerate @36

typedef int (*PFN_SetTag)(const void*, const SlViewport*, const SlResourceTag*, uint32_t, void*);
typedef int (*PFN_SetConstants)(const void*, const void*, const SlViewport*);
typedef int (*PFN_Evaluate)(uint32_t, const void*, const void**, uint32_t, void*);
typedef int (*PFN_GetFeatureFunction)(uint32_t, const char*, void**);
typedef int (*PFN_DlssgSetOptions)(const SlViewport*, const uint8_t*);

PFN_SetTag                 s_origTag = nullptr;
PFN_SetConstants           s_origConstants = nullptr;
PFN_Evaluate               s_origEvaluate = nullptr;
PFN_vkQueuePresentKHR      s_origPresent = nullptr;
PFN_vkGetSwapchainImagesKHR s_getImages = nullptr;
PFN_GetFeatureFunction     s_origGetFeature = nullptr;
PFN_DlssgSetOptions        s_origDlssgOptions = nullptr;

bool s_on = false;              // DLSS-G asked for, Streamline loaded, hooks in
bool s_streamline = false;      // Streamline loaded at all: DLSS alone too
unsigned s_generated = 0;
std::atomic<uint32_t> s_fgViewport{ kFgViewport };   // the viewport the game's DLSS-G options name

// One squeezed input per frame in flight: DLSS-G reads them on its own thread after the present returns.
const uint32_t kSlots = 6;
struct SbsImage { VkImage image = VK_NULL_HANDLE; VkDeviceMemory mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };
struct SbsSet { VkFormat format = VK_FORMAT_UNDEFINED; uint32_t w = 0, h = 0; SbsImage slot[kSlots]; };
struct Tagged { VkImage left = VK_NULL_HANDLE; VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED; VkImage out = VK_NULL_HANDLE; VkFormat format = VK_FORMAT_UNDEFINED; uint32_t w = 0, h = 0; };

struct FgFrame { VkCommandBuffer cb = VK_NULL_HANDLE; VkFence fence = VK_NULL_HANDLE; VkSemaphore done = VK_NULL_HANDLE; };

struct FgState
{
    DeviceData* d = nullptr;
    DWORD presentThread = 0;
    SbsSet sets[2];              // depth, motion
    Tagged tagged[2];
    uint32_t slot = 0;
    SlResource resources[2];     // what the swapped tags point at, for the call
    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> fake;
    VkImage scratch = VK_NULL_HANDLE;
    VkDeviceMemory scratchMem = VK_NULL_HANDLE;
    VkFormat scratchFormat = VK_FORMAT_UNDEFINED;
    uint32_t scratchW = 0, scratchH = 0;
    VkCommandPool pool = VK_NULL_HANDLE;
    uint32_t poolFamily = UINT32_MAX;
    FgFrame frames[kSlots];
    uint32_t frameIndex = 0;
    VkRect2D region = {};        // the game's picture in the swapchain image: the backbuffer tag's, else all of it
    bool regionTagged = false;
    uint64_t composed = 0, tagsSwapped = 0, noSibling = 0;
};
FgState s_fg;

VkImageAspectFlags BlitAspect(VkFormat f)
{
    return (f == VK_FORMAT_D32_SFLOAT || f == VK_FORMAT_D32_SFLOAT_S8_UINT) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
}

VkImageAspectFlags BarrierAspect(VkFormat f)
{
    return f == VK_FORMAT_D32_SFLOAT_S8_UINT ? (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : BlitAspect(f);
}

VkImageMemoryBarrier Barrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA, VkFormat f = VK_FORMAT_UNDEFINED)
{
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = srcA;
    b.dstAccessMask = dstA;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = { BarrierAspect(f), 0, 1, 0, 1 };
    return b;
}

bool MakeImage(DeviceData* d, VkFormat format, uint32_t w, uint32_t h, VkImageUsageFlags usage, VkImage* image, VkDeviceMemory* mem)
{
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = format;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d->CreateImage(d->device, &ci, nullptr, image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d->GetImageMemoryRequirements(d->device, *image, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->device, &mai, nullptr, mem) != VK_SUCCESS || d->BindImageMemory(d->device, *image, *mem, 0) != VK_SUCCESS)
    {
        Log("[fg] image %ux%u fmt=%d: no memory\n", w, h, (int)format);
        return false;
    }
    return true;
}

// The slot's squeezed copy of a tagged input, made when its size or format first appears.
const SbsImage* SqueezedInput(DeviceData* d, uint32_t which, VkFormat format, uint32_t w, uint32_t h)
{
    SbsSet& set = s_fg.sets[which];
    if (set.format != format || set.w != w || set.h != h)
    {
        if (set.format != VK_FORMAT_UNDEFINED) d->DeviceWaitIdle(d->device);
        for (SbsImage& s : set.slot)
        {
            if (s.view) d->DestroyImageView(d->device, s.view, nullptr);
            if (s.image) d->DestroyImage(d->device, s.image, nullptr);
            if (s.mem) d->FreeMemory(d->device, s.mem, nullptr);
            s = SbsImage();
        }
        set.format = format; set.w = w; set.h = h;
        for (SbsImage& s : set.slot)
        {
            if (!MakeImage(d, format, w, h, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, &s.image, &s.mem)) return nullptr;
            VkImageViewCreateInfo vi = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
            vi.image = s.image;
            vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
            vi.format = format;
            vi.subresourceRange = { BlitAspect(format), 0, 1, 0, 1 };
            if (d->CreateImageView(d->device, &vi, nullptr, &s.view) != VK_SUCCESS) return nullptr;
        }
        Log("[fg] squeezed %s: %ux%u fmt=%d x%u\n", which ? "motion" : "depth", w, h, (int)format, kSlots);
    }
    return &set.slot[s_fg.slot % kSlots];
}

// Vectors in UV units cross half the image's width in a squeezed view. Only DLSS-G's: DLSS has its own viewport.
int Hooked_slSetConstants(const void* values, const void* frame, const SlViewport* viewport)
{
    if (!s_on || !values || !viewport || viewport->value != s_fgViewport) return s_origConstants(values, frame, viewport);
    float* scale = (float*)((char*)values + kMvecScaleOffset);
    const float x = scale[0];
    scale[0] = x * 0.5f;
    const int r = s_origConstants(values, frame, viewport);
    scale[0] = x;
    return r;
}

int Hooked_slSetTagForFrame(const void* frame, const SlViewport* viewport, const SlResourceTag* tags, uint32_t count, void* cb)
{
    DeviceData* d = s_fg.d;
    if (!s_on || !d || !viewport || viewport->value != s_fgViewport || !tags || count > 8) return s_origTag(frame, viewport, tags, count, cb);
    SlResourceTag local[8];
    memcpy(local, tags, sizeof(SlResourceTag) * count);
    for (uint32_t i = 0; i < count; ++i)
    {
        SlResourceTag& t = local[i];
        if (t.type == kTagBackbuffer)
        {
            s_fg.region = { { (int32_t)t.extent.left, (int32_t)t.extent.top }, { t.extent.width, t.extent.height } };
            s_fg.regionTagged = t.extent.width && t.extent.height;
        }
        // Mono images over two squeezed views would be wrong in both (tagged only with KYTY_FG_HUDLESS=1).
        if (t.type == kTagHudless || t.type == kTagUi) t.resource = nullptr;
        if (t.type != kTagDepth && t.type != kTagMotion) continue;
        const uint32_t which = t.type == kTagMotion ? 1 : 0;
        s_fg.tagged[which] = Tagged();
        if (!t.resource || !t.resource->native) continue;
        const VkFormat format = (VkFormat)t.resource->nativeFormat;
        const SbsImage* out = SqueezedInput(d, which, format, t.resource->width, t.resource->height);
        if (!out) continue;
        Tagged& tagged = s_fg.tagged[which];
        tagged.left = (VkImage)t.resource->native;
        tagged.layout = (VkImageLayout)t.resource->state;
        tagged.out = out->image;
        tagged.format = format;
        tagged.w = t.resource->width;
        tagged.h = t.resource->height;
        s_fg.resources[which] = *t.resource;
        s_fg.resources[which].native = out->image;
        s_fg.resources[which].view = out->view;
        s_fg.resources[which].state = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        s_fg.resources[which].usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        t.resource = &s_fg.resources[which];
        ++s_fg.tagsSwapped;
    }
    return s_origTag(frame, viewport, local, count, cb);
}

// NGX records DLSS into the emulator's command buffer: its own dispatches are not the game's to run per eye.
int Hooked_slEvaluateFeature(uint32_t feature, const void* frame, const void** inputs, uint32_t count, void* cb)
{
    ++t_foreignCalls;
    const int r = s_origEvaluate(feature, frame, inputs, count, cb);
    --t_foreignCalls;
    return r;
}

bool EnsureFrames(DeviceData* d, VkQueue queue)
{
    uint32_t family = 0;
    {
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto it = d->queueFamilies.find(queue);
        if (it != d->queueFamilies.end()) family = it->second;
    }
    if (s_fg.pool && s_fg.poolFamily == family) return true;
    if (s_fg.pool) return false;   // one present queue so far; another would need its own pool, and the log says so
    VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family;
    if (d->CreateCommandPool(d->device, &pci, nullptr, &s_fg.pool) != VK_SUCCESS) return false;
    s_fg.poolFamily = family;
    for (FgFrame& f : s_fg.frames)
    {
        VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = s_fg.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        if (d->AllocateCommandBuffers(d->device, &ai, &f.cb) != VK_SUCCESS) return false;
        VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        if (d->CreateFence(d->device, &fi, nullptr, &f.fence) != VK_SUCCESS) return false;
        VkSemaphoreCreateInfo si = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        if (d->CreateSemaphore(d->device, &si, nullptr, &f.done) != VK_SUCCESS) return false;
    }
    return true;
}

// The fake image the emulator just finished, as two squeezed views; its tagged depth and motion likewise.
bool Compose(DeviceData* d, VkQueue queue, const VkPresentInfoKHR* pi, VkSemaphore* outDone)
{
    if (pi->swapchainCount != 1 || !s_getImages) return false;
    const uint32_t idx = pi->pImageIndices[0];
    VkImage img = VK_NULL_HANDLE;
    VkImage sib = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t W = 0, H = 0;
    VkImage depthR = VK_NULL_HANDLE, motionR = VK_NULL_HANDLE;
    {
        // Streamline reuses the handle when the window changes size: a swapchain made since, or an image the layer no
        // longer has, reads the image list again.
        auto known = [&](VkImage image)
        {
            std::shared_lock<std::shared_mutex> g(d->lock);
            auto it = d->images.find(image);
            return it != d->images.end() && it->second.sibling;
        };
        if (s_fg.swapchain != pi->pSwapchains[0] || idx >= s_fg.fake.size() || !known(s_fg.fake[idx]))
        {
            uint32_t n = 0;
            s_getImages(d->device, pi->pSwapchains[0], &n, nullptr);
            s_fg.fake.resize(n);
            s_getImages(d->device, pi->pSwapchains[0], &n, s_fg.fake.data());
            s_fg.swapchain = pi->pSwapchains[0];
            Log("[fg] swapchain %p: %u images the emulator draws into\n", (void*)s_fg.swapchain, n);
        }
        if (idx >= s_fg.fake.size()) return false;
        img = s_fg.fake[idx];
        std::shared_lock<std::shared_mutex> g(d->lock);
        auto it = d->images.find(img);
        if (it == d->images.end() || !it->second.sibling) { if (++s_fg.noSibling <= 3) Log("[fg] image %p has no right eye: presented as it is\n", (void*)img); return false; }
        sib = it->second.sibling;
        format = it->second.ci.format;
        W = it->second.ci.extent.width;
        H = it->second.ci.extent.height;
        auto dt = s_fg.tagged[0].left ? d->images.find(s_fg.tagged[0].left) : d->images.end();
        if (dt != d->images.end()) depthR = dt->second.sibling;
        auto mt = s_fg.tagged[1].left ? d->images.find(s_fg.tagged[1].left) : d->images.end();
        if (mt != d->images.end()) motionR = mt->second.sibling;
    }
    if (!EnsureFrames(d, queue)) return false;
    if (s_fg.scratchFormat != format || s_fg.scratchW != W || s_fg.scratchH != H)
    {
        if (s_fg.scratch)
        {
            d->DeviceWaitIdle(d->device);
            d->DestroyImage(d->device, s_fg.scratch, nullptr);
            d->FreeMemory(d->device, s_fg.scratchMem, nullptr);
            s_fg.scratch = VK_NULL_HANDLE;
            s_fg.scratchMem = VK_NULL_HANDLE;
        }
        if (!MakeImage(d, format, W, H, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, &s_fg.scratch, &s_fg.scratchMem)) { s_fg.scratchFormat = VK_FORMAT_UNDEFINED; return false; }
        s_fg.scratchFormat = format; s_fg.scratchW = W; s_fg.scratchH = H;
        Log("[fg] composing %ux%u fmt=%d: both eyes in one image, for DLSS-G\n", W, H, (int)format);
    }
    FgFrame& f = s_fg.frames[s_fg.frameIndex++ % kSlots];
    d->WaitForFences(d->device, 1, &f.fence, VK_TRUE, UINT64_MAX);
    d->ResetFences(d->device, 1, &f.fence);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    d->BeginCommandBuffer(f.cb, &bi);

    VkRect2D r = { { 0, 0 }, { W, H } };
    if (s_fg.regionTagged && (uint32_t)s_fg.region.offset.x + s_fg.region.extent.width <= W && (uint32_t)s_fg.region.offset.y + s_fg.region.extent.height <= H) r = s_fg.region;
    VkImageMemoryBarrier pre[3] = {
        Barrier(img, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
        Barrier(sib, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
        Barrier(s_fg.scratch, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT),
    };
    d->CmdPipelineBarrier(f.cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 3, pre);
    VkImage left = img, right = gInfo.Input.StereoActive ? sib : img;
    if (gInfo.Input.SwapEyes && gInfo.Input.StereoActive) { VkImage t = left; left = right; right = t; }
    const int32_t x0 = r.offset.x, y0 = r.offset.y, rw = (int32_t)r.extent.width, rh = (int32_t)r.extent.height;
    VkImageBlit blit = {};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstSubresource = blit.srcSubresource;
    blit.srcOffsets[0] = { x0, y0, 0 };
    blit.srcOffsets[1] = { x0 + rw, y0 + rh, 1 };
    blit.dstOffsets[0] = { x0, y0, 0 };
    blit.dstOffsets[1] = { x0 + rw / 2, y0 + rh, 1 };
    d->CmdBlitImage(f.cb, left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_fg.scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    blit.dstOffsets[0] = { x0 + rw / 2, y0, 0 };
    blit.dstOffsets[1] = { x0 + rw, y0 + rh, 1 };
    d->CmdBlitImage(f.cb, right, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_fg.scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
    VkImageMemoryBarrier mid[2] = {
        Barrier(s_fg.scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
        Barrier(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT),
    };
    d->CmdPipelineBarrier(f.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, mid);
    VkImageCopy copy = {};
    copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.dstSubresource = copy.srcSubresource;
    copy.srcOffset = { x0, y0, 0 };
    copy.dstOffset = copy.srcOffset;
    copy.extent = { (uint32_t)rw, (uint32_t)rh, 1 };
    d->CmdCopyImage(f.cb, s_fg.scratch, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    VkImageMemoryBarrier post[2] = {
        Barrier(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT),
        Barrier(sib, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0),
    };
    d->CmdPipelineBarrier(f.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, post);

    // Depth and motion: a right eye the copies mirrored, else the left in both views rather than an empty half.
    const VkImage inputR[2] = { depthR, motionR };
    for (uint32_t i = 0; i < 2; ++i)
    {
        const Tagged& t = s_fg.tagged[i];
        if (!t.left || !t.out) continue;
        const VkImage rightIn = inputR[i] ? inputR[i] : t.left;
        std::vector<VkImageMemoryBarrier> in;
        in.push_back(Barrier(t.left, t.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, t.format));
        if (rightIn != t.left) in.push_back(Barrier(rightIn, t.layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, t.format));
        in.push_back(Barrier(t.out, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT, t.format));
        d->CmdPipelineBarrier(f.cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, (uint32_t)in.size(), in.data());
        VkImageBlit b = {};
        b.srcSubresource = { BlitAspect(t.format), 0, 0, 1 };
        b.dstSubresource = b.srcSubresource;
        b.srcOffsets[1] = { (int32_t)t.w, (int32_t)t.h, 1 };
        b.dstOffsets[1] = { (int32_t)t.w / 2, (int32_t)t.h, 1 };
        d->CmdBlitImage(f.cb, t.left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_NEAREST);
        b.dstOffsets[0] = { (int32_t)t.w / 2, 0, 0 };
        b.dstOffsets[1] = { (int32_t)t.w, (int32_t)t.h, 1 };
        d->CmdBlitImage(f.cb, rightIn, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_NEAREST);
        std::vector<VkImageMemoryBarrier> out;
        out.push_back(Barrier(t.left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.layout, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, t.format));
        if (rightIn != t.left) out.push_back(Barrier(rightIn, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.layout, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_SHADER_READ_BIT, t.format));
        out.push_back(Barrier(t.out, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, t.format));
        d->CmdPipelineBarrier(f.cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, (uint32_t)out.size(), out.data());
    }
    d->EndCommandBuffer(f.cb);

    std::vector<VkPipelineStageFlags> stages(pi->waitSemaphoreCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = pi->waitSemaphoreCount;
    si.pWaitSemaphores = pi->pWaitSemaphores;
    si.pWaitDstStageMask = stages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &f.cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &f.done;
    VkResult r2;
    { const auto ql = FgQueueLock(); r2 = d->QueueSubmit(queue, 1, &si, f.fence); }
    if (r2 != VK_SUCCESS) { Log("[fg] compose submit failed %d\n", (int)r2); return false; }
    *outDone = f.done;
    return true;
}

VKAPI_ATTR VkResult VKAPI_CALL Hooked_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pi)
{
    DeviceData* d = s_fg.d;
    if (!s_on || !d || !d->stereoEnabled || !pi) return s_origPresent(queue, pi);
    s_fg.presentThread = GetCurrentThreadId();
    VkPresentInfoKHR info = *pi;
    VkSemaphore done = VK_NULL_HANDLE;
    if (Compose(d, queue, pi, &done))
    {
        info.waitSemaphoreCount = 1;
        info.pWaitSemaphores = &done;
        const uint64_t n = ++s_fg.composed;
        if (n <= 3 || n % 3600 == 0)
            Log("[fg] composed %llu frames, tagged inputs swapped %llu; right inputs depth=%d motion=%d\n", (unsigned long long)n, (unsigned long long)s_fg.tagsSwapped,
                (int)(s_fg.tagged[0].left != VK_NULL_HANDLE), (int)(s_fg.tagged[1].left != VK_NULL_HANDLE));
    }
    ++s_fg.slot;
    s_fg.tagged[0] = Tagged();
    s_fg.tagged[1] = Tagged();
    return s_origPresent(queue, &info);
}

bool HookExport(HMODULE m, const char* name, void* detour, void** orig)
{
    void* target = (void*)GetProcAddress(m, name);
    if (!target) { Log("[fg] %s: not exported\n", name); return false; }
    if (MH_CreateHook(target, detour, orig) != MH_OK || MH_EnableHook(target) != MH_OK) { Log("[fg] %s: hook failed\n", name); return false; }
    return true;
}

// The game's DLSS-G options: the viewport its tags and constants use, and its mode and count for the log.
int Hooked_DlssgSetOptions(const SlViewport* viewport, const uint8_t* options)
{
    static uint32_t s_seen[3] = { ~0u, ~0u, ~0u };
    if (viewport && options)
    {
        const uint32_t mode = *(const uint32_t*)(options + kDlssgModeOffset), frames = *(const uint32_t*)(options + kDlssgModeOffset + 4);
        s_fgViewport = viewport->value;
        if (mode != s_seen[0] || frames != s_seen[1] || viewport->value != s_seen[2])
        {
            s_seen[0] = mode; s_seen[1] = frames; s_seen[2] = viewport->value;
            Log("[fg] the game's DLSS-G options: viewport %u, mode %u (0 off, 1 on, 2 auto, 3 dynamic), %u generated per frame\n",
                viewport->value, mode, frames);
        }
    }
    return s_origDlssgOptions(viewport, options);
}

// DLSS-G's own functions come from slGetFeatureFunction, not from exports: the options setter is wrapped there.
int Hooked_slGetFeatureFunction(uint32_t feature, const char* name, void** fn)
{
    const int r = s_origGetFeature(feature, name, fn);
    if (r == 0 && fn && *fn && name && !strcmp(name, "slDLSSGSetOptions") && *fn != (void*)&Hooked_DlssgSetOptions)
    {
        s_origDlssgOptions = (PFN_DlssgSetOptions)*fn;
        *fn = (void*)&Hooked_DlssgSetOptions;
    }
    return r;
}

} // namespace

void FgInit(DeviceData* d)
{
    if (!Vk().VKFrameGen) return;
    HMODULE sl = GetModuleHandleW(L"sl.interposer.dll");
    if (!sl) return;
    s_streamline = true;
    if (MH_Initialize() != MH_OK && MH_Initialize() != MH_ERROR_ALREADY_INITIALIZED) Log("[fg] MH_Initialize failed (continuing)\n");
    HookExport(sl, "slEvaluateFeature", (void*)Hooked_slEvaluateFeature, (void**)&s_origEvaluate);
    HookExport(sl, "slGetFeatureFunction", (void*)Hooked_slGetFeatureFunction, (void**)&s_origGetFeature);
    const char* framegen = getenv("KYTY_FRAMEGEN");
    const char* backend = getenv("KYTY_FRAMEGEN_BACKEND");
    // Without the emulator's count, a loaded DLSS-G plugin is frame generation, counted at its 4x most so that sibling
    // retirement never runs short; the game's own count is logged when it sets its options.
    const bool dlssg = GetModuleHandleW(L"sl.dlss_g.dll") != nullptr;
    const int generated = framegen ? atoi(framegen) : dlssg ? 3 : 0;
    Log("[fg] Streamline above the layer: KYTY_FRAMEGEN=%s backend=%s, sl.dlss_g %s\n", framegen ? framegen : "-",
        backend ? backend : "-", dlssg ? "loaded" : "not loaded");
    if (generated <= 0 || (backend && !strcmp(backend, "fsr"))) return;
    s_getImages = (PFN_vkGetSwapchainImagesKHR)GetProcAddress(sl, "vkGetSwapchainImagesKHR");
    s_fg.d = d;
    s_generated = (unsigned)generated;
    if (!HookExport(sl, "slSetTagForFrame", (void*)Hooked_slSetTagForFrame, (void**)&s_origTag) ||
        !HookExport(sl, "slSetConstants", (void*)Hooked_slSetConstants, (void**)&s_origConstants) ||
        !HookExport(sl, "vkQueuePresentKHR", (void*)Hooked_QueuePresentKHR, (void**)&s_origPresent) || !s_getImages)
    {
        Log("[fg] hooks incomplete: frame generation stays mono-only\n");
        return;
    }
    s_on = true;
    Log("[fg] DLSS-G: both eyes squeezed into one image, %s%u generated per frame\n", framegen ? "" : "at most ", s_generated);
}

void FgDestroy(DeviceData* d)
{
    if (!s_fg.d || s_fg.d != d) return;
    s_on = false;
    for (SbsSet& set : s_fg.sets)
        for (SbsImage& s : set.slot)
        {
            if (s.view) d->DestroyImageView(d->device, s.view, nullptr);
            if (s.image) d->DestroyImage(d->device, s.image, nullptr);
            if (s.mem) d->FreeMemory(d->device, s.mem, nullptr);
        }
    if (s_fg.scratch) d->DestroyImage(d->device, s_fg.scratch, nullptr);
    if (s_fg.scratchMem) d->FreeMemory(d->device, s_fg.scratchMem, nullptr);
    for (FgFrame& f : s_fg.frames)
    {
        if (f.fence) d->DestroyFence(d->device, f.fence, nullptr);
        if (f.done) d->DestroySemaphore(d->device, f.done, nullptr);
    }
    if (s_fg.pool) d->DestroyCommandPool(d->device, s_fg.pool, nullptr);
    s_fg = FgState();
}

bool FgOn() { return s_on; }

std::unique_lock<std::mutex> FgQueueLock() { return s_on ? std::unique_lock<std::mutex>(s_queueLock) : std::unique_lock<std::mutex>(); }

// Image handles get reused as well as the swapchain's: what the emulator draws into is read again at the next present.
void FgSwapchainMade() { s_fg.swapchain = VK_NULL_HANDLE; }

unsigned FgPresentsPerFrame() { return s_on ? s_generated + 1 : 1; }

void FgRegion(uint32_t w, uint32_t h, VkRect2D* r)
{
    *r = { { 0, 0 }, { w, h } };
    if (s_fg.regionTagged && (uint32_t)s_fg.region.offset.x + s_fg.region.extent.width <= w && (uint32_t)s_fg.region.offset.y + s_fg.region.extent.height <= h) *r = s_fg.region;
}

// The depth and motion the emulator copies out at its TAA: the right eye's copies mirror them.
bool FgInputImage(const VkImageCreateInfo& ci)
{
    if (!s_on || ci.imageType != VK_IMAGE_TYPE_2D || ci.mipLevels != 1 || ci.arrayLayers != 1 || ci.extent.width < 640) return false;
    const VkImageUsageFlags base = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags usage = ci.usage & ~VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    if (usage != base && usage != (base | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)) return false;
    return ci.format == VK_FORMAT_R32_SFLOAT || ci.format == VK_FORMAT_D32_SFLOAT || ci.format == VK_FORMAT_D32_SFLOAT_S8_UINT || ci.format == VK_FORMAT_R16G16_SFLOAT;
}

// Streamline's and NGX's own command buffers run their own work once, whatever images they bind.
bool FgForeignCaller()
{
    if (!s_streamline) return false;
    void* frames[48];
    const USHORT n = CaptureStackBackTrace(1, 48, frames, nullptr);
    static const wchar_t* const kNames[] = { L"sl.common.dll", L"sl.dlss_g.dll", L"sl.dlss.dll", L"sl.reflex.dll", L"sl.pcl.dll", L"nvngx_dlssg.dll", L"nvngx_dlss.dll", L"_nvngx.dll" };
    HMODULE mods[_countof(kNames)];
    for (size_t i = 0; i < _countof(kNames); ++i) mods[i] = GetModuleHandleW(kNames[i]);
    for (USHORT i = 0; i < n; ++i)
    {
        HMODULE m = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)frames[i], &m)) continue;
        for (HMODULE own : mods) if (own && own == m) return true;
    }
    return false;
}

}} // namespace wiz3d::vk
