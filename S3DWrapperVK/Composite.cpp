/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Present: the presentable image (left eye) and its sibling (right) go side by side into the image, or to the weaver.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include "../S3DAPI/KeyboardHook.h"

namespace wiz3d { namespace vk {

static VKAPI_ATTR void VKAPI_CALL Hook_GetDeviceQueue(VkDevice device, uint32_t family, uint32_t index, VkQueue* pQueue)
{
    DeviceData* d = FindDevice(device);
    d->GetDeviceQueue(device, family, index, pQueue);
    if (pQueue && *pQueue) { std::lock_guard<std::shared_mutex> g(d->lock); d->queueFamilies[*pQueue] = family; }
}

static VKAPI_ATTR void VKAPI_CALL Hook_GetDeviceQueue2(VkDevice device, const VkDeviceQueueInfo2* pInfo, VkQueue* pQueue)
{
    DeviceData* d = FindDevice(device);
    d->GetDeviceQueue2(device, pInfo, pQueue);
    if (pQueue && *pQueue) { std::lock_guard<std::shared_mutex> g(d->lock); d->queueFamilies[*pQueue] = pInfo->queueFamilyIndex; }
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateSwapchainKHR(
    VkDevice device, const VkSwapchainCreateInfoKHR* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkSwapchainKHR* pSwapchain)
{
    DeviceData* d = FindDevice(device);
    VkSwapchainCreateInfoKHR ci = *pCreateInfo;
    // The composite blits into the presentable image and copies out of it; the OSD draws into it.
    if (d->stereoEnabled) ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VkResult r = d->CreateSwapchainKHR(device, &ci, pAllocator, pSwapchain);
    Log("vkCreateSwapchainKHR: %ux%u format=%d images>=%u usage=0x%x present=%d old=%p -> %d\n",
        pCreateInfo->imageExtent.width, pCreateInfo->imageExtent.height, pCreateInfo->imageFormat,
        pCreateInfo->minImageCount, ci.imageUsage, pCreateInfo->presentMode, (void*)pCreateInfo->oldSwapchain, r);
    if (r != VK_SUCCESS) return r;
    FgSwapchainMade();
    d->bbSize = pCreateInfo->imageExtent;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        SwapchainInfo& info = d->swapchains[*pSwapchain];
        info.ci = ci;
        info.ci.pNext = nullptr;
        std::lock_guard<std::mutex> gi(d->inst->lock);
        auto s = d->inst->surfaces.find(pCreateInfo->surface);
        if (s != d->inst->surfaces.end()) info.hwnd = s->second;
    }
    RegisterSwapchainImages(d, *pSwapchain);
    // The DX9/DX11 paths' hotkey thread; writes gInfo.Input, which the stereo
    // block reads at present.
    gKbdHook.initialize(&gInfo.Input);
    gKbdHook.setBackBufferWidth(pCreateInfo->imageExtent.width);
    return r;
}

void CompositeDestroy(DeviceData* d, SwapchainInfo& sc)
{
    for (VkFence f : sc.fences) if (f) { d->WaitForFences(d->device, 1, &f, VK_TRUE, UINT64_MAX); d->DestroyFence(d->device, f, nullptr); }
    for (VkSemaphore s : sc.done) if (s) d->DestroySemaphore(d->device, s, nullptr);
    OsdDestroySwapchain(d, sc);
    sc.fences.clear();
    sc.done.clear();
    sc.cmds.clear();   // freed with the pool at device teardown
    if (sc.scratch) d->DestroyImage(d->device, sc.scratch, nullptr);
    if (sc.scratchMem) d->FreeMemory(d->device, sc.scratchMem, nullptr);
    sc.scratch = VK_NULL_HANDLE;
    sc.scratchMem = VK_NULL_HANDLE;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroySwapchainKHR(VkDevice device, VkSwapchainKHR sc, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    if (sc)
    {
        std::vector<VkImage> siblings;
        std::vector<VkDeviceMemory> mems;
        {
            std::lock_guard<std::shared_mutex> g(d->lock);
            auto it = d->swapchains.find(sc);
            if (it != d->swapchains.end())
            {
                WeaveDestroySwapchain(d, it->second);
                CompositeDestroy(d, it->second);
                for (VkImage img : it->second.images)
                {
                    auto ii = d->images.find(img);
                    if (ii == d->images.end()) continue;
                    if (ii->second.sibling) siblings.push_back(ii->second.sibling);
                    if (ii->second.siblingMem) { mems.push_back(ii->second.siblingMem); d->perf.vramAdded.fetch_sub((long long)ii->second.siblingMemSize, std::memory_order_relaxed); }
                    d->images.erase(ii);
                }
                d->swapchains.erase(it);
            }
        }
        d->DeviceWaitIdle(device);
        for (VkImage s : siblings) d->DestroyImage(device, s, nullptr);
        for (VkDeviceMemory m : mems) d->FreeMemory(device, m, nullptr);
        gKbdHook.clear();
    }
    d->DestroySwapchainKHR(device, sc, pAllocator);
}

// A queue family with graphics: blits and render passes are recorded only for those.
static bool FamilyHasGraphics(const DeviceData* d, uint32_t family)
{
    return family >= d->familyProps.size() || (d->familyProps[family].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0;
}

// Lock held. Per-image command buffer, fence, semaphore and the scratch copy.
static bool EnsureCompositeResources(DeviceData* d, SwapchainInfo& sc, uint32_t family)
{
    if (!sc.cmds.empty() && sc.cmdFamily == family) return true;
    VkCommandPool pool = VK_NULL_HANDLE;
    auto pit = d->presentPools.find(family);
    if (pit != d->presentPools.end()) pool = pit->second;
    else
    {
        VkCommandPoolCreateInfo pci = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pci.queueFamilyIndex = family;
        if (d->CreateCommandPool(d->device, &pci, nullptr, &pool) != VK_SUCCESS) return false;
        d->presentPools[family] = pool;
    }
    if (!sc.cmds.empty())
    {
        // The game presents from another queue family now (its async compute queue): once every image's last
        // submit is done, the command buffers move to that family's pool.
        for (VkFence f : sc.fences) d->WaitForFences(d->device, 1, &f, VK_TRUE, UINT64_MAX);
        d->FreeCommandBuffers(d->device, sc.pool, (uint32_t)sc.cmds.size(), sc.cmds.data());
        WeaveFreeCommands(d, sc);
        VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = (uint32_t)sc.cmds.size();
        sc.pool = pool;
        sc.cmdFamily = family;
        if (d->AllocateCommandBuffers(d->device, &ai, sc.cmds.data()) != VK_SUCCESS) { sc.cmds.clear(); return false; }
        StampCommandBuffers(d, sc.cmds.data(), ai.commandBufferCount);
        Log("Composite: presents moved to queue family %u (%s)\n", family,
            FamilyHasGraphics(d, family) ? "graphics" : "no graphics: the eyes are copied, the OSD is left out");
        return true;
    }
    sc.pool = pool;
    sc.cmdFamily = family;
    const uint32_t n = (uint32_t)sc.images.size();
    sc.cmds.resize(n);
    sc.fences.resize(n);
    sc.done.resize(n);
    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = n;
    if (d->AllocateCommandBuffers(d->device, &ai, sc.cmds.data()) != VK_SUCCESS) { sc.cmds.clear(); return false; }
    StampCommandBuffers(d, sc.cmds.data(), n);
    for (uint32_t i = 0; i < n; ++i)
    {
        VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        d->CreateFence(d->device, &fci, nullptr, &sc.fences[i]);
        VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        d->CreateSemaphore(d->device, &sci, nullptr, &sc.done[i]);
    }
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = sc.ci.imageFormat;
    ci.extent = { sc.ci.imageExtent.width, sc.ci.imageExtent.height, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d->CreateImage(d->device, &ci, nullptr, &sc.scratch) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d->GetImageMemoryRequirements(d->device, sc.scratch, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UINT32_MAX);
    if (d->AllocateMemory(d->device, &mai, nullptr, &sc.scratchMem) != VK_SUCCESS) return false;
    if (d->BindImageMemory(d->device, sc.scratch, sc.scratchMem, 0) != VK_SUCCESS) return false;
    // Retirement waits one present longer than this; generated frames present too.
    d->swapImages = n * FgPresentsPerFrame();
    Log("Composite: %u images, family %u, scratch %ux%u\n", n, family, ci.extent.width, ci.extent.height);
    return true;
}

static VkImageMemoryBarrier ImageBarrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA)
{
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = srcA;
    b.dstAccessMask = dstA;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return b;
}

// Lock held. A present the composite leaves as it is (frame generation's images, an image with no right eye) still
// takes a screenshot's copy and the OSD while an element shows: one submit that reads the image and draws over it.
static void PresentOsdOnly(DeviceData* d, SwapchainInfo& sc, VkQueue queue, uint32_t family, uint32_t idx,
                           const VkPresentInfoKHR* info, std::vector<VkSemaphore>& outWait)
{
    if (idx >= sc.images.size() || !(OsdWanted(d) || ScreenshotArmed()) || !EnsureCompositeResources(d, sc, family)) return;
    d->WaitForFences(d->device, 1, &sc.fences[idx], VK_TRUE, UINT64_MAX);
    d->ResetFences(d->device, 1, &sc.fences[idx]);
    VkCommandBuffer cb = sc.cmds[idx];
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    d->BeginCommandBuffer(cb, &bi);
    const VkImage img = sc.images[idx];
    VkImageMemoryBarrier b = ImageBarrier(img, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    ScreenshotCopy(d, cb, img, VK_NULL_HANDLE, sc.ci.imageFormat, sc.ci.imageExtent.width, sc.ci.imageExtent.height, FgOn());
    b = ImageBarrier(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    if (FamilyHasGraphics(d, family)) OsdDraw(d, cb, img, sc.ci.imageFormat, sc.ci.imageExtent.width, sc.ci.imageExtent.height);
    b = ImageBarrier(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    d->EndCommandBuffer(cb);
    std::vector<VkPipelineStageFlags> stages(info->waitSemaphoreCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = outWait.empty() ? info->waitSemaphoreCount : 0;   // the first one takes the game's
    si.pWaitSemaphores = info->pWaitSemaphores;
    si.pWaitDstStageMask = stages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &sc.done[idx];
    VkResult r;
    { const auto ql = FgQueueLock(); r = d->QueueSubmit(queue, 1, &si, sc.fences[idx]); }
    if (r != VK_SUCCESS) { Log("OSD submit failed %d\n", r); return; }
    outWait.push_back(sc.done[idx]);
    ScreenshotWrite(d, queue);
}

void CompositePresent(DeviceData* d, VkQueue queue, VkPresentInfoKHR* info, std::vector<VkSemaphore>& outWait)
{
    outWait.clear();
    if (!d->stereoEnabled) return;
    uint32_t family = 0;
    std::lock_guard<std::shared_mutex> g(d->lock);
    {
        auto q = d->queueFamilies.find(queue);
        if (q != d->queueFamilies.end()) family = q->second;
    }
    const bool graphics = FamilyHasGraphics(d, family);
    OsdTick(d);
    ScreenshotTick();
    for (uint32_t i = 0; i < info->swapchainCount; ++i)
    {
        auto it = d->swapchains.find(info->pSwapchains[i]);
        if (it == d->swapchains.end()) continue;
        SwapchainInfo& sc = it->second;
        const uint32_t idx = info->pImageIndices[i];
        // Frame generation: every image here already holds both eyes; only the weave has more to do.
        if (FgOn() && !WeaveSelected()) { PresentOsdOnly(d, sc, queue, family, idx, info, outWait); continue; }
        if (idx >= sc.images.size())
        {
            if (FgOn()) { static bool s_once = false; if (!s_once) { s_once = true; Log("[fg] a present of an image the layer never listed: not woven\n"); } }
            continue;
        }
        VkImage img = sc.images[idx], sib = VK_NULL_HANDLE;
        if (!ImageIsStereo(d, img, &sib))
        {
            if (!FgOn()) { PresentOsdOnly(d, sc, queue, family, idx, info, outWait); continue; }
            sib = img;
        }
        if (!EnsureCompositeResources(d, sc, family)) return;

        // Weave when the output method asks for it and the interop is up;
        // otherwise (or on its failure) the side-by-side composite below.
        if (WeaveSelected())
        {
            VkSemaphore done = VK_NULL_HANDLE;
            if (WeaveFrame(d, sc, queue, idx, img, sib, info, outWait.empty(), graphics, &done)) { outWait.push_back(done); continue; }
            static bool s_fellBack = false;
            if (!s_fellBack) { s_fellBack = true; Log("Weave unavailable, presenting side-by-side\n"); }
        }
        // Side by side squeezes each eye with a blit, which a queue family with no graphics cannot record.
        if (!graphics)
        {
            static bool s_noBlit = false;
            if (!s_noBlit) { s_noBlit = true; Log("Composite: queue family %u has no graphics: the left eye is presented\n", family); }
            PresentOsdOnly(d, sc, queue, family, idx, info, outWait);
            continue;
        }

        d->WaitForFences(d->device, 1, &sc.fences[idx], VK_TRUE, UINT64_MAX);
        d->ResetFences(d->device, 1, &sc.fences[idx]);
        VkCommandBuffer cb = sc.cmds[idx];
        VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        d->BeginCommandBuffer(cb, &bi);

        const int32_t w = (int32_t)sc.ci.imageExtent.width, h = (int32_t)sc.ci.imageExtent.height;
        VkImageMemoryBarrier pre[3] = {
            ImageBarrier(img, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            ImageBarrier(sc.scratch, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT),
            ImageBarrier(sib, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
        };
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 3, pre);

        VkImageCopy copy = {};
        copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copy.dstSubresource = copy.srcSubresource;
        copy.extent = { (uint32_t)w, (uint32_t)h, 1 };
        d->CmdCopyImage(cb, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc.scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        VkImageMemoryBarrier mid[2] = {
            ImageBarrier(sc.scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
            ImageBarrier(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT),
        };
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, mid);
        ScreenshotCopy(d, cb, sc.scratch, gInfo.Input.StereoActive ? sib : VK_NULL_HANDLE, sc.ci.imageFormat, (uint32_t)w, (uint32_t)h, FgOn());

        // Stereo toggle presents the left eye to both sides; SwapEyes crosses them.
        VkImage left = sc.scratch, right = gInfo.Input.StereoActive ? sib : sc.scratch;
        if (gInfo.Input.SwapEyes && gInfo.Input.StereoActive) { VkImage t = left; left = right; right = t; }
        VkImageBlit blit = {};
        blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blit.dstSubresource = blit.srcSubresource;
        // Same source rect for both eyes: drop the band the shift uncovered in the right eye, which is otherwise stale
        // sibling bytes. Its left edge, or its right edge under a negative convergence.
        const int32_t crop = EdgeCropPx(d->lastSep, w, Vk().VKEdgeCrop);
        blit.srcOffsets[0] = { d->lastSep < 0.f ? 0 : crop, 0, 0 };
        blit.srcOffsets[1] = { d->lastSep < 0.f ? w - crop : w, h, 1 };
        blit.dstOffsets[0] = { 0, 0, 0 };
        blit.dstOffsets[1] = { w / 2, h, 1 };
        d->CmdBlitImage(cb, left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        blit.dstOffsets[0] = { w / 2, 0, 0 };
        blit.dstOffsets[1] = { w, h, 1 };
        d->CmdBlitImage(cb, right, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        OsdDraw(d, cb, img, sc.ci.imageFormat, (uint32_t)w, (uint32_t)h);

        VkImageMemoryBarrier post[2] = {
            ImageBarrier(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0),
            ImageBarrier(sib, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0),
        };
        d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 2, post);
        d->EndCommandBuffer(cb);

        std::vector<VkPipelineStageFlags> stages(info->waitSemaphoreCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
        VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
        si.waitSemaphoreCount = outWait.empty() ? info->waitSemaphoreCount : 0;   // the first one takes the game's
        si.pWaitSemaphores = info->pWaitSemaphores;
        si.pWaitDstStageMask = stages.data();
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cb;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &sc.done[idx];
        VkResult r;
        { const auto ql = FgQueueLock(); r = d->QueueSubmit(queue, 1, &si, sc.fences[idx]); }
        if (r != VK_SUCCESS) { Log("Composite submit failed %d\n", r); return; }
        outWait.push_back(sc.done[idx]);
        ScreenshotWrite(d, queue);
    }
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    DeviceData* d = FindDevice(queue);
    StereoParamsUpdate(d);
    VkPresentInfoKHR info = *pPresentInfo;
    std::vector<VkSemaphore> wait;
    {
        PerfScope ps(d->perf.composite);
        CompositePresent(d, queue, &info, wait);
    }
    if (!wait.empty())
    {
        info.waitSemaphoreCount = (uint32_t)wait.size();
        info.pWaitSemaphores = wait.data();
    }
    VkResult r;
    { const auto ql = FgQueueLock(); r = d->QueuePresentKHR(queue, &info); }
    // OUT_OF_DATE/SUBOPTIMAL are routine; anything else (DEVICE_LOST above all)
    // is the GPU rejecting work we recorded.
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR && r != VK_ERROR_OUT_OF_DATE_KHR)
    { static int n = 8; if (n-- > 0) Log("!! vkQueuePresentKHR returned %d\n", r); }
    {
        const long long now = PerfNow();
        PerfCounters& p = d->perf;
        if (p.lastPresent && p.frameMs.size() < 4096) p.frameMs.push_back((float)PerfMs(now - p.lastPresent));
        // clockFrameRate: a frame-rate row every 5 s by the clock, for games too slow to reach a 600-frame row.
        static long long s_from = now, s_last = now;
        static uint32_t s_frames = d->frames;
        static double s_worst = 0.0;
        if (Traits().clockFrameRate && gInfo.CollectDebugInformation)
        {
            const double one = PerfMs(now - s_last);
            if (one > s_worst) s_worst = one;
            const double span = PerfMs(now - s_from);
            if (span >= 5000.0)
            {
                SYSTEMTIME t;
                GetLocalTime(&t);
                Log("[fps] %02u:%02u:%02u %.1f fps, worst frame %.0f ms, frame %u, replayDraws so far %u\n", t.wHour, t.wMinute, t.wSecond,
                    (d->frames - s_frames) * 1000.0 / span, s_worst, d->frames, (unsigned)d->statReplayDraws);
                s_from = now; s_frames = d->frames; s_worst = 0.0;
            }
        }
        s_last = now;
        p.lastPresent = now;
    }
    ++d->frames;
    d->shadGtHudPhase = false;
    SpriteFrameEnd(d);
    RcbFrameEnd(d);
    DrainRetiredImages(d);
    DrainPendingBufferSiblings(d);
    if (gInfo.CollectDebugInformation && d->frames % 600 == 1) LogFrameStats(d);   // per-window rows, as DX11's
    return r;
}

const HookEntry g_swapchainHooks[] = {
    WIZ3D_HOOK(GetDeviceQueue), WIZ3D_HOOK(GetDeviceQueue2),
    WIZ3D_HOOK(CreateSwapchainKHR), WIZ3D_HOOK(DestroySwapchainKHR),
    WIZ3D_HOOK(QueuePresentKHR),
    { nullptr, nullptr },
};

}} // namespace wiz3d::vk
