/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// SR weave through a private D3D11 device: both eyes go into a shared side-by-side texture, the weaver writes a shared
// output, and Vulkan copies it into the presentable image; the stages wait on the CPU, or (VKWeaveGpuSync) on the GPU.

#include "Layer.h"
#include "../S3DAPI/GlobalData.h"
#include <d3d11_4.h>
#include <dxgi1_2.h>
#include <SR.hpp>
#include <tchar.h>

namespace wiz3d { namespace vk {

struct WeaveDevice
{
    ID3D11Device*        dev = nullptr;
    ID3D11DeviceContext* ctx = nullptr;
    ID3D11Query*         evt = nullptr;
    SimulatedReality::SRInterfaceDX11* sr = nullptr;
    HWND hwnd = nullptr;                               // the window `sr` was created for
    ID3D11ShaderResourceView* input = nullptr;         // the side-by-side texture `sr` currently weaves
    bool failed = false;
    // VKWeaveGpuSync: a D3D11 fence Vulkan imports as a semaphore; stage A signals odd values, the weave even ones.
    ID3D11DeviceContext4* ctx4 = nullptr;
    ID3D11Fence*         fence = nullptr;
    HANDLE               fenceHandle = nullptr;
    VkSemaphore          vkFence = VK_NULL_HANDLE;     // set only once the import succeeded
    uint64_t             value = 0;                    // the last value the weave signalled
};

struct WeaveState
{
    ID3D11Texture2D*          sbsTex = nullptr;
    ID3D11ShaderResourceView* sbsSRV = nullptr;
    ID3D11Texture2D*          outTex = nullptr;
    ID3D11RenderTargetView*   outRTV = nullptr;
    HANDLE                    sbsHandle = nullptr, outHandle = nullptr;
    VkImage        vkSbs = VK_NULL_HANDLE, vkOut = VK_NULL_HANDLE;
    VkDeviceMemory memSbs = VK_NULL_HANDLE, memOut = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> cmdsB;      // output -> swapchain, per image
    VkFence fence = VK_NULL_HANDLE;
    bool sbsInitialised = false;
    bool failed = false;
    unsigned frames = 0;
};

bool WeaveSelected()
{
    return _tcsicmp(gInfo.OutputMethodDll, _T("SimulatedRealityWeaveOutput")) == 0;
}

static DXGI_FORMAT ToDxgi(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_B8G8R8A8_UNORM: return DXGI_FORMAT_B8G8R8A8_UNORM;
    case VK_FORMAT_B8G8R8A8_SRGB:  return DXGI_FORMAT_B8G8R8A8_UNORM_SRGB;
    case VK_FORMAT_R8G8B8A8_UNORM: return DXGI_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8A8_SRGB:  return DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return DXGI_FORMAT_R10G10B10A2_UNORM;
    case VK_FORMAT_R16G16B16A16_SFLOAT: return DXGI_FORMAT_R16G16B16A16_FLOAT;
    default: return DXGI_FORMAT_UNKNOWN;
    }
}

// VKWeaveGpuSync: a shared D3D11 fence imported into Vulkan as a semaphore (D3D12_FENCE handle, values through
// VkD3D12FenceSubmitInfoKHR); when any step fails the stages keep waiting on the CPU.
static void ShareFence(DeviceData* d, WeaveDevice& w)
{
    const char* failed = nullptr;
    ID3D11Device5* dev5 = nullptr;
    VkSemaphore sem = VK_NULL_HANDLE;
    if (!d->externalSemaphoreWin32 || !d->ImportSemaphoreWin32HandleKHR) failed = "VK_KHR_external_semaphore_win32 is not enabled";
    else if (FAILED(w.dev->QueryInterface(__uuidof(ID3D11Device5), (void**)&dev5)) ||
             FAILED(w.ctx->QueryInterface(__uuidof(ID3D11DeviceContext4), (void**)&w.ctx4))) failed = "no ID3D11Device5";
    else if (FAILED(dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, __uuidof(ID3D11Fence), (void**)&w.fence)) ||
             FAILED(w.fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &w.fenceHandle))) failed = "the shared fence";
    else
    {
        VkSemaphoreCreateInfo sci = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        VkResult r = d->CreateSemaphore(d->device, &sci, nullptr, &sem);
        if (r == VK_SUCCESS)
        {
            VkImportSemaphoreWin32HandleInfoKHR imp = { VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR };
            imp.semaphore = sem;
            imp.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE_BIT;
            imp.handle = w.fenceHandle;
            r = d->ImportSemaphoreWin32HandleKHR(d->device, &imp);
        }
        if (r != VK_SUCCESS) failed = "the semaphore import";
    }
    if (dev5) dev5->Release();
    if (!failed) { w.vkFence = sem; Log("Weave: stages wait on the GPU through a shared D3D11 fence\n"); return; }
    if (sem) d->DestroySemaphore(d->device, sem, nullptr);
    if (w.fenceHandle) { CloseHandle(w.fenceHandle); w.fenceHandle = nullptr; }
    if (w.fence) { w.fence->Release(); w.fence = nullptr; }
    if (w.ctx4) { w.ctx4->Release(); w.ctx4 = nullptr; }
    Log("Weave: VKWeaveGpuSync: %s failed -- stages wait on the CPU\n", failed);
}

// D3D11 device on the Vulkan adapter (matched by LUID) and the SR weaver.
static bool EnsureWeaveDevice(DeviceData* d, HWND hwnd)
{
    if (d->weave)
    {
        WeaveDevice& w = *d->weave;
        // The weaver belongs to one window: a game may present a first swapchain on another window than its own, so
        // the weaver follows the window of the swapchain being presented.
        if (!w.failed && hwnd && w.hwnd != hwnd)
        {
            if (w.sr) { w.sr->Delete(); w.sr = nullptr; }
            w.input = nullptr;
            const HRESULT hr = SimulatedReality::CreateSRInterfaceDX11(w.ctx, hwnd, &w.sr);
            if (FAILED(hr) || !w.sr)
            {
                w.failed = true;
                Log("Weave: CreateSRInterfaceDX11 for the new window failed 0x%08lX (hwnd=%p, was %p) -- SBS fallback\n", hr, (void*)hwnd, (void*)w.hwnd);
            }
            else Log("Weave: SR weaver re-created for window %p (was %p)\n", (void*)hwnd, (void*)w.hwnd);
            w.hwnd = hwnd;
        }
        return !w.failed;
    }
    d->weave = new WeaveDevice();
    WeaveDevice& w = *d->weave;
    w.failed = true;

    VkPhysicalDeviceIDProperties idp = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
    VkPhysicalDeviceProperties2 p2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2 };
    p2.pNext = &idp;
    if (d->inst->GetPhysicalDeviceProperties2) d->inst->GetPhysicalDeviceProperties2(d->physDev, &p2);

    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory))) { Log("Weave: CreateDXGIFactory1 failed\n"); return false; }
    IDXGIAdapter1* adapter = nullptr;
    for (UINT i = 0; ; ++i)
    {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) != S_OK) break;
        DXGI_ADAPTER_DESC1 desc;
        a->GetDesc1(&desc);
        if (idp.deviceLUIDValid && memcmp(&desc.AdapterLuid, idp.deviceLUID, sizeof(LUID)) == 0) { adapter = a; break; }
        a->Release();
    }
    factory->Release();
    if (!adapter) Log("Weave: no DXGI adapter matched the Vulkan device LUID, using default\n");

    D3D_FEATURE_LEVEL fl;
    HRESULT hr = D3D11CreateDevice(adapter, adapter ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE, nullptr,
                                   0, nullptr, 0, D3D11_SDK_VERSION, &w.dev, &fl, &w.ctx);
    if (adapter) adapter->Release();
    if (FAILED(hr)) { Log("Weave: D3D11CreateDevice failed 0x%08lX\n", hr); return false; }

    D3D11_QUERY_DESC qd = { D3D11_QUERY_EVENT, 0 };
    w.dev->CreateQuery(&qd, &w.evt);

    // E_NOINTERFACE when the delay-loaded SR runtime is absent.
    hr = SimulatedReality::CreateSRInterfaceDX11(w.ctx, hwnd, &w.sr);
    if (FAILED(hr) || !w.sr) { Log("Weave: CreateSRInterfaceDX11 failed 0x%08lX (hwnd=%p) -- SBS fallback\n", hr, (void*)hwnd); return false; }
    w.failed = false;
    w.hwnd = hwnd;
    Log("Weave: D3D11 device + SR weaver ready (hwnd=%p, feature level 0x%x)\n", (void*)hwnd, fl);
    if (Vk().VKWeaveGpuSync) ShareFence(d, w);
    return true;
}

void WeaveDestroyDevice(DeviceData* d)
{
    if (!d->weave) return;
    WeaveDevice& w = *d->weave;
    if (w.sr) w.sr->Delete();
    if (w.evt) w.evt->Release();
    if (w.vkFence) d->DestroySemaphore(d->device, w.vkFence, nullptr);
    if (w.fenceHandle) CloseHandle(w.fenceHandle);
    if (w.fence) w.fence->Release();
    if (w.ctx4) w.ctx4->Release();
    if (w.ctx) w.ctx->Release();
    if (w.dev) w.dev->Release();
    delete d->weave;
    d->weave = nullptr;
}

// Shared textures: created on D3D11, imported into Vulkan.
static bool CreateShared(DeviceData* d, WeaveDevice& w, UINT width, UINT height, DXGI_FORMAT fmt, UINT bind,
                         ID3D11Texture2D** tex, HANDLE* handle, VkFormat vkFmt, VkImage* img, VkDeviceMemory* mem)
{
    D3D11_TEXTURE2D_DESC td = {};
    td.Width = width;
    td.Height = height;
    td.MipLevels = 1;
    td.ArraySize = 1;
    td.Format = fmt;
    td.SampleDesc.Count = 1;
    td.Usage = D3D11_USAGE_DEFAULT;
    td.BindFlags = bind;
    td.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
    HRESULT hr = w.dev->CreateTexture2D(&td, nullptr, tex);
    if (FAILED(hr)) { Log("Weave: shared texture %ux%u failed 0x%08lX\n", width, height, hr); return false; }
    IDXGIResource1* res = nullptr;
    if (FAILED((*tex)->QueryInterface(__uuidof(IDXGIResource1), (void**)&res))) return false;
    hr = res->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, handle);
    res->Release();
    if (FAILED(hr)) { Log("Weave: CreateSharedHandle failed 0x%08lX\n", hr); return false; }

    VkExternalMemoryImageCreateInfo ext = { VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO };
    ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.pNext = &ext;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = vkFmt;
    ci.extent = { width, height, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkResult r = d->CreateImage(d->device, &ci, nullptr, img);
    if (r != VK_SUCCESS) { Log("Weave: vkCreateImage (external) failed %d\n", r); return false; }

    VkMemoryRequirements req;
    d->GetImageMemoryRequirements(d->device, *img, &req);
    uint32_t bits = req.memoryTypeBits;
    if (d->GetMemoryWin32HandlePropertiesKHR)
    {
        VkMemoryWin32HandlePropertiesKHR hp = { VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR };
        if (d->GetMemoryWin32HandlePropertiesKHR(d->device, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT, *handle, &hp) == VK_SUCCESS && hp.memoryTypeBits)
            bits &= hp.memoryTypeBits;
    }
    VkImportMemoryWin32HandleInfoKHR imp = { VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR };
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
    imp.handle = *handle;
    VkMemoryDedicatedAllocateInfo ded = { VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO };
    ded.pNext = &imp;
    ded.image = *img;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.pNext = &ded;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = FindMemoryType(d, bits, 0, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX) { Log("Weave: no memory type for import (bits 0x%x)\n", bits); return false; }
    r = d->AllocateMemory(d->device, &mai, nullptr, mem);
    if (r != VK_SUCCESS) { Log("Weave: import vkAllocateMemory failed %d\n", r); return false; }
    r = d->BindImageMemory(d->device, *img, *mem, 0);
    if (r != VK_SUCCESS) { Log("Weave: bind imported memory failed %d\n", r); return false; }
    return true;
}

static bool EnsureWeaveState(DeviceData* d, SwapchainInfo& sc)
{
    if (sc.weave) return !sc.weave->failed;
    sc.weave = new WeaveState();
    WeaveState& s = *sc.weave;
    s.failed = true;
    WeaveDevice& w = *d->weave;
    const UINT W = sc.ci.imageExtent.width, H = sc.ci.imageExtent.height;
    const DXGI_FORMAT fmt = ToDxgi(sc.ci.imageFormat);
    if (fmt == DXGI_FORMAT_UNKNOWN) { Log("Weave: unsupported swapchain format %d\n", sc.ci.imageFormat); return false; }

    if (!CreateShared(d, w, W * 2, H, fmt, D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET,
                      &s.sbsTex, &s.sbsHandle, sc.ci.imageFormat, &s.vkSbs, &s.memSbs)) return false;
    if (!CreateShared(d, w, W, H, fmt, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
                      &s.outTex, &s.outHandle, sc.ci.imageFormat, &s.vkOut, &s.memOut)) return false;
    if (FAILED(w.dev->CreateShaderResourceView(s.sbsTex, nullptr, &s.sbsSRV))) return false;
    if (FAILED(w.dev->CreateRenderTargetView(s.outTex, nullptr, &s.outRTV))) return false;
    w.sr->SetInputTexture(s.sbsSRV);
    w.input = s.sbsSRV;

    VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    ai.commandPool = sc.pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = (uint32_t)sc.images.size();
    s.cmdsB.resize(sc.images.size());
    if (d->AllocateCommandBuffers(d->device, &ai, s.cmdsB.data()) != VK_SUCCESS) return false;
    VkFenceCreateInfo fci = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    if (d->CreateFence(d->device, &fci, nullptr, &s.fence) != VK_SUCCESS) return false;
    s.failed = false;
    Log("Weave: shared textures ready (%ux%u sbs, %ux%u out, fmt=%d)\n", W * 2, H, W, H, fmt);
    return true;
}

void WeaveDestroySwapchain(DeviceData* d, SwapchainInfo& sc)
{
    if (!sc.weave) return;
    WeaveState& s = *sc.weave;
    // The stage-A fence is waited and reset every frame, so a wait here would never return;
    // the per-image fences in CompositeDestroy cover the last stage-C submit.
    if (s.fence) d->DestroyFence(d->device, s.fence, nullptr);
    if (s.vkSbs) d->DestroyImage(d->device, s.vkSbs, nullptr);
    if (s.vkOut) d->DestroyImage(d->device, s.vkOut, nullptr);
    if (s.memSbs) d->FreeMemory(d->device, s.memSbs, nullptr);
    if (s.memOut) d->FreeMemory(d->device, s.memOut, nullptr);
    if (d->weave && d->weave->input == s.sbsSRV) d->weave->input = nullptr;   // the next frame registers its own again
    if (s.sbsSRV) s.sbsSRV->Release();
    if (s.outRTV) s.outRTV->Release();
    if (s.sbsTex) s.sbsTex->Release();
    if (s.outTex) s.outTex->Release();
    if (s.sbsHandle) CloseHandle(s.sbsHandle);
    if (s.outHandle) CloseHandle(s.outHandle);
    delete sc.weave;
    sc.weave = nullptr;
}

static VkImageMemoryBarrier Barrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA)
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

bool WeaveFrame(DeviceData* d, SwapchainInfo& sc, VkQueue queue, uint32_t idx, VkImage img, VkImage sib,
                const VkPresentInfoKHR* info, bool firstSwapchain, VkSemaphore* outDone)
{
    if (!d->externalMemoryWin32) return false;
    if (!EnsureWeaveDevice(d, sc.hwnd)) return false;
    if (!EnsureWeaveState(d, sc)) return false;
    WeaveDevice& w = *d->weave;
    WeaveState& s = *sc.weave;
    const int32_t W = (int32_t)sc.ci.imageExtent.width, H = (int32_t)sc.ci.imageExtent.height;
    const bool gpuSync = w.vkFence != VK_NULL_HANDLE;   // VKWeaveGpuSync with its fence shared

    // Stage A: both eyes, full size, into the shared side-by-side input.
    long long t0 = PerfNow();
    VkCommandBuffer cbA = sc.cmds[idx];
    d->WaitForFences(d->device, 1, &sc.fences[idx], VK_TRUE, UINT64_MAX);
    d->ResetFences(d->device, 1, &sc.fences[idx]);
    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    d->BeginCommandBuffer(cbA, &bi);
    // Frame generation: `img` holds both eyes, squeezed, real and generated frames alike; no sibling takes part.
    const bool fg = FgOn();
    VkImageMemoryBarrier pre[3] = {
        Barrier(img, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
        Barrier(s.vkSbs, s.sbsInitialised ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT),
        Barrier(sib, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
    };
    d->CmdPipelineBarrier(cbA, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, fg ? 2 : 3, pre);
    VkImageBlit blit = {};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstSubresource = blit.srcSubresource;
    if (fg)
    {
        VkRect2D r;
        FgRegion((uint32_t)W, (uint32_t)H, &r);
        if (r.extent.width != (uint32_t)W || r.extent.height != (uint32_t)H)
        {
            const VkClearColorValue black = {};
            const VkImageSubresourceRange all = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            d->CmdClearColorImage(cbA, s.vkSbs, VK_IMAGE_LAYOUT_GENERAL, &black, 1, &all);
        }
        const int32_t x0 = r.offset.x, y0 = r.offset.y, rw = (int32_t)r.extent.width, rh = (int32_t)r.extent.height;
        blit.srcOffsets[0] = { x0, y0, 0 };
        blit.srcOffsets[1] = { x0 + rw / 2, y0 + rh, 1 };
        blit.dstOffsets[0] = { x0, y0, 0 };
        blit.dstOffsets[1] = { x0 + rw, y0 + rh, 1 };
        d->CmdBlitImage(cbA, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.vkSbs, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_LINEAR);
        blit.srcOffsets[0] = { x0 + rw / 2, y0, 0 };
        blit.srcOffsets[1] = { x0 + rw, y0 + rh, 1 };
        blit.dstOffsets[0] = { W + x0, y0, 0 };
        blit.dstOffsets[1] = { W + x0 + rw, y0 + rh, 1 };
        d->CmdBlitImage(cbA, img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.vkSbs, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_LINEAR);
    }
    else
    {
        VkImage left = img, right = gInfo.Input.StereoActive ? sib : img;
        if (gInfo.Input.SwapEyes && gInfo.Input.StereoActive) { VkImage t = left; left = right; right = t; }
        // Same source rect for both eyes: drop the band the shift uncovered in the right eye, which is otherwise stale
        // sibling bytes. Its left edge, or its right edge under a negative convergence.
        const int32_t crop = EdgeCropPx(d->lastSep, W, Vk().VKEdgeCrop);
        blit.srcOffsets[0] = { d->lastSep < 0.f ? 0 : crop, 0, 0 };
        blit.srcOffsets[1] = { d->lastSep < 0.f ? W - crop : W, H, 1 };
        blit.dstOffsets[1] = { W, H, 1 };
        d->CmdBlitImage(cbA, left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.vkSbs, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
        blit.dstOffsets[0] = { W, 0, 0 };
        blit.dstOffsets[1] = { W * 2, H, 1 };
        d->CmdBlitImage(cbA, right, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s.vkSbs, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_NEAREST);
    }
    VkImageMemoryBarrier midA[2] = {
        Barrier(s.vkSbs, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT),
        Barrier(sib, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_READ_BIT, 0),
    };
    d->CmdPipelineBarrier(cbA, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, fg ? 1 : 2, midA);
    d->EndCommandBuffer(cbA);

    std::vector<VkPipelineStageFlags> stages(info->waitSemaphoreCount, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);
    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.waitSemaphoreCount = firstSwapchain ? info->waitSemaphoreCount : 0;
    si.pWaitSemaphores = info->pWaitSemaphores;
    si.pWaitDstStageMask = stages.data();
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cbA;
    // VKWeaveGpuSync: stage A signals the shared fence instead of a fence the CPU waits on; the game's semaphores take
    // no value.
    const uint64_t valA = w.value + 1, valB = w.value + 2;
    std::vector<uint64_t> noValues(si.waitSemaphoreCount, 0);
    VkD3D12FenceSubmitInfoKHR fenceA = { VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR };
    if (gpuSync)
    {
        fenceA.waitSemaphoreValuesCount = si.waitSemaphoreCount;
        fenceA.pWaitSemaphoreValues = noValues.data();
        fenceA.signalSemaphoreValuesCount = 1;
        fenceA.pSignalSemaphoreValues = &valA;
        si.pNext = &fenceA;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &w.vkFence;
    }
    VkResult r;
    { const auto ql = FgQueueLock(); r = d->QueueSubmit(queue, 1, &si, gpuSync ? VK_NULL_HANDLE : s.fence); }
    if (r != VK_SUCCESS) { Log("Weave: stage A submit failed %d\n", r); s.failed = true; return false; }
    s.sbsInitialised = true;
    if (!gpuSync)
    {
        d->WaitForFences(d->device, 1, &s.fence, VK_TRUE, UINT64_MAX);
        d->ResetFences(d->device, 1, &s.fence);
    }
    long long t1 = PerfNow();
    d->perf.weaveA.fetch_add(t1 - t0, std::memory_order_relaxed);

    // Stage B: the D3D11 weaver, this swapchain's input SRV registered (again after a re-created weaver or another
    // swapchain's frame), output RTV bound.
    if (gpuSync) w.ctx4->Wait(w.fence, valA);
    if (w.input != s.sbsSRV) { w.sr->SetInputTexture(s.sbsSRV); w.input = s.sbsSRV; }
    w.ctx->OMSetRenderTargets(1, &s.outRTV, nullptr);
    D3D11_VIEWPORT vp = { 0.f, 0.f, (float)W, (float)H, 0.f, 1.f };
    w.ctx->RSSetViewports(1, &vp);
    w.sr->Weave();
    if (gpuSync) { w.ctx4->Signal(w.fence, valB); w.value = valB; }
    w.ctx->Flush();
    if (w.evt && !gpuSync)
    {
        w.ctx->End(w.evt);
        while (w.ctx->GetData(w.evt, nullptr, 0, 0) == S_FALSE) Sleep(0);
    }
    long long t2 = PerfNow();
    d->perf.weaveB.fetch_add(t2 - t1, std::memory_order_relaxed);

    // Stage C: woven output into the presentable image.
    VkCommandBuffer cbB = s.cmdsB[idx];
    d->BeginCommandBuffer(cbB, &bi);
    VkImageMemoryBarrier preB[2] = {
        Barrier(img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT),
        Barrier(s.vkOut, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT),
    };
    d->CmdPipelineBarrier(cbB, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, preB);
    VkImageCopy copy = {};
    copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = { (uint32_t)W, (uint32_t)H, 1 };
    d->CmdCopyImage(cbB, s.vkOut, VK_IMAGE_LAYOUT_GENERAL, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    VkImageMemoryBarrier post = Barrier(img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_TRANSFER_WRITE_BIT, 0);
    d->CmdPipelineBarrier(cbB, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &post);
    d->EndCommandBuffer(cbB);

    VkSubmitInfo sb = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    sb.commandBufferCount = 1;
    sb.pCommandBuffers = &cbB;
    sb.signalSemaphoreCount = 1;
    sb.pSignalSemaphores = &sc.done[idx];
    // VKWeaveGpuSync: the copy waits on the GPU for the weave's fence value; the present semaphore takes no value.
    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const uint64_t noValue = 0;
    VkD3D12FenceSubmitInfoKHR fenceC = { VK_STRUCTURE_TYPE_D3D12_FENCE_SUBMIT_INFO_KHR };
    if (gpuSync)
    {
        fenceC.waitSemaphoreValuesCount = 1;
        fenceC.pWaitSemaphoreValues = &valB;
        fenceC.signalSemaphoreValuesCount = 1;
        fenceC.pSignalSemaphoreValues = &noValue;
        sb.pNext = &fenceC;
        sb.waitSemaphoreCount = 1;
        sb.pWaitSemaphores = &w.vkFence;
        sb.pWaitDstStageMask = &waitStage;
    }
    { const auto ql = FgQueueLock(); r = d->QueueSubmit(queue, 1, &sb, sc.fences[idx]); }
    if (r != VK_SUCCESS) { Log("Weave: stage C submit failed %d\n", r); s.failed = true; return false; }
    d->perf.weaveC.fetch_add(PerfNow() - t2, std::memory_order_relaxed);
    *outDone = sc.done[idx];
    if (++s.frames <= 3) Log("Weave: frame %u OK (%dx%d)\n", s.frames, W, H);
    return true;
}

}} // namespace wiz3d::vk
