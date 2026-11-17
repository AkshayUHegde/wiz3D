/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The stereo screenshot of the DX9/DX11 wrappers (Ctrl+F12): both eyes at full size, the right eye in the left half,
// in My Pictures\wiz3D Screenshots\<profile>, in the file type ScreenshotType names.

#include "Layer.h"
#include "../S3DAPI/GlobalData.h"
#include "../S3DAPI/KeyboardHook.h"
#include "../Shared/ProductNames.h"
#include <shlobj.h>
#include <shlwapi.h>
#include <wincodec.h>
#include <tchar.h>

#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "windowscodecs.lib")

namespace wiz3d { namespace vk {

namespace {

bool s_armed = false;   // a request waits for a present that copies its eyes out

// The copy in flight: one buffer holding the final picture's rows, `views` eyes of w x h side by side.
struct Shot
{
    VkBuffer       buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    void*          mapped = nullptr;
    VkFormat       format = VK_FORMAT_UNDEFINED;
    uint32_t       w = 0, h = 0, views = 0;
    bool           stereo = false;   // the picture holds two eyes
} s_shot;

uint32_t BytesPerPixel(VkFormat f)
{
    switch (f)
    {
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        return 4;
    case VK_FORMAT_R16G16B16A16_SFLOAT:
        return 8;
    default:
        return 0;
    }
}

float HalfToFloat(uint16_t v)
{
    const uint32_t e = (v >> 10) & 31, m = v & 1023;
    float f;
    if (e == 0) f = ldexpf((float)m, -24);
    else if (e == 31) f = m ? 0.f : 65504.f;
    else f = ldexpf((float)(m | 1024), (int)e - 25);
    return (v & 0x8000) ? -f : f;
}

// One row of the image's format into blue, green, red bytes; alpha is dropped, as a presented image's means nothing.
void RowToBgr(VkFormat f, const uint8_t* src, uint8_t* dst, uint32_t pixels)
{
    for (uint32_t x = 0; x < pixels; ++x, dst += 3)
    {
        switch (f)
        {
        case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
            dst[0] = src[x * 4]; dst[1] = src[x * 4 + 1]; dst[2] = src[x * 4 + 2];
            break;
        case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
            dst[0] = src[x * 4 + 2]; dst[1] = src[x * 4 + 1]; dst[2] = src[x * 4];
            break;
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2R10G10B10_UNORM_PACK32:
        {
            uint32_t v;
            memcpy(&v, src + x * 4, 4);
            const uint8_t lo = (uint8_t)((v & 1023) >> 2), mid = (uint8_t)(((v >> 10) & 1023) >> 2), hi = (uint8_t)(((v >> 20) & 1023) >> 2);
            const bool redLow = f == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
            dst[0] = redLow ? hi : lo; dst[1] = mid; dst[2] = redLow ? lo : hi;
            break;
        }
        default:
        {
            uint16_t c[3];
            memcpy(c, src + x * 8, 6);
            for (int k = 0; k < 3; ++k)
            {
                const float v = HalfToFloat(c[k]);
                dst[2 - k] = (uint8_t)(v <= 0.f ? 0 : v >= 1.f ? 255 : (int)(v * 255.f + 0.5f));
            }
            break;
        }
        }
    }
}

// DX11's file types: 2 and 3 are JPEG bytes (3 as .jps when stereo), 4 a bitmap, everything else PNG (1 as .pns).
const wchar_t* Extension(bool stereo)
{
    switch (gInfo.ScreenshotType)
    {
    case 1: return stereo ? L".pns" : L".png";
    case 2: return L".jpg";
    case 3: return stereo ? L".jps" : L".jpg";
    case 4: return L".bmp";
    default: return L".png";
    }
}

// The autotests' MakeScreenshotPath, else My Pictures\<product> Screenshots\<profile>\<profile> #NNNN<ext>.
bool ShotPath(bool stereo, wchar_t (&out)[MAX_PATH])
{
    if (gInfo.MakeScreenshot)
    {
        wcscpy_s(out, gInfo.MakeScreenshotPath);
        return true;
    }
    wchar_t dir[MAX_PATH] = {}, name[MAX_PATH] = {};
    if (FAILED(SHGetFolderPathW(nullptr, CSIDL_MYPICTURES | CSIDL_FLAG_CREATE, nullptr, 0, dir))) return false;
    PathAppendW(dir, _T(PRODUCT_NAME) L" Screenshots");
    CreateDirectoryW(dir, nullptr);
    wcscpy_s(name, gInfo.ProfileName);
    for (wchar_t* c = name; *c; ++c) if (wcschr(L"\\/:*?\"<>|", *c)) *c = L'_';
    PathAppendW(dir, name);
    CreateDirectoryW(dir, nullptr);
    static unsigned s_count = 0;
    do swprintf_s(out, L"%s\\%s #%04u%s", dir, name, s_count++, Extension(stereo));
    while (PathFileExistsW(out));
    return true;
}

// Windows' own codecs write the file: PNG, JPEG or BMP from 24-bit rows.
bool WriteImageFile(const wchar_t* path, const uint8_t* bgr, uint32_t w, uint32_t h)
{
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    const GUID& container = gInfo.ScreenshotType == 2 || gInfo.ScreenshotType == 3 ? GUID_ContainerFormatJpeg
                          : gInfo.ScreenshotType == 4 ? GUID_ContainerFormatBmp : GUID_ContainerFormatPng;
    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapEncoder* encoder = nullptr;
    IWICBitmapFrameEncode* frame = nullptr;
    IPropertyBag2* props = nullptr;
    WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
    HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory));
    if (SUCCEEDED(hr)) hr = factory->CreateStream(&stream);
    if (SUCCEEDED(hr)) hr = stream->InitializeFromFilename(path, GENERIC_WRITE);
    if (SUCCEEDED(hr)) hr = factory->CreateEncoder(container, nullptr, &encoder);
    if (SUCCEEDED(hr)) hr = encoder->Initialize(stream, WICBitmapEncoderNoCache);
    if (SUCCEEDED(hr)) hr = encoder->CreateNewFrame(&frame, &props);
    if (SUCCEEDED(hr) && container == GUID_ContainerFormatJpeg)
    {
        PROPBAG2 option = {};
        option.pstrName = const_cast<LPOLESTR>(L"ImageQuality");
        VARIANT quality;
        VariantInit(&quality);
        quality.vt = VT_R4;
        quality.fltVal = 0.95f;
        props->Write(1, &option, &quality);
    }
    if (SUCCEEDED(hr)) hr = frame->Initialize(props);
    if (SUCCEEDED(hr)) hr = frame->SetSize(w, h);
    if (SUCCEEDED(hr)) hr = frame->SetPixelFormat(&format);
    if (SUCCEEDED(hr) && format != GUID_WICPixelFormat24bppBGR) hr = E_FAIL;
    if (SUCCEEDED(hr)) hr = frame->WritePixels(h, w * 3, w * 3 * h, const_cast<BYTE*>(bgr));
    if (SUCCEEDED(hr)) hr = frame->Commit();
    if (SUCCEEDED(hr)) hr = encoder->Commit();
    if (props) props->Release();
    if (frame) frame->Release();
    if (encoder) encoder->Release();
    if (stream) stream->Release();
    if (factory) factory->Release();
    if (SUCCEEDED(init)) CoUninitialize();
    if (FAILED(hr)) Log("Screenshot: the image codec failed 0x%08lX\n", hr);
    return SUCCEEDED(hr);
}

// An autotest's request ends with its flag down and its callback called, written or not.
void RequestDone()
{
    if (!gInfo.MakeScreenshot) return;
    gInfo.MakeScreenshot = FALSE;
    if (gInfo.MakeScreenshotCallback) gInfo.MakeScreenshotCallback();
}

void ShotFree(DeviceData* d)
{
    if (s_shot.mapped) d->UnmapMemory(d->device, s_shot.memory);
    if (s_shot.buffer) d->DestroyBuffer(d->device, s_shot.buffer, nullptr);
    if (s_shot.memory) d->FreeMemory(d->device, s_shot.memory, nullptr);
    s_shot = Shot();
}

} // namespace

// Once per present: the hotkey's request (or an autotest's) stays armed until a present copies its eyes out.
void ScreenshotTick()
{
    if (gKbdHook.TakeScreenshotRequest() || gInfo.MakeScreenshot) s_armed = true;
}

bool ScreenshotArmed() { return s_armed; }

// Lock held. Both images are w x h in TRANSFER_SRC_OPTIMAL. `right` null: one eye. `squeezed`: `left` already holds
// both eyes, the left one first, and its halves are written swapped.
void ScreenshotCopy(DeviceData* d, VkCommandBuffer cb, VkImage left, VkImage right, VkFormat format, uint32_t w, uint32_t h, bool squeezed)
{
    if (!s_armed || s_shot.buffer) return;
    s_armed = false;
    const uint32_t bpp = BytesPerPixel(format);
    if (!bpp) { Log("Screenshot: format %d is not written\n", (int)format); RequestDone(); return; }
    const uint32_t views = (right && !squeezed) ? 2 : 1;
    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = (VkDeviceSize)w * views * h * bpp;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d->CreateBuffer(d->device, &bci, nullptr, &s_shot.buffer) != VK_SUCCESS) { s_shot = Shot(); RequestDone(); return; }
    VkMemoryRequirements req;
    d->GetBufferMemoryRequirements(d->device, s_shot.buffer, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    // Cached memory first: the whole picture is read back, which write-combined memory makes slow.
    const VkMemoryPropertyFlags coherent = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, coherent | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX) mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, coherent, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->device, &mai, nullptr, &s_shot.memory) != VK_SUCCESS ||
        d->BindBufferMemory(d->device, s_shot.buffer, s_shot.memory, 0) != VK_SUCCESS ||
        d->MapMemory(d->device, s_shot.memory, 0, VK_WHOLE_SIZE, 0, &s_shot.mapped) != VK_SUCCESS)
    {
        Log("Screenshot: no memory for a %ux%u picture\n", w * views, h);
        ShotFree(d);
        RequestDone();
        return;
    }
    s_shot.format = format; s_shot.w = w; s_shot.h = h; s_shot.views = views;
    s_shot.stereo = views == 2 || squeezed;
    // Each copy lands in its half of the picture's rows: the buffer's row length is the whole picture's width.
    VkBufferImageCopy copy = {};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.bufferRowLength = w * views;
    if (squeezed)
    {
        const uint32_t half = w / 2;
        copy.imageOffset = { (int32_t)half, 0, 0 };
        copy.imageExtent = { w - half, h, 1 };
        d->CmdCopyImageToBuffer(cb, left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_shot.buffer, 1, &copy);
        copy.bufferOffset = (VkDeviceSize)(w - half) * bpp;
        copy.imageOffset = { 0, 0, 0 };
        copy.imageExtent = { half, h, 1 };
        d->CmdCopyImageToBuffer(cb, left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_shot.buffer, 1, &copy);
        return;
    }
    copy.imageExtent = { w, h, 1 };
    d->CmdCopyImageToBuffer(cb, views == 2 ? right : left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_shot.buffer, 1, &copy);
    if (views == 2)
    {
        copy.bufferOffset = (VkDeviceSize)w * bpp;
        d->CmdCopyImageToBuffer(cb, left, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, s_shot.buffer, 1, &copy);
    }
}

// Lock held, after the submit that holds the copies: waits for them, as DX11's read-back does, and writes the file.
void ScreenshotWrite(DeviceData* d, VkQueue queue)
{
    if (!s_shot.buffer) return;
    { const auto ql = FgQueueLock(); d->QueueWaitIdle(queue); }
    const uint32_t width = s_shot.w * s_shot.views, bpp = BytesPerPixel(s_shot.format);
    std::vector<uint8_t> bgr((size_t)width * s_shot.h * 3);
    for (uint32_t y = 0; y < s_shot.h; ++y)
        RowToBgr(s_shot.format, (const uint8_t*)s_shot.mapped + (size_t)y * width * bpp, bgr.data() + (size_t)y * width * 3, width);
    wchar_t path[MAX_PATH] = {};
    const bool named = ShotPath(s_shot.stereo, path);
    const bool saved = named && WriteImageFile(path, bgr.data(), width, s_shot.h);
    Log("Screenshot: %ls (%ux%u, format %d, %s) saved=%d\n", named ? path : L"no path", width, s_shot.h, (int)s_shot.format,
        s_shot.stereo ? "right eye | left eye" : "one image", (int)saved);
    ShotFree(d);
    RequestDone();
}

void ScreenshotDestroy(DeviceData* d) { ShotFree(d); }

}} // namespace wiz3d::vk
