/* wiz3D - DX12 stereo swap-chain wrapper (implementation)
 *
 * Adapted from farmerarmor/RiseTombRaiderVR src/Dx12Stereo.cpp (LGPL 2.1-compatible,
 * our rev 981de7b vendored there). Rise's Publication/OpenXR companion-process
 * layer is replaced with an in-process desktop composite — first pass is a
 * mono-mirror of eye 0 (matches Rise's own desktop behaviour), stereo output
 * modes (SBS / TAB / Anaglyph) layer on top in a follow-up compositor file.
 *
 * See Dx12StereoChain.h for architectural overview.
 */

#include "Dx12StereoChain.h"
#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include <atomic>
#include <mutex>
#include <cstdio>
#include <cstdarg>
#include <new>

using Microsoft::WRL::ComPtr;

namespace wiz3d {
namespace Dx12Stereo {

namespace {

// ---------------------------------------------------------------------------
// Diagnostic log — writes to S3DWrapperD3D12.log beside this DLL. Appends so
// multiple proxies in the game folder share a single log.
// ---------------------------------------------------------------------------
HMODULE g_hSelfLog = nullptr;

void Log(const char* fmt, ...)
{
    wchar_t dir[MAX_PATH] = {};
    if (!g_hSelfLog)
    {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&Log), &g_hSelfLog);
    }
    if (g_hSelfLog) GetModuleFileNameW(g_hSelfLog, dir, MAX_PATH);
    wchar_t* p = wcsrchr(dir, L'\\');
    if (p) p[1] = L'\0'; else if (!GetTempPathW(MAX_PATH, dir)) return;

    wchar_t path[MAX_PATH];
    wcscpy_s(path, dir);
    wcscat_s(path, L"wiz3D_Logs");
    CreateDirectoryW(path, NULL);
    wcscat_s(path, L"\\S3DWrapperD3D12.log");

    HANDLE h = CreateFileW(path, FILE_APPEND_DATA,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;

    char line[512];
    va_list ap;
    va_start(ap, fmt);
    int n = _vsnprintf_s(line, sizeof(line), _TRUNCATE, fmt, ap);
    va_end(ap);
    if (n > 0)
    {
        char full[576];
        int m = _snprintf_s(full, sizeof(full), _TRUNCATE,
                            "[Dx12Stereo %llu] %s\n",
                            (unsigned long long)GetTickCount64(), line);
        if (m > 0)
        {
            DWORD written;
            WriteFile(h, full, (DWORD)m, &written, nullptr);
        }
    }
    CloseHandle(h);
}

// ---------------------------------------------------------------------------
// Runtime enable flag. Default on; D3d12VendorProxy clears via SetEnabled(false)
// when it reads the config XML and finds Dx12StereoEnabled = 0.
// ---------------------------------------------------------------------------
std::atomic<bool> g_enabled{true};

// Compositor output mode. Default to SBS so a shipped build produces visible
// stereo. MonoMirror remains available as a fallback / diagnostic knob.
std::atomic<OutputMode> g_outputMode{OutputMode::Sbs};

// ---------------------------------------------------------------------------
// Resource-barrier helper (lifted verbatim from Rise)
// ---------------------------------------------------------------------------
D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* r,
                                   D3D12_RESOURCE_STATES from,
                                   D3D12_RESOURCE_STATES to)
{
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource   = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = from;
    b.Transition.StateAfter  = to;
    return b;
}

// ---------------------------------------------------------------------------
// StereoChain — IDXGISwapChain3 wrapper that gives the game two eye slices
// per back buffer and composites to the (mono) desktop chain on Present.
// ---------------------------------------------------------------------------
class StereoChain final : public IDXGISwapChain3
{
    std::atomic<ULONG>              refs{1};
    ComPtr<IDXGISwapChain3>         real;
    ComPtr<ID3D12Device>            device;
    ComPtr<ID3D12CommandQueue>      queue;

    struct Buffer
    {
        ComPtr<ID3D12Resource>              eyes;       // ArraySize=2, per-eye W x H
        ComPtr<ID3D12CommandAllocator>      allocator;
        ComPtr<ID3D12GraphicsCommandList>   list;
        uint64_t                            submitted{};
    };
    std::vector<Buffer>             buffers;
    ComPtr<ID3D12Fence>             completed;
    HANDLE                          event{};
    uint64_t                        submission{};
    std::mutex                      guard;
    UINT                            tearingFlag{};

    // Game's requested per-eye dimensions. Separate from the real swap chain
    // dimensions because in SBS output mode the real chain is 2x the game's
    // width (half-SBS: DXGI stretches 2W to W for the window). GetDesc1 lies
    // to the game with these values + Stereo=TRUE.
    UINT                            m_gameWidth{0};
    UINT                            m_gameHeight{0};

    // Output mode captured at Init time (immutable per-chain; changing the
    // global SetOutputMode affects future chains, not this one, because
    // changing it mid-chain would require re-creating the underlying real
    // swap chain at a different width).
    OutputMode                      m_mode{OutputMode::Sbs};

    UINT PresentFlags(UINT flags) const
    {
        flags &= ~(DXGI_PRESENT_STEREO_PREFER_RIGHT |
                   DXGI_PRESENT_STEREO_TEMPORARY_MONO |
                   DXGI_PRESENT_ALLOW_TEARING);
        if (tearingFlag) flags |= DXGI_PRESENT_ALLOW_TEARING;
        return flags;
    }

    HRESULT Wait(uint64_t value)
    {
        if (!value || completed->GetCompletedValue() >= value) return S_OK;
        HRESULT hr = completed->SetEventOnCompletion(value, event);
        if (FAILED(hr)) return hr;
        DWORD r = WaitForSingleObject(event, 2000);
        return r == WAIT_OBJECT_0 ? S_OK :
               HRESULT_FROM_WIN32(r == WAIT_TIMEOUT ? ERROR_TIMEOUT : GetLastError());
    }

    HRESULT Drain()
    {
        if (!completed) return S_OK;
        uint64_t v = ++submission;
        HRESULT hr = queue->Signal(completed.Get(), v);
        return SUCCEEDED(hr) ? Wait(v) : hr;
    }

    HRESULT Allocate()
    {
        DXGI_SWAP_CHAIN_DESC1 desc{};
        HRESULT hr = real->GetDesc1(&desc);
        if (FAILED(hr)) return hr;
        tearingFlag = desc.Flags & DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

        // Eye textures are at the GAME's per-eye dimensions, not the real
        // chain's — in SBS mode the real chain is 2 x game width.
        const UINT eyeW = m_gameWidth  ? m_gameWidth  : desc.Width;
        const UINT eyeH = m_gameHeight ? m_gameHeight : desc.Height;

        if (!eyeW || !eyeH || eyeW > 16384 || eyeH > 8192 ||
            desc.SampleDesc.Count != 1 ||
            !(desc.Format == DXGI_FORMAT_R8G8B8A8_UNORM ||
              desc.Format == DXGI_FORMAT_B8G8R8A8_UNORM))
        {
            Log("Allocate: unsupported chain format=%u sample=%u game=%ux%u real=%ux%u",
                desc.Format, desc.SampleDesc.Count, eyeW, eyeH, desc.Width, desc.Height);
            return DXGI_ERROR_UNSUPPORTED;
        }

        buffers.clear();
        buffers.resize(desc.BufferCount);

        D3D12_HEAP_PROPERTIES heap{};
        heap.Type             = D3D12_HEAP_TYPE_DEFAULT;
        heap.CreationNodeMask = 1;
        heap.VisibleNodeMask  = 1;

        D3D12_RESOURCE_DESC target{};
        target.Dimension        = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        target.Width            = eyeW;
        target.Height           = eyeH;
        target.DepthOrArraySize = 2;
        target.MipLevels        = 1;
        target.Format           = desc.Format;
        target.SampleDesc.Count = 1;
        target.Flags            = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

        for (auto& b : buffers)
        {
            hr = device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &target,
                D3D12_RESOURCE_STATE_PRESENT, nullptr,
                IID_PPV_ARGS(&b.eyes));
            if (SUCCEEDED(hr))
                hr = device->CreateCommandAllocator(
                    D3D12_COMMAND_LIST_TYPE_DIRECT,
                    IID_PPV_ARGS(&b.allocator));
            if (SUCCEEDED(hr))
                hr = device->CreateCommandList(
                    0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                    b.allocator.Get(), nullptr,
                    IID_PPV_ARGS(&b.list));
            if (SUCCEEDED(hr))
                hr = b.list->Close();
            if (FAILED(hr))
            {
                Log("Allocate: per-buffer setup failed hr=%08lx", hr);
                return hr;
            }
        }

        Log("Allocate: %u back buffers (eye=%ux%u arr=2 fmt=%u, real desktop=%ux%u, mode=%s)",
            desc.BufferCount, eyeW, eyeH, desc.Format,
            desc.Width, desc.Height,
            m_mode == OutputMode::Sbs ? "SBS" : "MonoMirror");
        return S_OK;
    }

    // Compositor pass executed on each Present. Writes the two eye slices
    // into the real (desktop) back buffer according to the output mode
    // chosen at chain creation time:
    //
    //   MonoMirror: real BB is at game dims (W x H); slice 0 copies to
    //               the whole BB (user sees left eye only; diagnostic path).
    //   Sbs:        real BB is at 2W x H; slice 0 -> left half (0..W),
    //               slice 1 -> right half (W..2W). Window stretches 2W to W
    //               for standard half-SBS output on a 3D TV / anaglyph glasses
    //               / downstream SBS compositor (SR Weave etc.)
    HRESULT CopyPair()
    {
        UINT index = real->GetCurrentBackBufferIndex();
        if (index >= buffers.size()) return E_UNEXPECTED;

        auto& buffer = buffers[index];
        HRESULT hr = Wait(buffer.submitted);
        if (FAILED(hr)) return hr;

        ComPtr<ID3D12Resource> desktop;
        hr = real->GetBuffer(index, IID_PPV_ARGS(&desktop));
        if (FAILED(hr)) return hr;

        hr = buffer.allocator->Reset();
        if (SUCCEEDED(hr))
            hr = buffer.list->Reset(buffer.allocator.Get(), nullptr);
        if (FAILED(hr)) return hr;

        auto list = buffer.list.Get();

        D3D12_RESOURCE_BARRIER barriers[] = {
            Transition(buffer.eyes.Get(),
                       D3D12_RESOURCE_STATE_PRESENT,
                       D3D12_RESOURCE_STATE_COPY_SOURCE),
            Transition(desktop.Get(),
                       D3D12_RESOURCE_STATE_PRESENT,
                       D3D12_RESOURCE_STATE_COPY_DEST)
        };
        list->ResourceBarrier(2, barriers);

        D3D12_TEXTURE_COPY_LOCATION src{};
        src.pResource = buffer.eyes.Get();
        src.Type      = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;

        D3D12_TEXTURE_COPY_LOCATION dst{};
        dst.pResource        = desktop.Get();
        dst.Type             = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dst.SubresourceIndex = 0;

        if (m_mode == OutputMode::Sbs)
        {
            // Left eye -> left half of the doubled desktop
            src.SubresourceIndex = 0;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            // Right eye -> right half (DstX offset by game width)
            src.SubresourceIndex = 1;
            list->CopyTextureRegion(&dst, m_gameWidth, 0, 0, &src, nullptr);
        }
        else
        {
            // Mono-mirror: slice 0 -> whole desktop (same size as eye)
            src.SubresourceIndex = 0;
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }

        // Restore to PRESENT
        for (auto& b : barriers)
            std::swap(b.Transition.StateBefore, b.Transition.StateAfter);
        list->ResourceBarrier(2, barriers);

        hr = list->Close();
        if (FAILED(hr)) return hr;

        ID3D12CommandList* lists[] = { list };
        queue->ExecuteCommandLists(1, lists);

        buffer.submitted = ++submission;
        hr = queue->Signal(completed.Get(), submission);
        return hr;
    }

public:
    StereoChain(IDXGISwapChain3* swap, ID3D12CommandQueue* q,
                UINT gameWidth, UINT gameHeight, OutputMode mode)
        : real(swap), queue(q),
          m_gameWidth(gameWidth), m_gameHeight(gameHeight), m_mode(mode) {}

    ~StereoChain()
    {
        Drain();
        if (event) CloseHandle(event);
    }

    HRESULT Init()
    {
        HRESULT hr = queue->GetDevice(IID_PPV_ARGS(&device));
        if (FAILED(hr)) return hr;

        // Rise rejects multi-node and non-direct queues; same constraints apply.
        if (device->GetNodeCount() != 1 ||
            queue->GetDesc().Type != D3D12_COMMAND_LIST_TYPE_DIRECT)
            return DXGI_ERROR_UNSUPPORTED;

        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) return HRESULT_FROM_WIN32(GetLastError());

        hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                 IID_PPV_ARGS(&completed));
        return SUCCEEDED(hr) ? Allocate() : hr;
    }

    // --- IUnknown -----------------------------------------------------------
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID iid, void** result) override
    {
        if (!result) return E_POINTER;
        *result = nullptr;
        if (iid == __uuidof(IUnknown) ||
            iid == __uuidof(IDXGIObject) ||
            iid == __uuidof(IDXGIDeviceSubObject) ||
            iid == __uuidof(IDXGISwapChain) ||
            iid == __uuidof(IDXGISwapChain1) ||
            iid == __uuidof(IDXGISwapChain2) ||
            iid == __uuidof(IDXGISwapChain3))
        {
            *result = static_cast<IDXGISwapChain3*>(this);
            AddRef();
            return S_OK;
        }
        return E_NOINTERFACE;
    }
    ULONG STDMETHODCALLTYPE AddRef()  override { return ++refs; }
    ULONG STDMETHODCALLTYPE Release() override
    {
        ULONG n = --refs;
        if (!n) delete this;
        return n;
    }

    // --- IDXGIObject --------------------------------------------------------
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID a, UINT b, const void* c) override { return real->SetPrivateData(a, b, c); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID a, const IUnknown* b) override { return real->SetPrivateDataInterface(a, b); }
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID a, UINT* b, void* c) override { return real->GetPrivateData(a, b, c); }
    HRESULT STDMETHODCALLTYPE GetParent(REFIID a, void** b) override { return real->GetParent(a, b); }

    // --- IDXGIDeviceSubObject ----------------------------------------------
    HRESULT STDMETHODCALLTYPE GetDevice(REFIID a, void** b) override { return real->GetDevice(a, b); }

    // Report the game's per-eye dimensions back to the game, not the real
    // desktop dimensions (which in SBS mode are 2 x game width).
    void ApplyGameDimensions(DXGI_SWAP_CHAIN_DESC* desc) const
    {
        if (desc && m_gameWidth && m_gameHeight)
        {
            desc->BufferDesc.Width  = m_gameWidth;
            desc->BufferDesc.Height = m_gameHeight;
        }
    }
    void ApplyGameDimensions1(DXGI_SWAP_CHAIN_DESC1* desc) const
    {
        if (desc && m_gameWidth && m_gameHeight)
        {
            desc->Width  = m_gameWidth;
            desc->Height = m_gameHeight;
        }
    }

    // --- IDXGISwapChain -----------------------------------------------------
    HRESULT STDMETHODCALLTYPE Present(UINT interval, UINT flags) override
    {
        if (flags & DXGI_PRESENT_TEST)
            return real->Present(interval, flags);

        std::lock_guard<std::mutex> lock(guard);
        HRESULT hr = CopyPair();
        if (FAILED(hr)) { Log("Present: CopyPair failed hr=%08lx", hr); return hr; }
        return real->Present(interval, PresentFlags(flags));
    }

    HRESULT STDMETHODCALLTYPE GetBuffer(UINT index, REFIID iid, void** result) override
    {
        std::lock_guard<std::mutex> lock(guard);
        if (!result) return E_POINTER;
        *result = nullptr;
        if (index >= buffers.size()) return DXGI_ERROR_INVALID_CALL;
        return buffers[index].eyes->QueryInterface(iid, result);
    }

    // Monitor stereo tool: allow fullscreen (Rise refuses because OpenXR owns
    // the HMD display). We just forward through.
    HRESULT STDMETHODCALLTYPE SetFullscreenState(BOOL enabled, IDXGIOutput* output) override
    {
        Log("SetFullscreenState(%d) forwarded", enabled);
        return real->SetFullscreenState(enabled, output);
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenState(BOOL* a, IDXGIOutput** b) override { return real->GetFullscreenState(a, b); }

    HRESULT STDMETHODCALLTYPE GetDesc(DXGI_SWAP_CHAIN_DESC* desc) override
    {
        HRESULT hr = real->GetDesc(desc);
        if (SUCCEEDED(hr)) ApplyGameDimensions(desc);
        return hr;
    }

    HRESULT STDMETHODCALLTYPE ResizeBuffers(UINT count, UINT w, UINT h,
                                            DXGI_FORMAT format, UINT flags) override
    {
        // Preserve tearing flag through the resize; w/h are the game's per-eye
        // dimensions. In SBS mode the real chain goes to 2 x game width.
        flags = (flags & ~DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING) | tearingFlag;

        std::lock_guard<std::mutex> lock(guard);
        if (w) m_gameWidth  = w;
        if (h) m_gameHeight = h;

        UINT realW = w;
        UINT realH = h;
        if (m_mode == OutputMode::Sbs && realW) realW *= 2;

        Log("ResizeBuffers begin count=%u game=%ux%u real=%ux%u mode=%s submitted=%llu completed=%llu",
            count, w, h, realW, realH,
            m_mode == OutputMode::Sbs ? "SBS" : "MonoMirror",
            (unsigned long long)submission,
            (unsigned long long)completed->GetCompletedValue());

        HRESULT hr = Drain();
        if (FAILED(hr)) return hr;
        buffers.clear();
        hr = real->ResizeBuffers(count, realW, realH, format, flags);
        Log("ResizeBuffers real->ResizeBuffers hr=%08lx", hr);
        if (FAILED(hr)) return hr;
        return Allocate();
    }

    HRESULT STDMETHODCALLTYPE ResizeTarget(const DXGI_MODE_DESC* a) override { return real->ResizeTarget(a); }
    HRESULT STDMETHODCALLTYPE GetContainingOutput(IDXGIOutput** a) override { return real->GetContainingOutput(a); }
    HRESULT STDMETHODCALLTYPE GetFrameStatistics(DXGI_FRAME_STATISTICS* a) override { return real->GetFrameStatistics(a); }
    HRESULT STDMETHODCALLTYPE GetLastPresentCount(UINT* a) override { return real->GetLastPresentCount(a); }

    // --- IDXGISwapChain1 ----------------------------------------------------
    // The lie that makes the game believe it's rendering stereo at its own
    // per-eye dimensions (not the doubled real desktop in SBS mode).
    HRESULT STDMETHODCALLTYPE GetDesc1(DXGI_SWAP_CHAIN_DESC1* desc) override
    {
        HRESULT hr = real->GetDesc1(desc);
        if (SUCCEEDED(hr) && desc)
        {
            desc->Stereo = TRUE;
            ApplyGameDimensions1(desc);
        }
        return hr;
    }
    HRESULT STDMETHODCALLTYPE GetFullscreenDesc(DXGI_SWAP_CHAIN_FULLSCREEN_DESC* a) override { return real->GetFullscreenDesc(a); }
    HRESULT STDMETHODCALLTYPE GetHwnd(HWND* a) override { return real->GetHwnd(a); }
    HRESULT STDMETHODCALLTYPE GetCoreWindow(REFIID a, void** b) override { return real->GetCoreWindow(a, b); }

    HRESULT STDMETHODCALLTYPE Present1(UINT interval, UINT flags, const DXGI_PRESENT_PARAMETERS* p) override
    {
        if (flags & DXGI_PRESENT_TEST)
            return real->Present1(interval, flags, p);

        std::lock_guard<std::mutex> lock(guard);
        HRESULT hr = CopyPair();
        if (FAILED(hr)) return hr;
        return real->Present1(interval, PresentFlags(flags), p);
    }

    BOOL    STDMETHODCALLTYPE IsTemporaryMonoSupported() override { return TRUE; }
    HRESULT STDMETHODCALLTYPE GetRestrictToOutput(IDXGIOutput** a) override { return real->GetRestrictToOutput(a); }
    HRESULT STDMETHODCALLTYPE SetBackgroundColor(const DXGI_RGBA* a) override { return real->SetBackgroundColor(a); }
    HRESULT STDMETHODCALLTYPE GetBackgroundColor(DXGI_RGBA* a) override { return real->GetBackgroundColor(a); }
    HRESULT STDMETHODCALLTYPE SetRotation(DXGI_MODE_ROTATION a) override { return real->SetRotation(a); }
    HRESULT STDMETHODCALLTYPE GetRotation(DXGI_MODE_ROTATION* a) override { return real->GetRotation(a); }

    // --- IDXGISwapChain2 ----------------------------------------------------
    HRESULT STDMETHODCALLTYPE SetSourceSize(UINT a, UINT b) override { return real->SetSourceSize(a, b); }
    HRESULT STDMETHODCALLTYPE GetSourceSize(UINT* a, UINT* b) override { return real->GetSourceSize(a, b); }
    HRESULT STDMETHODCALLTYPE SetMaximumFrameLatency(UINT a) override { return real->SetMaximumFrameLatency(a); }
    HRESULT STDMETHODCALLTYPE GetMaximumFrameLatency(UINT* a) override { return real->GetMaximumFrameLatency(a); }
    HANDLE  STDMETHODCALLTYPE GetFrameLatencyWaitableObject() override { return real->GetFrameLatencyWaitableObject(); }
    HRESULT STDMETHODCALLTYPE SetMatrixTransform(const DXGI_MATRIX_3X2_F* a) override { return real->SetMatrixTransform(a); }
    HRESULT STDMETHODCALLTYPE GetMatrixTransform(DXGI_MATRIX_3X2_F* a) override { return real->GetMatrixTransform(a); }

    // --- IDXGISwapChain3 ----------------------------------------------------
    UINT    STDMETHODCALLTYPE GetCurrentBackBufferIndex() override { return real->GetCurrentBackBufferIndex(); }
    HRESULT STDMETHODCALLTYPE CheckColorSpaceSupport(DXGI_COLOR_SPACE_TYPE a, UINT* b) override { return real->CheckColorSpaceSupport(a, b); }
    HRESULT STDMETHODCALLTYPE SetColorSpace1(DXGI_COLOR_SPACE_TYPE a) override { return real->SetColorSpace1(a); }
    HRESULT STDMETHODCALLTYPE ResizeBuffers1(UINT count, UINT w, UINT h,
                                              DXGI_FORMAT format, UINT flags,
                                              const UINT* masks,
                                              IUnknown* const* queues) override
    {
        if (queues || masks) return DXGI_ERROR_UNSUPPORTED;
        return ResizeBuffers(count, w, h, format, flags);
    }
};

} // anonymous namespace

bool IsEnabled() { return g_enabled.load(); }
void SetEnabled(bool v) { g_enabled.store(v); }

OutputMode GetOutputMode() { return g_outputMode.load(); }
void SetOutputMode(OutputMode m) { g_outputMode.store(m); }

HRESULT Create(CreateForHwndFn native,
               IDXGIFactory2*  factory,
               IUnknown*       opaque,
               HWND            window,
               const DXGI_SWAP_CHAIN_DESC1* desc,
               const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
               IDXGIOutput*    output,
               IDXGISwapChain1** result)
{
    if (!desc || !opaque || !IsEnabled())
        return native(factory, opaque, window, desc, fullscreen, output, result);

    // Must be a DX12 command queue; DX11 chains fall through unchanged.
    ComPtr<ID3D12CommandQueue> q;
    if (FAILED(opaque->QueryInterface(IID_PPV_ARGS(&q))))
        return native(factory, opaque, window, desc, fullscreen, output, result);

    const OutputMode mode = g_outputMode.load();
    Log("Create: game DX12 chain request %ux%u fmt=%u stereo=%d buffers=%u mode=%s",
        desc->Width, desc->Height, desc->Format,
        desc->Stereo, desc->BufferCount,
        mode == OutputMode::Sbs ? "SBS" : "MonoMirror");

    // Game didn't ask for stereo — pass through, no wrapping.
    if (!desc->Stereo)
        return native(factory, opaque, window, desc, fullscreen, output, result);

    if (!result) return E_POINTER;
    *result = nullptr;

    // Mono copy of the desc for the DESKTOP chain. Game sees GetDesc1 report
    // Stereo=TRUE at per-eye dimensions via our wrapper's lie.
    DXGI_SWAP_CHAIN_DESC1 mono = *desc;
    mono.Stereo  = FALSE;
    mono.Scaling = DXGI_SCALING_STRETCH;

    // SBS: real desktop is 2 x per-eye width. DXGI stretches down to window
    // size for standard half-SBS output. Window dimensions are unchanged
    // (the hwnd stays at the game's expected size).
    if (mode == OutputMode::Sbs)
        mono.Width = desc->Width * 2;

    // Preserve tearing flag if supported.
    ComPtr<IDXGIFactory5> factory5;
    BOOL tearing{};
    if (SUCCEEDED(factory->QueryInterface(IID_PPV_ARGS(&factory5))) &&
        SUCCEEDED(factory5->CheckFeatureSupport(
            DXGI_FEATURE_PRESENT_ALLOW_TEARING, &tearing, sizeof(tearing))) &&
        tearing)
        mono.Flags |= DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    else
        mono.Flags &= ~DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;

    // Honour the game's fullscreen request (Rise forces windowed because
    // OpenXR owns the display; we're a monitor tool so we pass through).
    ComPtr<IDXGISwapChain1> real;
    HRESULT hr = native(factory, opaque, window, &mono, fullscreen, output, &real);
    ComPtr<IDXGISwapChain3> real3;
    if (SUCCEEDED(hr)) hr = real.As(&real3);
    if (FAILED(hr))
    {
        Log("Create: desktop chain creation failed hr=%08lx (requested %ux%u)",
            hr, mono.Width, mono.Height);
        return hr;
    }

    auto* wrapper = new (std::nothrow) StereoChain(
        real3.Get(), q.Get(),
        desc->Width, desc->Height, mode);
    if (!wrapper) return E_OUTOFMEMORY;

    hr = wrapper->Init();
    if (FAILED(hr))
    {
        wrapper->Release();
        Log("Create: StereoChain Init failed hr=%08lx", hr);
        return hr;
    }

    *result = wrapper;
    Log("Create: stereo wrapper installed; game=%ux%u Stereo=TRUE, real desktop=%ux%u, mode=%s",
        desc->Width, desc->Height, mono.Width, mono.Height,
        mode == OutputMode::Sbs ? "SBS half (DXGI stretches 2W to W)" : "MonoMirror eye 0");
    return S_OK;
}

} // namespace Dx12Stereo
} // namespace wiz3d
