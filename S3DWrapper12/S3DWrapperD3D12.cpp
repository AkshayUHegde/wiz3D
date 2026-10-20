/* wiz3D - S3DWrapperD3D12.dll
 *
 * The DX12 stereo wrapper. Currently passthrough-only — establishes the
 * InitializeExchangeServer protocol that wiz3D-proxy/d3d12/d3d12.dll uses
 * to hand off device creation, but performs no stereo injection yet.
 *
 * M0 groundwork. See PLAN.md for the full architecture this is building
 * toward (ReShade-based command-list interception, matrix manager,
 * engine-aware paths, etc.).
 *
 * The DllMain pattern mirrors S3DWrapperD3D9.cpp's loader-lock-safe design:
 * trivial DllMain DLL_PROCESS_ATTACH, all heavy work deferred to
 * EnsureWrapperInitialized() called from InitializeExchangeServer or the
 * D3D12CreateDevice export — i.e. AFTER LoadLibraryW returns and the
 * loader lock is released. (The DX9 wrapper used to call Direct3DCreate9
 * inside DllMain, which deadlocked AMD's UMD on launch. We don't want to
 * repeat that mistake on DX12.)
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include "Dx12StereoChain.h"

// ---------------------------------------------------------------------------
// Diagnostic log
// ---------------------------------------------------------------------------
static HMODULE g_hSelf = nullptr;

static void WriteLog(const char* fmt, ...)
{
    wchar_t dir[MAX_PATH] = {};
    if (g_hSelf)
    {
        GetModuleFileNameW(g_hSelf, dir, MAX_PATH);
        wchar_t* p = wcsrchr(dir, L'\\');
        if (p) p[1] = L'\0';
    }
    if (!dir[0] && !GetTempPathW(MAX_PATH, dir)) return;

    wchar_t path[MAX_PATH];
    wcscpy_s(path, dir);
    wcscat_s(path, L"wiz3D_Logs");
    CreateDirectoryW(path, NULL);
    wcscat_s(path, L"\\S3DWrapperD3D12.log");
    HANDLE h = CreateFileW(path, FILE_APPEND_DATA,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    SetFilePointer(h, 0, nullptr, FILE_END);

    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
    va_end(ap);
    if (n > 0)
    {
        DWORD written;
        WriteFile(h, buf, (DWORD)n, &written, nullptr);
    }
    CloseHandle(h);
}

// ---------------------------------------------------------------------------
// Lazy init — runs the first time the proxy calls into us. AFTER loader lock
// is released. Currently a no-op apart from one-time logging.
// ---------------------------------------------------------------------------
static volatile LONG g_bInitialized = 0;
static HMODULE g_hRealD3D12 = nullptr;

typedef HRESULT (WINAPI *PFN_D3D12CreateDevice)(void*, int, const GUID*, void**);
static PFN_D3D12CreateDevice g_pfnRealCreateDevice = nullptr;

static BOOL EnsureRealD3D12Loaded()
{
    if (g_pfnRealCreateDevice) return TRUE;

    wchar_t sysDir[MAX_PATH];
    GetSystemDirectoryW(sysDir, MAX_PATH);
    wchar_t path[MAX_PATH];
    wcscpy_s(path, sysDir);
    wcscat_s(path, L"\\d3d12.dll");

    g_hRealD3D12 = LoadLibraryW(path);
    if (!g_hRealD3D12)
    {
        WriteLog("[wiz3D-D3D12] FAIL: could not load real d3d12.dll (err=%lu)\n", GetLastError());
        return FALSE;
    }

    g_pfnRealCreateDevice = (PFN_D3D12CreateDevice)
        GetProcAddress(g_hRealD3D12, "D3D12CreateDevice");
    if (!g_pfnRealCreateDevice)
    {
        WriteLog("[wiz3D-D3D12] FAIL: real d3d12.dll has no D3D12CreateDevice export\n");
        return FALSE;
    }
    WriteLog("[wiz3D-D3D12] Real d3d12.dll resolved at %p\n", g_pfnRealCreateDevice);
    return TRUE;
}

void EnsureWrapperInitialized()
{
    if (InterlockedCompareExchange(&g_bInitialized, 1, 0) != 0) return;
    WriteLog("[wiz3D-D3D12] EnsureWrapperInitialized — passthrough mode\n");
    EnsureRealD3D12Loaded();
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------

// Router type:
//   0 → use this wrapper's D3D12CreateDevice export
//   1 → reserved (matches S3DInjector contract from DX9 wrapper)
//  >1 → wrapper declines, proxy falls back to real DLL
// Currently always 0 — passthrough wrapper still wants to be in the chain so
// future stereo work can hook the device creation point without changing
// the proxy.
extern "C" __declspec(dllexport) DWORD WINAPI InitializeExchangeServer()
{
    EnsureWrapperInitialized();
    return 0;
}

extern "C" __declspec(dllexport) HRESULT WINAPI D3D12CreateDevice(
    void* pAdapter, int MinimumFeatureLevel, const GUID* riid, void** ppDevice)
{
    EnsureWrapperInitialized();
    if (!g_pfnRealCreateDevice) return E_FAIL;

    HRESULT hr = g_pfnRealCreateDevice(pAdapter, MinimumFeatureLevel, riid, ppDevice);
    WriteLog("[wiz3D-D3D12] D3D12CreateDevice (passthrough) -> 0x%08lX\n", hr);
    // TODO: when stereo work begins, wrap the returned ID3D12Device here
    return hr;
}

// ---------------------------------------------------------------------------
// Dx12Stereo_Create — C export called by D3d12VendorProxy's hooked
// CreateSwapChainForHwnd. Forwards to the wrapper namespace. Signature must
// match wiz3d::Dx12Stereo::CreateForHwndFn.
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport) HRESULT WINAPI Dx12Stereo_Create(
    wiz3d::Dx12Stereo::CreateForHwndFn native,
    IDXGIFactory2*  factory,
    IUnknown*       opaque,
    HWND            window,
    const DXGI_SWAP_CHAIN_DESC1* desc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
    IDXGIOutput*    output,
    IDXGISwapChain1** result)
{
    EnsureWrapperInitialized();
    return wiz3d::Dx12Stereo::Create(native, factory, opaque, window,
                                      desc, fullscreen, output, result);
}

extern "C" __declspec(dllexport) void WINAPI Dx12Stereo_SetEnabled(int enabled)
{
    wiz3d::Dx12Stereo::SetEnabled(enabled != 0);
    WriteLog("[wiz3D-D3D12] Dx12Stereo_SetEnabled(%d)\n", enabled);
}

// Set compositor output mode: 0 = MonoMirror (debug), 1 = SBS (default).
// Only affects swap chains created after this call. In-flight chains keep
// their mode because changing it mid-chain would require re-creating the
// underlying real DXGI swap chain at a different width.
extern "C" __declspec(dllexport) void WINAPI Dx12Stereo_SetOutputMode(int mode)
{
    wiz3d::Dx12Stereo::SetOutputMode(
        mode == 0 ? wiz3d::Dx12Stereo::OutputMode::MonoMirror
                  : wiz3d::Dx12Stereo::OutputMode::Sbs);
    WriteLog("[wiz3D-D3D12] Dx12Stereo_SetOutputMode(%d)\n", mode);
}

// ---------------------------------------------------------------------------
// DllMain — kept trivial. No D3D calls, no LoadLibrary of system DLLs, no
// thread spawns. See header comment + the DX9 wrapper's loader-lock fix.
// ---------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_hSelf = hModule;
        DisableThreadLibraryCalls(hModule);
        WriteLog("[wiz3D-D3D12] DLL_PROCESS_ATTACH\n");
    }
    else if (reason == DLL_PROCESS_DETACH)
    {
        if (g_hRealD3D12)
        {
            FreeLibrary(g_hRealD3D12);
            g_hRealD3D12 = nullptr;
            g_pfnRealCreateDevice = nullptr;
        }
    }
    return TRUE;
}
