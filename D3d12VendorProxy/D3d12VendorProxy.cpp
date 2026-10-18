// D3d12VendorProxy.cpp
// d3d12.dll proxy for AMD HD3D on DX12 titles (Sniper Elite 4 DX12 mode).
//
// Why d3d12.dll:
//   SE4 DX12 mode loads d3d12.dll for device creation.  d3d12.dll is NOT
//   pre-loaded by Steam's GameOverlayRenderer (which only pre-loads dxgi.dll),
//   so a local d3d12.dll wins the DLL search order and loads before the game's
//   import table is resolved.  We then install MinHook patches on the already-
//   loaded system dxgi.dll so CreateDXGIFactory* calls are intercepted regardless
//   of which DLL triggers them — same technique as D3d11VendorProxy.
//
// All hook infrastructure (DXGI VendorId spoof, LoadLibrary redirect, NvAPI
// blocking, registry spoofing) is identical to D3d11VendorProxy.  The only
// difference is which DLL we forward to, the export set, and the log name.
//
// AmdQbProxy compatibility with DX12:
//   AmdQbProxy hooks IDXGISwapChain::Present (a DXGI-level hook), so it fires
//   regardless of whether the swap chain was created by D3D11 or D3D12.
//   If the game creates a D3D11 device alongside its D3D12 device (e.g. via
//   D3D11On12CreateDevice) and calls AmdDxExtCreate11 with that D3D11 device,
//   AmdQbProxy's D3D11 compositor works as-is.  If the game is pure DX12 with
//   no D3D11 device, this needs further investigation — the log will show whether
//   AmdDxExtCreate11 is ever called.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <dxgi.h>
#include <dxgi1_2.h>
#include "MinHook.h"

static HINSTANCE g_hSelf     = nullptr;
static wchar_t   g_sysDir[MAX_PATH] = {};
static HMODULE   g_hRealD3D12 = nullptr;
static HMODULE   g_hRealDXGI  = nullptr;

// ---- Diagnostic log --------------------------------------------------------

static void WriteLog(const char* msg)
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
#ifdef BLOCK_NVAPI_STEREO
    wcscat_s(path, L"HD3D_dx12.log");
#else
    wcscat_s(path, L"3DV_dx12.log");
#endif
    HANDLE h = CreateFileW(path, FILE_APPEND_DATA,
                           FILE_SHARE_READ | FILE_SHARE_WRITE,
                           nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    DWORD written;
    WriteFile(h, msg, static_cast<DWORD>(strlen(msg)), &written, nullptr);
    CloseHandle(h);
}

static FARPROC RealD3D12(const char* name)
{
    if (!g_hRealD3D12 && g_sysDir[0])
    {
        wchar_t path[MAX_PATH];
        wcscpy_s(path, g_sysDir);
        wcscat_s(path, L"\\d3d12.dll");
        g_hRealD3D12 = LoadLibraryW(path);
        char buf[128];
        wsprintfA(buf, "[D3d12Proxy] real d3d12.dll %s (hMod=%p self=%p)\n",
                  g_hRealD3D12 ? "loaded OK" : "LOAD FAILED",
                  g_hRealD3D12, g_hSelf);
        WriteLog(buf);
    }
    return g_hRealD3D12 ? GetProcAddress(g_hRealD3D12, name) : nullptr;
}

// ---- DXGI vtable hook state ------------------------------------------------

typedef HRESULT (WINAPI* PFN_GetDesc)(IDXGIAdapter*, DXGI_ADAPTER_DESC*);
typedef HRESULT (WINAPI* PFN_GetDesc1)(IDXGIAdapter1*, DXGI_ADAPTER_DESC1*);
typedef HRESULT (WINAPI* PFN_EnumAdapters)(IDXGIFactory*, UINT, IDXGIAdapter**);
typedef HRESULT (WINAPI* PFN_EnumAdapters1)(IDXGIFactory1*, UINT, IDXGIAdapter1**);

static PFN_GetDesc       g_pfnGetDesc       = nullptr;
static PFN_GetDesc1      g_pfnGetDesc1      = nullptr;
static PFN_EnumAdapters  g_pfnEnumAdapters  = nullptr;
static PFN_EnumAdapters1 g_pfnEnumAdapters1 = nullptr;

static bool g_bGetDescHooked       = false;
static bool g_bGetDesc1Hooked      = false;
static bool g_bEnumAdaptersHooked  = false;
static bool g_bEnumAdapters1Hooked = false;

typedef BOOL (WINAPI* PFN_IsWindowedStereoEnabled)(IDXGIFactory2*);
typedef HRESULT (WINAPI* PFN_CreateSwapChain)(
    IDXGIFactory*, IUnknown*, DXGI_SWAP_CHAIN_DESC*, IDXGISwapChain**);
typedef HRESULT (WINAPI* PFN_CreateSwapChainForHwnd)(
    IDXGIFactory2*, IUnknown*, HWND, const DXGI_SWAP_CHAIN_DESC1*,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*, IDXGIOutput*, IDXGISwapChain1**);

static PFN_IsWindowedStereoEnabled g_pfnIsWindowedStereoEnabled = nullptr;
static PFN_CreateSwapChain         g_pfnCreateSwapChain         = nullptr;
static PFN_CreateSwapChainForHwnd  g_pfnCreateSwapChainForHwnd  = nullptr;

static bool g_bIsWindowedStereoHooked = false;
static bool g_bCreateSCHooked         = false;
static bool g_bCreateSCForHwndHooked  = false;

typedef HRESULT (WINAPI* PFN_CreateFactory)(REFIID, void**);
typedef HRESULT (WINAPI* PFN_CreateFactory2)(UINT, REFIID, void**);

static PFN_CreateFactory  g_pfnOrigCreateFactory  = nullptr;
static PFN_CreateFactory  g_pfnOrigCreateFactory1 = nullptr;
static PFN_CreateFactory2 g_pfnOrigCreateFactory2 = nullptr;

// ---- Hook implementations --------------------------------------------------

static HRESULT WINAPI HookedGetDesc(IDXGIAdapter* pAdapter, DXGI_ADAPTER_DESC* pDesc)
{
    HRESULT hr = g_pfnGetDesc(pAdapter, pDesc);
    if (SUCCEEDED(hr) && pDesc && pDesc->VendorId != 0x1002)
    {
        char buf[80];
        wsprintfA(buf, "[D3d12Proxy] GetDesc: VendorId 0x%04X->0x1002\n", pDesc->VendorId);
        WriteLog(buf);
        pDesc->VendorId = 0x1002;
        wcscpy_s(pDesc->Description, 128, L"AMD Radeon RX 580 Series");
    }
    return hr;
}

static HRESULT WINAPI HookedGetDesc1(IDXGIAdapter1* pAdapter, DXGI_ADAPTER_DESC1* pDesc)
{
    HRESULT hr = g_pfnGetDesc1(pAdapter, pDesc);
    if (SUCCEEDED(hr) && pDesc && pDesc->VendorId != 0x1002)
    {
        char buf[80];
        wsprintfA(buf, "[D3d12Proxy] GetDesc1: VendorId 0x%04X->0x1002\n", pDesc->VendorId);
        WriteLog(buf);
        pDesc->VendorId = 0x1002;
        wcscpy_s(pDesc->Description, 128, L"AMD Radeon RX 580 Series");
    }
    return hr;
}

static void HookAdapter(IDXGIAdapter* pA);

static HRESULT WINAPI HookedEnumAdapters(IDXGIFactory* pFactory, UINT Adapter, IDXGIAdapter** ppAdapter)
{
    HRESULT hr = g_pfnEnumAdapters(pFactory, Adapter, ppAdapter);
    if (SUCCEEDED(hr) && ppAdapter && *ppAdapter)
        HookAdapter(*ppAdapter);
    return hr;
}

static HRESULT WINAPI HookedEnumAdapters1(IDXGIFactory1* pFactory, UINT Adapter, IDXGIAdapter1** ppAdapter)
{
    HRESULT hr = g_pfnEnumAdapters1(pFactory, Adapter, ppAdapter);
    if (SUCCEEDED(hr) && ppAdapter && *ppAdapter)
        HookAdapter(static_cast<IDXGIAdapter*>(*ppAdapter));
    return hr;
}

static BOOL WINAPI HookedIsWindowedStereoEnabled(IDXGIFactory2* pFactory)
{
    BOOL real = g_pfnIsWindowedStereoEnabled(pFactory);
    char buf[96];
#ifdef BLOCK_NVAPI_STEREO
    wsprintfA(buf, "[D3d12Proxy] IsWindowedStereoEnabled: real=%d -> returning TRUE\n", real);
    WriteLog(buf);
    return TRUE;
#else
    wsprintfA(buf, "[D3d12Proxy] IsWindowedStereoEnabled: real=%d -> pass-through\n", real);
    WriteLog(buf);
    return real;
#endif
}

static HRESULT WINAPI HookedCreateSwapChain(
    IDXGIFactory* pFactory, IUnknown* pDevice, DXGI_SWAP_CHAIN_DESC* pDesc,
    IDXGISwapChain** ppSwapChain)
{
    char buf[256];
    if (pDesc)
    {
        wsprintfA(buf, "[D3d12Proxy] CreateSwapChain: %ux%u fmt=%u windowed=%d bufs=%u hwnd=%p\n",
                  pDesc->BufferDesc.Width, pDesc->BufferDesc.Height,
                  pDesc->BufferDesc.Format, pDesc->Windowed,
                  pDesc->BufferCount, pDesc->OutputWindow);
    }
    else
    {
        wsprintfA(buf, "[D3d12Proxy] CreateSwapChain: pDesc=NULL\n");
    }
    WriteLog(buf);

    HRESULT hr = g_pfnCreateSwapChain(pFactory, pDevice, pDesc, ppSwapChain);
    wsprintfA(buf, "[D3d12Proxy] CreateSwapChain -> 0x%08X\n", hr);
    WriteLog(buf);
    return hr;
}

// Lazy-loaded Dx12Stereo_Create export from S3DWrapperD3D12.dll. Loaded on
// first HookedCreateSwapChainForHwnd fire, cached thereafter. If the wrapper
// DLL isn't beside us or doesn't expose the export, we fall through to the
// native CreateSwapChainForHwnd directly (passthrough with logging only).
typedef HRESULT (WINAPI* PFN_Dx12Stereo_Create)(
    void* native,                                       /* native CreateSwapChainForHwnd fn ptr */
    IDXGIFactory2*, IUnknown*, HWND,
    const DXGI_SWAP_CHAIN_DESC1*,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*,
    IDXGIOutput*, IDXGISwapChain1**);

static HMODULE               g_hWrapperD3D12       = nullptr;
static PFN_Dx12Stereo_Create g_pfnDx12StereoCreate = nullptr;
static bool                  g_bWrapperProbed      = false;

static void LoadStereoWrapper()
{
    if (g_bWrapperProbed) return;
    g_bWrapperProbed = true;

    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(g_hSelf, path, MAX_PATH);
    wchar_t* sep = wcsrchr(path, L'\\');
    if (!sep) return;
    size_t rem = MAX_PATH - static_cast<size_t>(sep - path) - 1;
    wcscpy_s(sep + 1, rem, L"S3DWrapperD3D12.dll");

    g_hWrapperD3D12 = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!g_hWrapperD3D12)
    {
        WriteLog("[D3d12Proxy] S3DWrapperD3D12.dll not found beside proxy - stereo passthrough\n");
        return;
    }

    g_pfnDx12StereoCreate = reinterpret_cast<PFN_Dx12Stereo_Create>(
        GetProcAddress(g_hWrapperD3D12, "Dx12Stereo_Create"));

    char log[160];
    wsprintfA(log, "[D3d12Proxy] S3DWrapperD3D12.dll loaded (hMod=%p) Dx12Stereo_Create=%p\n",
              g_hWrapperD3D12, g_pfnDx12StereoCreate);
    WriteLog(log);
}

static HRESULT WINAPI HookedCreateSwapChainForHwnd(
    IDXGIFactory2* pFactory, IUnknown* pDevice, HWND hWnd,
    const DXGI_SWAP_CHAIN_DESC1* pDesc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* pFullscreenDesc,
    IDXGIOutput* pRestrictToOutput, IDXGISwapChain1** ppSwapChain)
{
    char buf[256];
    if (pDesc)
    {
        wsprintfA(buf, "[D3d12Proxy] CreateSwapChainForHwnd: %ux%u fmt=%u stereo=%d bufCnt=%u hwnd=%p\n",
                  pDesc->Width, pDesc->Height, pDesc->Format, pDesc->Stereo,
                  pDesc->BufferCount, hWnd);
    }
    else
    {
        wsprintfA(buf, "[D3d12Proxy] CreateSwapChainForHwnd: pDesc=NULL hwnd=%p\n", hWnd);
    }
    WriteLog(buf);

    LoadStereoWrapper();

    HRESULT hr;
    if (g_pfnDx12StereoCreate)
    {
        // Wrapper decides internally: if desc.Stereo and opaque is a DX12 queue,
        // wraps with StereoChain; otherwise falls through to native unchanged.
        hr = g_pfnDx12StereoCreate(
            reinterpret_cast<void*>(g_pfnCreateSwapChainForHwnd),
            pFactory, pDevice, hWnd, pDesc,
            pFullscreenDesc, pRestrictToOutput, ppSwapChain);
    }
    else
    {
        hr = g_pfnCreateSwapChainForHwnd(pFactory, pDevice, hWnd, pDesc,
                                          pFullscreenDesc, pRestrictToOutput, ppSwapChain);
    }
    wsprintfA(buf, "[D3d12Proxy] CreateSwapChainForHwnd -> 0x%08X (wrapperActive=%d)\n",
              hr, g_pfnDx12StereoCreate ? 1 : 0);
    WriteLog(buf);
    return hr;
}

// vtable slots (same as D3d11VendorProxy — DXGI vtables are identical):
//   IDXGIAdapter:  GetDesc[8]
//   IDXGIAdapter1: GetDesc1[10]
//   IDXGIFactory:  EnumAdapters[7]
//   IDXGIFactory1: EnumAdapters1[12]
//   IDXGIFactory2: IsWindowedStereoEnabled[14], CreateSwapChainForHwnd[15]

static void HookAdapter(IDXGIAdapter* pA)
{
    void** vt = *reinterpret_cast<void***>(pA);

    if (!g_bGetDescHooked)
    {
        if (MH_CreateHook(vt[8], &HookedGetDesc,
                reinterpret_cast<void**>(&g_pfnGetDesc)) == MH_OK)
        {
            MH_EnableHook(vt[8]);
            g_bGetDescHooked = true;
            WriteLog("[D3d12Proxy] IDXGIAdapter::GetDesc hooked\n");
        }
    }

    IDXGIAdapter1* pA1 = nullptr;
    if (!g_bGetDesc1Hooked &&
        SUCCEEDED(pA->QueryInterface(__uuidof(IDXGIAdapter1),
                reinterpret_cast<void**>(&pA1))))
    {
        void** vt1 = *reinterpret_cast<void***>(pA1);
        if (MH_CreateHook(vt1[10], &HookedGetDesc1,
                reinterpret_cast<void**>(&g_pfnGetDesc1)) == MH_OK)
        {
            MH_EnableHook(vt1[10]);
            g_bGetDesc1Hooked = true;
            WriteLog("[D3d12Proxy] IDXGIAdapter1::GetDesc1 hooked\n");
        }
        pA1->Release();
    }
}

static void HookFactory(void* pRaw)
{
    if (!pRaw) return;

    IDXGIFactory* pF = nullptr;
    if (FAILED(static_cast<IUnknown*>(pRaw)->QueryInterface(
            __uuidof(IDXGIFactory), reinterpret_cast<void**>(&pF))))
    {
        WriteLog("[D3d12Proxy] HookFactory: QI(IDXGIFactory) failed\n");
        return;
    }
    WriteLog("[D3d12Proxy] HookFactory: installing hooks\n");

    IDXGIAdapter* pA0 = nullptr;
    if (SUCCEEDED(pF->EnumAdapters(0, &pA0)))
    {
        HookAdapter(pA0);
        pA0->Release();
    }

    void** vt = *reinterpret_cast<void***>(pF);
    if (!g_bEnumAdaptersHooked)
    {
        if (MH_CreateHook(vt[7], &HookedEnumAdapters,
                reinterpret_cast<void**>(&g_pfnEnumAdapters)) == MH_OK)
        {
            MH_EnableHook(vt[7]);
            g_bEnumAdaptersHooked = true;
            WriteLog("[D3d12Proxy] IDXGIFactory::EnumAdapters hooked\n");
        }
    }

    if (!g_bCreateSCHooked)
    {
        if (MH_CreateHook(vt[10], &HookedCreateSwapChain,
                reinterpret_cast<void**>(&g_pfnCreateSwapChain)) == MH_OK)
        {
            MH_EnableHook(vt[10]);
            g_bCreateSCHooked = true;
            WriteLog("[D3d12Proxy] IDXGIFactory::CreateSwapChain hooked\n");
        }
    }

    IDXGIFactory1* pF1 = nullptr;
    if (!g_bEnumAdapters1Hooked &&
        SUCCEEDED(pF->QueryInterface(__uuidof(IDXGIFactory1),
                reinterpret_cast<void**>(&pF1))))
    {
        void** vt1 = *reinterpret_cast<void***>(pF1);
        if (MH_CreateHook(vt1[12], &HookedEnumAdapters1,
                reinterpret_cast<void**>(&g_pfnEnumAdapters1)) == MH_OK)
        {
            MH_EnableHook(vt1[12]);
            g_bEnumAdapters1Hooked = true;
            WriteLog("[D3d12Proxy] IDXGIFactory1::EnumAdapters1 hooked\n");
        }
        pF1->Release();
    }

    IDXGIFactory2* pF2 = nullptr;
    if (SUCCEEDED(pF->QueryInterface(__uuidof(IDXGIFactory2),
            reinterpret_cast<void**>(&pF2))))
    {
        void** vt2 = *reinterpret_cast<void***>(pF2);
        if (!g_bIsWindowedStereoHooked)
        {
            if (MH_CreateHook(vt2[14], &HookedIsWindowedStereoEnabled,
                    reinterpret_cast<void**>(&g_pfnIsWindowedStereoEnabled)) == MH_OK)
            {
                MH_EnableHook(vt2[14]);
                g_bIsWindowedStereoHooked = true;
                WriteLog("[D3d12Proxy] IDXGIFactory2::IsWindowedStereoEnabled hooked\n");
            }
        }
        if (!g_bCreateSCForHwndHooked)
        {
            if (MH_CreateHook(vt2[15], &HookedCreateSwapChainForHwnd,
                    reinterpret_cast<void**>(&g_pfnCreateSwapChainForHwnd)) == MH_OK)
            {
                MH_EnableHook(vt2[15]);
                g_bCreateSCForHwndHooked = true;
                WriteLog("[D3d12Proxy] IDXGIFactory2::CreateSwapChainForHwnd hooked\n");
            }
        }
        pF2->Release();
    }

    pF->Release();
}

static HRESULT WINAPI HookedCreateDXGIFactory(REFIID riid, void** ppFactory)
{
    WriteLog("[D3d12Proxy] CreateDXGIFactory\n");
    HRESULT hr = g_pfnOrigCreateFactory(riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory && *ppFactory)
        HookFactory(*ppFactory);
    return hr;
}

static HRESULT WINAPI HookedCreateDXGIFactory1(REFIID riid, void** ppFactory)
{
    WriteLog("[D3d12Proxy] CreateDXGIFactory1\n");
    HRESULT hr = g_pfnOrigCreateFactory1(riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory && *ppFactory)
        HookFactory(*ppFactory);
    return hr;
}

static HRESULT WINAPI HookedCreateDXGIFactory2(UINT Flags, REFIID riid, void** ppFactory)
{
    WriteLog("[D3d12Proxy] CreateDXGIFactory2\n");
    HRESULT hr = g_pfnOrigCreateFactory2(Flags, riid, ppFactory);
    if (SUCCEEDED(hr) && ppFactory && *ppFactory)
        HookFactory(*ppFactory);
    return hr;
}

static void InstallDxgiHooks()
{
    if (!g_hRealDXGI) return;

    void* pf1  = GetProcAddress(g_hRealDXGI, "CreateDXGIFactory");
    void* pf1a = GetProcAddress(g_hRealDXGI, "CreateDXGIFactory1");
    void* pf2  = GetProcAddress(g_hRealDXGI, "CreateDXGIFactory2");

    if (pf1)  MH_CreateHook(pf1,  &HookedCreateDXGIFactory,
                    reinterpret_cast<void**>(&g_pfnOrigCreateFactory));
    if (pf1a) MH_CreateHook(pf1a, &HookedCreateDXGIFactory1,
                    reinterpret_cast<void**>(&g_pfnOrigCreateFactory1));
    if (pf2)  MH_CreateHook(pf2,  &HookedCreateDXGIFactory2,
                    reinterpret_cast<void**>(&g_pfnOrigCreateFactory2));

    char buf[128];
    wsprintfA(buf, "[D3d12Proxy] DXGI hook ptrs: f=%p f1=%p f2=%p\n", pf1, pf1a, pf2);
    WriteLog(buf);

    MH_STATUS s = MH_EnableHook(MH_ALL_HOOKS);
    WriteLog(s == MH_OK
        ? "[D3d12Proxy] CreateDXGIFactory hooks installed OK\n"
        : "[D3d12Proxy] CreateDXGIFactory hooks FAILED\n");
}

// ---- GetProcAddress trampolines -------------------------------------------
typedef FARPROC (WINAPI* PFN_GetProcAddr)(HMODULE, LPCSTR);
static PFN_GetProcAddr g_pfnGetProcAddr   = nullptr;
static PFN_GetProcAddr g_pfnGetProcAddrKB = nullptr;

static HMODULE g_hAtidxx = nullptr;

// ---- AMD extension IAT intercept ------------------------------------------
// SE4 may have AmdDxExtCreate11 as a static IAT import (from its DX11 code path).
// Walk the exe IAT and redirect it to our proxy so it doesn't hit system atidxx.

typedef HRESULT (__cdecl* PFN_AmdDxExtCreate11)(void*, void**);
static HMODULE              g_hAmdProxy          = nullptr;
static PFN_AmdDxExtCreate11 g_pfnProxyExtCreate11 = nullptr;

typedef HMODULE (WINAPI* PFN_LoadLibraryExW)(LPCWSTR, HANDLE, DWORD);
static PFN_LoadLibraryExW g_pfnLoadLibraryExW = nullptr;

static HMODULE LoadAmdProxy()
{
    if (g_hAmdProxy) return g_hAmdProxy;
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(g_hSelf, path, MAX_PATH);
    wchar_t* sep = wcsrchr(path, L'\\');
    if (!sep) return nullptr;
    size_t rem = MAX_PATH - static_cast<size_t>(sep - path) - 1;
    wcscpy_s(sep + 1, rem, L"atidxx64.dll");  // DX12 is always x64
    if (g_pfnLoadLibraryExW)
        g_hAmdProxy = g_pfnLoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    else
        g_hAmdProxy = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (g_hAmdProxy)
    {
        HMODULE hPin = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                           reinterpret_cast<LPCWSTR>(g_hAmdProxy), &hPin);
    }
    WriteLog(g_hAmdProxy
        ? "[D3d12Proxy] AmdQbProxy (atidxx64) loaded OK\n"
        : "[D3d12Proxy] AmdQbProxy (atidxx64) load FAILED\n");
    return g_hAmdProxy;
}

static HMODULE g_hAmdAdl = nullptr;

static HMODULE LoadAmdAdlProxy()
{
    if (g_hAmdAdl) return g_hAmdAdl;
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(g_hSelf, path, MAX_PATH);
    wchar_t* sep = wcsrchr(path, L'\\');
    if (!sep) return nullptr;
    size_t rem = MAX_PATH - static_cast<size_t>(sep - path) - 1;
    wcscpy_s(sep + 1, rem, L"atiadlxx.dll");  // DX12 is always x64
    if (g_pfnLoadLibraryExW)
        g_hAmdAdl = g_pfnLoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    else
        g_hAmdAdl = LoadLibraryExW(path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (g_hAmdAdl)
    {
        HMODULE hPin = nullptr;
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                           reinterpret_cast<LPCWSTR>(g_hAmdAdl), &hPin);
    }
    WriteLog(g_hAmdAdl
        ? "[D3d12Proxy] AmdAdlProxy (atiadlxx) loaded OK\n"
        : "[D3d12Proxy] AmdAdlProxy (atiadlxx) load FAILED\n");
    return g_hAmdAdl;
}

static HRESULT __cdecl ProxiedAmdDxExtCreate11(void* pDevice, void** ppExt)
{
    WriteLog("[D3d12Proxy] AmdDxExtCreate11 -> proxy\n");
    if (!g_pfnProxyExtCreate11)
    {
        HMODULE h = LoadAmdProxy();
        if (h)
        {
            PFN_GetProcAddr resolveProc = g_pfnGetProcAddrKB ? g_pfnGetProcAddrKB : g_pfnGetProcAddr;
            g_pfnProxyExtCreate11 =
                reinterpret_cast<PFN_AmdDxExtCreate11>(resolveProc(h, "AmdDxExtCreate11"));
        }
    }
    if (!g_pfnProxyExtCreate11)
    {
        WriteLog("[D3d12Proxy] AmdDxExtCreate11: proxy unavailable -> E_NOTIMPL\n");
        if (ppExt) *ppExt = nullptr;
        return E_NOTIMPL;
    }
    return g_pfnProxyExtCreate11(pDevice, ppExt);
}

static void PatchAmdIAT()
{
    HMODULE hExe = GetModuleHandleW(nullptr);
    if (!hExe) return;

    BYTE* base = reinterpret_cast<BYTE*>(hExe);
    auto* dos = reinterpret_cast<PIMAGE_DOS_HEADER>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) return;
    auto* nt  = reinterpret_cast<PIMAGE_NT_HEADERS>(base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!dir.VirtualAddress) { WriteLog("[D3d12Proxy] PatchAmdIAT: no import dir\n"); return; }

    auto* imp = reinterpret_cast<PIMAGE_IMPORT_DESCRIPTOR>(base + dir.VirtualAddress);
    bool patched = false;

    for (; imp->Name; ++imp)
    {
        const char* dllName = reinterpret_cast<const char*>(base + imp->Name);
        char lc[MAX_PATH] = {};
        for (int i = 0; dllName[i] && i < MAX_PATH - 1; ++i)
            lc[i] = (dllName[i] >= 'A' && dllName[i] <= 'Z') ? dllName[i] + 32 : dllName[i];
        if (!strstr(lc, "atidxx")) continue;

        char mbuf[128];
        wsprintfA(mbuf, "[D3d12Proxy] PatchAmdIAT: found %s\n", dllName);
        WriteLog(mbuf);

        if (!imp->OriginalFirstThunk) continue;
        auto* orig  = reinterpret_cast<PIMAGE_THUNK_DATA>(base + imp->OriginalFirstThunk);
        auto* thunk = reinterpret_cast<PIMAGE_THUNK_DATA>(base + imp->FirstThunk);

        for (; orig->u1.Function; ++orig, ++thunk)
        {
            if (orig->u1.Ordinal & IMAGE_ORDINAL_FLAG) continue;
            auto* ibn = reinterpret_cast<PIMAGE_IMPORT_BY_NAME>(
                base + orig->u1.AddressOfData);
            if (_stricmp(reinterpret_cast<const char*>(ibn->Name), "AmdDxExtCreate11") != 0)
                continue;

            WriteLog("[D3d12Proxy] PatchAmdIAT: patching AmdDxExtCreate11\n");
            DWORD old;
            VirtualProtect(&thunk->u1.Function, sizeof(void*), PAGE_READWRITE, &old);
            thunk->u1.Function = reinterpret_cast<ULONG_PTR>(&ProxiedAmdDxExtCreate11);
            VirtualProtect(&thunk->u1.Function, sizeof(void*), old, &old);
            patched = true;
        }
    }

    if (!patched)
        WriteLog("[D3d12Proxy] PatchAmdIAT: AmdDxExtCreate11 not in static imports\n");
}

// ---- GetProcAddress hook ---------------------------------------------------

static FARPROC WINAPI HookedGetProcAddr(HMODULE hMod, LPCSTR name)
{
    if (name && !IS_INTRESOURCE(name))
    {
        if (_stricmp(name, "AmdDxExtCreate11") == 0)
        {
            WriteLog("[D3d12Proxy] GetProcAddress(AmdDxExtCreate11) intercepted\n");
            return reinterpret_cast<FARPROC>(&ProxiedAmdDxExtCreate11);
        }
        if (_strnicmp(name, "Amd", 3) == 0)
        {
            char logbuf[128];
            wsprintfA(logbuf, "[D3d12Proxy] GetProcAddress(%s) - unhandled AMD name\n", name);
            WriteLog(logbuf);
        }
        if (_strnicmp(name, "ADL_", 4) == 0 || _strnicmp(name, "ADL2_", 5) == 0)
        {
            PFN_GetProcAddr pfnR = g_pfnGetProcAddrKB ? g_pfnGetProcAddrKB : g_pfnGetProcAddr;
            FARPROC result = pfnR(hMod, name);
            char logbuf[256];
            wsprintfA(logbuf, "[D3d12Proxy] GetProcAddress(%s) hMod=%p result=%p\n",
                      name, hMod, result);
            WriteLog(logbuf);
            return result;
        }
        if (g_hAtidxx && hMod == g_hAtidxx)
        {
            char logbuf[128];
            wsprintfA(logbuf, "[D3d12Proxy] GetProcAddress(atidxx, %s)\n", name);
            WriteLog(logbuf);
        }
    }
    else if (IS_INTRESOURCE(name) && g_hAtidxx && hMod == g_hAtidxx)
    {
        char logbuf[64];
        wsprintfA(logbuf, "[D3d12Proxy] GetProcAddress(atidxx, ordinal#%u)\n",
                   static_cast<UINT>(reinterpret_cast<ULONG_PTR>(name)));
        WriteLog(logbuf);
    }
    PFN_GetProcAddr pfnReal = g_pfnGetProcAddrKB ? g_pfnGetProcAddrKB : g_pfnGetProcAddr;
    return pfnReal(hMod, name);
}

typedef HMODULE (WINAPI* PFN_GetModuleHandleA)(LPCSTR);
static PFN_GetModuleHandleA g_pfnGetModuleHandleA = nullptr;

static HMODULE WINAPI HookedGetModuleHandleA(LPCSTR name)
{
    HMODULE h = g_pfnGetModuleHandleA(name);
    if (name)
    {
        char lc[MAX_PATH] = {};
        for (int i = 0; name[i] && i < MAX_PATH - 1; ++i)
            lc[i] = (name[i] >= 'A' && name[i] <= 'Z') ? name[i] + 32 : name[i];
        if (strstr(lc, "atidxx"))
        {
            if (h)
            {
                g_hAtidxx = h;
                char buf[128];
                wsprintfA(buf, "[D3d12Proxy] GetModuleHandleA(%s) = %p\n", name, h);
                WriteLog(buf);
            }
            else
            {
                char buf[256];
                wsprintfA(buf, "[D3d12Proxy] GetModuleHandleA(%s) = NULL -> loading AmdQbProxy\n", name);
                WriteLog(buf);
                h = LoadAmdProxy();
                if (h) g_hAtidxx = h;
            }
        }
    }
    return h;
}

// ---- LoadLibrary hooks -----------------------------------------------------

typedef HMODULE (WINAPI* PFN_LoadLibraryA)(LPCSTR);
typedef HMODULE (WINAPI* PFN_LoadLibraryW)(LPCWSTR);
typedef HMODULE (WINAPI* PFN_LoadLibraryExA)(LPCSTR, HANDLE, DWORD);

static PFN_LoadLibraryA   g_pfnLoadLibraryA   = nullptr;
static PFN_LoadLibraryW   g_pfnLoadLibraryW   = nullptr;
static PFN_LoadLibraryExA g_pfnLoadLibraryExA = nullptr;

static bool ContainsAtidxxA(LPCSTR s)
{
    if (!s) return false;
    for (const char* p = s; *p; ++p)
    {
        if ((p[0]|0x20)=='a' && (p[1]|0x20)=='t' && (p[2]|0x20)=='i' &&
            (p[3]|0x20)=='d' && (p[4]|0x20)=='x' && (p[5]|0x20)=='x')
            return true;
    }
    return false;
}

static bool ContainsAtidxxW(LPCWSTR s)
{
    if (!s) return false;
    for (const wchar_t* p = s; *p; ++p)
    {
        if ((p[0]|0x20)==L'a' && (p[1]|0x20)==L't' && (p[2]|0x20)==L'i' &&
            (p[3]|0x20)==L'd' && (p[4]|0x20)==L'x' && (p[5]|0x20)==L'x')
            return true;
    }
    return false;
}

static bool ContainsAtiadlA(LPCSTR s)
{
    if (!s) return false;
    for (const char* p = s; *p; ++p)
    {
        if ((p[0]|0x20)=='a' && (p[1]|0x20)=='t' && (p[2]|0x20)=='i' &&
            (p[3]|0x20)=='a' && (p[4]|0x20)=='d' && (p[5]|0x20)=='l')
            return true;
    }
    return false;
}

static bool ContainsAtiadlW(LPCWSTR s)
{
    if (!s) return false;
    for (const wchar_t* p = s; *p; ++p)
    {
        if ((p[0]|0x20)==L'a' && (p[1]|0x20)==L't' && (p[2]|0x20)==L'i' &&
            (p[3]|0x20)==L'a' && (p[4]|0x20)==L'd' && (p[5]|0x20)==L'l')
            return true;
    }
    return false;
}

static bool IsAmdRelatedA(LPCSTR s)
{
    if (!s) return false;
    const char* base = s;
    for (const char* p = s; *p; ++p)
        if (*p == '\\' || *p == '/') base = p + 1;
    if ((base[0]|0x20) == 'a')
    {
        if (((base[1]|0x20)=='t' && (base[2]|0x20)=='i') ||
            ((base[1]|0x20)=='m' && (base[2]|0x20)=='d'))
            return true;
    }
    return false;
}

static bool IsAmdRelatedW(LPCWSTR s)
{
    if (!s) return false;
    const wchar_t* base = s;
    for (const wchar_t* p = s; *p; ++p)
        if (*p == L'\\' || *p == L'/') base = p + 1;
    if ((base[0]|0x20) == L'a')
    {
        if (((base[1]|0x20)==L't' && (base[2]|0x20)==L'i') ||
            ((base[1]|0x20)==L'm' && (base[2]|0x20)==L'd'))
            return true;
    }
    return false;
}

#ifdef BLOCK_NVAPI_STEREO
static bool ContainsNvStereo3DA(LPCSTR s)
{
    if (!s) return false;
    for (const char* p = s; *p; ++p)
    {
        if ((p[0]|0x20)=='s' && (p[1]|0x20)=='t' && (p[2]|0x20)=='e' && (p[3]|0x20)=='r' &&
            (p[4]|0x20)=='e' && (p[5]|0x20)=='o' && p[6]=='3' && (p[7]|0x20)=='d')
            return true;
    }
    return false;
}

static bool ContainsNvStereo3DW(LPCWSTR s)
{
    if (!s) return false;
    for (const wchar_t* p = s; *p; ++p)
    {
        if ((p[0]|0x20)==L's' && (p[1]|0x20)==L't' && (p[2]|0x20)==L'e' && (p[3]|0x20)==L'r' &&
            (p[4]|0x20)==L'e' && (p[5]|0x20)==L'o' && p[6]==L'3' && (p[7]|0x20)==L'd')
            return true;
    }
    return false;
}

static bool IsNvapiDllA(LPCSTR s)
{
    if (!s) return false;
    const char* base = s;
    for (const char* p = s; *p; ++p)
        if (*p == '\\' || *p == '/') base = p + 1;
    if ((base[0]|0x20)=='n' && (base[1]|0x20)=='v' && (base[2]|0x20)=='a' &&
        (base[3]|0x20)=='p' && (base[4]|0x20)=='i')
    {
        if ((base[5]|0x20)=='.' || (base[5]=='6' && base[6]=='4' && (base[7]|0x20)=='.'))
            return true;
    }
    return false;
}

static bool IsNvapiDllW(LPCWSTR s)
{
    if (!s) return false;
    const wchar_t* base = s;
    for (const wchar_t* p = s; *p; ++p)
        if (*p == L'\\' || *p == L'/') base = p + 1;
    if ((base[0]|0x20)==L'n' && (base[1]|0x20)==L'v' && (base[2]|0x20)==L'a' &&
        (base[3]|0x20)==L'p' && (base[4]|0x20)==L'i')
    {
        if ((base[5]|0x20)==L'.' || (base[5]==L'6' && base[6]==L'4' && (base[7]|0x20)==L'.'))
            return true;
    }
    return false;
}

static int __cdecl NvApiStub()
{
    WriteLog("[D3d12Proxy] NvApiStub called -> NVAPI_ERROR\n");
    return -1; // NVAPI_ERROR
}

extern "C" __declspec(dllexport) void* __cdecl nvapi_QueryInterface(unsigned int id)
{
    char buf[128];
    wsprintfA(buf, "[D3d12Proxy] nvapi_QueryInterface(0x%08X) -> stub fn\n", id);
    WriteLog(buf);
    return reinterpret_cast<void*>(&NvApiStub);
}
#endif // BLOCK_NVAPI_STEREO

static HMODULE WINAPI HookedLoadLibraryA(LPCSTR lpFileName)
{
#ifdef BLOCK_NVAPI_STEREO
    if (IsNvapiDllA(lpFileName))
    {
        char buf[300];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryA(%s) -> NULL (nvapi blocked)\n", lpFileName);
        WriteLog(buf);
        SetLastError(ERROR_MOD_NOT_FOUND);
        return NULL;
    }
#endif
    if (ContainsAtidxxA(lpFileName))
    {
        char buf[300];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryA(%s) -> redirecting to local proxy\n", lpFileName);
        WriteLog(buf);
        HMODULE h = LoadAmdProxy();
        if (h) return h;
        WriteLog("[D3d12Proxy] LoadLibraryA redirect failed, falling through\n");
    }
    else if (ContainsAtiadlA(lpFileName))
    {
        char buf[300];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryA(%s) -> redirecting to ADL proxy\n", lpFileName);
        WriteLog(buf);
        HMODULE h = LoadAmdAdlProxy();
        if (h) return h;
        WriteLog("[D3d12Proxy] LoadLibraryA ADL redirect failed, falling through\n");
    }
    else if (IsAmdRelatedA(lpFileName))
    {
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryA(%s) [AMD-related, not redirected]\n", lpFileName);
        WriteLog(buf);
    }
    return g_pfnLoadLibraryA(lpFileName);
}

static HMODULE WINAPI HookedLoadLibraryW(LPCWSTR lpFileName)
{
#ifdef BLOCK_NVAPI_STEREO
    if (IsNvapiDllW(lpFileName))
    {
        char narrow[300] = {};
        WideCharToMultiByte(CP_ACP, 0, lpFileName, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryW(%s) -> NULL (nvapi blocked)\n", narrow);
        WriteLog(buf);
        SetLastError(ERROR_MOD_NOT_FOUND);
        return NULL;
    }
#endif
    if (ContainsAtidxxW(lpFileName))
    {
        char narrow[300] = {};
        WideCharToMultiByte(CP_ACP, 0, lpFileName, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryW(%s) -> redirecting to local proxy\n", narrow);
        WriteLog(buf);
        HMODULE h = LoadAmdProxy();
        if (h) return h;
        WriteLog("[D3d12Proxy] LoadLibraryW redirect failed, falling through\n");
    }
    else if (ContainsAtiadlW(lpFileName))
    {
        char narrow[300] = {};
        WideCharToMultiByte(CP_ACP, 0, lpFileName, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryW(%s) -> redirecting to ADL proxy\n", narrow);
        WriteLog(buf);
        HMODULE h = LoadAmdAdlProxy();
        if (h) return h;
        WriteLog("[D3d12Proxy] LoadLibraryW ADL redirect failed, falling through\n");
    }
    else if (IsAmdRelatedW(lpFileName))
    {
        char narrow[300] = {};
        WideCharToMultiByte(CP_ACP, 0, lpFileName, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryW(%s) [AMD-related, not redirected]\n", narrow);
        WriteLog(buf);
    }
    return g_pfnLoadLibraryW(lpFileName);
}

static HMODULE WINAPI HookedLoadLibraryExA(LPCSTR lpFileName, HANDLE hFile, DWORD dwFlags)
{
#ifdef BLOCK_NVAPI_STEREO
    if (IsNvapiDllA(lpFileName))
    {
        char buf[300];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryExA(%s) -> NULL (nvapi blocked)\n", lpFileName);
        WriteLog(buf);
        SetLastError(ERROR_MOD_NOT_FOUND);
        return NULL;
    }
#endif
    if (ContainsAtidxxA(lpFileName))
    {
        char buf[300];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryExA(%s) -> redirecting to local proxy\n", lpFileName);
        WriteLog(buf);
        HMODULE h = LoadAmdProxy();
        if (h) return h;
        WriteLog("[D3d12Proxy] LoadLibraryExA redirect failed, falling through\n");
    }
    else if (ContainsAtiadlA(lpFileName))
    {
        char buf[300];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryExA(%s) -> redirecting to ADL proxy\n", lpFileName);
        WriteLog(buf);
        HMODULE h = LoadAmdAdlProxy();
        if (h) return h;
        WriteLog("[D3d12Proxy] LoadLibraryExA ADL redirect failed, falling through\n");
    }
    else if (IsAmdRelatedA(lpFileName))
    {
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryExA(%s) [AMD-related, not redirected]\n", lpFileName);
        WriteLog(buf);
    }
    return g_pfnLoadLibraryExA(lpFileName, hFile, dwFlags);
}

static HMODULE WINAPI HookedLoadLibraryExW(LPCWSTR lpFileName, HANDLE hFile, DWORD dwFlags)
{
#ifdef BLOCK_NVAPI_STEREO
    if (IsNvapiDllW(lpFileName))
    {
        char narrow[300] = {};
        WideCharToMultiByte(CP_ACP, 0, lpFileName, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryExW(%s) -> NULL (nvapi blocked)\n", narrow);
        WriteLog(buf);
        SetLastError(ERROR_MOD_NOT_FOUND);
        return NULL;
    }
#endif
    if (ContainsAtidxxW(lpFileName))
    {
        char narrow[300] = {};
        WideCharToMultiByte(CP_ACP, 0, lpFileName, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryExW(%s) -> redirecting to local proxy\n", narrow);
        WriteLog(buf);
        HMODULE h = LoadAmdProxy();
        if (h) return h;
        WriteLog("[D3d12Proxy] LoadLibraryExW redirect failed, falling through\n");
    }
    else if (ContainsAtiadlW(lpFileName))
    {
        char narrow[300] = {};
        WideCharToMultiByte(CP_ACP, 0, lpFileName, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryExW(%s) -> redirecting to ADL proxy\n", narrow);
        WriteLog(buf);
        HMODULE h = LoadAmdAdlProxy();
        if (h) return h;
        WriteLog("[D3d12Proxy] LoadLibraryExW ADL redirect failed, falling through\n");
    }
    else if (IsAmdRelatedW(lpFileName))
    {
        char narrow[300] = {};
        WideCharToMultiByte(CP_ACP, 0, lpFileName, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[400];
        wsprintfA(buf, "[D3d12Proxy] LoadLibraryExW(%s) [AMD-related, not redirected]\n", narrow);
        WriteLog(buf);
    }
    return g_pfnLoadLibraryExW(lpFileName, hFile, dwFlags);
}

// ---- Registry monitoring hooks ---------------------------------------------

typedef LSTATUS (WINAPI* PFN_RegOpenKeyExA)(HKEY, LPCSTR, DWORD, REGSAM, PHKEY);
typedef LSTATUS (WINAPI* PFN_RegOpenKeyExW)(HKEY, LPCWSTR, DWORD, REGSAM, PHKEY);
typedef LSTATUS (WINAPI* PFN_RegQueryValueExA)(HKEY, LPCSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);
typedef LSTATUS (WINAPI* PFN_RegQueryValueExW)(HKEY, LPCWSTR, LPDWORD, LPDWORD, LPBYTE, LPDWORD);
typedef LSTATUS (WINAPI* PFN_RegGetValueA)(HKEY, LPCSTR, LPCSTR, DWORD, LPDWORD, PVOID, LPDWORD);
typedef LSTATUS (WINAPI* PFN_RegGetValueW)(HKEY, LPCWSTR, LPCWSTR, DWORD, LPDWORD, PVOID, LPDWORD);

static PFN_RegOpenKeyExA    g_pfnRegOpenKeyExA    = nullptr;
static PFN_RegOpenKeyExW    g_pfnRegOpenKeyExW    = nullptr;
static PFN_RegQueryValueExA g_pfnRegQueryValueExA = nullptr;
static PFN_RegQueryValueExW g_pfnRegQueryValueExW = nullptr;
static PFN_RegGetValueA     g_pfnRegGetValueA     = nullptr;
static PFN_RegGetValueW     g_pfnRegGetValueW     = nullptr;

static HKEY  g_trackedKeys[64] = {};
static int   g_trackedKeyCount = 0;

static void TrackKey(HKEY k)
{
    if (g_trackedKeyCount < 64)
        g_trackedKeys[g_trackedKeyCount++] = k;
}

static bool IsTrackedKey(HKEY k)
{
    for (int i = 0; i < g_trackedKeyCount; ++i)
        if (g_trackedKeys[i] == k) return true;
    return false;
}

static bool IsAmdRegistryPathA(LPCSTR s)
{
    if (!s) return false;
    for (const char* p = s; *p; ++p)
    {
        char c0 = p[0] | 0x20;
        if (c0 == 'a')
        {
            bool boundary = (p == s) ||
                !((p[-1]>='A'&&p[-1]<='Z')||(p[-1]>='a'&&p[-1]<='z')||(p[-1]>='0'&&p[-1]<='9'));
            if (boundary)
            {
                char c1 = p[1] | 0x20;
                if ((c1=='t'&&(p[2]|0x20)=='i') || (c1=='m'&&(p[2]|0x20)=='d'))
                    return true;
            }
        }
        if (c0=='r'&&(p[1]|0x20)=='a'&&(p[2]|0x20)=='d'&&
            (p[3]|0x20)=='e'&&(p[4]|0x20)=='o'&&(p[5]|0x20)=='n')
            return true;
        if (p[0]=='4'&&(p[1]|0x20)=='d'&&p[2]=='3'&&p[3]=='6'&&(p[4]|0x20)=='e'&&p[5]=='9')
            return true;
    }
    return false;
}

static bool IsAmdRegistryPathW(LPCWSTR s)
{
    if (!s) return false;
    for (const wchar_t* p = s; *p; ++p)
    {
        wchar_t c0 = p[0] | 0x20;
        if (c0 == L'a')
        {
            bool boundary = (p == s) ||
                !((p[-1]>=L'A'&&p[-1]<=L'Z')||(p[-1]>=L'a'&&p[-1]<=L'z')||(p[-1]>=L'0'&&p[-1]<=L'9'));
            if (boundary)
            {
                wchar_t c1 = p[1] | 0x20;
                if ((c1==L't'&&(p[2]|0x20)==L'i') || (c1==L'm'&&(p[2]|0x20)==L'd'))
                    return true;
            }
        }
        if (c0==L'r'&&(p[1]|0x20)==L'a'&&(p[2]|0x20)==L'd'&&
            (p[3]|0x20)==L'e'&&(p[4]|0x20)==L'o'&&(p[5]|0x20)==L'n')
            return true;
        if (p[0]==L'4'&&(p[1]|0x20)==L'd'&&p[2]==L'3'&&p[3]==L'6'&&(p[4]|0x20)==L'e'&&p[5]==L'9')
            return true;
    }
    return false;
}

static LSTATUS WINAPI HookedRegOpenKeyExA(HKEY hKey, LPCSTR lpSubKey,
    DWORD ulOptions, REGSAM samDesired, PHKEY phkResult)
{
#ifdef BLOCK_NVAPI_STEREO
    if (lpSubKey && ContainsNvStereo3DA(lpSubKey))
    {
        char buf[512];
        wsprintfA(buf, "[D3d12Proxy] RegOpenKeyExA(%s) -> BLOCKED (NvStereo3D)\n", lpSubKey);
        WriteLog(buf);
        return ERROR_FILE_NOT_FOUND;
    }
#endif
    LSTATUS r = g_pfnRegOpenKeyExA(hKey, lpSubKey, ulOptions, samDesired, phkResult);
    bool tracked = IsTrackedKey(hKey);
    if (lpSubKey && (IsAmdRegistryPathA(lpSubKey) || tracked))
    {
        char buf[512];
        wsprintfA(buf, "[D3d12Proxy] RegOpenKeyExA(%s) -> %ld%s\n",
                  lpSubKey, r, tracked ? " [child of tracked]" : "");
        WriteLog(buf);
        if (r == ERROR_SUCCESS && phkResult)
            TrackKey(*phkResult);
    }
    return r;
}

static LSTATUS WINAPI HookedRegOpenKeyExW(HKEY hKey, LPCWSTR lpSubKey,
    DWORD ulOptions, REGSAM samDesired, PHKEY phkResult)
{
#ifdef BLOCK_NVAPI_STEREO
    if (lpSubKey && ContainsNvStereo3DW(lpSubKey))
    {
        char narrow[400] = {};
        WideCharToMultiByte(CP_ACP, 0, lpSubKey, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[512];
        wsprintfA(buf, "[D3d12Proxy] RegOpenKeyExW(%s) -> BLOCKED (NvStereo3D)\n", narrow);
        WriteLog(buf);
        return ERROR_FILE_NOT_FOUND;
    }
#endif
    LSTATUS r = g_pfnRegOpenKeyExW(hKey, lpSubKey, ulOptions, samDesired, phkResult);
    bool tracked = IsTrackedKey(hKey);
    if (lpSubKey && (IsAmdRegistryPathW(lpSubKey) || tracked))
    {
        char narrow[400] = {};
        WideCharToMultiByte(CP_ACP, 0, lpSubKey, -1, narrow, sizeof(narrow)-1, nullptr, nullptr);
        char buf[512];
        wsprintfA(buf, "[D3d12Proxy] RegOpenKeyExW(%s) -> %ld%s\n",
                  narrow, r, tracked ? " [child of tracked]" : "");
        WriteLog(buf);
        if (r == ERROR_SUCCESS && phkResult)
            TrackKey(*phkResult);
    }
    return r;
}

static LSTATUS WINAPI HookedRegQueryValueExA(HKEY hKey, LPCSTR lpValueName,
    LPDWORD lpReserved, LPDWORD lpType, LPBYTE lpData, LPDWORD lpcbData)
{
    LSTATUS r = g_pfnRegQueryValueExA(hKey, lpValueName, lpReserved, lpType, lpData, lpcbData);
    if (IsTrackedKey(hKey) && lpValueName)
    {
        char buf[512];
        if (r == ERROR_SUCCESS && lpData && lpcbData && *lpcbData > 0 &&
            lpType && *lpType == REG_SZ)
        {
            wsprintfA(buf, "[D3d12Proxy] RegQueryValueExA(%s) = \"%s\"\n",
                      lpValueName, reinterpret_cast<const char*>(lpData));
            WriteLog(buf);
#ifdef BLOCK_NVAPI_STEREO
            char* str = reinterpret_cast<char*>(lpData);
            bool hasNv = false;
            for (const char* p = str; *p; ++p)
            {
                if ((p[0]|0x20)=='n'&&(p[1]|0x20)=='v'&&(p[2]|0x20)=='i'&&
                    (p[3]|0x20)=='d'&&(p[4]|0x20)=='i'&&(p[5]|0x20)=='a')
                { hasNv = true; break; }
            }
            if (hasNv)
            {
                if (lstrcmpiA(lpValueName, "ProviderName") == 0)
                { memcpy(lpData, "AMD", 4); *lpcbData = 4; WriteLog("[D3d12Proxy]   -> SPOOFED ProviderName='AMD'\n"); }
                else if (lstrcmpiA(lpValueName, "DriverDesc") == 0)
                { memcpy(lpData, "AMD Radeon", 11); *lpcbData = 11; WriteLog("[D3d12Proxy]   -> SPOOFED DriverDesc='AMD Radeon'\n"); }
                else
                {
                    for (char* p = str; *p; ++p)
                    {
                        if ((p[0]|0x20)=='n'&&(p[1]|0x20)=='v'&&(p[2]|0x20)=='i'&&
                            (p[3]|0x20)=='d'&&(p[4]|0x20)=='i'&&(p[5]|0x20)=='a')
                        { p[0]='A';p[1]='M';p[2]='D';p[3]=' ';p[4]=' ';p[5]=' '; }
                    }
                    wsprintfA(buf, "[D3d12Proxy]   -> SPOOFED %s (NVIDIA->AMD)\n", lpValueName);
                    WriteLog(buf);
                }
            }
            if (lstrcmpiA(lpValueName, "MatchingDeviceId") == 0)
            {
                bool spoofed = false;
                for (char* p = str; *p; ++p)
                {
                    if ((p[0]|0x20)=='v'&&(p[1]|0x20)=='e'&&(p[2]|0x20)=='n'&&
                        p[3]=='_'&&p[4]=='1'&&p[5]=='0'&&(p[6]|0x20)=='d'&&(p[7]|0x20)=='e')
                    { p[6]='0';p[7]='2'; spoofed=true; break; }
                }
                if (spoofed) WriteLog("[D3d12Proxy]   -> SPOOFED MatchingDeviceId ven_1002\n");
            }
#endif
        }
        else
        {
            wsprintfA(buf, "[D3d12Proxy] RegQueryValueExA(%s) -> status=%ld\n", lpValueName, r);
            WriteLog(buf);
        }
    }
    return r;
}

static LSTATUS WINAPI HookedRegQueryValueExW(HKEY hKey, LPCWSTR lpValueName,
    LPDWORD lpReserved, LPDWORD lpType, LPBYTE lpData, LPDWORD lpcbData)
{
    LSTATUS r = g_pfnRegQueryValueExW(hKey, lpValueName, lpReserved, lpType, lpData, lpcbData);
    if (IsTrackedKey(hKey) && lpValueName)
    {
        char narrowName[256] = {};
        WideCharToMultiByte(CP_ACP, 0, lpValueName, -1, narrowName, sizeof(narrowName)-1, nullptr, nullptr);
        char buf[512];
        if (r == ERROR_SUCCESS && lpData && lpcbData && *lpcbData > 0 &&
            lpType && *lpType == REG_SZ)
        {
            char narrowVal[256] = {};
            WideCharToMultiByte(CP_ACP, 0, reinterpret_cast<LPCWSTR>(lpData), -1,
                                narrowVal, sizeof(narrowVal)-1, nullptr, nullptr);
            wsprintfA(buf, "[D3d12Proxy] RegQueryValueExW(%s) = \"%s\"\n", narrowName, narrowVal);
            WriteLog(buf);
#ifdef BLOCK_NVAPI_STEREO
            wchar_t* wstr = reinterpret_cast<wchar_t*>(lpData);
            bool hasNv = false;
            for (const wchar_t* p = wstr; *p; ++p)
            {
                if ((p[0]|0x20)==L'n'&&(p[1]|0x20)==L'v'&&(p[2]|0x20)==L'i'&&
                    (p[3]|0x20)==L'd'&&(p[4]|0x20)==L'i'&&(p[5]|0x20)==L'a')
                { hasNv = true; break; }
            }
            if (hasNv)
            {
                if (lstrcmpiW(lpValueName, L"ProviderName") == 0)
                { memcpy(lpData,L"AMD",4*sizeof(wchar_t));*lpcbData=4*sizeof(wchar_t);WriteLog("[D3d12Proxy]   -> SPOOFED ProviderName='AMD'\n"); }
                else if (lstrcmpiW(lpValueName, L"DriverDesc") == 0)
                { memcpy(lpData,L"AMD Radeon",11*sizeof(wchar_t));*lpcbData=11*sizeof(wchar_t);WriteLog("[D3d12Proxy]   -> SPOOFED DriverDesc='AMD Radeon'\n"); }
                else
                {
                    for (wchar_t* p = wstr; *p; ++p)
                    {
                        if ((p[0]|0x20)==L'n'&&(p[1]|0x20)==L'v'&&(p[2]|0x20)==L'i'&&
                            (p[3]|0x20)==L'd'&&(p[4]|0x20)==L'i'&&(p[5]|0x20)==L'a')
                        { p[0]=L'A';p[1]=L'M';p[2]=L'D';p[3]=L' ';p[4]=L' ';p[5]=L' '; }
                    }
                    wsprintfA(buf, "[D3d12Proxy]   -> SPOOFED %s (NVIDIA->AMD)\n", narrowName);
                    WriteLog(buf);
                }
            }
            if (lstrcmpiW(lpValueName, L"MatchingDeviceId") == 0)
            {
                bool spoofed = false;
                for (wchar_t* p = wstr; *p; ++p)
                {
                    if ((p[0]|0x20)==L'v'&&(p[1]|0x20)==L'e'&&(p[2]|0x20)==L'n'&&
                        p[3]==L'_'&&p[4]==L'1'&&p[5]==L'0'&&(p[6]|0x20)==L'd'&&(p[7]|0x20)==L'e')
                    { p[6]=L'0';p[7]=L'2'; spoofed=true; break; }
                }
                if (spoofed) WriteLog("[D3d12Proxy]   -> SPOOFED MatchingDeviceId ven_1002\n");
            }
#endif
        }
        else
        {
            wsprintfA(buf, "[D3d12Proxy] RegQueryValueExW(%s) -> status=%ld\n", narrowName, r);
            WriteLog(buf);
        }
    }
    return r;
}

static LSTATUS WINAPI HookedRegGetValueA(HKEY hKey, LPCSTR lpSubKey,
    LPCSTR lpValue, DWORD dwFlags, LPDWORD pdwType, PVOID pvData, LPDWORD pcbData)
{
    LSTATUS r = g_pfnRegGetValueA(hKey, lpSubKey, lpValue, dwFlags, pdwType, pvData, pcbData);
    bool tracked = IsTrackedKey(hKey);
    if (!tracked && lpSubKey && IsAmdRegistryPathA(lpSubKey)) tracked = true;
    if (!tracked || !lpValue) return r;

    char buf[512];
    if (r == ERROR_SUCCESS && pvData && pcbData && *pcbData > 0 &&
        pdwType && *pdwType == REG_SZ)
    {
        wsprintfA(buf, "[D3d12Proxy] RegGetValueA(%s) = \"%s\"\n",
                  lpValue, reinterpret_cast<const char*>(pvData));
        WriteLog(buf);
#ifdef BLOCK_NVAPI_STEREO
        char* str = reinterpret_cast<char*>(pvData);
        bool hasNv = false;
        for (const char* p = str; *p; ++p)
        {
            if ((p[0]|0x20)=='n'&&(p[1]|0x20)=='v'&&(p[2]|0x20)=='i'&&
                (p[3]|0x20)=='d'&&(p[4]|0x20)=='i'&&(p[5]|0x20)=='a')
            { hasNv = true; break; }
        }
        if (hasNv)
        {
            if (lstrcmpiA(lpValue, "ProviderName") == 0)
            { memcpy(pvData,"AMD",4);*pcbData=4;WriteLog("[D3d12Proxy]   -> SPOOFED ProviderName='AMD'\n"); }
            else if (lstrcmpiA(lpValue, "DriverDesc") == 0)
            { memcpy(pvData,"AMD Radeon",11);*pcbData=11;WriteLog("[D3d12Proxy]   -> SPOOFED DriverDesc='AMD Radeon'\n"); }
            else
            {
                for (char* p = str; *p; ++p)
                {
                    if ((p[0]|0x20)=='n'&&(p[1]|0x20)=='v'&&(p[2]|0x20)=='i'&&
                        (p[3]|0x20)=='d'&&(p[4]|0x20)=='i'&&(p[5]|0x20)=='a')
                    { p[0]='A';p[1]='M';p[2]='D';p[3]=' ';p[4]=' ';p[5]=' '; }
                }
                wsprintfA(buf, "[D3d12Proxy]   -> SPOOFED %s (NVIDIA->AMD)\n", lpValue);
                WriteLog(buf);
            }
        }
        if (lstrcmpiA(lpValue, "MatchingDeviceId") == 0)
        {
            bool spoofed = false;
            for (char* p = str; *p; ++p)
            {
                if ((p[0]|0x20)=='v'&&(p[1]|0x20)=='e'&&(p[2]|0x20)=='n'&&
                    p[3]=='_'&&p[4]=='1'&&p[5]=='0'&&(p[6]|0x20)=='d'&&(p[7]|0x20)=='e')
                { p[6]='0';p[7]='2'; spoofed=true; break; }
            }
            if (spoofed) WriteLog("[D3d12Proxy]   -> SPOOFED MatchingDeviceId ven_1002\n");
        }
#endif
    }
    else
    {
        wsprintfA(buf, "[D3d12Proxy] RegGetValueA(%s) -> status=%ld\n", lpValue, r);
        WriteLog(buf);
    }
    return r;
}

static LSTATUS WINAPI HookedRegGetValueW(HKEY hKey, LPCWSTR lpSubKey,
    LPCWSTR lpValue, DWORD dwFlags, LPDWORD pdwType, PVOID pvData, LPDWORD pcbData)
{
    LSTATUS r = g_pfnRegGetValueW(hKey, lpSubKey, lpValue, dwFlags, pdwType, pvData, pcbData);
    bool tracked = IsTrackedKey(hKey);
    if (!tracked && lpSubKey && IsAmdRegistryPathW(lpSubKey)) tracked = true;
    if (!tracked || !lpValue) return r;

    char narrowName[256] = {};
    WideCharToMultiByte(CP_ACP, 0, lpValue, -1, narrowName, sizeof(narrowName)-1, nullptr, nullptr);
    char buf[512];
    if (r == ERROR_SUCCESS && pvData && pcbData && *pcbData > 0 &&
        pdwType && *pdwType == REG_SZ)
    {
        char narrowVal[256] = {};
        WideCharToMultiByte(CP_ACP, 0, reinterpret_cast<LPCWSTR>(pvData), -1,
                            narrowVal, sizeof(narrowVal)-1, nullptr, nullptr);
        wsprintfA(buf, "[D3d12Proxy] RegGetValueW(%s) = \"%s\"\n", narrowName, narrowVal);
        WriteLog(buf);
#ifdef BLOCK_NVAPI_STEREO
        wchar_t* wstr = reinterpret_cast<wchar_t*>(pvData);
        bool hasNv = false;
        for (const wchar_t* p = wstr; *p; ++p)
        {
            if ((p[0]|0x20)==L'n'&&(p[1]|0x20)==L'v'&&(p[2]|0x20)==L'i'&&
                (p[3]|0x20)==L'd'&&(p[4]|0x20)==L'i'&&(p[5]|0x20)==L'a')
            { hasNv = true; break; }
        }
        if (hasNv)
        {
            if (lstrcmpiW(lpValue, L"ProviderName") == 0)
            { memcpy(pvData,L"AMD",4*sizeof(wchar_t));*pcbData=4*sizeof(wchar_t);WriteLog("[D3d12Proxy]   -> SPOOFED ProviderName='AMD'\n"); }
            else if (lstrcmpiW(lpValue, L"DriverDesc") == 0)
            { memcpy(pvData,L"AMD Radeon",11*sizeof(wchar_t));*pcbData=11*sizeof(wchar_t);WriteLog("[D3d12Proxy]   -> SPOOFED DriverDesc='AMD Radeon'\n"); }
            else
            {
                for (wchar_t* p = wstr; *p; ++p)
                {
                    if ((p[0]|0x20)==L'n'&&(p[1]|0x20)==L'v'&&(p[2]|0x20)==L'i'&&
                        (p[3]|0x20)==L'd'&&(p[4]|0x20)==L'i'&&(p[5]|0x20)==L'a')
                    { p[0]=L'A';p[1]=L'M';p[2]=L'D';p[3]=L' ';p[4]=L' ';p[5]=L' '; }
                }
                wsprintfA(buf, "[D3d12Proxy]   -> SPOOFED %s (NVIDIA->AMD)\n", narrowName);
                WriteLog(buf);
            }
        }
        if (lstrcmpiW(lpValue, L"MatchingDeviceId") == 0)
        {
            bool spoofed = false;
            for (wchar_t* p = wstr; *p; ++p)
            {
                if ((p[0]|0x20)==L'v'&&(p[1]|0x20)==L'e'&&(p[2]|0x20)==L'n'&&
                    p[3]==L'_'&&p[4]==L'1'&&p[5]==L'0'&&(p[6]|0x20)==L'd'&&(p[7]|0x20)==L'e')
                { p[6]=L'0';p[7]=L'2'; spoofed=true; break; }
            }
            if (spoofed) WriteLog("[D3d12Proxy]   -> SPOOFED MatchingDeviceId ven_1002\n");
        }
#endif
    }
    else
    {
        wsprintfA(buf, "[D3d12Proxy] RegGetValueW(%s) -> status=%ld\n", narrowName, r);
        WriteLog(buf);
    }
    return r;
}

static void InstallRegistryHooks()
{
    HMODULE hKB = GetModuleHandleW(L"KernelBase.dll");
    if (!hKB) hKB = GetModuleHandleW(L"advapi32.dll");
    if (!hKB) return;

    PFN_GetProcAddr resolveProc = g_pfnGetProcAddrKB ? g_pfnGetProcAddrKB : g_pfnGetProcAddr;
    if (!resolveProc) resolveProc = reinterpret_cast<PFN_GetProcAddr>(&GetProcAddress);

    void* pfnROA = resolveProc(hKB, "RegOpenKeyExA");
    void* pfnROW = resolveProc(hKB, "RegOpenKeyExW");
    void* pfnRQA = resolveProc(hKB, "RegQueryValueExA");
    void* pfnRQW = resolveProc(hKB, "RegQueryValueExW");
    void* pfnRGA = resolveProc(hKB, "RegGetValueA");
    void* pfnRGW = resolveProc(hKB, "RegGetValueW");

    if (pfnROA && MH_CreateHook(pfnROA, &HookedRegOpenKeyExA, reinterpret_cast<void**>(&g_pfnRegOpenKeyExA)) == MH_OK)
    { MH_EnableHook(pfnROA); WriteLog("[D3d12Proxy] RegOpenKeyExA hook installed\n"); }
    if (pfnROW && MH_CreateHook(pfnROW, &HookedRegOpenKeyExW, reinterpret_cast<void**>(&g_pfnRegOpenKeyExW)) == MH_OK)
    { MH_EnableHook(pfnROW); WriteLog("[D3d12Proxy] RegOpenKeyExW hook installed\n"); }
    if (pfnRQA && MH_CreateHook(pfnRQA, &HookedRegQueryValueExA, reinterpret_cast<void**>(&g_pfnRegQueryValueExA)) == MH_OK)
    { MH_EnableHook(pfnRQA); WriteLog("[D3d12Proxy] RegQueryValueExA hook installed\n"); }
    if (pfnRQW && MH_CreateHook(pfnRQW, &HookedRegQueryValueExW, reinterpret_cast<void**>(&g_pfnRegQueryValueExW)) == MH_OK)
    { MH_EnableHook(pfnRQW); WriteLog("[D3d12Proxy] RegQueryValueExW hook installed\n"); }
    if (pfnRGA && MH_CreateHook(pfnRGA, &HookedRegGetValueA, reinterpret_cast<void**>(&g_pfnRegGetValueA)) == MH_OK)
    { MH_EnableHook(pfnRGA); WriteLog("[D3d12Proxy] RegGetValueA hook installed\n"); }
    if (pfnRGW && MH_CreateHook(pfnRGW, &HookedRegGetValueW, reinterpret_cast<void**>(&g_pfnRegGetValueW)) == MH_OK)
    { MH_EnableHook(pfnRGW); WriteLog("[D3d12Proxy] RegGetValueW hook installed\n"); }
}

static void InstallLoadLibraryHooks(HMODULE hK)
{
    PFN_GetProcAddr resolveProc = g_pfnGetProcAddrKB ? g_pfnGetProcAddrKB : g_pfnGetProcAddr;
    if (!resolveProc) resolveProc = reinterpret_cast<PFN_GetProcAddr>(&GetProcAddress);

    HMODULE hKB = GetModuleHandleW(L"KernelBase.dll");
    HMODULE hTarget = hKB ? hKB : hK;

    void* pfnLLA  = resolveProc(hTarget, "LoadLibraryA");
    void* pfnLLW  = resolveProc(hTarget, "LoadLibraryW");
    void* pfnLLEA = resolveProc(hTarget, "LoadLibraryExA");
    void* pfnLLEW = resolveProc(hTarget, "LoadLibraryExW");

    if (pfnLLA && MH_CreateHook(pfnLLA, &HookedLoadLibraryA, reinterpret_cast<void**>(&g_pfnLoadLibraryA)) == MH_OK)
    { MH_EnableHook(pfnLLA); WriteLog("[D3d12Proxy] LoadLibraryA hook installed\n"); }
    if (pfnLLW && MH_CreateHook(pfnLLW, &HookedLoadLibraryW, reinterpret_cast<void**>(&g_pfnLoadLibraryW)) == MH_OK)
    { MH_EnableHook(pfnLLW); WriteLog("[D3d12Proxy] LoadLibraryW hook installed\n"); }
    if (pfnLLEA && MH_CreateHook(pfnLLEA, &HookedLoadLibraryExA, reinterpret_cast<void**>(&g_pfnLoadLibraryExA)) == MH_OK)
    { MH_EnableHook(pfnLLEA); WriteLog("[D3d12Proxy] LoadLibraryExA hook installed\n"); }
    if (pfnLLEW && MH_CreateHook(pfnLLEW, &HookedLoadLibraryExW, reinterpret_cast<void**>(&g_pfnLoadLibraryExW)) == MH_OK)
    { MH_EnableHook(pfnLLEW); WriteLog("[D3d12Proxy] LoadLibraryExW hook installed\n"); }
}

static void InstallGetProcHook()
{
    HMODULE hK = GetModuleHandleW(L"kernel32.dll");
    if (!hK) return;
    void* pfn = GetProcAddress(hK, "GetProcAddress");
    if (!pfn) return;
    if (MH_CreateHook(pfn, &HookedGetProcAddr,
            reinterpret_cast<void**>(&g_pfnGetProcAddr)) == MH_OK)
    {
        MH_EnableHook(pfn);
        WriteLog("[D3d12Proxy] GetProcAddress hook installed\n");
    }
    else
    {
        WriteLog("[D3d12Proxy] GetProcAddress hook FAILED\n");
        return;
    }

    HMODULE hKB = GetModuleHandleW(L"KernelBase.dll");
    if (hKB)
    {
        void* pfnKB = g_pfnGetProcAddr(hKB, "GetProcAddress");
        if (pfnKB && pfnKB != pfn)
        {
            if (MH_CreateHook(pfnKB, &HookedGetProcAddr,
                    reinterpret_cast<void**>(&g_pfnGetProcAddrKB)) == MH_OK)
            {
                MH_EnableHook(pfnKB);
                WriteLog("[D3d12Proxy] KernelBase GetProcAddress hook installed\n");
            }
        }
    }

    void* pfnGMH = g_pfnGetProcAddr(hK, "GetModuleHandleA");
    if (pfnGMH)
    {
        if (MH_CreateHook(pfnGMH, &HookedGetModuleHandleA,
                reinterpret_cast<void**>(&g_pfnGetModuleHandleA)) == MH_OK)
        {
            MH_EnableHook(pfnGMH);
            WriteLog("[D3d12Proxy] GetModuleHandleA hook installed\n");
        }
    }

    InstallLoadLibraryHooks(hK);
}

// ---- DllMain ---------------------------------------------------------------

BOOL WINAPI DllMain(HINSTANCE hInst, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_hSelf = hInst;
        DisableThreadLibraryCalls(hInst);
        WriteLog("[D3d12Proxy] DLL_PROCESS_ATTACH\n");

        GetSystemDirectoryW(g_sysDir, MAX_PATH);
        WriteLog("[D3d12Proxy] GetSystemDirectoryW OK\n");

        if (!GetModuleHandleExW(0, L"dxgi.dll", &g_hRealDXGI))
        {
            wchar_t p[MAX_PATH];
            wcscpy_s(p, g_sysDir);
            wcscat_s(p, L"\\dxgi.dll");
            g_hRealDXGI = LoadLibraryW(p);
        }
        WriteLog(g_hRealDXGI
            ? "[D3d12Proxy] dxgi.dll handle acquired\n"
            : "[D3d12Proxy] WARNING: dxgi.dll not available\n");

        MH_Initialize();
        WriteLog("[D3d12Proxy] MH_Initialize OK\n");
        InstallDxgiHooks();
        PatchAmdIAT();
        InstallGetProcHook();
        InstallRegistryHooks();
    }
    else if (reason == DLL_PROCESS_DETACH)
    {
        WriteLog("[D3d12Proxy] DLL_PROCESS_DETACH\n");
        MH_Uninitialize();
        if (g_hAmdAdl)   { FreeLibrary(g_hAmdAdl);   g_hAmdAdl   = nullptr; }
        if (g_hAmdProxy) { FreeLibrary(g_hAmdProxy);  g_hAmdProxy = nullptr; }
        g_hRealDXGI  = nullptr;
        g_hRealD3D12 = nullptr;
    }
    return TRUE;
}

// ---- d3d12.dll export thunks -----------------------------------------------
// All named exports of system d3d12.dll forwarded to the real module.
// void* parameters avoid pulling in d3d12.h while producing correct x64 frames.
// D3D12 is always x64 so no Win32/x64 split needed.

extern "C" HRESULT WINAPI D3D12CreateDevice(void* pAdapter, UINT MinFeatureLevel,
                                             const void* riid, void** ppDevice)
{
    typedef HRESULT(WINAPI*PFN)(void*, UINT, const void*, void**);
    static PFN fn = (PFN)RealD3D12("D3D12CreateDevice");
    WriteLog("[D3d12Proxy] D3D12CreateDevice\n");
    return fn ? fn(pAdapter, MinFeatureLevel, riid, ppDevice) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12GetDebugInterface(const void* riid, void** ppvDebug)
{
    typedef HRESULT(WINAPI*PFN)(const void*, void**);
    static PFN fn = (PFN)RealD3D12("D3D12GetDebugInterface");
    return fn ? fn(riid, ppvDebug) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12GetInterface(const void* rclsid, const void* riid, void** ppvDebug)
{
    typedef HRESULT(WINAPI*PFN)(const void*, const void*, void**);
    static PFN fn = (PFN)RealD3D12("D3D12GetInterface");
    return fn ? fn(rclsid, riid, ppvDebug) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12SerializeRootSignature(const void* pRootSignature,
    UINT Version, void** ppBlob, void** ppErrorBlob)
{
    typedef HRESULT(WINAPI*PFN)(const void*, UINT, void**, void**);
    static PFN fn = (PFN)RealD3D12("D3D12SerializeRootSignature");
    return fn ? fn(pRootSignature, Version, ppBlob, ppErrorBlob) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12CreateRootSignatureDeserializer(const void* pSrcData,
    SIZE_T SrcDataSizeInBytes, const void* pRootSignatureDeserializerInterface, void** ppRootSignatureDeserializer)
{
    typedef HRESULT(WINAPI*PFN)(const void*, SIZE_T, const void*, void**);
    static PFN fn = (PFN)RealD3D12("D3D12CreateRootSignatureDeserializer");
    return fn ? fn(pSrcData, SrcDataSizeInBytes, pRootSignatureDeserializerInterface,
                   ppRootSignatureDeserializer) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12SerializeVersionedRootSignature(const void* pRootSignature,
    void** ppBlob, void** ppErrorBlob)
{
    typedef HRESULT(WINAPI*PFN)(const void*, void**, void**);
    static PFN fn = (PFN)RealD3D12("D3D12SerializeVersionedRootSignature");
    return fn ? fn(pRootSignature, ppBlob, ppErrorBlob) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12CreateVersionedRootSignatureDeserializer(const void* pSrcData,
    SIZE_T SrcDataSizeInBytes, const void* pRootSignatureDeserializerInterface, void** ppRootSignatureDeserializer)
{
    typedef HRESULT(WINAPI*PFN)(const void*, SIZE_T, const void*, void**);
    static PFN fn = (PFN)RealD3D12("D3D12CreateVersionedRootSignatureDeserializer");
    return fn ? fn(pSrcData, SrcDataSizeInBytes, pRootSignatureDeserializerInterface,
                   ppRootSignatureDeserializer) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12EnableExperimentalFeatures(UINT NumFeatures,
    const void* pIIDs, void* pConfigurationStructs, UINT* pConfigurationStructSizes)
{
    typedef HRESULT(WINAPI*PFN)(UINT, const void*, void*, UINT*);
    static PFN fn = (PFN)RealD3D12("D3D12EnableExperimentalFeatures");
    return fn ? fn(NumFeatures, pIIDs, pConfigurationStructs, pConfigurationStructSizes) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12CoreCreateLayeredDevice(const void* pInput, DWORD InputSize,
    const void* pLayerInput, const void* riid, void** ppvObj)
{
    typedef HRESULT(WINAPI*PFN)(const void*, DWORD, const void*, const void*, void**);
    static PFN fn = (PFN)RealD3D12("D3D12CoreCreateLayeredDevice");
    return fn ? fn(pInput, InputSize, pLayerInput, riid, ppvObj) : E_NOTIMPL;
}

extern "C" SIZE_T WINAPI D3D12CoreGetLayeredDeviceSize(const void* pInput, DWORD InputSize)
{
    typedef SIZE_T(WINAPI*PFN)(const void*, DWORD);
    static PFN fn = (PFN)RealD3D12("D3D12CoreGetLayeredDeviceSize");
    return fn ? fn(pInput, InputSize) : 0;
}

extern "C" HRESULT WINAPI D3D12CoreRegisterLayers(const void* pLayerDesc, DWORD NumLayers)
{
    typedef HRESULT(WINAPI*PFN)(const void*, DWORD);
    static PFN fn = (PFN)RealD3D12("D3D12CoreRegisterLayers");
    return fn ? fn(pLayerDesc, NumLayers) : E_NOTIMPL;
}

extern "C" HRESULT WINAPI D3D12DeviceRemovedExtendedData(void* a0, void* a1)
{
    typedef HRESULT(WINAPI*PFN)(void*, void*);
    static PFN fn = (PFN)RealD3D12("D3D12DeviceRemovedExtendedData");
    return fn ? fn(a0, a1) : E_NOTIMPL;
}

// PIX profiling functions — forwarded verbatim, never called by SE4 HD3D path.
extern "C" void* WINAPI D3D12PIXEventsReplaceBlock(void* a0, void* a1)
{
    typedef void*(WINAPI*PFN)(void*, void*);
    static PFN fn = (PFN)RealD3D12("D3D12PIXEventsReplaceBlock");
    return fn ? fn(a0, a1) : nullptr;
}

extern "C" void* WINAPI D3D12PIXGetThreadInfo()
{
    typedef void*(WINAPI*PFN)();
    static PFN fn = (PFN)RealD3D12("D3D12PIXGetThreadInfo");
    return fn ? fn() : nullptr;
}

extern "C" void WINAPI D3D12PIXNotifyWakeFromFenceSignal(void* a0)
{
    typedef void(WINAPI*PFN)(void*);
    static PFN fn = (PFN)RealD3D12("D3D12PIXNotifyWakeFromFenceSignal");
    if (fn) fn(a0);
}

extern "C" void WINAPI D3D12PIXReportCounter(void* a0, float a1)
{
    typedef void(WINAPI*PFN)(void*, float);
    static PFN fn = (PFN)RealD3D12("D3D12PIXReportCounter");
    if (fn) fn(a0, a1);
}

extern "C" void* WINAPI GetBehaviorValue(void* a0)
{
    typedef void*(WINAPI*PFN)(void*);
    static PFN fn = (PFN)RealD3D12("GetBehaviorValue");
    return fn ? fn(a0) : nullptr;
}

extern "C" void WINAPI SetAppCompatStringPointer(void* a0, void* a1)
{
    typedef void(WINAPI*PFN)(void*, void*);
    static PFN fn = (PFN)RealD3D12("SetAppCompatStringPointer");
    if (fn) fn(a0, a1);
}
