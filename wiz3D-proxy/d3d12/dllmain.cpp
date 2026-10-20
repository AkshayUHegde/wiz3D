/* wiz3D - d3d12.dll Proxy Loader
 *
 * Drop this d3d12.dll into a game's folder alongside S3DWrapperD3D12.dll
 * and the output plugins. Forwards all D3D12 calls to the real system
 * d3d12.dll, and routes D3D12CreateDevice through the wiz3D wrapper using
 * the same InitializeExchangeServer protocol as the d3d9 proxy.
 *
 * M0 groundwork — passthrough only. No stereo injection yet. See
 * S3DWrapper12/PLAN.md for the architecture this is building toward.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../proxy_version.h"
#include <stdio.h>
#include <stdarg.h>

// ---------------------------------------------------------------------------
// Log to wiz3D_proxy.log next to the game exe
// ---------------------------------------------------------------------------
static FILE* g_logFile = nullptr;

static void LogOpen(void)
{
    if (g_logFile) return;
    WCHAR dir[MAX_PATH];
    GetModuleFileNameW(NULL, dir, MAX_PATH);
    WCHAR* pSlash = wcsrchr(dir, L'\\');
    if (pSlash) *(pSlash + 1) = L'\0';
    lstrcatW(dir, L"wiz3D_Logs");
    CreateDirectoryW(dir, NULL);
    lstrcatW(dir, L"\\wiz3D_proxy.log");
    // Append rather than truncate, so the d3d9 / d3d12 / vk proxies can
    // share a single log file when more than one is in the same game folder.
    g_logFile = _wfopen(dir, L"a");
}

static void Log(const char* fmt, ...)
{
    if (!g_logFile) LogOpen();
    if (!g_logFile) return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(g_logFile, fmt, ap);
    va_end(ap);
    fflush(g_logFile);
}

// ---------------------------------------------------------------------------
// Handles + forwarded function pointers
// ---------------------------------------------------------------------------
static HMODULE g_hRealD3D12 = nullptr;
static HMODULE g_hWrapper   = nullptr;
static HMODULE g_hProxy     = nullptr;
static BOOL    g_bWrapperActive = FALSE;

// We treat all D3D12 entry-point arguments as opaque void* / UINT / GUID-like
// blobs so this proxy doesn't need the d3d12.h header. Forwarding only.
typedef HRESULT (WINAPI *PFN_D3D12CreateDevice)(void*, int, const GUID*, void**);
typedef HRESULT (WINAPI *PFN_D3D12GetDebugInterface)(const GUID*, void**);
typedef HRESULT (WINAPI *PFN_D3D12GetInterface)(const GUID*, const GUID*, void**);
typedef HRESULT (WINAPI *PFN_D3D12CreateRootSignatureDeserializer)(LPCVOID, SIZE_T, const GUID*, void**);
typedef HRESULT (WINAPI *PFN_D3D12CreateVersionedRootSignatureDeserializer)(LPCVOID, SIZE_T, const GUID*, void**);
typedef HRESULT (WINAPI *PFN_D3D12SerializeRootSignature)(const void*, int, void**, void**);
typedef HRESULT (WINAPI *PFN_D3D12SerializeVersionedRootSignature)(const void*, void**, void**);
typedef HRESULT (WINAPI *PFN_D3D12EnableExperimentalFeatures)(UINT, const GUID*, void*, UINT*);

static PFN_D3D12CreateDevice                            g_pfnRealCreateDevice                            = nullptr;
static PFN_D3D12GetDebugInterface                       g_pfnRealGetDebugInterface                       = nullptr;
static PFN_D3D12GetInterface                            g_pfnRealGetInterface                            = nullptr;
static PFN_D3D12CreateRootSignatureDeserializer         g_pfnRealCreateRootSigDeserializer               = nullptr;
static PFN_D3D12CreateVersionedRootSignatureDeserializer g_pfnRealCreateVersionedRootSigDeserializer     = nullptr;
static PFN_D3D12SerializeRootSignature                  g_pfnRealSerializeRootSig                        = nullptr;
static PFN_D3D12SerializeVersionedRootSignature         g_pfnRealSerializeVersionedRootSig               = nullptr;
static PFN_D3D12EnableExperimentalFeatures              g_pfnRealEnableExperimentalFeatures              = nullptr;

// Wrapper's D3D12CreateDevice (only this one is wrapper-overridable for now).
static PFN_D3D12CreateDevice g_pfnWrapCreateDevice = nullptr;

// ---------------------------------------------------------------------------
// Load real system d3d12.dll and resolve every export we forward
// ---------------------------------------------------------------------------
static BOOL LoadRealD3D12(void)
{
    if (g_hRealD3D12) return TRUE;

    WCHAR sysDir[MAX_PATH];
    GetSystemDirectoryW(sysDir, MAX_PATH);

    WCHAR path[MAX_PATH];
    wcscpy_s(path, sysDir);
    wcscat_s(path, L"\\d3d12.dll");

    g_hRealD3D12 = LoadLibraryW(path);
    if (!g_hRealD3D12)
    {
        Log("FAIL: real d3d12.dll load (err=%lu)\n", GetLastError());
        return FALSE;
    }
    Log("OK: Real d3d12.dll loaded from %ls\n", path);

    g_pfnRealCreateDevice                       = (PFN_D3D12CreateDevice)                            GetProcAddress(g_hRealD3D12, "D3D12CreateDevice");
    g_pfnRealGetDebugInterface                  = (PFN_D3D12GetDebugInterface)                       GetProcAddress(g_hRealD3D12, "D3D12GetDebugInterface");
    g_pfnRealGetInterface                       = (PFN_D3D12GetInterface)                            GetProcAddress(g_hRealD3D12, "D3D12GetInterface");
    g_pfnRealCreateRootSigDeserializer          = (PFN_D3D12CreateRootSignatureDeserializer)         GetProcAddress(g_hRealD3D12, "D3D12CreateRootSignatureDeserializer");
    g_pfnRealCreateVersionedRootSigDeserializer = (PFN_D3D12CreateVersionedRootSignatureDeserializer)GetProcAddress(g_hRealD3D12, "D3D12CreateVersionedRootSignatureDeserializer");
    g_pfnRealSerializeRootSig                   = (PFN_D3D12SerializeRootSignature)                  GetProcAddress(g_hRealD3D12, "D3D12SerializeRootSignature");
    g_pfnRealSerializeVersionedRootSig          = (PFN_D3D12SerializeVersionedRootSignature)         GetProcAddress(g_hRealD3D12, "D3D12SerializeVersionedRootSignature");
    g_pfnRealEnableExperimentalFeatures         = (PFN_D3D12EnableExperimentalFeatures)              GetProcAddress(g_hRealD3D12, "D3D12EnableExperimentalFeatures");
    return TRUE;
}

// ---------------------------------------------------------------------------
// LoadWrapper — same protocol as d3d9 proxy: load S3DWrapperD3D12.dll,
// call InitializeExchangeServer, look up D3D12CreateDevice export.
// ---------------------------------------------------------------------------
typedef DWORD (WINAPI *pfnInitializeExchangeServer)();

static void GetProxyDirectory(WCHAR* out, DWORD cch)
{
    GetModuleFileNameW(g_hProxy, out, cch);
    WCHAR* pSlash = wcsrchr(out, L'\\');
    if (pSlash) *(pSlash + 1) = L'\0';
}

static void LoadWrapper(void)
{
    if (g_bWrapperActive) return;
    g_bWrapperActive = TRUE;

    WCHAR wrapPath[MAX_PATH];
    GetProxyDirectory(wrapPath, MAX_PATH);
    lstrcatW(wrapPath, L"S3DWrapperD3D12.dll");
    Log("Loading wrapper: %ls\n", wrapPath);

    g_hWrapper = LoadLibraryW(wrapPath);
    if (!g_hWrapper)
    {
        Log("WARN: S3DWrapperD3D12.dll not found (err=%lu) — passthrough\n", GetLastError());
        return;
    }
    Log("OK: Wrapper loaded\n");

    pfnInitializeExchangeServer pInit =
        (pfnInitializeExchangeServer)GetProcAddress(g_hWrapper, "InitializeExchangeServer");
    if (pInit)
    {
        Log("Calling InitializeExchangeServer...\n");
        DWORD routerType = pInit();
        Log("InitializeExchangeServer returned routerType=%lu\n", routerType);
        if (routerType <= 1)
        {
            g_pfnWrapCreateDevice = (PFN_D3D12CreateDevice)
                GetProcAddress(g_hWrapper, "D3D12CreateDevice");
            Log("Wrapper exports: D3D12CreateDevice=%p\n", g_pfnWrapCreateDevice);
        }
    }
    else
    {
        Log("FAIL: InitializeExchangeServer not in wrapper — passthrough\n");
    }
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport) HRESULT WINAPI D3D12CreateDevice(
    void* pAdapter, int MinimumFeatureLevel, const GUID* riid, void** ppDevice)
{
    Log("D3D12CreateDevice(adapter=%p, FL=0x%X) called\n", pAdapter, MinimumFeatureLevel);

    if (!LoadRealD3D12()) return E_FAIL;
    LoadWrapper();

    if (g_pfnWrapCreateDevice)
    {
        Log("Routing through WRAPPER D3D12CreateDevice...\n");
        return g_pfnWrapCreateDevice(pAdapter, MinimumFeatureLevel, riid, ppDevice);
    }
    if (g_pfnRealCreateDevice)
    {
        Log("Routing through REAL D3D12CreateDevice (no wrapper)\n");
        return g_pfnRealCreateDevice(pAdapter, MinimumFeatureLevel, riid, ppDevice);
    }
    return E_FAIL;
}

extern "C" __declspec(dllexport) HRESULT WINAPI D3D12GetDebugInterface(const GUID* riid, void** ppvDebug)
{
    if (!LoadRealD3D12() || !g_pfnRealGetDebugInterface) return E_FAIL;
    return g_pfnRealGetDebugInterface(riid, ppvDebug);
}

extern "C" __declspec(dllexport) HRESULT WINAPI D3D12GetInterface(const GUID* rclsid, const GUID* riid, void** ppvDebug)
{
    if (!LoadRealD3D12() || !g_pfnRealGetInterface) return E_FAIL;
    return g_pfnRealGetInterface(rclsid, riid, ppvDebug);
}

extern "C" __declspec(dllexport) HRESULT WINAPI D3D12CreateRootSignatureDeserializer(
    LPCVOID pSrcData, SIZE_T SrcDataSizeInBytes, const GUID* iface, void** pp)
{
    if (!LoadRealD3D12() || !g_pfnRealCreateRootSigDeserializer) return E_FAIL;
    return g_pfnRealCreateRootSigDeserializer(pSrcData, SrcDataSizeInBytes, iface, pp);
}

extern "C" __declspec(dllexport) HRESULT WINAPI D3D12CreateVersionedRootSignatureDeserializer(
    LPCVOID pSrcData, SIZE_T SrcDataSizeInBytes, const GUID* iface, void** pp)
{
    if (!LoadRealD3D12() || !g_pfnRealCreateVersionedRootSigDeserializer) return E_FAIL;
    return g_pfnRealCreateVersionedRootSigDeserializer(pSrcData, SrcDataSizeInBytes, iface, pp);
}

extern "C" __declspec(dllexport) HRESULT WINAPI D3D12SerializeRootSignature(
    const void* pRootSig, int Version, void** ppBlob, void** ppErrorBlob)
{
    if (!LoadRealD3D12() || !g_pfnRealSerializeRootSig) return E_FAIL;
    return g_pfnRealSerializeRootSig(pRootSig, Version, ppBlob, ppErrorBlob);
}

extern "C" __declspec(dllexport) HRESULT WINAPI D3D12SerializeVersionedRootSignature(
    const void* pRootSig, void** ppBlob, void** ppErrorBlob)
{
    if (!LoadRealD3D12() || !g_pfnRealSerializeVersionedRootSig) return E_FAIL;
    return g_pfnRealSerializeVersionedRootSig(pRootSig, ppBlob, ppErrorBlob);
}

extern "C" __declspec(dllexport) HRESULT WINAPI D3D12EnableExperimentalFeatures(
    UINT NumFeatures, const GUID* pIIDs, void* pConfigStructs, UINT* pConfigStructSizes)
{
    if (!LoadRealD3D12() || !g_pfnRealEnableExperimentalFeatures) return E_FAIL;
    return g_pfnRealEnableExperimentalFeatures(NumFeatures, pIIDs, pConfigStructs, pConfigStructSizes);
}

// ---------------------------------------------------------------------------
// DllMain
// ---------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        g_hProxy = hModule;
        DisableThreadLibraryCalls(hModule);
        LogOpen();
        {
            WCHAR exePath[MAX_PATH];
            GetModuleFileNameW(NULL, exePath, MAX_PATH);
            Log("=== wiz3D " DISPLAYED_VERSION " - d3d12 proxy loaded ===\n");
            Log("Game exe: %ls\n", exePath);
            WCHAR proxyPath[MAX_PATH];
            GetModuleFileNameW(hModule, proxyPath, MAX_PATH);
            Log("Proxy DLL: %ls\n\n", proxyPath);
        }
        break;

    case DLL_PROCESS_DETACH:
        Log("=== wiz3D d3d12 proxy unloading ===\n");
        if (g_hWrapper)   { FreeLibrary(g_hWrapper);   g_hWrapper   = nullptr; }
        if (g_hRealD3D12) { FreeLibrary(g_hRealD3D12); g_hRealD3D12 = nullptr; }
        if (g_logFile)    { fclose(g_logFile);         g_logFile    = nullptr; }
        break;
    }
    return TRUE;
}
