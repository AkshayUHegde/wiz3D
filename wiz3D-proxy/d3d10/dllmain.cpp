/* wiz3D - d3d10.dll Proxy Loader
 *
 * Drop this d3d10.dll into a game's folder alongside S3DWrapperD3D10.dll
 * and the output plugins. Forwards every d3d10.dll export to the real
 * system d3d10.dll. Intercepts D3D10CreateDevice and
 * D3D10CreateDeviceAndSwapChain — Stage 1 just logs and passes through;
 * Stage 2 will route the swap-chain creation through the wrapper.
 *
 * This proxy fills a gap the dxgi.dll proxy can't: many DX10 games create
 * their swap chain via D3D10CreateDeviceAndSwapChain, which uses an
 * internal-private DXGI path that bypasses IDXGIFactory::CreateSwapChain
 * (and therefore the wrapper's vtable hook). See architectural note in
 * memory.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../proxy_version.h"
#include <stdio.h>
#include <stdarg.h>

// ---------------------------------------------------------------------------
// Log to wiz3D_proxy.log next to the game exe. Append mode so the d3d10,
// d3d11, d3d9 and dxgi proxies share a single file when multiple are dropped
// into the same game folder.
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
// Real d3d10.dll loader (cached)
// ---------------------------------------------------------------------------
static HMODULE g_hRealD3D10 = nullptr;
static HMODULE g_hWrapper   = nullptr;
static HMODULE g_hProxy     = nullptr;

static BOOL LoadRealD3D10(void)
{
    if (g_hRealD3D10) return TRUE;
    WCHAR sysDir[MAX_PATH];
    GetSystemDirectoryW(sysDir, MAX_PATH);
    WCHAR path[MAX_PATH];
    wcscpy_s(path, sysDir);
    wcscat_s(path, L"\\d3d10.dll");
    g_hRealD3D10 = LoadLibraryW(path);
    if (!g_hRealD3D10)
    {
        Log("FAIL: real d3d10.dll load (err=%lu)\n", GetLastError());
        return FALSE;
    }
    Log("OK: Real d3d10.dll loaded from %ls\n", path);
    return TRUE;
}

// ---------------------------------------------------------------------------
// Lazy GetProcAddress helper — returns cached pointer or resolves it.
// ---------------------------------------------------------------------------
static FARPROC GetReal(FARPROC* pCache, const char* name)
{
    if (*pCache) return *pCache;
    if (!LoadRealD3D10()) return nullptr;
    *pCache = GetProcAddress(g_hRealD3D10, name);
    return *pCache;
}

// ---------------------------------------------------------------------------
// Pass-through forwarders. Each exported function lazy-resolves the real
// d3d10.dll entry point and calls through. Stage 1: even the special
// CreateDevice ones are pass-through, just logged.
//
// Signatures use opaque types (void*, UINT, etc.) so we don't need to pull
// in d3d10.h or d3d10misc.h — sizes match for the WINAPI calling convention.
// ---------------------------------------------------------------------------
#define DECLARE_FWD(name) static FARPROC g_pfn_##name = nullptr

DECLARE_FWD(D3D10CompileEffectFromMemory);
DECLARE_FWD(D3D10CompileShader);
DECLARE_FWD(D3D10CreateBlob);
DECLARE_FWD(D3D10CreateDevice);
DECLARE_FWD(D3D10CreateDeviceAndSwapChain);
DECLARE_FWD(D3D10CreateEffectFromMemory);
DECLARE_FWD(D3D10CreateEffectPoolFromMemory);
DECLARE_FWD(D3D10CreateStateBlock);
DECLARE_FWD(D3D10DisassembleEffect);
DECLARE_FWD(D3D10DisassembleShader);
DECLARE_FWD(D3D10GetGeometryShaderProfile);
DECLARE_FWD(D3D10GetInputAndOutputSignatureBlob);
DECLARE_FWD(D3D10GetInputSignatureBlob);
DECLARE_FWD(D3D10GetOutputSignatureBlob);
DECLARE_FWD(D3D10GetPixelShaderProfile);
DECLARE_FWD(D3D10GetShaderDebugInfo);
DECLARE_FWD(D3D10GetVersion);
DECLARE_FWD(D3D10GetVertexShaderProfile);
DECLARE_FWD(D3D10PreprocessShader);
DECLARE_FWD(D3D10ReflectShader);
DECLARE_FWD(D3D10RegisterLayers);
DECLARE_FWD(D3D10StateBlockMaskDifference);
DECLARE_FWD(D3D10StateBlockMaskDisableAll);
DECLARE_FWD(D3D10StateBlockMaskDisableCapture);
DECLARE_FWD(D3D10StateBlockMaskEnableAll);
DECLARE_FWD(D3D10StateBlockMaskEnableCapture);
DECLARE_FWD(D3D10StateBlockMaskGetSetting);
DECLARE_FWD(D3D10StateBlockMaskIntersect);
DECLARE_FWD(D3D10StateBlockMaskUnion);

// ---------------------------------------------------------------------------
// Special: D3D10CreateDevice — routes the returned ID3D10Device through
// S3DWrapperD3D10's wiz3D_WrapD3D10Device export so games on the Option B
// COM-wrap path (UseCOMWrap=1) get our Device10Proxy instead of the raw
// real device. Symmetric with the D3D11 dllmain's MaybeWrapDeviceAndContext.
// ---------------------------------------------------------------------------
typedef void (*PFN_wiz3D_WrapD3D10Device)(void**);
typedef void (*PFN_wiz3D_WrapD3D10SwapChain)(void**, void*);
static PFN_wiz3D_WrapD3D10Device    g_pfn_wiz3D_WrapD3D10    = nullptr;
static PFN_wiz3D_WrapD3D10SwapChain g_pfn_wiz3D_WrapD3D10SC  = nullptr;
static bool                          g_pfn_wiz3D_WrapD3D10_resolved = false;

static void MaybeWrapD3D10Device(void** ppDevice)
{
    if (!ppDevice || !*ppDevice) return;
    if (!g_pfn_wiz3D_WrapD3D10_resolved)
    {
        g_pfn_wiz3D_WrapD3D10_resolved = true;
        HMODULE hWrap = GetModuleHandleW(L"S3DWrapperD3D10.dll");
        if (hWrap)
        {
            g_pfn_wiz3D_WrapD3D10 = (PFN_wiz3D_WrapD3D10Device)
                GetProcAddress(hWrap, "wiz3D_WrapD3D10Device");
            g_pfn_wiz3D_WrapD3D10SC = (PFN_wiz3D_WrapD3D10SwapChain)
                GetProcAddress(hWrap, "wiz3D_WrapD3D10SwapChain");
            Log("  wiz3D Option B: wiz3D_WrapD3D10Device=%p wiz3D_WrapD3D10SwapChain=%p (hWrap=%p)\n",
                (void*)g_pfn_wiz3D_WrapD3D10, (void*)g_pfn_wiz3D_WrapD3D10SC, (void*)hWrap);
        }
        else
        {
            Log("  wiz3D Option B: S3DWrapperD3D10.dll not loaded -- passing through unwrapped\n");
        }
    }
    if (g_pfn_wiz3D_WrapD3D10)
        g_pfn_wiz3D_WrapD3D10(ppDevice);
}

static void MaybeWrapD3D10SwapChain(void** ppSwapChain, void* pWrappedDevice)
{
    if (!ppSwapChain || !*ppSwapChain || !pWrappedDevice) return;
    if (g_pfn_wiz3D_WrapD3D10SC)
        g_pfn_wiz3D_WrapD3D10SC(ppSwapChain, pWrappedDevice);
}

typedef HRESULT(WINAPI* PFN_D3D10CreateDevice)(void*, int, HMODULE, UINT, UINT, void**);
extern "C" __declspec(dllexport) HRESULT WINAPI D3D10CreateDevice(
    void* pAdapter, int DriverType, HMODULE Software, UINT Flags,
    UINT SDKVersion, void** ppDevice)
{
    Log("D3D10CreateDevice called (DriverType=%d, Flags=0x%X, SDKVersion=%u)\n",
        DriverType, Flags, SDKVersion);
    auto p = (PFN_D3D10CreateDevice)GetReal(&g_pfn_D3D10CreateDevice, "D3D10CreateDevice");
    if (!p) return E_FAIL;
    HRESULT hr = p(pAdapter, DriverType, Software, Flags, SDKVersion, ppDevice);
    Log("  D3D10CreateDevice returned 0x%08lX, *ppDevice=%p\n",
        hr, ppDevice ? *ppDevice : nullptr);
    if (SUCCEEDED(hr)) MaybeWrapD3D10Device(ppDevice);
    return hr;
}

typedef HRESULT(WINAPI* PFN_D3D10CreateDeviceAndSwapChain)(
    void*, int, HMODULE, UINT, UINT, void*, void**, void**);
extern "C" __declspec(dllexport) HRESULT WINAPI D3D10CreateDeviceAndSwapChain(
    void* pAdapter, int DriverType, HMODULE Software, UINT Flags,
    UINT SDKVersion, void* pSwapChainDesc, void** ppSwapChain, void** ppDevice)
{
    Log("D3D10CreateDeviceAndSwapChain called (DriverType=%d, Flags=0x%X, SDKVersion=%u)\n",
        DriverType, Flags, SDKVersion);
    auto p = (PFN_D3D10CreateDeviceAndSwapChain)GetReal(
        &g_pfn_D3D10CreateDeviceAndSwapChain, "D3D10CreateDeviceAndSwapChain");
    if (!p) return E_FAIL;
    HRESULT hr = p(pAdapter, DriverType, Software, Flags, SDKVersion,
                   pSwapChainDesc, ppSwapChain, ppDevice);
    Log("  D3D10CreateDeviceAndSwapChain returned 0x%08lX, *ppSwapChain=%p, *ppDevice=%p\n",
        hr, ppSwapChain ? *ppSwapChain : nullptr, ppDevice ? *ppDevice : nullptr);
    if (SUCCEEDED(hr))
    {
        MaybeWrapD3D10Device(ppDevice);
        if (ppDevice && *ppDevice) MaybeWrapD3D10SwapChain(ppSwapChain, *ppDevice);
    }
    return hr;
}

// ---------------------------------------------------------------------------
// Plain pass-through forwarders. Macro generates a typedef + extern-C export
// that lazy-resolves and calls through with the same signature.
// ---------------------------------------------------------------------------
#define FWD_RET(rt, name, params, args)                                       \
    typedef rt(WINAPI* PFN_##name) params;                                    \
    extern "C" __declspec(dllexport) rt WINAPI name params {                  \
        auto p = (PFN_##name)GetReal(&g_pfn_##name, #name);                   \
        if (!p) return (rt)0;                                                 \
        return p args;                                                        \
    }

FWD_RET(HRESULT, D3D10CompileEffectFromMemory,
    (void* pData, SIZE_T DataLength, LPCSTR pSrcFileName,
     const void* pDefines, void* pInclude, UINT HLSLFlags, UINT FXFlags,
     void** ppCompiledEffect, void** ppErrors),
    (pData, DataLength, pSrcFileName, pDefines, pInclude,
     HLSLFlags, FXFlags, ppCompiledEffect, ppErrors))

FWD_RET(HRESULT, D3D10CompileShader,
    (LPCSTR pSrcData, SIZE_T SrcDataSize, LPCSTR pFileName,
     const void* pDefines, void* pInclude, LPCSTR pFunctionName,
     LPCSTR pProfile, UINT Flags, void** ppShader, void** ppErrorMsgs),
    (pSrcData, SrcDataSize, pFileName, pDefines, pInclude,
     pFunctionName, pProfile, Flags, ppShader, ppErrorMsgs))

FWD_RET(HRESULT, D3D10CreateBlob,
    (SIZE_T NumBytes, void** ppBuffer), (NumBytes, ppBuffer))

FWD_RET(HRESULT, D3D10CreateEffectFromMemory,
    (void* pData, SIZE_T DataLength, UINT FXFlags,
     void* pDevice, void* pEffectPool, void** ppEffect),
    (pData, DataLength, FXFlags, pDevice, pEffectPool, ppEffect))

FWD_RET(HRESULT, D3D10CreateEffectPoolFromMemory,
    (void* pData, SIZE_T DataLength, UINT FXFlags,
     void* pDevice, void** ppEffectPool),
    (pData, DataLength, FXFlags, pDevice, ppEffectPool))

FWD_RET(HRESULT, D3D10CreateStateBlock,
    (void* pDevice, void* pStateBlockMask, void** ppStateBlock),
    (pDevice, pStateBlockMask, ppStateBlock))

FWD_RET(HRESULT, D3D10DisassembleEffect,
    (void* pEffect, BOOL EnableColorCode, void** ppDisassembly),
    (pEffect, EnableColorCode, ppDisassembly))

FWD_RET(HRESULT, D3D10DisassembleShader,
    (const void* pShader, SIZE_T BytecodeLength, BOOL EnableColorCode,
     LPCSTR pComments, void** ppDisassembly),
    (pShader, BytecodeLength, EnableColorCode, pComments, ppDisassembly))

FWD_RET(LPCSTR, D3D10GetGeometryShaderProfile, (void* pDevice), (pDevice))

FWD_RET(HRESULT, D3D10GetInputAndOutputSignatureBlob,
    (const void* pShaderBytecode, SIZE_T BytecodeLength, void** ppSignatureBlob),
    (pShaderBytecode, BytecodeLength, ppSignatureBlob))

FWD_RET(HRESULT, D3D10GetInputSignatureBlob,
    (const void* pShaderBytecode, SIZE_T BytecodeLength, void** ppSignatureBlob),
    (pShaderBytecode, BytecodeLength, ppSignatureBlob))

FWD_RET(HRESULT, D3D10GetOutputSignatureBlob,
    (const void* pShaderBytecode, SIZE_T BytecodeLength, void** ppSignatureBlob),
    (pShaderBytecode, BytecodeLength, ppSignatureBlob))

FWD_RET(LPCSTR, D3D10GetPixelShaderProfile, (void* pDevice), (pDevice))

FWD_RET(HRESULT, D3D10GetShaderDebugInfo,
    (const void* pShaderBytecode, SIZE_T BytecodeLength, void** ppDebugInfo),
    (pShaderBytecode, BytecodeLength, ppDebugInfo))

FWD_RET(DWORD, D3D10GetVersion, (void), ())

FWD_RET(LPCSTR, D3D10GetVertexShaderProfile, (void* pDevice), (pDevice))

FWD_RET(HRESULT, D3D10PreprocessShader,
    (LPCSTR pSrcData, SIZE_T SrcDataSize, LPCSTR pFileName,
     const void* pDefines, void* pInclude,
     void** ppShaderText, void** ppErrorMsgs),
    (pSrcData, SrcDataSize, pFileName, pDefines, pInclude,
     ppShaderText, ppErrorMsgs))

FWD_RET(HRESULT, D3D10ReflectShader,
    (const void* pShaderBytecode, SIZE_T BytecodeLength, void** ppReflector),
    (pShaderBytecode, BytecodeLength, ppReflector))

FWD_RET(HRESULT, D3D10RegisterLayers, (void), ())

FWD_RET(HRESULT, D3D10StateBlockMaskDifference,
    (const void* pA, const void* pB, void* pResult), (pA, pB, pResult))

FWD_RET(HRESULT, D3D10StateBlockMaskDisableAll,
    (void* pMask), (pMask))

FWD_RET(HRESULT, D3D10StateBlockMaskDisableCapture,
    (void* pMask, int StateType, UINT RangeStart, UINT RangeLength),
    (pMask, StateType, RangeStart, RangeLength))

FWD_RET(HRESULT, D3D10StateBlockMaskEnableAll,
    (void* pMask), (pMask))

FWD_RET(HRESULT, D3D10StateBlockMaskEnableCapture,
    (void* pMask, int StateType, UINT RangeStart, UINT RangeLength),
    (pMask, StateType, RangeStart, RangeLength))

FWD_RET(BOOL, D3D10StateBlockMaskGetSetting,
    (const void* pMask, int StateType, UINT Entry),
    (pMask, StateType, Entry))

FWD_RET(HRESULT, D3D10StateBlockMaskIntersect,
    (const void* pA, const void* pB, void* pResult), (pA, pB, pResult))

FWD_RET(HRESULT, D3D10StateBlockMaskUnion,
    (const void* pA, const void* pB, void* pResult), (pA, pB, pResult))

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
            Log("\n=== wiz3D " DISPLAYED_VERSION " - d3d10 proxy loaded ===\n");
            Log("Game exe: %ls\n", exePath);
            WCHAR proxyPath[MAX_PATH];
            GetModuleFileNameW(hModule, proxyPath, MAX_PATH);
            Log("Proxy DLL: %ls\n", proxyPath);
        }
        break;

    case DLL_PROCESS_DETACH:
        Log("=== wiz3D d3d10 proxy unloading ===\n");
        if (g_hWrapper)   { FreeLibrary(g_hWrapper);   g_hWrapper   = nullptr; }
        if (g_hRealD3D10) { FreeLibrary(g_hRealD3D10); g_hRealD3D10 = nullptr; }
        if (g_logFile)    { fclose(g_logFile);         g_logFile    = nullptr; }
        break;
    }
    return TRUE;
}
