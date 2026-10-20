/* wiz3D - vulkan-1.dll Proxy Loader
 *
 * Drop this vulkan-1.dll into a game's folder alongside S3DWrapperVK.dll.
 * Forwards Vulkan loader entry points to the real system vulkan-1.dll, and
 * routes vkCreateInstance through the wiz3D wrapper using the same
 * InitializeExchangeServer protocol as the d3d9/d3d12 proxies.
 *
 * Why DLL proxy and not implicit layer:
 *   - Same drop-in pattern as the rest of wiz3D — no registry edits, no
 *     admin needed. User copies file into game folder, done.
 *   - The Vulkan implicit-layer manifest path (registered globally for all
 *     Vulkan apps) is in the long-term plan but more invasive. Saved for
 *     later. See S3DWrapper12/PLAN.md.
 *
 * Most Vulkan apps directly import only a small set of loader entry points
 * (vkGetInstanceProcAddr is the gateway for everything else). We forward
 * the canonical six and let the game fetch the rest via proc-addr lookup.
 *
 * M0 groundwork — passthrough only. No stereo injection yet.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../proxy_version.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Log to wiz3D_proxy.log (shared with d3d9 / d3d12 proxies via append mode)
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
// Local Vulkan typedefs — kept minimal so this file doesn't need vulkan.h
// ---------------------------------------------------------------------------
// VkResult is a signed enum, int-compatible. VK_SUCCESS = 0, errors negative.
typedef int      VkResult;
typedef uint32_t VkU32;

// Opaque handles — we never dereference, just pass through.
struct VkInstance_T;
struct VkDevice_T;
typedef struct VkInstance_T* VkInstance;
typedef struct VkDevice_T*   VkDevice;

// On Windows the Vulkan calling convention is __stdcall.
typedef void (__stdcall *PFN_vkVoidFunction)(void);

typedef PFN_vkVoidFunction (__stdcall *PFN_vkGetInstanceProcAddr)(VkInstance, const char*);
typedef PFN_vkVoidFunction (__stdcall *PFN_vkGetDeviceProcAddr)(VkDevice, const char*);
typedef VkResult (__stdcall *PFN_vkCreateInstance)(const void*, const void*, VkInstance*);
typedef VkResult (__stdcall *PFN_vkEnumerateInstanceExtensionProperties)(const char*, VkU32*, void*);
typedef VkResult (__stdcall *PFN_vkEnumerateInstanceLayerProperties)(VkU32*, void*);
typedef VkResult (__stdcall *PFN_vkEnumerateInstanceVersion)(VkU32*);

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
static HMODULE g_hRealVulkan = nullptr;
static HMODULE g_hWrapper    = nullptr;
static HMODULE g_hProxy      = nullptr;
static BOOL    g_bWrapperActive = FALSE;

static PFN_vkGetInstanceProcAddr                  g_pfnRealGetInstanceProcAddr             = nullptr;
static PFN_vkGetDeviceProcAddr                    g_pfnRealGetDeviceProcAddr               = nullptr;
static PFN_vkCreateInstance                       g_pfnRealCreateInstance                  = nullptr;
static PFN_vkEnumerateInstanceExtensionProperties g_pfnRealEnumInstanceExtensionProperties = nullptr;
static PFN_vkEnumerateInstanceLayerProperties     g_pfnRealEnumInstanceLayerProperties     = nullptr;
static PFN_vkEnumerateInstanceVersion             g_pfnRealEnumInstanceVersion             = nullptr;

static PFN_vkCreateInstance g_pfnWrapCreateInstance = nullptr;

// ---------------------------------------------------------------------------
// Load real system vulkan-1.dll
// ---------------------------------------------------------------------------
static BOOL LoadRealVulkan(void)
{
    if (g_hRealVulkan) return TRUE;

    WCHAR sysDir[MAX_PATH];
    GetSystemDirectoryW(sysDir, MAX_PATH);

    WCHAR path[MAX_PATH];
    wcscpy_s(path, sysDir);
    wcscat_s(path, L"\\vulkan-1.dll");

    g_hRealVulkan = LoadLibraryW(path);
    if (!g_hRealVulkan)
    {
        Log("FAIL: real vulkan-1.dll load (err=%lu)\n", GetLastError());
        return FALSE;
    }
    Log("OK: Real vulkan-1.dll loaded from %ls\n", path);

    g_pfnRealGetInstanceProcAddr             = (PFN_vkGetInstanceProcAddr)                  GetProcAddress(g_hRealVulkan, "vkGetInstanceProcAddr");
    g_pfnRealGetDeviceProcAddr               = (PFN_vkGetDeviceProcAddr)                    GetProcAddress(g_hRealVulkan, "vkGetDeviceProcAddr");
    g_pfnRealCreateInstance                  = (PFN_vkCreateInstance)                       GetProcAddress(g_hRealVulkan, "vkCreateInstance");
    g_pfnRealEnumInstanceExtensionProperties = (PFN_vkEnumerateInstanceExtensionProperties) GetProcAddress(g_hRealVulkan, "vkEnumerateInstanceExtensionProperties");
    g_pfnRealEnumInstanceLayerProperties     = (PFN_vkEnumerateInstanceLayerProperties)     GetProcAddress(g_hRealVulkan, "vkEnumerateInstanceLayerProperties");
    g_pfnRealEnumInstanceVersion             = (PFN_vkEnumerateInstanceVersion)             GetProcAddress(g_hRealVulkan, "vkEnumerateInstanceVersion");
    return TRUE;
}

// ---------------------------------------------------------------------------
// LoadWrapper — same protocol as d3d9 / d3d12 proxies
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
    lstrcatW(wrapPath, L"S3DWrapperVK.dll");
    Log("Loading wrapper: %ls\n", wrapPath);

    g_hWrapper = LoadLibraryW(wrapPath);
    if (!g_hWrapper)
    {
        Log("WARN: S3DWrapperVK.dll not found (err=%lu) — passthrough\n", GetLastError());
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
            g_pfnWrapCreateInstance = (PFN_vkCreateInstance)
                GetProcAddress(g_hWrapper, "vkCreateInstance");
            Log("Wrapper exports: vkCreateInstance=%p\n", g_pfnWrapCreateInstance);
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
extern "C" __declspec(dllexport) VkResult __stdcall vkCreateInstance(
    const void* pCreateInfo, const void* pAllocator, VkInstance* pInstance)
{
    Log("vkCreateInstance called\n");
    if (!LoadRealVulkan()) return -1;  // VK_ERROR_INITIALIZATION_FAILED
    LoadWrapper();

    if (g_pfnWrapCreateInstance)
    {
        Log("Routing through WRAPPER vkCreateInstance...\n");
        return g_pfnWrapCreateInstance(pCreateInfo, pAllocator, pInstance);
    }
    if (g_pfnRealCreateInstance)
    {
        Log("Routing through REAL vkCreateInstance (no wrapper)\n");
        return g_pfnRealCreateInstance(pCreateInfo, pAllocator, pInstance);
    }
    return -1;
}

extern "C" __declspec(dllexport) PFN_vkVoidFunction __stdcall vkGetInstanceProcAddr(
    VkInstance instance, const char* pName)
{
    if (!LoadRealVulkan() || !g_pfnRealGetInstanceProcAddr) return nullptr;

    // Return our overridden vkCreateInstance when the game asks via
    // the proc-addr mechanism instead of importing it directly. This is
    // how most modern Vulkan apps actually call vkCreateInstance.
    if (pName)
    {
        if (lstrcmpA(pName, "vkCreateInstance") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(&vkCreateInstance);
        if (lstrcmpA(pName, "vkGetInstanceProcAddr") == 0)
            return reinterpret_cast<PFN_vkVoidFunction>(&vkGetInstanceProcAddr);
    }
    return g_pfnRealGetInstanceProcAddr(instance, pName);
}

extern "C" __declspec(dllexport) PFN_vkVoidFunction __stdcall vkGetDeviceProcAddr(
    VkDevice device, const char* pName)
{
    if (!LoadRealVulkan() || !g_pfnRealGetDeviceProcAddr) return nullptr;
    return g_pfnRealGetDeviceProcAddr(device, pName);
}

extern "C" __declspec(dllexport) VkResult __stdcall vkEnumerateInstanceExtensionProperties(
    const char* pLayerName, VkU32* pPropertyCount, void* pProperties)
{
    if (!LoadRealVulkan() || !g_pfnRealEnumInstanceExtensionProperties) return -1;
    return g_pfnRealEnumInstanceExtensionProperties(pLayerName, pPropertyCount, pProperties);
}

extern "C" __declspec(dllexport) VkResult __stdcall vkEnumerateInstanceLayerProperties(
    VkU32* pPropertyCount, void* pProperties)
{
    if (!LoadRealVulkan() || !g_pfnRealEnumInstanceLayerProperties) return -1;
    return g_pfnRealEnumInstanceLayerProperties(pPropertyCount, pProperties);
}

extern "C" __declspec(dllexport) VkResult __stdcall vkEnumerateInstanceVersion(VkU32* pApiVersion)
{
    if (!LoadRealVulkan() || !g_pfnRealEnumInstanceVersion)
    {
        // Vulkan 1.0 loaders don't export this — return 1.0.0 directly.
        if (pApiVersion) *pApiVersion = (1u << 22);  // VK_MAKE_VERSION(1,0,0)
        return 0;
    }
    return g_pfnRealEnumInstanceVersion(pApiVersion);
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
            Log("=== wiz3D " DISPLAYED_VERSION " - vulkan-1 proxy loaded ===\n");
            Log("Game exe: %ls\n", exePath);
            WCHAR proxyPath[MAX_PATH];
            GetModuleFileNameW(hModule, proxyPath, MAX_PATH);
            Log("Proxy DLL: %ls\n\n", proxyPath);
        }
        break;

    case DLL_PROCESS_DETACH:
        Log("=== wiz3D vulkan-1 proxy unloading ===\n");
        if (g_hWrapper)    { FreeLibrary(g_hWrapper);    g_hWrapper    = nullptr; }
        if (g_hRealVulkan) { FreeLibrary(g_hRealVulkan); g_hRealVulkan = nullptr; }
        if (g_logFile)     { fclose(g_logFile);          g_logFile     = nullptr; }
        break;
    }
    return TRUE;
}
