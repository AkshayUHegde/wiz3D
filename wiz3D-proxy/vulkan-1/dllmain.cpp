/* wiz3D - vulkan-1.dll proxy loader: forwards every loader export to the system vulkan-1.dll and, before its first
 * layer scan, enables VK_LAYER_wiz3D_stereo from this folder (VK_ADD_LAYER_PATH, VK_LOADER_LAYERS_ENABLE). */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "../proxy_version.h"
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>
#include <string>

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
    if (g_logFile) return;
    // Game folders under Program Files-style ACLs refuse writes.
    WCHAR local[MAX_PATH] = {};
    if (!GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH)) return;
    lstrcatW(local, L"\\wiz3D");
    CreateDirectoryW(local, NULL);
    lstrcatW(local, L"\\wiz3D_proxy.log");
    g_logFile = _wfopen(local, L"a");
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

// ---------------------------------------------------------------------------
// Load real system vulkan-1.dll
// ---------------------------------------------------------------------------
static void GetProxyDirectory(WCHAR* out, DWORD cch)
{
    GetModuleFileNameW(g_hProxy, out, cch);
    WCHAR* pSlash = wcsrchr(out, L'\\');
    if (pSlash) *(pSlash + 1) = L'\0';
}

// Puts `item` first in the `sep`-separated list in environment variable `name`, keeping the value set before; an
// item already listed (case aside) leaves the variable as it is.
static void PrependEnvItem(const WCHAR* name, const WCHAR* item, WCHAR sep)
{
    const DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (!n) { SetEnvironmentVariableW(name, item); return; }
    std::wstring cur(n, L'\0');
    cur.resize(GetEnvironmentVariableW(name, &cur[0], n));
    const size_t len = wcslen(item);
    for (size_t at = 0; at <= cur.size(); )
    {
        size_t end = cur.find(sep, at);
        if (end == std::wstring::npos) end = cur.size();
        if (end - at == len && _wcsnicmp(cur.c_str() + at, item, len) == 0) return;
        at = end + 1;
    }
    Log("%ls: %ls put first, the value set before kept (%ls)\n", name, item, cur.c_str());
    SetEnvironmentVariableW(name, (std::wstring(item) + sep + cur).c_str());
}

// Points the loader at our manifest and enables the layer, before any call into the real vulkan-1.dll (read on its
// first layer scan); a VK_LAYER_PATH, which would override VK_ADD_LAYER_PATH, is reported.
static void EnableWiz3DLayer(void)
{
    WCHAR dir[MAX_PATH];
    GetProxyDirectory(dir, MAX_PATH);
    WCHAR manifest[MAX_PATH];
    wcscpy_s(manifest, dir);
    wcscat_s(manifest, L"VK_LAYER_wiz3D_stereo.json");
    if (GetFileAttributesW(manifest) == INVALID_FILE_ATTRIBUTES)
    {
        Log("WARN: %ls not found — layer not enabled, passthrough\n", manifest);
        return;
    }
    PrependEnvItem(L"VK_ADD_LAYER_PATH", dir, L';');
    PrependEnvItem(L"VK_LOADER_LAYERS_ENABLE", L"VK_LAYER_wiz3D_stereo", L',');
    WCHAR existing[MAX_PATH];
    if (GetEnvironmentVariableW(L"VK_LAYER_PATH", existing, MAX_PATH))
        Log("WARN: VK_LAYER_PATH is set (%ls); it overrides VK_ADD_LAYER_PATH\n", existing);
    Log("Layer enabled: VK_ADD_LAYER_PATH=%ls\n", dir);
}

// Trampoline table for the 259 exports games import directly (vk_forwards.asm), filled once the real loader is
// loaded; a stub hit before that resolves its own slot through wiz3D_ResolveForward.
#include "vk_forward_names.h"
extern "C" void* g_vkForward[WIZ3D_VK_FORWARD_COUNT] = {};
static BOOL LoadRealVulkan(void);

static void FillForwardTable(void)
{
    for (int i = 0; i < WIZ3D_VK_FORWARD_COUNT; ++i)
        g_vkForward[i] = (void*)GetProcAddress(g_hRealVulkan, g_vkForwardNames[i]);
}

extern "C" void* wiz3D_ResolveForward(unsigned index)
{
    LoadRealVulkan();
    void* p = index < WIZ3D_VK_FORWARD_COUNT ? g_vkForward[index] : nullptr;
    if (!p)
    {
        Log("FAIL: forward slot %u (%s) unresolved\n", index,
            index < WIZ3D_VK_FORWARD_COUNT ? g_vkForwardNames[index] : "?");
        RaiseException(STATUS_ENTRYPOINT_NOT_FOUND, EXCEPTION_NONCONTINUABLE, 0, nullptr);
    }
    return p;
}

static BOOL LoadRealVulkan(void)
{
    if (g_hRealVulkan) return TRUE;

    EnableWiz3DLayer();

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
    FillForwardTable();
    return TRUE;
}

// LoadWrapper: the InitializeExchangeServer probe of the other proxies; the loader loads the DLL as a layer, this only
// confirms it is loadable, so wiz3D_proxy.log explains a missing layer.
typedef DWORD (WINAPI *pfnInitializeExchangeServer)();

static void LoadWrapper(void)
{
    if (g_bWrapperActive) return;
    g_bWrapperActive = TRUE;

    WCHAR wrapPath[MAX_PATH];
    GetProxyDirectory(wrapPath, MAX_PATH);
    lstrcatW(wrapPath, L"S3DWrapperVK.dll");

    g_hWrapper = LoadLibraryW(wrapPath);
    if (!g_hWrapper)
    {
        Log("WARN: S3DWrapperVK.dll not loadable (err=%lu) — passthrough\n", GetLastError());
        return;
    }
    pfnInitializeExchangeServer pInit =
        (pfnInitializeExchangeServer)GetProcAddress(g_hWrapper, "InitializeExchangeServer");
    if (pInit)
        Log("OK: Wrapper present, InitializeExchangeServer returned %lu\n", pInit());
    else
        Log("FAIL: InitializeExchangeServer not in wrapper\n");
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
    if (!g_pfnRealCreateInstance) return -1;
    VkResult r = g_pfnRealCreateInstance(pCreateInfo, pAllocator, pInstance);
    Log("vkCreateInstance -> %d\n", r);
    return r;
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
