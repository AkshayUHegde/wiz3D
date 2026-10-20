/* wiz3D - S3DWrapperVK.dll
 *
 * The Vulkan stereo wrapper. Currently passthrough-only — establishes the
 * InitializeExchangeServer protocol that wiz3D-proxy/vulkan-1/vulkan-1.dll
 * uses to hand off instance creation, but performs no stereo injection yet.
 *
 * Per S3DWrapper12/PLAN.md, this shares almost all logic with S3DWrapper12 —
 * only the proxy DLL entry point and (eventually) the Vulkan implicit-layer
 * manifest differ. The actual stereo work will live in a shared add-on
 * codebase common to DX12 and Vulkan.
 *
 * M0 groundwork. Loader-lock-safe DllMain, lazy init in
 * EnsureWrapperInitialized() — same pattern as the DX12 wrapper.
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Local Vulkan types — same minimal subset as the proxy
// ---------------------------------------------------------------------------
typedef int      VkResult;
typedef uint32_t VkU32;
struct VkInstance_T;
typedef struct VkInstance_T* VkInstance;

typedef VkResult (__stdcall *PFN_vkCreateInstance)(const void*, const void*, VkInstance*);

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
    wcscat_s(path, L"\\S3DWrapperVK.log");
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
// Lazy init
// ---------------------------------------------------------------------------
static volatile LONG g_bInitialized = 0;
static HMODULE g_hRealVulkan = nullptr;
static PFN_vkCreateInstance g_pfnRealCreateInstance = nullptr;

static BOOL EnsureRealVulkanLoaded()
{
    if (g_pfnRealCreateInstance) return TRUE;

    wchar_t sysDir[MAX_PATH];
    GetSystemDirectoryW(sysDir, MAX_PATH);
    wchar_t path[MAX_PATH];
    wcscpy_s(path, sysDir);
    wcscat_s(path, L"\\vulkan-1.dll");

    g_hRealVulkan = LoadLibraryW(path);
    if (!g_hRealVulkan)
    {
        WriteLog("[wiz3D-VK] FAIL: could not load real vulkan-1.dll (err=%lu)\n", GetLastError());
        return FALSE;
    }

    g_pfnRealCreateInstance = (PFN_vkCreateInstance)
        GetProcAddress(g_hRealVulkan, "vkCreateInstance");
    if (!g_pfnRealCreateInstance)
    {
        WriteLog("[wiz3D-VK] FAIL: real vulkan-1.dll has no vkCreateInstance export\n");
        return FALSE;
    }
    WriteLog("[wiz3D-VK] Real vulkan-1.dll resolved at %p\n", g_pfnRealCreateInstance);
    return TRUE;
}

void EnsureWrapperInitialized()
{
    if (InterlockedCompareExchange(&g_bInitialized, 1, 0) != 0) return;
    WriteLog("[wiz3D-VK] EnsureWrapperInitialized — passthrough mode\n");
    EnsureRealVulkanLoaded();
}

// ---------------------------------------------------------------------------
// Exports
// ---------------------------------------------------------------------------
extern "C" __declspec(dllexport) DWORD WINAPI InitializeExchangeServer()
{
    EnsureWrapperInitialized();
    return 0;
}

extern "C" __declspec(dllexport) VkResult __stdcall vkCreateInstance(
    const void* pCreateInfo, const void* pAllocator, VkInstance* pInstance)
{
    EnsureWrapperInitialized();
    if (!g_pfnRealCreateInstance) return -1;  // VK_ERROR_INITIALIZATION_FAILED

    VkResult r = g_pfnRealCreateInstance(pCreateInfo, pAllocator, pInstance);
    WriteLog("[wiz3D-VK] vkCreateInstance (passthrough) -> %d\n", r);
    // TODO: when stereo work begins, install our pfnGetInstanceProcAddr
    // chain here to intercept vkCreateDevice / vkCreateSwapchainKHR / etc.
    return r;
}

// ---------------------------------------------------------------------------
// DllMain
// ---------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_hSelf = hModule;
        DisableThreadLibraryCalls(hModule);
        WriteLog("[wiz3D-VK] DLL_PROCESS_ATTACH\n");
    }
    else if (reason == DLL_PROCESS_DETACH)
    {
        if (g_hRealVulkan)
        {
            FreeLibrary(g_hRealVulkan);
            g_hRealVulkan = nullptr;
            g_pfnRealCreateInstance = nullptr;
        }
    }
    return TRUE;
}
