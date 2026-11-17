/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// S3DWrapperVK.dll: an explicit Vulkan layer (VK_LAYER_wiz3D_stereo.json) that the vulkan-1.dll proxy enables; the
// hooks live in Layer.cpp. DllMain only opens the log, as in the other wrappers.

#include "Layer.h"
#include "../S3DAPI/GlobalData.h"
#include <stdio.h>
#include <stdarg.h>

GlobalInfo& gInfo = _gInfo;

static HMODULE g_hSelf = nullptr;
static HANDLE  g_log   = INVALID_HANDLE_VALUE;
static CRITICAL_SECTION g_logLock;

namespace wiz3d { namespace vk {

// What the layer was last doing, so a fault can say whether it happened inside
// our work or in a call the game made next.
char g_breadcrumb[160] = "none";
void Breadcrumb(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    _vsnprintf_s(g_breadcrumb, sizeof(g_breadcrumb), _TRUNCATE, fmt, ap);
    va_end(ap);
}

void Log(const char* fmt, ...)
{
    if (g_log == INVALID_HANDLE_VALUE) return;
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = _vsnprintf_s(buf, sizeof(buf), _TRUNCATE, fmt, ap);
    va_end(ap);
    if (n < 0) n = (int)strlen(buf);   // too long for buf: the part that fits, not nothing
    if (n <= 0) return;
    EnterCriticalSection(&g_logLock);
    DWORD written;
    WriteFile(g_log, buf, (DWORD)n, &written, nullptr);
    LeaveCriticalSection(&g_logLock);
}

// A game with its own unhandled-exception filter exits on a fault before Windows records it; a vectored handler
// runs ahead of that filter, so the fault gets named in the log.
static PVOID g_veh = nullptr;
static LONG CALLBACK FaultLogger(EXCEPTION_POINTERS* ep)
{
    const DWORD code = ep->ExceptionRecord->ExceptionCode;
    // Only real faults; C++ throws and debugger notifications pass through.
    if (code != EXCEPTION_ACCESS_VIOLATION && code != EXCEPTION_ILLEGAL_INSTRUCTION &&
        code != EXCEPTION_STACK_OVERFLOW && code != EXCEPTION_INT_DIVIDE_BY_ZERO &&
        code != EXCEPTION_PRIV_INSTRUCTION && code != EXCEPTION_IN_PAGE_ERROR)
        return EXCEPTION_CONTINUE_SEARCH;
    static LONG once = 0;
    if (InterlockedExchange(&once, 1)) return EXCEPTION_CONTINUE_SEARCH;

    void* addr = ep->ExceptionRecord->ExceptionAddress;
    char module[MAX_PATH] = "?";
    HMODULE mod = nullptr;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)addr, &mod) && mod)
        GetModuleFileNameA(mod, module, MAX_PATH);
    wiz3d::vk::Log("\n!!!! FAULT code=0x%08X at %p in %s (base %p, +0x%llX)\n",
                   code, addr, module, (void*)mod,
                   mod ? (unsigned long long)((char*)addr - (char*)mod) : 0ull);
    wiz3d::vk::Log("!!!! layer was: %s\n", wiz3d::vk::g_breadcrumb);
    if (code == EXCEPTION_ACCESS_VIOLATION || code == EXCEPTION_IN_PAGE_ERROR)
        wiz3d::vk::Log("!!!! %s address %p\n",
                       ep->ExceptionRecord->ExceptionInformation[0] ? "writing" : "reading",
                       (void*)ep->ExceptionRecord->ExceptionInformation[1]);
    return EXCEPTION_CONTINUE_SEARCH;   // let the game's own handler run as before
}

}} // namespace wiz3d::vk

static void OpenLog()
{
    wchar_t path[MAX_PATH] = {};
    GetModuleFileNameW(g_hSelf, path, MAX_PATH);
    wchar_t* p = wcsrchr(path, L'\\');
    if (p) p[1] = L'\0';
    wcscat_s(path, L"wiz3D_Logs");
    CreateDirectoryW(path, nullptr);
    wcscat_s(path, L"\\S3DWrapperVK.log");
    g_log = CreateFileW(path, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (g_log != INVALID_HANDLE_VALUE) return;
    // Game folders under Program Files-style ACLs refuse writes.
    wchar_t local[MAX_PATH] = {};
    if (!GetEnvironmentVariableW(L"LOCALAPPDATA", local, MAX_PATH)) return;
    wcscat_s(local, L"\\wiz3D");
    CreateDirectoryW(local, nullptr);
    wcscat_s(local, L"\\S3DWrapperVK.log");
    g_log = CreateFileW(local, FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE,
                        nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
}

// Exports
extern "C" __declspec(dllexport) DWORD WINAPI InitializeExchangeServer()
{
    wiz3d::vk::Log("InitializeExchangeServer (proxy probe)\n");
    return 0;
}

// Loader interface v2: the loader asks for our proc-addr functions through the
// entry point named in the manifest instead of fixed exports.
extern "C" __declspec(dllexport) VkResult VKAPI_CALL wiz3D_NegotiateLoaderLayerInterfaceVersion(VkNegotiateLayerInterface* pVersionStruct)
{
    if (!pVersionStruct || pVersionStruct->sType != LAYER_NEGOTIATE_INTERFACE_STRUCT)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (pVersionStruct->loaderLayerInterfaceVersion < 2)
        return VK_ERROR_INITIALIZATION_FAILED;
    pVersionStruct->loaderLayerInterfaceVersion = 2;
    pVersionStruct->pfnGetInstanceProcAddr = &wiz3d::vk::LayerGetInstanceProcAddr;
    pVersionStruct->pfnGetDeviceProcAddr   = &wiz3d::vk::LayerGetDeviceProcAddr;
    pVersionStruct->pfnGetPhysicalDeviceProcAddr = nullptr;
    wiz3d::vk::Log("Loader negotiated layer interface v2\n");
    return VK_SUCCESS;
}

extern "C" __declspec(dllexport) PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* pName)
{
    return wiz3d::vk::LayerGetInstanceProcAddr(instance, pName);
}

extern "C" __declspec(dllexport) PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(VkDevice device, const char* pName)
{
    return wiz3d::vk::LayerGetDeviceProcAddr(device, pName);
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID)
{
    if (reason == DLL_PROCESS_ATTACH)
    {
        g_hSelf = hModule;
        DisableThreadLibraryCalls(hModule);
        InitializeCriticalSection(&g_logLock);
        OpenLog();
        wiz3d::vk::Log("=== S3DWrapperVK attached (pid %lu) ===\n", GetCurrentProcessId());
        wiz3d::vk::g_veh = AddVectoredExceptionHandler(1, wiz3d::vk::FaultLogger);
    }
    else if (reason == DLL_PROCESS_DETACH)
    {
        if (wiz3d::vk::g_veh) { RemoveVectoredExceptionHandler(wiz3d::vk::g_veh); wiz3d::vk::g_veh = nullptr; }
        wiz3d::vk::Log("=== S3DWrapperVK detached ===\n");
        if (g_log != INVALID_HANDLE_VALUE) CloseHandle(g_log);
        g_log = INVALID_HANDLE_VALUE;
        DeleteCriticalSection(&g_logLock);
    }
    return TRUE;
}
