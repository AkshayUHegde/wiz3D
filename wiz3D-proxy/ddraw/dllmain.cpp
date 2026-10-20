/* wiz3D - ddraw.dll Proxy Loader
 *
 * Drop this ddraw.dll into a game's folder alongside S3DWrapperD3D7.dll
 * and the support DLLs (plus the full DX9 stereo stack).
 *
 * The proxy loads the iZ3D DX7/DDraw wrapper (S3DWrapperD3D7.dll) and
 * routes DirectDrawCreate / DirectDrawCreateEx through it. The wrapper
 * wraps the real DirectDraw objects with stereo-capable COM wrappers
 * and uses D3D9 internally for stereo rendering.
 *
 * IMPORTANT: Because the wrapper links ddraw.lib, its internal calls
 * to DirectDrawCreate resolve back to THIS proxy (since we ARE ddraw.dll
 * in the process). A reentry guard prevents infinite recursion — when
 * the wrapper calls DirectDrawCreate, the proxy forwards it directly
 * to the real system ddraw.dll.
 *
 * All other 20 ddraw exports are forwarded to the real system ddraw.dll
 * via naked jmp thunks (Win32 only — DX7 games are always 32-bit).
 */

#define WIN32_LEAN_AND_MEAN
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include "../proxy_version.h"
#include <stdio.h>
#include <psapi.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#pragma comment(lib, "psapi.lib")

// ---------------------------------------------------------------------------
// Simple diagnostic log — writes to wiz3D_ddraw_proxy.log in the proxy dir
// ---------------------------------------------------------------------------
static FILE* g_logFile = NULL;

static void LogOpen(void)
{
    if (g_logFile) return;
    WCHAR dir[MAX_PATH];
    GetModuleFileNameW(NULL, dir, MAX_PATH);
    WCHAR* pSlash = wcsrchr(dir, L'\\');
    if (pSlash) *(pSlash + 1) = L'\0';
    lstrcatW(dir, L"wiz3D_Logs");
    CreateDirectoryW(dir, NULL);
    lstrcatW(dir, L"\\wiz3D_ddraw_proxy.log");
    g_logFile = _wfopen(dir, L"a");  // append: shared with other proxies
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
// Vectored exception handler — logs fatal exceptions for crash diagnostics
// ---------------------------------------------------------------------------
static PVOID g_hVEH = NULL;
static volatile LONG g_crashLogged = 0;

static LONG CALLBACK VectoredCrashHandler(EXCEPTION_POINTERS* pExInfo)
{
    DWORD code = pExInfo->ExceptionRecord->ExceptionCode;

    switch (code)
    {
    // PRIV_INSTRUCTION / ILLEGAL_INSTRUCTION excluded — see d3d9 dllmain for rationale
    case EXCEPTION_ACCESS_VIOLATION:
    case EXCEPTION_STACK_OVERFLOW:
    case EXCEPTION_INT_DIVIDE_BY_ZERO:
    case EXCEPTION_IN_PAGE_ERROR:
    case EXCEPTION_ARRAY_BOUNDS_EXCEEDED:
    case EXCEPTION_FLT_DIVIDE_BY_ZERO:
    case EXCEPTION_FLT_INVALID_OPERATION:
        break;
    default:
        return EXCEPTION_CONTINUE_SEARCH;
    }

    if (InterlockedCompareExchange(&g_crashLogged, 1, 0) != 0)
        return EXCEPTION_CONTINUE_SEARCH;

    Log("\n!!! FATAL EXCEPTION (VEH) !!!\n");
    Log("Exception code: 0x%08lX\n", code);
    void* crashAddr = pExInfo->ExceptionRecord->ExceptionAddress;
    Log("Crash address:  %p\n", crashAddr);

    HMODULE hMod = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)crashAddr, &hMod))
    {
        WCHAR modName[MAX_PATH];
        GetModuleFileNameW(hMod, modName, MAX_PATH);
        BYTE* base = (BYTE*)hMod;
        DWORD_PTR offset = (BYTE*)crashAddr - base;
        Log("Faulting module: %ls + 0x%IX\n", modName, offset);
    }
    else
    {
        Log("Faulting module: UNKNOWN\n");
    }

    if (code == EXCEPTION_ACCESS_VIOLATION &&
        pExInfo->ExceptionRecord->NumberParameters >= 2)
    {
        ULONG_PTR accessType = pExInfo->ExceptionRecord->ExceptionInformation[0];
        const char* op = "UNKNOWN";
        if (accessType == 0) op = "READ";
        else if (accessType == 1) op = "WRITE";
        else if (accessType == 8) op = "DEP (execute non-executable memory)";
        Log("Access violation: %s of address %p\n", op,
            (void*)pExInfo->ExceptionRecord->ExceptionInformation[1]);
    }

    CONTEXT* ctx = pExInfo->ContextRecord;
    Log("EAX=%08lX EBX=%08lX ECX=%08lX EDX=%08lX\n", ctx->Eax, ctx->Ebx, ctx->Ecx, ctx->Edx);
    Log("ESI=%08lX EDI=%08lX EBP=%08lX ESP=%08lX\n", ctx->Esi, ctx->Edi, ctx->Ebp, ctx->Esp);
    Log("EIP=%08lX\n", ctx->Eip);

    Log("--- Stack trace ---\n");
    {
        void* stack[64];
        USHORT frames = CaptureStackBackTrace(0, 64, stack, NULL);
        for (USHORT i = 0; i < frames; i++)
        {
            HMODULE hFrameMod = NULL;
            if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   (LPCSTR)stack[i], &hFrameMod))
            {
                WCHAR modName[MAX_PATH];
                GetModuleFileNameW(hFrameMod, modName, MAX_PATH);
                WCHAR* pSlash2 = wcsrchr(modName, L'\\');
                DWORD_PTR offset = (BYTE*)stack[i] - (BYTE*)hFrameMod;
                Log("  [%2u] %p  %ls+0x%IX\n", i, stack[i],
                    pSlash2 ? pSlash2 + 1 : modName, offset);
            }
            else
            {
                Log("  [%2u] %p  (unknown)\n", i, stack[i]);
            }
        }
    }
    Log("--- End stack trace ---\n");

    Log("--- Loaded modules ---\n");
    HMODULE hMods[256];
    DWORD cbNeeded;
    if (EnumProcessModules(GetCurrentProcess(), hMods, sizeof(hMods), &cbNeeded))
    {
        DWORD count = cbNeeded / sizeof(HMODULE);
        for (DWORD i = 0; i < count && i < 256; i++)
        {
            WCHAR name[MAX_PATH];
            MODULEINFO mi;
            if (GetModuleFileNameW(hMods[i], name, MAX_PATH) &&
                GetModuleInformation(GetCurrentProcess(), hMods[i], &mi, sizeof(mi)))
            {
                Log("  %p-%p  %ls\n", mi.lpBaseOfDll,
                    (BYTE*)mi.lpBaseOfDll + mi.SizeOfImage, name);
            }
        }
    }
    Log("--- End modules ---\n");

    {
        WCHAR dumpPath[MAX_PATH];
        GetModuleFileNameW(NULL, dumpPath, MAX_PATH);
        WCHAR* pSlash3 = wcsrchr(dumpPath, L'\\');
        if (pSlash3) *(pSlash3 + 1) = L'\0';
        lstrcatW(dumpPath, L"wiz3D_ddraw_crash.dmp");

        HANDLE hFile = CreateFileW(dumpPath, GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, 0, NULL);
        if (hFile != INVALID_HANDLE_VALUE)
        {
            MINIDUMP_EXCEPTION_INFORMATION mei;
            mei.ThreadId = GetCurrentThreadId();
            mei.ExceptionPointers = pExInfo;
            mei.ClientPointers = FALSE;
            MiniDumpWriteDump(GetCurrentProcess(), GetCurrentProcessId(),
                              hFile,
                              (MINIDUMP_TYPE)(MiniDumpWithDataSegs |
                                              MiniDumpWithHandleData |
                                              MiniDumpWithThreadInfo |
                                              MiniDumpWithFullMemoryInfo),
                              &mei, NULL, NULL);
            CloseHandle(hFile);
            Log("Minidump written to: %ls\n", dumpPath);
        }
    }

    Log("=== CRASH END ===\n");
    if (g_logFile) fflush(g_logFile);
    return EXCEPTION_CONTINUE_SEARCH;
}

// ---------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------
static HMODULE g_hRealDDraw = NULL;   // real system ddraw.dll
static HMODULE g_hWrapper   = NULL;   // S3DWrapperD3D7.dll
static HMODULE g_hProxy     = NULL;   // our own HMODULE
static BOOL    g_bWrapperActive = FALSE;

// ---------------------------------------------------------------------------
// Original function addresses from real ddraw.dll (for pass-through)
// ---------------------------------------------------------------------------
static FARPROC g_pOrig_AcquireDDThreadLock         = NULL;
static FARPROC g_pOrig_CompleteCreateSysmemSurface = NULL;
static FARPROC g_pOrig_D3DParseUnknownCommand      = NULL;
static FARPROC g_pOrig_DDGetAttachedSurfaceLcl     = NULL;
static FARPROC g_pOrig_DDInternalLock              = NULL;
static FARPROC g_pOrig_DDInternalUnlock            = NULL;
static FARPROC g_pOrig_DSoundHelp                  = NULL;
static FARPROC g_pOrig_DirectDrawCreate            = NULL;
static FARPROC g_pOrig_DirectDrawCreateClipper     = NULL;
static FARPROC g_pOrig_DirectDrawCreateEx          = NULL;
static FARPROC g_pOrig_DirectDrawEnumerateA        = NULL;
static FARPROC g_pOrig_DirectDrawEnumerateExA      = NULL;
static FARPROC g_pOrig_DirectDrawEnumerateExW      = NULL;
static FARPROC g_pOrig_DirectDrawEnumerateW        = NULL;
static FARPROC g_pOrig_DllCanUnloadNow             = NULL;
static FARPROC g_pOrig_DllGetClassObject           = NULL;
static FARPROC g_pOrig_GetDDSurfaceLocal           = NULL;
static FARPROC g_pOrig_GetOLEThunkData             = NULL;
static FARPROC g_pOrig_GetSurfaceFromDC            = NULL;
static FARPROC g_pOrig_RegisterSpecialCase         = NULL;
static FARPROC g_pOrig_ReleaseDDThreadLock         = NULL;
static FARPROC g_pOrig_SetAppCompatData            = NULL;

// ---------------------------------------------------------------------------
// Wrapper function pointers
// ---------------------------------------------------------------------------
typedef HRESULT (WINAPI *pfnDirectDrawCreate)(void*, void**, void*);
typedef HRESULT (WINAPI *pfnDirectDrawCreateEx)(void*, void**, const void*, void*);
typedef DWORD   (WINAPI *pfnInitializeExchangeServer)(void);

static pfnDirectDrawCreate   g_pfnWrapCreate   = NULL;
static pfnDirectDrawCreateEx g_pfnWrapCreateEx  = NULL;

// Reentry guard: prevents infinite loop when wrapper calls DirectDrawCreate
// back through our proxy
static volatile LONG g_inWrapperCall = 0;

// ---------------------------------------------------------------------------
// Get the directory containing this proxy DLL
// ---------------------------------------------------------------------------
static void GetProxyDirectory(WCHAR* dir, DWORD maxLen)
{
    GetModuleFileNameW(g_hProxy, dir, maxLen);
    WCHAR* pSlash = wcsrchr(dir, L'\\');
    if (pSlash)
        *(pSlash + 1) = L'\0';
}

// ---------------------------------------------------------------------------
// Load real ddraw.dll from System32 (for pass-through functions)
// ---------------------------------------------------------------------------
static BOOL LoadRealDDraw(void)
{
    if (g_hRealDDraw)
        return TRUE;

    WCHAR sysDir[MAX_PATH];
    GetSystemDirectoryW(sysDir, MAX_PATH);
    lstrcatW(sysDir, L"\\ddraw.dll");

    g_hRealDDraw = LoadLibraryW(sysDir);
    if (!g_hRealDDraw)
    {
        Log("FAIL: Could not load real ddraw.dll from %ls (error %lu)\n", sysDir, GetLastError());
        return FALSE;
    }
    Log("OK: Real ddraw.dll loaded from %ls\n", sysDir);

    // Resolve all 22 exports
    g_pOrig_AcquireDDThreadLock         = GetProcAddress(g_hRealDDraw, "AcquireDDThreadLock");
    g_pOrig_CompleteCreateSysmemSurface = GetProcAddress(g_hRealDDraw, "CompleteCreateSysmemSurface");
    g_pOrig_D3DParseUnknownCommand      = GetProcAddress(g_hRealDDraw, "D3DParseUnknownCommand");
    g_pOrig_DDGetAttachedSurfaceLcl     = GetProcAddress(g_hRealDDraw, "DDGetAttachedSurfaceLcl");
    g_pOrig_DDInternalLock              = GetProcAddress(g_hRealDDraw, "DDInternalLock");
    g_pOrig_DDInternalUnlock            = GetProcAddress(g_hRealDDraw, "DDInternalUnlock");
    g_pOrig_DSoundHelp                  = GetProcAddress(g_hRealDDraw, "DSoundHelp");
    g_pOrig_DirectDrawCreate            = GetProcAddress(g_hRealDDraw, "DirectDrawCreate");
    g_pOrig_DirectDrawCreateClipper     = GetProcAddress(g_hRealDDraw, "DirectDrawCreateClipper");
    g_pOrig_DirectDrawCreateEx          = GetProcAddress(g_hRealDDraw, "DirectDrawCreateEx");
    g_pOrig_DirectDrawEnumerateA        = GetProcAddress(g_hRealDDraw, "DirectDrawEnumerateA");
    g_pOrig_DirectDrawEnumerateExA      = GetProcAddress(g_hRealDDraw, "DirectDrawEnumerateExA");
    g_pOrig_DirectDrawEnumerateExW      = GetProcAddress(g_hRealDDraw, "DirectDrawEnumerateExW");
    g_pOrig_DirectDrawEnumerateW        = GetProcAddress(g_hRealDDraw, "DirectDrawEnumerateW");
    g_pOrig_DllCanUnloadNow             = GetProcAddress(g_hRealDDraw, "DllCanUnloadNow");
    g_pOrig_DllGetClassObject           = GetProcAddress(g_hRealDDraw, "DllGetClassObject");
    g_pOrig_GetDDSurfaceLocal           = GetProcAddress(g_hRealDDraw, "GetDDSurfaceLocal");
    g_pOrig_GetOLEThunkData             = GetProcAddress(g_hRealDDraw, "GetOLEThunkData");
    g_pOrig_GetSurfaceFromDC            = GetProcAddress(g_hRealDDraw, "GetSurfaceFromDC");
    g_pOrig_RegisterSpecialCase         = GetProcAddress(g_hRealDDraw, "RegisterSpecialCase");
    g_pOrig_ReleaseDDThreadLock         = GetProcAddress(g_hRealDDraw, "ReleaseDDThreadLock");
    g_pOrig_SetAppCompatData            = GetProcAddress(g_hRealDDraw, "SetAppCompatData");

    Log("Resolved %d of 22 ddraw exports\n",
        (g_pOrig_AcquireDDThreadLock ? 1 : 0) + (g_pOrig_CompleteCreateSysmemSurface ? 1 : 0) +
        (g_pOrig_D3DParseUnknownCommand ? 1 : 0) + (g_pOrig_DDGetAttachedSurfaceLcl ? 1 : 0) +
        (g_pOrig_DDInternalLock ? 1 : 0) + (g_pOrig_DDInternalUnlock ? 1 : 0) +
        (g_pOrig_DSoundHelp ? 1 : 0) + (g_pOrig_DirectDrawCreate ? 1 : 0) +
        (g_pOrig_DirectDrawCreateClipper ? 1 : 0) + (g_pOrig_DirectDrawCreateEx ? 1 : 0) +
        (g_pOrig_DirectDrawEnumerateA ? 1 : 0) + (g_pOrig_DirectDrawEnumerateExA ? 1 : 0) +
        (g_pOrig_DirectDrawEnumerateExW ? 1 : 0) + (g_pOrig_DirectDrawEnumerateW ? 1 : 0) +
        (g_pOrig_DllCanUnloadNow ? 1 : 0) + (g_pOrig_DllGetClassObject ? 1 : 0) +
        (g_pOrig_GetDDSurfaceLocal ? 1 : 0) + (g_pOrig_GetOLEThunkData ? 1 : 0) +
        (g_pOrig_GetSurfaceFromDC ? 1 : 0) + (g_pOrig_RegisterSpecialCase ? 1 : 0) +
        (g_pOrig_ReleaseDDThreadLock ? 1 : 0) + (g_pOrig_SetAppCompatData ? 1 : 0));

    return TRUE;
}

// ---------------------------------------------------------------------------
// Load S3DWrapperD3D7.dll and run the InitializeExchangeServer handshake
// ---------------------------------------------------------------------------
static void LoadWrapper(void)
{
    if (g_bWrapperActive)
        return;
    g_bWrapperActive = TRUE;

    WCHAR wrapPath[MAX_PATH];
    GetProxyDirectory(wrapPath, MAX_PATH);
    lstrcatW(wrapPath, L"S3DWrapperD3D7.dll");

    Log("Loading wrapper: %ls\n", wrapPath);

    HMODULE hTest = LoadLibraryExW(wrapPath, NULL, DONT_RESOLVE_DLL_REFERENCES);
    if (hTest)
    {
        Log("OK: Wrapper PE is loadable (DONT_RESOLVE_DLL_REFERENCES)\n");
        FreeLibrary(hTest);
    }
    else
    {
        Log("FAIL: Wrapper PE not loadable even without imports (error %lu)\n", GetLastError());
    }

    // Pre-check known local dependencies
    {
        WCHAR depPath[MAX_PATH];
        static const WCHAR* localDeps[] = {
            L"ZLOg.dll", NULL
        };
        static const WCHAR* systemDeps[] = {
            L"PSAPI.DLL", L"d3dx9_43.dll", L"d3d9.dll", NULL
        };

        Log("--- Checking local dependencies ---\n");
        for (int i = 0; localDeps[i]; i++)
        {
            // Probing log helps distinguish a hang inside LoadLibrary's DllMain
            // from external process termination — without it, a deadlocked dep
            // produces an indistinguishable truncated log.
            Log("  Probing %ls...\n", localDeps[i]);
            GetProxyDirectory(depPath, MAX_PATH);
            lstrcatW(depPath, localDeps[i]);

            DWORD attrs = GetFileAttributesW(depPath);
            if (attrs == INVALID_FILE_ATTRIBUTES)
            {
                Log("    %ls: FILE NOT FOUND\n", localDeps[i]);
                continue;
            }

            HMODULE hDep = LoadLibraryW(depPath);
            if (hDep)
            {
                Log("    %ls: OK (loaded)\n", localDeps[i]);
                FreeLibrary(hDep);
            }
            else
            {
                Log("    %ls: FAIL (error %lu)\n", localDeps[i], GetLastError());
            }
        }

        Log("--- Checking system dependencies ---\n");
        for (int i = 0; systemDeps[i]; i++)
        {
            HMODULE hDep = LoadLibraryW(systemDeps[i]);
            if (hDep)
            {
                Log("  %ls: OK\n", systemDeps[i]);
                FreeLibrary(hDep);
            }
            else
            {
                Log("  %ls: FAIL (error %lu)\n", systemDeps[i], GetLastError());
            }
        }
        Log("--- End dependency check ---\n");
    }

    // Add the proxy's directory to the DLL search path
    {
        WCHAR proxyDir[MAX_PATH];
        GetProxyDirectory(proxyDir, MAX_PATH);
        size_t len = wcslen(proxyDir);
        if (len > 0 && proxyDir[len - 1] == L'\\')
            proxyDir[len - 1] = L'\0';
        SetDllDirectoryW(proxyDir);
        Log("SetDllDirectory: %ls\n", proxyDir);
    }

    g_hWrapper = LoadLibraryW(wrapPath);

    SetDllDirectoryW(NULL);

    if (!g_hWrapper)
    {
        Log("FAIL: Could not load wrapper (error %lu)\n", GetLastError());
        return;
    }
    Log("OK: Wrapper loaded\n");

    pfnInitializeExchangeServer pGetRouter =
        (pfnInitializeExchangeServer)GetProcAddress(g_hWrapper, "InitializeExchangeServer");
    if (pGetRouter)
    {
        Log("Calling InitializeExchangeServer...\n");
        DWORD routerType = pGetRouter();
        Log("InitializeExchangeServer returned routerType=%lu\n", routerType);
        if (routerType <= 1)
        {
            g_pfnWrapCreate   = (pfnDirectDrawCreate)  GetProcAddress(g_hWrapper, "DirectDrawCreate");
            g_pfnWrapCreateEx = (pfnDirectDrawCreateEx) GetProcAddress(g_hWrapper, "DirectDrawCreateEx");
            Log("Wrapper exports: DirectDrawCreate=%p  DirectDrawCreateEx=%p\n",
                g_pfnWrapCreate, g_pfnWrapCreateEx);
        }
        else
        {
            Log("WARN: routerType=%lu > 1, not using wrapper functions\n", routerType);
        }
    }
    else
    {
        Log("FAIL: InitializeExchangeServer not found in wrapper\n");
    }

    // Diagnostic: log per-source profile match flags (base/community/user).
    // DX7 (ddraw) wrapper has its own minimal local gInfo, so we query
    // S3DWrapperD3D9 which has S3DAPI. See wiz3D-proxy/d3d8/dllmain.cpp
    // for the same rationale.
    HMODULE hWrap9 = LoadLibraryW(L"S3DWrapperD3D9.dll");
    if (hWrap9)
    {
        typedef void (__stdcall *pfnGetProfile)(char*, size_t, int*, int*, int*);
        pfnGetProfile pGetProfile =
            (pfnGetProfile)GetProcAddress(hWrap9, "wiz3D_GetActiveProfileInfo");
        if (pGetProfile)
        {
            char profileName[260] = {};
            int matchedBase = 0, matchedCommunity = 0, matchedUser = 0;
            pGetProfile(profileName, sizeof(profileName), &matchedBase, &matchedCommunity, &matchedUser);
            int anyMatched = matchedBase || matchedCommunity || matchedUser;
            Log("ProfileLoad: ProfileName='%s' base=%d community=%d user=%d\n",
                anyMatched ? profileName : "",
                matchedBase, matchedCommunity, matchedUser);
        }
        // No FreeLibrary — DDraw wrapper will pull it in via Direct3DCreate9.
    }
    else
    {
        Log("ProfileLoad: S3DWrapperD3D9.dll not loadable — DX7 profile state unknown\n");
    }

    if (!g_pfnWrapCreate && !g_pfnWrapCreateEx)
    {
        Log("WARN: No wrapper DDraw functions — falling through to real DLL\n");
        FreeLibrary(g_hWrapper);
        g_hWrapper = NULL;
    }

    // Post-load file checks
    {
        WCHAR chkPath[MAX_PATH];
        // Config: prefer wiz3D_Config.xml, accept legacy Config.xml.
        GetProxyDirectory(chkPath, MAX_PATH);
        lstrcatW(chkPath, L"wiz3D_Config.xml");
        DWORD attrNew = GetFileAttributesW(chkPath);
        GetProxyDirectory(chkPath, MAX_PATH);
        lstrcatW(chkPath, L"Config.xml");
        DWORD attr = GetFileAttributesW(chkPath);
        Log("--- Post-load file check ---\n");
        Log("  wiz3D_Config.xml: %s\n", (attrNew != INVALID_FILE_ATTRIBUTES) ? "FOUND" : "missing");
        Log("  Config.xml (legacy): %s\n", (attr != INVALID_FILE_ATTRIBUTES) ? "FOUND" : "missing");
        if (attrNew == INVALID_FILE_ATTRIBUTES && attr == INVALID_FILE_ATTRIBUTES)
            Log("  WARNING: no config file found\n");
        Log("--- End post-load check ---\n");
    }
}

// ===========================================================================
// Exported functions — DirectDrawCreate and DirectDrawCreateEx
// (with reentry guard to prevent infinite loop with wrapper)
// ===========================================================================

extern "C" HRESULT WINAPI DirectDrawCreate(void* lpGUID, void** lplpDD, void* pUnkOuter)
{
    Log("DirectDrawCreate(lpGUID=%p, lplpDD=%p, pUnkOuter=%p) called\n", lpGUID, lplpDD, pUnkOuter);

    LoadRealDDraw();

    // Reentry check: if the wrapper is calling us, forward to real ddraw
    if (g_inWrapperCall)
    {
        Log("  -> Reentry detected, forwarding to real ddraw.dll\n");
        if (g_pOrig_DirectDrawCreate)
            return ((pfnDirectDrawCreate)g_pOrig_DirectDrawCreate)(lpGUID, lplpDD, pUnkOuter);
        return E_FAIL;
    }

    LoadWrapper();

    if (g_pfnWrapCreate)
    {
        Log("  -> Routing through WRAPPER DirectDrawCreate...\n");
        InterlockedExchange(&g_inWrapperCall, 1);
        HRESULT hr = g_pfnWrapCreate(lpGUID, lplpDD, pUnkOuter);
        InterlockedExchange(&g_inWrapperCall, 0);
        Log("  -> Wrapper returned 0x%08lX, *lplpDD=%p\n", hr, lplpDD ? *lplpDD : NULL);
        return hr;
    }

    Log("  -> Fallback to real ddraw.dll\n");
    if (g_pOrig_DirectDrawCreate)
        return ((pfnDirectDrawCreate)g_pOrig_DirectDrawCreate)(lpGUID, lplpDD, pUnkOuter);
    return E_FAIL;
}

extern "C" HRESULT WINAPI DirectDrawCreateEx(void* lpGUID, void** lplpDD, const void* iid, void* pUnkOuter)
{
    Log("DirectDrawCreateEx(lpGUID=%p, lplpDD=%p, iid=%p, pUnkOuter=%p) called\n",
        lpGUID, lplpDD, iid, pUnkOuter);

    LoadRealDDraw();

    if (g_inWrapperCall)
    {
        Log("  -> Reentry detected, forwarding to real ddraw.dll\n");
        if (g_pOrig_DirectDrawCreateEx)
            return ((pfnDirectDrawCreateEx)g_pOrig_DirectDrawCreateEx)(lpGUID, lplpDD, iid, pUnkOuter);
        return E_FAIL;
    }

    LoadWrapper();

    if (g_pfnWrapCreateEx)
    {
        Log("  -> Routing through WRAPPER DirectDrawCreateEx...\n");
        InterlockedExchange(&g_inWrapperCall, 1);
        HRESULT hr = g_pfnWrapCreateEx(lpGUID, lplpDD, iid, pUnkOuter);
        InterlockedExchange(&g_inWrapperCall, 0);
        Log("  -> Wrapper returned 0x%08lX, *lplpDD=%p\n", hr, lplpDD ? *lplpDD : NULL);
        return hr;
    }

    Log("  -> Fallback to real ddraw.dll\n");
    if (g_pOrig_DirectDrawCreateEx)
        return ((pfnDirectDrawCreateEx)g_pOrig_DirectDrawCreateEx)(lpGUID, lplpDD, iid, pUnkOuter);
    return E_FAIL;
}

// ===========================================================================
// Pass-through exports — naked jmp thunks (Win32 x86 only)
// These preserve the entire stack/registers regardless of calling convention
// or parameter count, by jumping directly to the real ddraw function.
// ===========================================================================

#define NAKED_THUNK(exportName, pOrigVar) \
    extern "C" __declspec(naked) void exportName() \
    { \
        __asm mov eax, dword ptr [pOrigVar] \
        __asm test eax, eax \
        __asm jz _skip_##exportName \
        __asm jmp eax \
        __asm _skip_##exportName: \
        __asm ret \
    }

NAKED_THUNK(AcquireDDThreadLock,         g_pOrig_AcquireDDThreadLock)
NAKED_THUNK(CompleteCreateSysmemSurface, g_pOrig_CompleteCreateSysmemSurface)
NAKED_THUNK(D3DParseUnknownCommand,      g_pOrig_D3DParseUnknownCommand)
NAKED_THUNK(DDGetAttachedSurfaceLcl,     g_pOrig_DDGetAttachedSurfaceLcl)
NAKED_THUNK(DDInternalLock,              g_pOrig_DDInternalLock)
NAKED_THUNK(DDInternalUnlock,            g_pOrig_DDInternalUnlock)
NAKED_THUNK(DSoundHelp,                  g_pOrig_DSoundHelp)
NAKED_THUNK(DirectDrawCreateClipper,     g_pOrig_DirectDrawCreateClipper)
NAKED_THUNK(DirectDrawEnumerateA,        g_pOrig_DirectDrawEnumerateA)
NAKED_THUNK(DirectDrawEnumerateExA,      g_pOrig_DirectDrawEnumerateExA)
NAKED_THUNK(DirectDrawEnumerateExW,      g_pOrig_DirectDrawEnumerateExW)
NAKED_THUNK(DirectDrawEnumerateW,        g_pOrig_DirectDrawEnumerateW)
NAKED_THUNK(DllCanUnloadNow,            g_pOrig_DllCanUnloadNow)
NAKED_THUNK(DllGetClassObject,           g_pOrig_DllGetClassObject)
NAKED_THUNK(GetDDSurfaceLocal,           g_pOrig_GetDDSurfaceLocal)
NAKED_THUNK(GetOLEThunkData,             g_pOrig_GetOLEThunkData)
NAKED_THUNK(GetSurfaceFromDC,            g_pOrig_GetSurfaceFromDC)
NAKED_THUNK(RegisterSpecialCase,         g_pOrig_RegisterSpecialCase)
NAKED_THUNK(ReleaseDDThreadLock,         g_pOrig_ReleaseDDThreadLock)
NAKED_THUNK(SetAppCompatData,            g_pOrig_SetAppCompatData)

// ---------------------------------------------------------------------------
// DllMain
// ---------------------------------------------------------------------------
BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved)
{
    switch (reason)
    {
    case DLL_PROCESS_ATTACH:
        g_hProxy = hModule;
        DisableThreadLibraryCalls(hModule);
        LogOpen();
        g_hVEH = AddVectoredExceptionHandler(1, VectoredCrashHandler);
        {
            WCHAR exePath[MAX_PATH];
            GetModuleFileNameW(NULL, exePath, MAX_PATH);
            Log("=== wiz3D " DISPLAYED_VERSION " - ddraw proxy loaded ===\n");
            Log("Game exe: %ls\n", exePath);
            WCHAR proxyPath[MAX_PATH];
            GetModuleFileNameW(hModule, proxyPath, MAX_PATH);
            Log("Proxy DLL: %ls\n", proxyPath);
        }
        // Load real ddraw.dll early so all thunks have valid addresses
        LoadRealDDraw();
        break;

    case DLL_PROCESS_DETACH:
        Log("=== wiz3D ddraw proxy unloading ===\n");
        if (g_hVEH)
        {
            RemoveVectoredExceptionHandler(g_hVEH);
            g_hVEH = NULL;
        }
        if (g_hWrapper)
        {
            FreeLibrary(g_hWrapper);
            g_hWrapper = NULL;
        }
        if (g_hRealDDraw)
        {
            FreeLibrary(g_hRealDDraw);
            g_hRealDDraw = NULL;
        }
        if (g_logFile)
        {
            fclose(g_logFile);
            g_logFile = NULL;
        }
        break;
    }
    return TRUE;
}
