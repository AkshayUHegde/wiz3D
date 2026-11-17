/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// DLSS-SR at the NGX exports: each feature gets a right-eye twin with its own history, evaluated on the siblings.
// The upscale itself is opaque binary-import kernels, so it cannot be replayed per eye (S3DWrapper12/NGX12.cpp).

#include "Layer.h"
#include "../S3DAPI/GlobalData.h"
#include <MinHook.h>

namespace wiz3d { namespace vk {

// Generic 8-slot forwarders: x64 passes everything in registers/stack the same
// way, so forwarding extra UINT64s past the real arity is harmless.
typedef UINT64 (*PFN_Ngx8)(UINT64, UINT64, UINT64, UINT64, UINT64, UINT64, UINT64, UINT64);

// Two routes: the driver's _nvngx.dll and the DLSS snippet (nvngx_dlss.dll exports the whole VULKAN API), which
// Streamline calls directly, so the driver sees no CreateFeature (S3DWrapper12's HookNgxSnippets).
struct NgxRoute
{
    PFN_Ngx8 create = nullptr;     // NVSDK_NGX_VULKAN_CreateFeature(cmdbuf, feature, params, handle*)
    PFN_Ngx8 create1 = nullptr;    // NVSDK_NGX_VULKAN_CreateFeature1(device, cmdbuf, feature, params, handle*)
    PFN_Ngx8 evaluate = nullptr;   // NVSDK_NGX_VULKAN_EvaluateFeature(cmdbuf, handle, params, callback)
    PFN_Ngx8 release = nullptr;    // NVSDK_NGX_VULKAN_ReleaseFeature(handle)
};
static NgxRoute s_route[2];
static thread_local int t_ngxNest = 0;   // a route calling the other: the outer one speaks for both

struct ID3D11Resource;   // pointer-only vtable slots; the d3d headers stay out
struct ID3D12Resource;

// NVSDK_NGX_Parameter, declared exactly as the public nvsdk_ngx_params.h does:
// driver and wrapper both build this with MSVC, so the vtable layouts agree.
struct NgxParameter
{
    virtual void Set(const char*, unsigned long long) = 0;
    virtual void Set(const char*, float) = 0;
    virtual void Set(const char*, double) = 0;
    virtual void Set(const char*, unsigned int) = 0;
    virtual void Set(const char*, int) = 0;
    virtual void Set(const char*, ID3D11Resource*) = 0;
    virtual void Set(const char*, ID3D12Resource*) = 0;
    virtual void Set(const char*, void*) = 0;
    virtual int Get(const char*, unsigned long long*) const = 0;
    virtual int Get(const char*, float*) const = 0;
    virtual int Get(const char*, double*) const = 0;
    virtual int Get(const char*, unsigned int*) const = 0;
    virtual int Get(const char*, int*) const = 0;
    virtual int Get(const char*, ID3D11Resource**) const = 0;
    virtual int Get(const char*, ID3D12Resource**) const = 0;
    virtual int Get(const char*, void**) const = 0;
    virtual void Reset() = 0;
};

// NVSDK_NGX_Resource_VK (nvsdk_ngx_vk.h): what the void* param slots point at.
struct NgxResourceVK
{
    union
    {
        struct { VkImageView view; VkImage image; VkImageSubresourceRange range; VkFormat format; unsigned width, height; } img;
        struct { VkBuffer buffer; unsigned size; } buf;
    } u;
    int  type;        // 0 = image view, 1 = buffer
    bool readWrite;
};

// The DLSS-SR resource slots (nvsdk_ngx_defs.h key strings).
static const int kKeyCount = 7;
static const char* const kResKeys[kKeyCount] = {
    "Color", "Output", "Depth", "MotionVectors",
    "ExposureTexture", "TransparencyMask", "DLSS.Input.Bias.Current.Color.Mask",
};

// Left handle -> right twin, and every handle's feature id: DLSS-G (feature 11) evaluates through this same export
// with Streamline's parameter object, which must pass through untouched.
static SRWLOCK s_lock = SRWLOCK_INIT;
static void*   s_pairLeft[8];
static void*   s_pairRight[8];
static int     s_pairCount = 0;
static void*   s_featHandle[32];
static int     s_featId[32];
static int     s_featCount = 0;

static void NoteFeature(void* handle, int id)
{
    if (!handle) return;
    AcquireSRWLockExclusive(&s_lock);
    for (int i = 0; i < s_featCount; ++i)
        if (s_featHandle[i] == handle) { s_featId[i] = id; ReleaseSRWLockExclusive(&s_lock); return; }
    if (s_featCount == 32)
    {
        for (int i = 1; i < 32; ++i) { s_featHandle[i - 1] = s_featHandle[i]; s_featId[i - 1] = s_featId[i]; }
        --s_featCount;
    }
    s_featHandle[s_featCount] = handle; s_featId[s_featCount++] = id;
    ReleaseSRWLockExclusive(&s_lock);
}

static void ForgetFeature(void* handle)
{
    if (!handle) return;
    AcquireSRWLockExclusive(&s_lock);
    for (int i = 0; i < s_featCount; ++i)
        if (s_featHandle[i] == handle)
        { s_featHandle[i] = s_featHandle[s_featCount - 1]; s_featId[i] = s_featId[--s_featCount]; break; }
    ReleaseSRWLockExclusive(&s_lock);
}

static int FeatureIdFor(void* handle)
{
    int id = -1;
    AcquireSRWLockShared(&s_lock);
    for (int i = 0; i < s_featCount; ++i) if (s_featHandle[i] == handle) { id = s_featId[i]; break; }
    ReleaseSRWLockShared(&s_lock);
    return id;
}

static void* RightHandleFor(void* left)
{
    void* out = nullptr;
    AcquireSRWLockShared(&s_lock);
    for (int i = 0; i < s_pairCount; ++i) if (s_pairLeft[i] == left) { out = s_pairRight[i]; break; }
    ReleaseSRWLockShared(&s_lock);
    return out;
}

static void* TakePairFor(void* left)
{
    void* out = nullptr;
    AcquireSRWLockExclusive(&s_lock);
    for (int i = 0; i < s_pairCount; ++i)
        if (s_pairLeft[i] == left)
        {
            out = s_pairRight[i];
            s_pairLeft[i] = s_pairLeft[s_pairCount - 1];
            s_pairRight[i] = s_pairRight[--s_pairCount];
            break;
        }
    ReleaseSRWLockExclusive(&s_lock);
    return out;
}

template <int R>
static UINT64 Hooked_Create(UINT64 a0, UINT64 a1, UINT64 a2, UINT64 a3, UINT64 a4, UINT64 a5, UINT64 a6, UINT64 a7)
{
    if (t_ngxNest) return s_route[R].create(a0, a1, a2, a3, a4, a5, a6, a7);
    ++t_ngxNest; ++t_foreignCalls;
    const UINT64 r = s_route[R].create(a0, a1, a2, a3, a4, a5, a6, a7);
    Log("NGX VK CreateFeature (route %d): feature=%d -> 0x%08X handle=%p\n", R, (int)a1, (unsigned)r, a3 ? *(void**)a3 : nullptr);
    if (r == 0x1 && a3 && *(void**)a3) NoteFeature(*(void**)a3, (int)a1);
    if (r == 0x1 && a1 == 1 && a3 && *(void**)a3 && Vk().VKStereoDLSS)
    {
        // The twin's init work is eye-agnostic; created in the same cb.
        void* rightHandle = nullptr;
        const UINT64 rr = s_route[R].create(a0, a1, a2, (UINT64)&rightHandle, a4, a5, a6, a7);
        if (rr == 0x1 && rightHandle && s_pairCount < 8)
        {
            AcquireSRWLockExclusive(&s_lock);
            s_pairLeft[s_pairCount] = *(void**)a3;
            s_pairRight[s_pairCount++] = rightHandle;
            ReleaseSRWLockExclusive(&s_lock);
            NoteFeature(rightHandle, (int)a1);
        }
        Log("NGX VK right twin -> 0x%08X handle=%p (left %p)\n", (unsigned)rr, rightHandle, *(void**)a3);
    }
    --t_ngxNest; --t_foreignCalls;
    return r;
}

template <int R>
static UINT64 Hooked_Create1(UINT64 a0, UINT64 a1, UINT64 a2, UINT64 a3, UINT64 a4, UINT64 a5, UINT64 a6, UINT64 a7)
{
    if (t_ngxNest) return s_route[R].create1(a0, a1, a2, a3, a4, a5, a6, a7);
    ++t_ngxNest; ++t_foreignCalls;
    const UINT64 r = s_route[R].create1(a0, a1, a2, a3, a4, a5, a6, a7);
    Log("NGX VK CreateFeature1 (route %d): feature=%d -> 0x%08X handle=%p\n", R, (int)a2, (unsigned)r, a4 ? *(void**)a4 : nullptr);
    if (r == 0x1 && a4 && *(void**)a4) NoteFeature(*(void**)a4, (int)a2);
    if (r == 0x1 && a2 == 1 && a4 && *(void**)a4 && Vk().VKStereoDLSS)
    {
        void* rightHandle = nullptr;
        const UINT64 rr = s_route[R].create1(a0, a1, a2, a3, (UINT64)&rightHandle, a5, a6, a7);
        if (rr == 0x1 && rightHandle && s_pairCount < 8)
        {
            AcquireSRWLockExclusive(&s_lock);
            s_pairLeft[s_pairCount] = *(void**)a4;
            s_pairRight[s_pairCount++] = rightHandle;
            ReleaseSRWLockExclusive(&s_lock);
            NoteFeature(rightHandle, (int)a2);
        }
        Log("NGX VK right twin -> 0x%08X handle=%p (left %p)\n", (unsigned)rr, rightHandle, *(void**)a4);
    }
    --t_ngxNest; --t_foreignCalls;
    return r;
}

struct NgxScope { NgxScope() { ++t_ngxNest; ++t_foreignCalls; } ~NgxScope() { --t_ngxNest; --t_foreignCalls; } };

template <int R>
static UINT64 Hooked_Evaluate(UINT64 a0, UINT64 a1, UINT64 a2, UINT64 a3, UINT64 a4, UINT64 a5, UINT64 a6, UINT64 a7)
{
    if (t_ngxNest) return s_route[R].evaluate(a0, a1, a2, a3, a4, a5, a6, a7);
    NgxScope scope;
    const int fid = FeatureIdFor((void*)a1);
    if (fid >= 0 && fid != 1)
        return s_route[R].evaluate(a0, a1, a2, a3, a4, a5, a6, a7);

    const UINT64 r = s_route[R].evaluate(a0, a1, a2, a3, a4, a5, a6, a7);
    void* rightHandle = RightHandleFor((void*)a1);
    if (!rightHandle || r != 0x1) return r;
    DeviceData* d = FindDevice((const void*)a0);
    if (!d || !d->stereoEnabled) return r;

    // Re-aim every param that has a sibling; the Output must have one or the
    // right evaluate would stomp the left eye's upscale.
    auto* p = (NgxParameter*)a2;
    static NgxResourceVK s_right[kKeyCount];   // driver reads them during the call
    void* leftPtr[kKeyCount] = {};
    bool  present[kKeyCount] = {};
    bool  outputOk = false;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (int i = 0; i < kKeyCount; ++i)
        {
            void* v = nullptr;
            if (!p || p->Get(kResKeys[i], &v) != 0x1 || !v) continue;
            auto* res = (NgxResourceVK*)v;
            if (res->type != 0) continue;                      // image views only
            VkImage sibImg = VK_NULL_HANDLE;
            VkImageView sibView = VK_NULL_HANDLE;
            if (!ImageIsStereo(d, res->u.img.image, &sibImg)) continue;
            if (!ViewSibling(d, res->u.img.view, &sibView)) continue;
            s_right[i] = *res;
            s_right[i].u.img.image = sibImg;
            s_right[i].u.img.view = sibView;
            leftPtr[i] = v;
            present[i] = true;
            if (i == 1) outputOk = true;
        }
    }
    static volatile LONG s_n = 0;
    const LONG n = InterlockedIncrement(&s_n);
    if (!outputOk)
    {
        if (n <= 3 || n % 2000 == 0) Log("NGX VK evaluate #%ld: no Output sibling, right eye skipped\n", n);
        return r;
    }
    for (int i = 0; i < kKeyCount; ++i) if (present[i]) p->Set(kResKeys[i], (void*)&s_right[i]);
    const UINT64 rr = s_route[R].evaluate(a0, (UINT64)rightHandle, a2, a3, a4, a5, a6, a7);
    for (int i = 0; i < kKeyCount; ++i) if (present[i]) p->Set(kResKeys[i], leftPtr[i]);
    if (n <= 3 || n % 2000 == 0 || rr != 0x1)
        Log("NGX VK right evaluate #%ld -> 0x%08X (in %d/%d sibling'd)\n", n, (unsigned)rr,
            (int)(present[0] + present[2] + present[3]), 3);
    return r;
}

template <int R>
static UINT64 Hooked_Release(UINT64 a0, UINT64 a1, UINT64 a2, UINT64 a3, UINT64 a4, UINT64 a5, UINT64 a6, UINT64 a7)
{
    if (t_ngxNest) return s_route[R].release(a0, a1, a2, a3, a4, a5, a6, a7);
    NgxScope scope;
    void* rightHandle = TakePairFor((void*)a0);
    ForgetFeature((void*)a0);
    if (rightHandle) ForgetFeature(rightHandle);
    const UINT64 r = s_route[R].release(a0, a1, a2, a3, a4, a5, a6, a7);
    if (rightHandle) s_route[R].release((UINT64)rightHandle, a1, a2, a3, a4, a5, a6, a7);
    return r;
}

static bool HookExport(HMODULE m, const char* name, void* detour, PFN_Ngx8* orig)
{
    void* target = (void*)GetProcAddress(m, name);
    if (!target) { Log("  NGX VK %s: not exported\n", name); return false; }
    if (MH_CreateHook(target, detour, (void**)orig) != MH_OK || MH_EnableHook(target) != MH_OK)
    { Log("  NGX VK %s: hook failed\n", name); return false; }
    Log("  NGX VK %s @ %p\n", name, target);
    return true;
}

template <int R>
static void HookRoute(HMODULE m)
{
    wchar_t path[MAX_PATH] = L"?";
    GetModuleFileNameW(m, path, MAX_PATH);
    Log("NGX VK watcher: module loaded, hooking route %d (%ls)\n", R, path);
    if (MH_Initialize() != MH_OK && MH_Initialize() != MH_ERROR_ALREADY_INITIALIZED)
        Log("NGX VK watcher: MH_Initialize failed (continuing)\n");
    HookExport(m, "NVSDK_NGX_VULKAN_CreateFeature",   (void*)Hooked_Create<R>,   &s_route[R].create);
    HookExport(m, "NVSDK_NGX_VULKAN_CreateFeature1",  (void*)Hooked_Create1<R>,  &s_route[R].create1);
    HookExport(m, "NVSDK_NGX_VULKAN_EvaluateFeature", (void*)Hooked_Evaluate<R>, &s_route[R].evaluate);
    HookExport(m, "NVSDK_NGX_VULKAN_ReleaseFeature",  (void*)Hooked_Release<R>,  &s_route[R].release);
}

static DWORD WINAPI NgxWatchThread(void*)
{
    bool core = false, snippet = false;
    for (int i = 0; i < 2400 && !(core && snippet); ++i)
    {
        if (!core)
        {
            HMODULE m = GetModuleHandleW(L"_nvngx.dll");
            if (!m) m = GetModuleHandleW(L"nvngx.dll");
            if (m) { HookRoute<0>(m); core = true; }
        }
        if (!snippet)
        {
            HMODULE m = GetModuleHandleW(L"nvngx_dlss.dll");
            if (m) { HookRoute<1>(m); snippet = true; }
        }
        Sleep(core && snippet ? 0 : 250);
    }
    if (!core && !snippet) Log("NGX VK watcher: no nvngx module after the wait (DLSS off or not NVIDIA)\n");
    return 0;
}

void InstallNgxWatcherVK()
{
    if (!Vk().VKStereoDLSS) return;
    static volatile LONG s_once = 0;
    if (InterlockedCompareExchange(&s_once, 1, 0) != 0) return;
    HANDLE t = CreateThread(nullptr, 0, NgxWatchThread, nullptr, 0, nullptr);
    if (t) CloseHandle(t); else Log("NGX VK watcher: thread creation failed\n");
}

}} // namespace wiz3d::vk
