/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*/

#include "StdAfx.h"
#include "SRDisplayInfo.h"
#include <windows.h>

namespace wiz3d
{

// LeiaSR SDK C-API types. Both are opaque void* in the C header, so we
// forward-declare them here and skip pulling in the SDK headers.
typedef void* SRContext_t;
typedef void* SRDisplay_t;

typedef SRContext_t  (*PFN_newSRContext)        (void);
typedef void         (*PFN_initializeSRContext) (SRContext_t);
typedef void         (*PFN_deleteSRContext)     (SRContext_t);
typedef SRDisplay_t  (*PFN_createDisplay)       (SRContext_t);
typedef int          (*PFN_getRecommendedWidth) (SRDisplay_t);
typedef int          (*PFN_getRecommendedHeight)(SRDisplay_t);

static bool                     s_resolved             = false;
static bool                     s_available            = false;
static PFN_newSRContext         s_newSRContext         = nullptr;
static PFN_initializeSRContext  s_initializeSRContext  = nullptr;
static PFN_deleteSRContext      s_deleteSRContext      = nullptr;
static PFN_createDisplay        s_createDisplay        = nullptr;
static PFN_getRecommendedWidth  s_getRecommendedWidth  = nullptr;
static PFN_getRecommendedHeight s_getRecommendedHeight = nullptr;

// Resolve the SR runtime function pointers on first call. Idempotent and
// cheap after the first call (two bool checks). Only entered from the
// game's Present thread via SwapChain11Proxy, so thread-safety isn't a
// concern for the sentinel.
static void ResolveOnce()
{
    if (s_resolved) return;
    s_resolved = true;

    HMODULE hCore     = LoadLibraryW(L"SimulatedRealityCore.dll");
    HMODULE hDisplays = LoadLibraryW(L"SimulatedRealityDisplays.dll");
    if (!hCore || !hDisplays) return;

    s_newSRContext         = reinterpret_cast<PFN_newSRContext>        (GetProcAddress(hCore,     "newSRContext"));
    s_initializeSRContext  = reinterpret_cast<PFN_initializeSRContext> (GetProcAddress(hCore,     "initializeSRContext"));
    s_deleteSRContext      = reinterpret_cast<PFN_deleteSRContext>     (GetProcAddress(hCore,     "deleteSRContext"));
    s_createDisplay        = reinterpret_cast<PFN_createDisplay>       (GetProcAddress(hDisplays, "createDisplay"));
    s_getRecommendedWidth  = reinterpret_cast<PFN_getRecommendedWidth> (GetProcAddress(hDisplays, "getRecommendedViewsTextureWidth"));
    s_getRecommendedHeight = reinterpret_cast<PFN_getRecommendedHeight>(GetProcAddress(hDisplays, "getRecommendedViewsTextureHeight"));

    s_available = s_newSRContext && s_initializeSRContext && s_deleteSRContext &&
                  s_createDisplay && s_getRecommendedWidth && s_getRecommendedHeight;
}

bool QueryRecommendedSRViewsSize(int& outWidthSBS, int& outHeight)
{
    ResolveOnce();
    if (!s_available) return false;

    SRContext_t ctx = s_newSRContext();
    if (!ctx) return false;
    s_initializeSRContext(ctx);

    SRDisplay_t display = s_createDisplay(ctx);
    if (!display)
    {
        s_deleteSRContext(ctx);
        return false;
    }

    const int w = s_getRecommendedWidth(display);
    const int h = s_getRecommendedHeight(display);
    s_deleteSRContext(ctx);

    if (w <= 0 || h <= 0) return false;
    outWidthSBS = w;
    outHeight   = h;
    return true;
}

}  // namespace wiz3d
