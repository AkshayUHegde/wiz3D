// wiz3D stereo bridge - exposed exports of S3DWrapperD3D10.dll for other
// in-process modules (specifically NvApiProxy) to read and write the active
// renderer's stereo state. Mirrors S3DWrapperD3D9.dll's export surface
// 1:1 (same names, same semantics) so NvApiProxy can probe either DLL
// depending on which wrapper the running game loaded.
//
// All getters return wiz3D's authoritative gInfo.Input values; all setters
// update them so NVAPI calls from 3D Vision Auto Mode games (GTA V, Max
// Payne 3, BioShock Infinite's 3D settings, etc.) are kept in sync with
// the wiz3D config / OSD slider and the hotkey thread. gInfo.Input is
// shared across all wiz3D wrappers via S3DAPI, so the DX11 implementation
// of these entry points is identical in logic to the DX9 one.
//
// Resolved at runtime by NvApiProxy via GetModuleHandle("S3DWrapperD3D10.dll")
// + GetProcAddress; see NvApiProxy.cpp for the per-DLL probe chain.

#pragma once

namespace wiz3d { class Device11Proxy; }

extern "C"
{
    // Lifetime: called by Device11Proxy ctor / dtor so the bridge knows
    // the wrapper is initialised enough to honour state changes. Internal
    // to S3DWrapperD3D10; not for external callers.
    void Wiz3D_RegisterActiveRenderer(wiz3d::Device11Proxy* pRenderer);
    void Wiz3D_UnregisterActiveRenderer(wiz3d::Device11Proxy* pRenderer);

    // Exports for NvApiProxy. Defaults returned before the wrapper is
    // fully initialised (IsActivated=1, Separation=50%, Convergence=1.0)
    // chosen so 3D Vision-aware games / HelixMod don't bail out on an
    // early "stereo isn't usable" probe at startup.
    __declspec(dllexport) int   __cdecl Wiz3D_GetStereoActive();
    __declspec(dllexport) void  __cdecl Wiz3D_SetStereoActive(int active);

    // Separation expressed as a percent (0..100) to match NvAPI's convention.
    __declspec(dllexport) float __cdecl Wiz3D_GetSeparationPercent();
    __declspec(dllexport) void  __cdecl Wiz3D_SetSeparationPercent(float percent);

    // Convergence in world-depth units (matches NvAPI semantics: depth at
    // which both eyes converge, i.e. objects appear at the screen plane).
    // Internally wiz3D stores 1/depth (One_div_ZPS); the bridge inverts.
    __declspec(dllexport) float __cdecl Wiz3D_GetConvergence();
    __declspec(dllexport) void  __cdecl Wiz3D_SetConvergence(float depth);

    // Hotkey-equivalent step adjustments routed to STEP_STEREOBASE /
    // STEP_ONE_DIV_ZPS so NvAPI Increase/Decrease matches the magnitude
    // of wiz3D's own Num+/- hotkeys. dir > 0 increases, dir <= 0 decreases.
    // For convergence, "increase" pushes the parallax plane FURTHER from
    // the viewer = SMALLER One_div_ZPS.
    __declspec(dllexport) void  __cdecl Wiz3D_StepSeparation(int dir);
    __declspec(dllexport) void  __cdecl Wiz3D_StepConvergence(int dir);

    // Profile-match probe: returns 1 if Base/Community/User profile matched
    // the running game, else 0. NvApiProxy uses this to decide whether to
    // accept game-side init Sets (no profile = accept as our default,
    // profile exists = ignore until user-initiated).
    __declspec(dllexport) int   __cdecl Wiz3D_HasProfileEntry();
}
