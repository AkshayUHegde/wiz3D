// wiz3D stereo bridge - exposed exports of S3DWrapperD3D9.dll for other
// in-process modules (specifically NvApiProxy) to read and write the active
// renderer's stereo state.
//
// All getters return wiz3D's authoritative values; all setters update them
// so that calls coming via NvAPI (3D Vision-aware games, HelixMod) are kept
// in sync with the wiz3D config / OSD slider.
//
// Resolved at runtime by NvApiProxy via GetModuleHandle("S3DWrapperD3D9.dll")
// + GetProcAddress. When the wrapper isn't loaded NvApiProxy falls back to
// its own local FakeStereoState so the proxy works standalone.

#pragma once

#include <d3d9.h>

extern "C"
{
	// Lifetime: called by CBaseStereoRenderer ctor / dtor so the bridge knows
	// which renderer to read state from. Internal to S3DWrapperD3D9; not for
	// external callers.
	void Wiz3D_RegisterActiveRenderer(class CBaseStereoRenderer* pRenderer);
	void Wiz3D_UnregisterActiveRenderer(class CBaseStereoRenderer* pRenderer);

	// Exports for NvApiProxy. All return 0 / false if no renderer is active.
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

	// Hotkey-equivalent step adjustments. NvAPI's Increase/Decrease entry
	// points come from the Ctrl+F3/F4/F5/F6 default 3D Vision keybinds —
	// route these to wiz3D's STEP_STEREOBASE / STEP_ONE_DIV_ZPS so they
	// match the magnitude of wiz3D's own Num+/- hotkeys. dir > 0 increases,
	// dir <= 0 decreases. NB: for convergence, "increase" pushes the
	// parallax plane FURTHER from viewer = SMALLER One_div_ZPS.
	__declspec(dllexport) void  __cdecl Wiz3D_StepSeparation(int dir);
	__declspec(dllexport) void  __cdecl Wiz3D_StepConvergence(int dir);

	// Profile-match probe: returns 1 if Base/Community/User profile matched
	// the running game, else 0. NvApiProxy uses this to decide whether to
	// accept game-side init Sets (no profile → accept as our default,
	// profile exists → ignore until user-initiated, detected via "did the
	// game call Get first?").
	__declspec(dllexport) int   __cdecl Wiz3D_HasProfileEntry();

	// NvAPI_D3D9_StretchRectEx reverse-blit bridge. Called by NvApiProxy's
	// Spoof_D3D9_StretchRectEx when the game toggled ReverseStereoBlitControl
	// ON; the wrapper is expected to pack both eyes into dst as SBS. Returns
	// E_NOTIMPL (0x80004001) to say "fall back to single-eye stretch" if the
	// full composition isn't wired yet, otherwise the HRESULT of the compose.
	__declspec(dllexport) HRESULT __cdecl Wiz3D_StretchRectExSBS(
		IDirect3DDevice9* pDevice,
		IDirect3DSurface9* pSrc, const RECT* pSrcRect,
		IDirect3DSurface9* pDst, const RECT* pDstRect,
		D3DTEXTUREFILTERTYPE filter);
}
