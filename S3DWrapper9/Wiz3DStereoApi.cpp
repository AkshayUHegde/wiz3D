#include "stdafx.h"
#include "Wiz3DStereoApi.h"
#include "BaseStereoRenderer.h"
#include "BaseSwapChain.h"         // m_pLeftBackBuffer / m_pRightBackBuffer / m_pPrimaryBackBuffer
#include "..\S3DAPI\GlobalData.h"
#include "..\S3DAPI\KeyboardHook.h" // gKbdHook.m_Access lock + extern decl
#include "..\S3DAPI\ReadData.h"     // WriteInputData for UserProfile.xml persistence

// Renderer-presence sentinel. We don't actually mutate the renderer's
// m_Input — that gets overwritten from gInfo.Input every Present (see
// BaseStereoRenderer_Main.cpp around line 305: `m_Input = updatedInput`,
// where updatedInput came from gInfo.Input). The * hotkey and our bridge
// must both write to gInfo.Input or their changes disappear on the next
// frame. This pointer just tells us the wrapper is initialised enough
// to honour state changes; before that, NvApi callers see static
// defaults so they don't bail out (HelixMod is unforgiving here).
static CBaseStereoRenderer* g_pActiveRenderer = nullptr;

extern "C"
{

void Wiz3D_RegisterActiveRenderer(CBaseStereoRenderer* pRenderer)
{
	g_pActiveRenderer = pRenderer;
}

void Wiz3D_UnregisterActiveRenderer(CBaseStereoRenderer* pRenderer)
{
	if (g_pActiveRenderer == pRenderer)
		g_pActiveRenderer = nullptr;
}

// Defaults returned before the wrapper is fully initialised. Chosen so
// 3D Vision-aware games / HelixMod don't bail out on a "stereo isn't
// usable" probe at startup:
//   - IsActivated = 1   : caller proceeds with stereo setup
//   - Separation  = 50% : neutral mid-slider value
//   - Convergence = 1.0 : 1 world-unit depth

int Wiz3D_GetStereoActive()
{
	if (!g_pActiveRenderer)
		return 1;
	return gInfo.Input.StereoActive ? 1 : 0;
}

void Wiz3D_SetStereoActive(int active)
{
	if (!g_pActiveRenderer)
		return; // wrapper not alive yet; caller will retry once it is
	CriticalSectionLocker locker(gKbdHook.m_Access);
	gInfo.Input.StereoActive = active ? true : false;
	WriteInputData(&gInfo.Input);
}

float Wiz3D_GetSeparationPercent()
{
	if (!g_pActiveRenderer)
		return 50.0f;
	const CameraPreset* p = gInfo.Input.GetActivePreset();
	return SEPARATION_TO_PERCENT(p->StereoBase);
}

void Wiz3D_SetSeparationPercent(float percent)
{
	if (!g_pActiveRenderer)
		return;
	if (percent < 0.0f) percent = 0.0f;
	if (percent > 100.0f) percent = 100.0f;
	CriticalSectionLocker locker(gKbdHook.m_Access);
	CameraPreset* p = gInfo.Input.GetActivePreset();
	p->StereoBase = PERCENT_TO_SEPARATION(percent);
	WriteInputData(&gInfo.Input);
}

float Wiz3D_GetConvergence()
{
	if (!g_pActiveRenderer)
		return 1.0f;
	const CameraPreset* p = gInfo.Input.GetActivePreset();
	if (p->One_div_ZPS == 0.0f)
		return 1.0f;
	return 1.0f / p->One_div_ZPS;
}

void Wiz3D_SetConvergence(float depth)
{
	if (!g_pActiveRenderer)
		return;
	if (depth <= 0.0f)
		return;
	CriticalSectionLocker locker(gKbdHook.m_Access);
	CameraPreset* p = gInfo.Input.GetActivePreset();
	p->One_div_ZPS = 1.0f / depth;
	WriteInputData(&gInfo.Input);
}

void Wiz3D_StepSeparation(int dir)
{
	if (!g_pActiveRenderer)
		return;
	CriticalSectionLocker locker(gKbdHook.m_Access);
	CameraPreset* p = gInfo.Input.GetActivePreset();
	float step = (dir > 0) ? STEP_STEREOBASE : -STEP_STEREOBASE;
	p->StereoBase += step;
	if (p->StereoBase < 0.0f) p->StereoBase = 0.0f;
	if (p->StereoBase > MAX_STEREOBASE) p->StereoBase = MAX_STEREOBASE;
	WriteInputData(&gInfo.Input);
}

void Wiz3D_StepConvergence(int dir)
{
	// "Increase convergence" pushes parallax plane FURTHER away (Z_conv up).
	// Since One_div_ZPS = 1/Z_conv, that maps to One_div_ZPS DECREASING.
	if (!g_pActiveRenderer)
		return;
	CriticalSectionLocker locker(gKbdHook.m_Access);
	CameraPreset* p = gInfo.Input.GetActivePreset();
	float step = (dir > 0) ? -STEP_ONE_DIV_ZPS : STEP_ONE_DIV_ZPS;
	p->One_div_ZPS += step;
	if (p->One_div_ZPS < MIN_ONE_DIV_ZPS) p->One_div_ZPS = MIN_ONE_DIV_ZPS;
	if (p->One_div_ZPS > MAX_ONE_DIV_ZPS) p->One_div_ZPS = MAX_ONE_DIV_ZPS;
	WriteInputData(&gInfo.Input);
}

int Wiz3D_HasProfileEntry()
{
	// gInfo flags are set during ReadProfilesAllParts when the running game's
	// exe matches an entry under <Profile><File Name="..."> in any of the
	// three profile files. No renderer required — these reflect static load
	// state from process startup, even if the renderer hasn't registered yet.
	return (gInfo.bMatchedInBase || gInfo.bMatchedInCommunity || gInfo.bMatchedInUser) ? 1 : 0;
}

// NvAPI_D3D9_StretchRectEx reverse-blit SBS bridge. Called by NvApiProxy's
// Spoof_D3D9_StretchRectEx when the game has toggled
// NvAPI_Stereo_ReverseStereoBlitControl ON. The game expects dst to come
// back as 2W x H packed side-by-side (left eye in the left half, right eye
// in the right half) -- that's how EGO engine (DIRT / GRID / Dirt Rally /
// F1) and Oil Rush read the right eye for depth reprojection / post-process.
//
// This completes the "wiz3D's StereoInjector as 3D Vision Automatic" chain:
//   1. NvApiProxy spoofs NVAPI_Stereo_* so the game thinks 3D Vision is live
//   2. The wiz3D DX9 StereoInjector duplicates draws and applies per-eye
//      projection shift, filling the active CBaseSwapChain's left + right
//      back-buffer surfaces with distinct content
//   3. When the game calls NvAPI_D3D9_StretchRectEx with reverse-blit ON,
//      NvApiProxy forwards to this bridge, which composes SBS from the pair
//
// The source resource the game passes is typically its own "stereo surface"
// which our wrapper has wrapped around the active swap-chain. The match
// here is heuristic: if `pSrc` is the active renderer's primary back buffer
// (or one of the stereo pair), we treat the request as a stereo readback.
// Otherwise E_NOTIMPL falls back to a plain StretchRect.
HRESULT Wiz3D_StretchRectExSBS(IDirect3DDevice9* pDevice,
                               IDirect3DSurface9* pSrc, const RECT* pSrcRect,
                               IDirect3DSurface9* pDst, const RECT* pDstRect,
                               D3DTEXTUREFILTERTYPE filter)
{
	if (!g_pActiveRenderer || !pDevice || !pSrc || !pDst) return E_NOTIMPL;

	// Locate the active CBaseSwapChain and its stereo pair. We only cover the
	// primary swap chain here -- a game using multiple stereo surfaces would
	// need more sophisticated matching, which no shipping EGO/Oil-Rush title
	// does.
	CBaseSwapChain* pSC = g_pActiveRenderer->GetBaseSC(0);
	if (!pSC) return E_NOTIMPL;
	IDirect3DSurface9* pLeft  = pSC->m_pLeftBackBuffer;
	IDirect3DSurface9* pRight = pSC->m_pRightBackBuffer;
	if (!pLeft || !pRight) return E_NOTIMPL;

	// Match the source. Accept any of: the primary BB, the raw left BB, or
	// the raw right BB -- a game that already has the stereo-surface handle
	// may pass any of these.
	if (pSrc != pLeft && pSrc != pRight && pSrc != pSC->m_pPrimaryBackBuffer)
		return E_NOTIMPL;

	// Compute dst halves. If pDstRect is null, dst covers the whole surface.
	D3DSURFACE_DESC dstDesc = {};
	if (FAILED(pDst->GetDesc(&dstDesc))) return E_NOTIMPL;
	RECT dstFull  = { 0, 0, (LONG)dstDesc.Width, (LONG)dstDesc.Height };
	const RECT* dstRectEff = pDstRect ? pDstRect : &dstFull;
	const LONG dstW = dstRectEff->right - dstRectEff->left;
	const LONG dstH = dstRectEff->bottom - dstRectEff->top;
	if (dstW <= 1 || dstH <= 0) return E_NOTIMPL;
	const LONG halfW = dstW / 2;
	RECT dstLeft  = { dstRectEff->left,         dstRectEff->top,
	                  dstRectEff->left + halfW, dstRectEff->bottom };
	RECT dstRight = { dstRectEff->left + halfW, dstRectEff->top,
	                  dstRectEff->right,        dstRectEff->bottom };

	// Issue the pair of StretchRects. If either fails, the dst may be in a
	// half-written state -- that's acceptable (the game sees garbage in one
	// half, same as the prior no-op) and better than tearing the whole frame.
	HRESULT hr1 = pDevice->StretchRect(pLeft,  pSrcRect, pDst, &dstLeft,  filter);
	HRESULT hr2 = pDevice->StretchRect(pRight, pSrcRect, pDst, &dstRight, filter);
	return (SUCCEEDED(hr1) && SUCCEEDED(hr2)) ? S_OK
	                                          : (FAILED(hr1) ? hr1 : hr2);
}

} // extern "C"
