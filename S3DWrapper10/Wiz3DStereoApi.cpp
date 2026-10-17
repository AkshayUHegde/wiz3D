/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*
* Mirror of S3DWrapper9/Wiz3DStereoApi.cpp. The DX11 wrapper reads and
* writes the same gInfo.Input the DX9 wrapper does, so the bridge logic
* is identical; the only difference is the renderer-presence sentinel
* type (Device11Proxy instead of CBaseStereoRenderer).
*/

#include "StdAfx.h"
#include "Wiz3DStereoApi.h"
#include "Device11Proxy.h"
#include "..\S3DAPI\GlobalData.h"
#include "..\S3DAPI\KeyboardHook.h"  // gKbdHook.m_Access lock + extern decl
#include "..\S3DAPI\ReadData.h"      // WriteInputData for UserProfile.xml persistence

// Renderer-presence sentinel. See the DX9 counterpart for why we don't
// mutate the renderer's state directly: wiz3D reads from gInfo.Input every
// frame, so writes have to land there. This pointer just tells us the
// wrapper is alive enough to honour state changes; before that, NvApi
// callers see static defaults so they don't bail out early (HelixMod and
// 3D Vision Auto Mode games are unforgiving about this).
static wiz3d::Device11Proxy* g_pActiveRenderer = nullptr;

extern "C"
{

void Wiz3D_RegisterActiveRenderer(wiz3d::Device11Proxy* pRenderer)
{
    g_pActiveRenderer = pRenderer;
}

void Wiz3D_UnregisterActiveRenderer(wiz3d::Device11Proxy* pRenderer)
{
    if (g_pActiveRenderer == pRenderer)
        g_pActiveRenderer = nullptr;
}

int Wiz3D_GetStereoActive()
{
    if (!g_pActiveRenderer)
        return 1;
    return gInfo.Input.StereoActive ? 1 : 0;
}

void Wiz3D_SetStereoActive(int active)
{
    if (!g_pActiveRenderer)
        return;  // wrapper not alive yet; caller will retry once it is
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
    if (percent < 0.0f)   percent = 0.0f;
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
    if (p->StereoBase < 0.0f)           p->StereoBase = 0.0f;
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
    // Reflects static load state from process startup; renderer registration
    // not required (profile flags are set during ReadProfilesAllParts well
    // before any Device creation).
    return (gInfo.bMatchedInBase || gInfo.bMatchedInCommunity || gInfo.bMatchedInUser) ? 1 : 0;
}

}  // extern "C"
