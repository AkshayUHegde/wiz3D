/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <ClipSpaceRays Value="1"/>: a draw passing NDC (w = 1) and a world ray per vertex through (Frostbite's sky quad, its
// panorama looked up by the ray) gets, for the right eye, the ray moved by the eye offset at infinity.

#include "StdAfx.h"
#include <stdlib.h>
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog

namespace wiz3d
{
namespace
{

class ClipSpaceRaysFeature : public ProfileFeature
{
public:
    void OnRightDraw(Context11Proxy& ctx) override
    {
        m_bound = ctx.BindClipSpaceRayShift();
    }

    void OnRightDrawDone(Context11Proxy& ctx) override
    {
        if (m_bound) ctx.UnbindClipSpaceShift();
        m_bound = false;
    }

    void OnFrameEnd(Context11Proxy& ctx, unsigned frame, bool report) override
    {
        const unsigned n = ctx.ClipSpaceRayCount();
        if (report) DDILog("  [frame %u] clip-space rays: right draws with the ray moved to infinity %u\n", frame, n);
    }

private:
    bool m_bound = false;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeClipSpaceRays(const ProfileEntry* e)
{
    if (!e || atoi(EntryValue(*e, "Value").c_str()) <= 0) return nullptr;
    return std::unique_ptr<ProfileFeature>(new ClipSpaceRaysFeature());
}

} // namespace wiz3d
