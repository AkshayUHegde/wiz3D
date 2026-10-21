/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <LightViewPasses Value="1" ColorSize="WxH"/>: a depth-only pass into a target not the presented size renders a light's
// view; its right copy reads the left eye's geometry constants, and registers only such passes read are never shifted.
// ColorSize: so does any pass into a colour target of that size, another camera's picture shown on a panel (NieR's call
// portrait, its minimap).

#include "StdAfx.h"
#include <math.h>
#include <stdio.h>
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Device11Proxy.h"
#include "AdapterFunctions.h"   // DDILog

namespace wiz3d
{
namespace
{

class LightViewPassesFeature : public ProfileFeature
{
public:
    LightViewPassesFeature(UINT colorW, UINT colorH) : m_colorW(colorW), m_colorH(colorH) {}

    void OnTargetsSet(Context11Proxy& ctx) override
    {
        m_pass = false;
        Device11Proxy* dev = ctx.Parent();
        if (!dev || !dev->GetPresentWidth()) return;
        // ColorSize: with or without depth (a minimap's cards bind none).
        if (m_colorW && ctx.NumTargets() && ctx.LeftTarget(0))
        {
            ID3D11Resource* c = nullptr;
            ctx.LeftTarget(0)->GetResource(&c);
            if (!c) return;
            D3D11_RESOURCE_DIMENSION cdim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
            c->GetType(&cdim);
            D3D11_TEXTURE2D_DESC cd = {};
            if (cdim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) static_cast<ID3D11Texture2D*>(c)->GetDesc(&cd);
            c->Release();
            m_pass = cd.Width == m_colorW && cd.Height == m_colorH;
            return;
        }
        if (!ctx.LeftDepth()) return;
        for (UINT i = 0; i < ctx.NumTargets(); ++i) if (ctx.LeftTarget(i)) return;
        ID3D11Resource* r = nullptr;
        ctx.LeftDepth()->GetResource(&r);
        if (!r) return;
        D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
        r->GetType(&dim);
        D3D11_TEXTURE2D_DESC td = {};
        if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) static_cast<ID3D11Texture2D*>(r)->GetDesc(&td);
        r->Release();
        // Nor the presented aspect: a half-size depth pre-pass is a view of the scene (Metro's 1920x1080 depth under
        // a 3840x2160 present, which its sun quad is depth-tested against).
        const bool sameAspect = td.Height && dev->GetPresentHeight() &&
            fabsf((float)td.Width / td.Height - (float)dev->GetPresentWidth() / dev->GetPresentHeight()) < 0.01f;
        m_pass = td.Width && !sameAspect && (td.Width != dev->GetPresentWidth() || td.Height != dev->GetPresentHeight());
    }

    void OnRightDraw(Context11Proxy& ctx) override
    {
        if (!m_pass) return;
        ctx.BindLeftGeometryConstants();
        ++m_rightDraws;
    }

    void OnFrameEnd(Context11Proxy& ctx, unsigned frame, bool report) override
    {
        if (report)
        {
            Device11Proxy* dev = ctx.Parent();
            DDILog("  [frame %u] light-view passes (this frame): right-eye depth-only draws off the presented size%s, given the"
                   " left constants %u (present %ux%u)\n", frame, m_colorW ? " or into the colour size" : "", m_rightDraws,
                   dev ? dev->GetPresentWidth() : 0u, dev ? dev->GetPresentHeight() : 0u);
        }
        m_rightDraws = 0;
    }

    bool LightViewPass() const override { return m_pass; }

private:
    UINT     m_colorW, m_colorH;   // ColorSize, 0 = none
    bool     m_pass = false;
    unsigned m_rightDraws = 0;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeLightViewPasses(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    unsigned w = 0, h = 0;
    if (sscanf_s(EntryValue(*e, "ColorSize").c_str(), "%ux%u", &w, &h) != 2) w = h = 0;
    return std::unique_ptr<ProfileFeature>(new LightViewPassesFeature(w, h));
}

} // namespace wiz3d
