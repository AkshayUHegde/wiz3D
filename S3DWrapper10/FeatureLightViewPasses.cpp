/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <LightViewPasses Value="1"/>: a depth-only pass into a target not the presented size renders a light's view; its right
// copy reads the left eye's geometry constants, and registers only such passes read are never shifted.

#include "StdAfx.h"
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
    void OnTargetsSet(Context11Proxy& ctx) override
    {
        m_pass = false;
        Device11Proxy* dev = ctx.Parent();
        if (!ctx.LeftDepth() || !dev || !dev->GetPresentWidth()) return;
        for (UINT i = 0; i < ctx.NumTargets(); ++i) if (ctx.LeftTarget(i)) return;
        ID3D11Resource* r = nullptr;
        ctx.LeftDepth()->GetResource(&r);
        if (!r) return;
        D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
        r->GetType(&dim);
        D3D11_TEXTURE2D_DESC td = {};
        if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) static_cast<ID3D11Texture2D*>(r)->GetDesc(&td);
        r->Release();
        m_pass = td.Width && (td.Width != dev->GetPresentWidth() || td.Height != dev->GetPresentHeight());
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
            DDILog("  [frame %u] light-view passes (this frame): right-eye depth-only draws off the presented size, given the"
                   " left constants %u (present %ux%u)\n", frame, m_rightDraws, dev ? dev->GetPresentWidth() : 0u,
                   dev ? dev->GetPresentHeight() : 0u);
        }
        m_rightDraws = 0;
    }

    bool LightViewPass() const override { return m_pass; }

private:
    bool     m_pass = false;
    unsigned m_rightDraws = 0;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeLightViewPasses(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new LightViewPassesFeature());
}

} // namespace wiz3d
