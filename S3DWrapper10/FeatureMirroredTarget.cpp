/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <MirroredTarget Value="texture"/>: a planar reflection rendered x-flipped (sampled at u = 0.5 - 0.5*x/w); draws into it
// take the opposite eye shift. Targets are learned as the game recreates them, 16 at most.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <vector>

namespace wiz3d
{
namespace
{

class MirroredTargetFeature : public ProfileFeature
{
public:
    explicit MirroredTargetFeature(const ProfileEntry& e) : m_name(EntryValue(e, "Value")) {}

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        info.reflectTexSlot = rdef.TextureSlot(m_name);
    }

    void OnTargetsSet(Context11Proxy& ctx) override
    {
        m_inMirror = false;
        for (UINT i = 0; i < ctx.NumTargets() && !m_tex.empty() && !m_inMirror; ++i)
        {
            ID3D11RenderTargetView* rtv = ctx.LeftTarget(i);
            if (!rtv) continue;
            ID3D11Resource* res = nullptr;
            rtv->GetResource(&res);
            if (!res) continue;
            for (const Known& t : m_tex) if (t.res == res && Same(t, Desc(res))) m_inMirror = true;
            res->Release();
        }
    }

    void OnLeftDraw(Context11Proxy& ctx) override
    {
        Learn(ctx);
        if (m_inMirror) ++m_draws;
    }

    void OnFrameEnd(Context11Proxy& ctx, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] mirror (this frame): reflection targets known %u, draws into one %u, right-eye CB writes with"
                   " the shift flipped %u\n", frame, (unsigned)m_tex.size(), m_draws, ctx.FlippedWritesThisFrame());
        m_draws = 0;
    }

    float EyeShiftSign() const override { return m_inMirror ? -1.f : 1.f; }

private:
    // A reflection target: the real left texture, identity only (not AddRef'd), with its shape to reject a reused address.
    struct Known { void* res; UINT w, h; DXGI_FORMAT fmt; };
    static const size_t kMaxTargets = 16;
    std::string        m_name;
    std::vector<Known> m_tex;   // oldest first
    bool               m_inMirror = false;
    unsigned           m_draws = 0;

    static D3D11_TEXTURE2D_DESC Desc(ID3D11Resource* res)
    {
        D3D11_TEXTURE2D_DESC td = {};
        D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
        res->GetType(&dim);
        if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) static_cast<ID3D11Texture2D*>(res)->GetDesc(&td);
        return td;
    }
    static bool Same(const Known& k, const D3D11_TEXTURE2D_DESC& td)
    {
        return td.Width == k.w && td.Height == k.h && td.Format == k.fmt;
    }

    // The texture behind the bound pixel shader's reflection slot is a reflection target; learned by identity.
    void Learn(Context11Proxy& ctx)
    {
        const ShaderAnalysis11Result* ps = ctx.BoundShader(true);
        if (!ps || ps->reflectTexSlot < 0 || ps->reflectTexSlot >= (int)Context11Proxy::kMaxSRVs) return;
        ID3D11ShaderResourceView* srv = ctx.LeftSRV(true, (UINT)ps->reflectTexSlot);
        if (!srv) return;
        ID3D11Resource* res = nullptr;
        srv->GetResource(&res);
        if (!res) return;
        const D3D11_TEXTURE2D_DESC td = Desc(res);
        res->Release();
        for (size_t i = 0; i < m_tex.size(); ++i)
            if (m_tex[i].res == res)
            {
                if (Same(m_tex[i], td)) return;
                m_tex.erase(m_tex.begin() + i);   // the address now holds another texture
                break;
            }
        if (m_tex.size() >= kMaxTargets) m_tex.erase(m_tex.begin());
        m_tex.push_back({ (void*)res, td.Width, td.Height, td.Format });
        FeatureLog("  mirror: reflection target learned tex=%p %ux%u fmt=%d bind=0x%X (PS t%d, right SRV %s), %u known\n",
                   (void*)res, td.Width, td.Height, (int)td.Format, td.BindFlags, ps->reflectTexSlot,
                   ctx.RightSRV(true, (UINT)ps->reflectTexSlot) != srv ? "per eye" : "SHARED -- mono reflection",
                   (unsigned)m_tex.size());
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeMirroredTarget(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new MirroredTargetFeature(*e));
}

} // namespace wiz3d
