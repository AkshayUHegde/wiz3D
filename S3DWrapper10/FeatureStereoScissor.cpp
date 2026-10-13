/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <StereoScissor Value="1"/>: a right scene draw's scissor rectangles widen by the parallax range (far field to the near
// plane). The game sizes them for the mono view (UE3 bounds each light's pass); the right image shifts past them.

#include "StdAfx.h"
#include <math.h>
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "AdapterFunctions.h"   // DDILog

namespace wiz3d
{
namespace
{

class StereoScissorFeature : public ProfileFeature
{
public:
    ~StereoScissorFeature() { if (m_widthView) m_widthView->Release(); }

    void OnRightDraw(Context11Proxy& ctx) override
    {
        m_saved = 0;
        if (ctx.OverlayPass() || ctx.LightViewPass()) return;
        UINT n = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
        ctx.Real()->RSGetScissorRects(&n, m_rects);
        const float p00 = stereo::SceneP00();
        const UINT w = n ? TargetWidth(ctx) : 0;
        if (!w || !(p00 > 0.f)) return;
        // The NDC shift b + m41 / w: b far away, b (1 - C / n) at the near plane n (where z = 0, or 1 reversed).
        const float b = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign() * p00;
        float nearShift = b, m22 = 0.f, m23 = 0.f;
        if (stereo::SceneZRow(m22, m23))
        {
            const float zn = m23 < 0.f ? -m23 / m22 : m23 / (1.f - m22);
            if (zn > 0.f && zn < 1e30f) nearShift = b * (1.f - stereo::Convergence() / zn);
        }
        const float half = 0.5f * (float)w;
        const LONG toRight = (LONG)ceilf((std::max)((std::max)(b, nearShift), 0.f) * half) + 1;
        const LONG toLeft = (LONG)ceilf((std::max)(-(std::min)(b, nearShift), 0.f) * half) + 1;
        D3D11_RECT wide[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
        for (UINT i = 0; i < n; ++i)
        {
            wide[i] = m_rects[i];
            wide[i].left = (std::max)(m_rects[i].left - toLeft, 0L);
            wide[i].right = m_rects[i].right + toRight;
        }
        ctx.Real()->RSSetScissorRects(n, wide);
        m_saved = n;
        ++m_widened;
    }

    // The left eye's next draw sees the game's rectangles again.
    void OnRightDrawDone(Context11Proxy& ctx) override
    {
        if (!m_saved) return;
        ctx.Real()->RSSetScissorRects(m_saved, m_rects);
        m_saved = 0;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] stereo scissor (this frame): right scene draws given wider scissor rects %u\n", frame,
                   m_widened);
        m_widened = 0;
    }

private:
    D3D11_RECT m_rects[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
    UINT       m_saved = 0;
    unsigned   m_widened = 0;
    ID3D11View* m_widthView = nullptr;   // the view m_width was read from (held, so its address is never reused)
    UINT        m_width = 0;

    // The bound target's width in pixels (the depth target's for a depth-only pass).
    UINT TargetWidth(Context11Proxy& ctx)
    {
        ID3D11View* v = ctx.LeftTarget(0);
        if (!v) v = ctx.LeftDepth();
        if (!v) return 0;
        if (v == m_widthView) return m_width;
        ID3D11Resource* res = nullptr;
        v->GetResource(&res);
        ID3D11Texture2D* tex = nullptr;
        if (res) { res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex); res->Release(); }
        D3D11_TEXTURE2D_DESC d = {};
        if (tex) { tex->GetDesc(&d); tex->Release(); }
        if (m_widthView) m_widthView->Release();
        m_widthView = v;
        m_widthView->AddRef();
        m_width = d.Width;
        return m_width;
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeStereoScissor(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new StereoScissorFeature());
}

} // namespace wiz3d
