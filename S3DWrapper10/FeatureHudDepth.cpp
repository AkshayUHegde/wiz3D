/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <HudDepth Value="depthVariable" Rect="rectVariable"/>: HUD elements tagged with a 3D Vision depth move by b * (1 - C/depth);
// the untagged sprite over the screen centre is the reticle and takes the frame's tagged depth (Max Payne 3).

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Buffer11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog

namespace wiz3d
{
namespace
{

class HudDepthFeature : public ProfileFeature
{
public:
    explicit HudDepthFeature(const ProfileEntry& e) : m_depthVar(EntryValue(e, "Value")), m_rectVar(EntryValue(e, "Rect")) {}

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        rdef.Variable(m_depthVar, true, info.apparentDepthCB, info.apparentDepthOffset);
        rdef.Variable(m_rectVar, true, info.transPosCB, info.transPosOffset);
        if (info.apparentDepthCB >= 0)
            FeatureLog("  HudDepth: CRC=0x%08lX %s at cb%d+%u\n", info.crc32, m_depthVar.c_str(), info.apparentDepthCB,
                       info.apparentDepthOffset);
    }

    // An element at depth > 0 moves in the right eye by our shift at that depth; 0 or less stays at screen depth.
    void OnRightDraw(Context11Proxy& ctx) override
    {
        const ShaderAnalysis11Result* vs = ctx.BoundShader(false);
        if (!vs || vs->apparentDepthCB < 0 || vs->apparentDepthCB >= (int)Context11Proxy::kMaxCBs) return;
        Buffer11Proxy* cb = ctx.BoundCB(false, (UINT)vs->apparentDepthCB);
        if (!cb || cb->Shadow().size() < vs->apparentDepthOffset + 4) return;
        const unsigned char* sh = cb->Shadow().data();
        float depth = 0.f;
        memcpy(&depth, sh + vs->apparentDepthOffset, 4);
        if (!m_draws++) m_depthMin = m_depthMax = depth;
        else { if (depth < m_depthMin) m_depthMin = depth; if (depth > m_depthMax) m_depthMax = depth; }
        const bool tagged = depth > 0.f && depth < 1e7f;
        if (tagged) m_reticleDepthFrame = depth;
        // A small sprite over the screen centre, for either corner convention ([0,1]: zw..zw+xy, [-1,1]: zw+-xy).
        float tp[4] = {};
        bool centered = false;
        if (vs->transPosCB == vs->apparentDepthCB && cb->Shadow().size() >= vs->transPosOffset + 16)
        {
            memcpy(tp, sh + vs->transPosOffset, 16);
            const bool unitCorners = fminf(tp[2], tp[2] + tp[0]) <= 0.f && fmaxf(tp[2], tp[2] + tp[0]) >= 0.f &&
                                     fminf(tp[3], tp[3] + tp[1]) <= 0.f && fmaxf(tp[3], tp[3] + tp[1]) >= 0.f;
            const bool signedCorners = fabsf(tp[2]) <= fabsf(tp[0]) && fabsf(tp[3]) <= fabsf(tp[1]);
            centered = (unitCorners || signedCorners) && fabsf(tp[0]) < 0.25f && fabsf(tp[1]) < 0.45f;
        }
        if (centered) ++m_centered;
        if (gInfo.CollectDebugInformation && m_listLen < sizeof m_list - 1)
        {
            const int w = _snprintf_s(m_list + m_listLen, sizeof m_list - m_listLen, _TRUNCATE,
                                      " %.1f@[%.3f %.3f %.3f %.3f]%s", depth, tp[0], tp[1], tp[2], tp[3], centered ? "*" : "");
            if (w > 0) m_listLen += (size_t)w;
        }
        // Its own depth if the game tagged it, else the reticle's target depth when it sits over the centre.
        const float use = tagged ? depth : (centered ? m_reticleDepth : 0.f);
        if (!(use > 0.f)) return;
        const float eyeShift = wiz3D_GetEffectiveEyeShift(), p00 = ctx.CameraP00();
        const float b = p00 > 0.f ? eyeShift * p00 : stereo::FallbackB(eyeShift);
        const float dNdc = b * (1.f - stereo::Convergence() / use);
        UINT n = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
        if (ctx.Recording()) n = ctx.GameViewports(m_vp);   // BatchRightEye: the recorder reads nothing back
        else ctx.Real()->RSGetViewports(&n, m_vp);
        if (!n) return;
        D3D11_VIEWPORT vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE];
        for (UINT i = 0; i < n; ++i) { vp[i] = m_vp[i]; vp[i].TopLeftX += dNdc * vp[i].Width * 0.5f; }
        ctx.Real()->RSSetViewports(n, vp);
        m_vpCount = n;
        m_shifted = true;
        m_lastPx = dNdc * vp[0].Width * 0.5f;
        ++m_moved;
    }

    void OnRightDrawDone(Context11Proxy& ctx) override
    {
        if (m_shifted) { ctx.Real()->RSSetViewports(m_vpCount, m_vp); m_shifted = false; }
    }

    // A Present that drew the HUD publishes its reticle depth (0 = nothing targeted); others keep the last one.
    void OnFrameEnd(Context11Proxy& ctx, unsigned frame, bool report) override
    {
        if (m_draws)
        {
            m_reticleDepth = m_reticleDepthFrame;
            memcpy(m_listLast, m_list, sizeof m_listLast);
        }
        if (report)
            DDILog("  [frame %u] HUD depth (this frame): right-eye HUD draws %u, moved %u, over the centre %u,"
                   " depth %.3f..%.3f, reticle depth %.2f, last move %.1f px, camera p00 %.4f | last HUD frame, depth@[rect]:%s\n",
                   frame, m_draws, m_moved, m_centered, m_depthMin, m_depthMax, m_reticleDepth, m_lastPx, ctx.CameraP00(), m_listLast);
        m_draws = m_moved = m_centered = 0;
        m_reticleDepthFrame = 0.f;
        m_listLen = 0;
        m_list[0] = 0;
    }

private:
    std::string    m_depthVar, m_rectVar;
    D3D11_VIEWPORT m_vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE] = {};
    UINT           m_vpCount = 0;
    bool           m_shifted = false;
    unsigned       m_draws = 0, m_moved = 0, m_centered = 0;
    float          m_depthMin = 0.f, m_depthMax = 0.f, m_lastPx = 0.f;
    // The depth the game gave its one tagged element (the target under the reticle): published, and this frame's.
    float          m_reticleDepth = 0.f, m_reticleDepthFrame = 0.f;
    char           m_list[640] = {}, m_listLast[640] = {};
    size_t         m_listLen = 0;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeHudDepth(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new HudDepthFeature(*e));
}

} // namespace wiz3d
