/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <DeferredLightCamera Buffer InvView ViewProjection Pass Rays/>: deferred lights rebuild position as InvView row 3 + ray *
// depth; in passes binding Pass, the right eye's copies hold the right camera (origin moved, rays from its pixels).

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Device11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <cmath>
#include <mutex>
#include <set>
#include <sstream>

namespace wiz3d
{
namespace
{

const char* const kKey = "deferredlightcamera";
enum { kInvView, kViewProj, kRay0, kRay1, kRay2, kCount };

std::mutex         s_logLock;
std::set<unsigned> s_logged;   // shader CRCs already logged

float Length3(const float* v) { return sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

class DeferredLightCameraFeature : public ProfileFeature
{
public:
    explicit DeferredLightCameraFeature(const ProfileEntry& e)
        : m_buffer(EntryValue(e, "Buffer")), m_invView(EntryValue(e, "InvView")), m_viewProj(EntryValue(e, "ViewProjection")),
          m_pass(EntryValue(e, "Pass"))
    {
        std::istringstream in(EntryValue(e, "Rays"));
        for (int i = 0; i < 3 && (in >> m_rays[i]); ++i) {}
    }
    bool Valid() const { return !m_buffer.empty() && !m_invView.empty() && !m_pass.empty(); }

    // A shader binding Pass draws a deferred light. Fields are found as declared: the first stage edited writes the copy
    // both stages read.
    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        const int passSlot = rdef.CBufferSlot(m_pass);
        if (passSlot < 0) return;
        const int camSlot = rdef.CBufferSlot(m_buffer);
        auto find = [&](const std::string& n, int slot, bool usedOnly) -> int
        {
            if (n.empty() || slot < 0) return -1;
            for (const auto& v : rdef.Variables(n, usedOnly)) if (v.first == slot) return (int)v.second;
            return -1;
        };
        std::vector<int> off(kCount, -1);
        off[kInvView] = find(m_invView, camSlot, false);
        off[kViewProj] = find(m_viewProj, camSlot, false);
        for (int i = 0; i < 3; ++i) off[kRay0 + i] = find(m_rays[i], passSlot, false);
        const bool camera = off[kInvView] >= 0 && (find(m_invView, camSlot, true) >= 0 || find(m_viewProj, camSlot, true) >= 0);
        const bool rays = off[kRay0] >= 0 && off[kRay1] >= 0 && off[kRay2] >= 0 && find(m_rays[0], passSlot, true) >= 0;
        if (!camera && !rays) return;
        if (camera) info.NoteRightCB(m_buffer, camSlot);
        if (rays) info.NoteRightCB(m_pass, passSlot);
        info.tags.push_back(kKey);
        info.featureOffsets.emplace_back(kKey, off);
        std::lock_guard<std::mutex> g(s_logLock);
        if (s_logged.insert(info.crc32).second)
            FeatureLog("  DeferredLightCamera: CRC=0x%08lX %s (cb%d) InvView +%d ViewProjection +%d | %s (cb%d) rays %s\n",
                       info.crc32, m_buffer.c_str(), camSlot, camera ? off[kInvView] : -1, camera ? off[kViewProj] : -1,
                       m_pass.c_str(), passSlot, rays ? "read" : "not read");
    }

    bool MovedOriginPass(Context11Proxy& ctx) const override { return ctx.BoundShadersTagged(kKey); }

    bool EditRightConstants(Context11Proxy& ctx, const std::string& cbuffer, unsigned char* data, size_t size) override
    {
        const ShaderAnalysis11Result* shader = ctx.EditingShader();
        if (!ctx.MovedOrigin() || !shader || (cbuffer != m_buffer && cbuffer != m_pass)) return false;
        const std::vector<int>* off = nullptr;
        for (const auto& f : shader->featureOffsets) if (f.first == kKey) off = &f.second;
        if (!off) return false;
        const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign();
        if (eyeShift == 0.f) return false;
        const float d = eyeShift * stereo::Convergence();
        auto at = [&](int i, size_t n) -> float*
        {
            const int o = (*off)[i];
            return (o >= 0 && (size_t)o + n <= size) ? (float*)(data + o) : nullptr;
        };

        if (cbuffer == m_buffer)
        {
            // Row-major: InvView row 0 is the right axis and row 3 the position; the moved copy starts from the game's bytes,
            // so a camera's view-projection takes the core's shift here (x column += b * w column, x translation -= p00*d).
            float* iv = at(kInvView, 64);
            if (!iv) return false;
            const float rn = Length3(iv);
            if (!(rn > 0.9f && rn < 1.1f)) { ++m_notCamera; return false; }
            const float R[3] = { iv[0] / rn, iv[1] / rn, iv[2] / rn };
            if (float* vp = at(kViewProj, 64))
            {
                const float x[3] = { vp[0], vp[4], vp[8] }, y[3] = { vp[1], vp[5], vp[9] }, w[3] = { vp[3], vp[7], vp[11] };
                const float wn = Length3(w);
                const float p00 = wn > 1e-6f ? Length3(x) / wn : 0.f, p11 = wn > 1e-6f ? Length3(y) / wn : 0.f;
                const float along = p00 > 0.f ? (x[0] * R[0] + x[1] * R[1] + x[2] * R[2]) / (p00 * wn) : 0.f;
                if (p00 > 0.05f && p00 < 20.f && fabsf(along - 1.f) < 0.02f && AtAspect(ctx, p00, p11))
                {
                    const float b = eyeShift * p00;
                    for (int r = 0; r < 4; ++r) vp[r * 4] += b * vp[r * 4 + 3];
                    vp[12] -= p00 * d;
                    ++m_shifted;
                }
                else ++m_unshifted;
            }
            for (int k = 0; k < 3; ++k) iv[12 + k] += d * R[k];
            ++m_moved;
            return true;
        }

        // Rays: rows of ray = M * (ndc x, ndc y, 1), column 0 = right / p00, column 2 the unit forward; the right pixel's
        // ray is the left's at ndc x - b, so column 2 -= b * column 0.
        float* r[3] = { at(kRay0, 12), at(kRay1, 12), at(kRay2, 12) };
        if (!r[0] || !r[1] || !r[2]) return false;
        const float c0[3] = { r[0][0], r[1][0], r[2][0] }, c2[3] = { r[0][2], r[1][2], r[2][2] };
        const float n0 = Length3(c0), n2 = Length3(c2);
        if (!(n0 > 1e-4f) || fabsf(n2 - 1.f) > 0.05f || fabsf(c0[0] * c2[0] + c0[1] * c2[1] + c0[2] * c2[2]) > 0.05f * n0)
        {
            ++m_raysOdd;
            return false;
        }
        const float b = eyeShift / n0;
        for (int i = 0; i < 3; ++i) r[i][2] -= b * r[i][0];
        ++m_raysMoved;
        return true;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] deferred light camera (this frame): origins moved %u (view-projections shifted %u, not a"
                   " camera's %u), no right axis %u | ray bases moved %u, not ray-shaped %u\n", frame, m_moved, m_shifted,
                   m_unshifted, m_notCamera, m_raysMoved, m_raysOdd);
        m_moved = m_shifted = m_unshifted = m_notCamera = m_raysMoved = m_raysOdd = 0;
    }

private:
    std::string m_buffer, m_invView, m_viewProj, m_pass, m_rays[3];
    unsigned    m_moved = 0, m_shifted = 0, m_unshifted = 0, m_notCamera = 0, m_raysMoved = 0, m_raysOdd = 0;

    // p11 / p00 is the presented aspect within 3%: the scene camera's projection.
    static bool AtAspect(Context11Proxy& ctx, float p00, float p11)
    {
        Device11Proxy* dev = ctx.Parent();
        const float present = (dev && dev->GetPresentHeight()) ? (float)dev->GetPresentWidth() / dev->GetPresentHeight() : 0.f;
        return present > 0.f && fabsf(p11 / p00 - present) <= 0.03f * present;
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeDeferredLightCamera(const ProfileEntry* e)
{
    if (!e) return nullptr;
    std::unique_ptr<DeferredLightCameraFeature> f(new DeferredLightCameraFeature(*e));
    if (!f->Valid()) return nullptr;
    return std::unique_ptr<ProfileFeature>(f.release());
}

} // namespace wiz3d
