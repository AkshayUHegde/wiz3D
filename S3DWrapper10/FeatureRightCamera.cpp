/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <RightCamera Buffer Position InvView View Projection InvProjection ViewProjection RelativeViewProjection MoveOriginFor
// CameraFrom NearPlane FocalRays/>: the right eye's copy of a per-view block describes the right camera (see Transform).

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Device11Proxy.h"
#include "Buffer11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <cmath>
#include <map>
#include <mutex>
#include <sstream>

namespace wiz3d
{
namespace
{

// Byte offsets in the camera block, found once from the first shader declaring it; shared by the device-side and
// per-context instances.
struct CameraLayout
{
    bool              known = false;
    bool              fromRelVP = false;   // CameraFrom="RelativeViewProjection"
    std::vector<UINT> positions, relVP, nearPlanes;
    int               invView = -1, view = -1, proj = -1, invProj = -1, viewProj = -1;
};
std::mutex   s_layoutLock;
CameraLayout s_layout;

std::vector<std::string> SplitNames(const std::string& s)
{
    std::vector<std::string> out;
    std::istringstream in(s);
    std::string n;
    while (in >> n) out.push_back(n);
    return out;
}

// 4x4 inverse (Gauss-Jordan, partial pivot); false when singular.
bool Invert4(const double m[4][4], double out[4][4])
{
    double a[4][8];
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 8; ++c) a[r][c] = c < 4 ? m[r][c] : (c - 4 == r ? 1.0 : 0.0);
    for (int c = 0; c < 4; ++c)
    {
        int p = c;
        for (int r = c + 1; r < 4; ++r) if (fabs(a[r][c]) > fabs(a[p][c])) p = r;
        if (fabs(a[p][c]) < 1e-12) return false;
        if (p != c) for (int k = 0; k < 8; ++k) { const double t = a[c][k]; a[c][k] = a[p][k]; a[p][k] = t; }
        const double inv = 1.0 / a[c][c];
        for (int k = 0; k < 8; ++k) a[c][k] *= inv;
        for (int r = 0; r < 4; ++r)
            if (r != c && a[r][c] != 0.0)
            {
                const double f = a[r][c];
                for (int k = 0; k < 8; ++k) a[r][k] -= f * a[c][k];
            }
    }
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) out[r][c] = a[r][c + 4];
    return true;
}

class RightCameraFeature : public ProfileFeature
{
public:
    explicit RightCameraFeature(const ProfileEntry& e)
        : m_buffer(EntryValue(e, "Buffer")), m_positions(SplitNames(EntryValue(e, "Position"))),
          m_invView(EntryValue(e, "InvView")), m_view(EntryValue(e, "View")), m_proj(EntryValue(e, "Projection")),
          m_invProj(EntryValue(e, "InvProjection")), m_viewProj(EntryValue(e, "ViewProjection")),
          m_relVP(SplitNames(EntryValue(e, "RelativeViewProjection"))), m_moveOriginFor(SplitNames(EntryValue(e, "MoveOriginFor"))),
          m_cameraFrom(EntryValue(e, "CameraFrom")), m_nearPlanes(SplitNames(EntryValue(e, "NearPlane"))),
          m_focalRays(EntryValue(e, "FocalRays") == "1") {}

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        // A MoveOriginFor name is a cbuffer (its copy goes through the moved-origin edit too), a texture or a variable.
        for (const std::string& n : m_moveOriginFor)
        {
            const int raySlot = rdef.CBufferSlot(n);
            if (raySlot >= 0) info.NoteRightCB(n, raySlot);
            std::string cb;
            int varSlot = -1;
            UINT varOff = 0, varSize = 0;
            if (raySlot >= 0 || rdef.TextureSlot(n) >= 0 || rdef.FindVariable(n, cb, varSlot, varOff, varSize)) info.tags.push_back(n);
        }
        const int slot = rdef.CBufferSlot(m_buffer);
        if (slot < 0) return;
        info.NoteRightCB(m_buffer, slot);
        std::lock_guard<std::mutex> g(s_layoutLock);
        if (s_layout.known) return;
        auto off = [&](const std::string& n) -> int
        {
            for (const auto& v : rdef.Variables(n, false)) if (v.first == slot) return (int)v.second;
            return -1;
        };
        for (const std::string& n : m_positions) if (off(n) >= 0) s_layout.positions.push_back((UINT)off(n));
        for (const std::string& n : m_relVP) if (off(n) >= 0) s_layout.relVP.push_back((UINT)off(n));
        for (const std::string& n : m_nearPlanes) if (off(n) >= 0) s_layout.nearPlanes.push_back((UINT)off(n));
        s_layout.invView = off(m_invView);
        s_layout.view = off(m_view);
        s_layout.proj = off(m_proj);
        s_layout.invProj = off(m_invProj);
        s_layout.viewProj = off(m_viewProj);
        s_layout.fromRelVP = m_cameraFrom == "RelativeViewProjection";
        s_layout.known = true;
        FeatureLog("  RightCamera: %s (cb%d in CRC=0x%08lX): positions %u, InvView +%d, View +%d, Projection +%d,"
               " InvProjection +%d, ViewProjection +%d, relative VPs %u, near planes %u; origin moved for passes reading %u"
               " name(s)%s; camera from %s\n",
               m_buffer.c_str(), slot, info.crc32, (unsigned)s_layout.positions.size(), s_layout.invView, s_layout.view,
               s_layout.proj, s_layout.invProj, s_layout.viewProj, (unsigned)s_layout.relVP.size(),
               (unsigned)s_layout.nearPlanes.size(), (unsigned)m_moveOriginFor.size(),
               m_focalRays ? " (focal rays: only theirs, skew in the view)" : "",
               s_layout.fromRelVP ? "the relative view-projection" : "Projection / InvView");
    }

    bool MovedOriginPass(Context11Proxy& ctx) const override
    {
        for (const std::string& n : m_moveOriginFor) if (ctx.BoundShadersTagged(n)) return true;
        return false;
    }

    // The camera block the bound shader reads gets the right camera.
    bool EditRightConstants(Context11Proxy& ctx, const std::string& cbuffer, unsigned char* data, size_t size) override
    {
        if (cbuffer != m_buffer || !s_layout.known) return false;
        // FocalRays: screen passes rebuild from their own eye's depth with the game's projection; position chains are the core's.
        if (m_focalRays && !ctx.MovedOrigin()) return false;
        if (!Transform(ctx, data, size, wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign(), ctx.MovedOrigin())) return false;
        ++(ctx.MovedOrigin() ? m_moved : ctx.EditAtDispatch() ? m_atDispatch : m_atDraw);
        return true;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        // The scene camera, for shaders that bind no camera block: the projection moved most often this frame.
        unsigned best = 0;
        for (const auto& c : m_byP00) if (c.second.count > best) { best = c.second.count; m_scene = c.second.latest; m_sceneValid = true; }
        if (report)
            DDILog("  [frame %u] right camera (this frame): moved at draws %u, at dispatches %u, origin-moved copies %u,"
                   " skipped %u (not the camera's projection), scene p00 %.4f, move %.4f, cameras %u\n", frame, m_atDraw,
                   m_atDispatch, m_moved, m_skipped, m_scene.p00, m_scene.d, (unsigned)m_byP00.size());
        m_atDraw = m_atDispatch = m_moved = m_skipped = 0;
        m_byP00.clear();
    }

    float CameraP00() const override { return m_scene.p00; }

    // From the game's bytes of the camera block the bound shaders read; else the frame's scene camera.
    bool BoundCameraMove(Context11Proxy& ctx, CameraMove& out) const override
    {
        Buffer11Proxy* b = s_layout.known ? ctx.BoundNamedCB(m_buffer) : nullptr;
        if (b && !b->Shadow().empty() && Camera(ctx, b->Shadow().data(), b->Shadow().size(), out)) return true;
        if (!m_sceneValid) return false;
        out = m_scene;
        return true;
    }

private:
    struct PerP00 { unsigned count = 0; CameraMove latest; };
    std::string              m_buffer;
    std::vector<std::string> m_positions;
    std::string              m_invView, m_view, m_proj, m_invProj, m_viewProj;
    std::vector<std::string> m_relVP;
    std::vector<std::string> m_moveOriginFor;
    std::string              m_cameraFrom;
    std::vector<std::string> m_nearPlanes;
    bool                     m_focalRays = false;
    std::map<int, PerP00>    m_byP00;   // this frame's moved cameras, by p00 in thousandths
    CameraMove               m_scene;
    bool                     m_sceneValid = false;
    unsigned                 m_atDraw = 0, m_atDispatch = 0, m_moved = 0, m_skipped = 0;

    static const float* Row(const unsigned char* data, size_t size, int off, int r)
    {
        const size_t o = (size_t)off + (size_t)r * 16;
        return (off >= 0 && o + 16 <= size) ? (const float*)(data + o) : nullptr;
    }

    // The camera in a block's (left) bytes: p00 and the move d = eyeShift * conv along its right axis; false = not the
    // scene camera's projection (not at the presented aspect) or no eye shift.
    static bool Camera(Context11Proxy& ctx, const unsigned char* data, size_t size, CameraMove& out)
    {
        const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign();
        if (eyeShift == 0.f) return false;
        Device11Proxy* dev = ctx.Parent();
        const float present = (dev && dev->GetPresentHeight()) ? (float)dev->GetPresentWidth() / dev->GetPresentHeight() : 0.f;
        if (!(present > 0.f)) return false;
        auto atAspect = [present](float x, float y) { return x > 1e-3f && x < 100.f && fabsf(fabsf(y / x) - present) <= 0.03f * present; };
        if (s_layout.fromRelVP)
        {
            if (!FromRelativeViewProjection(data, size, atAspect, out)) return false;
        }
        else
        {
            const float* r0 = Row(data, size, s_layout.proj, 0);
            const float* r1 = Row(data, size, s_layout.proj, 1);
            const float p00 = r0 ? r0[0] : 0.f, p11 = r1 ? r1[1] : 0.f;
            if (!atAspect(p00, p11)) return false;
            float R[3] = {};
            for (int r = 0; r < 3; ++r) if (const float* iv = Row(data, size, s_layout.invView, r)) R[r] = iv[0];
            const float rn = sqrtf(R[0] * R[0] + R[1] * R[1] + R[2] * R[2]);
            if (!(rn > 0.5f && rn < 2.f)) return false;
            out.p00 = p00;
            for (int k = 0; k < 3; ++k) out.right[k] = R[k] / rn;
        }
        out.d = eyeShift * stereo::Convergence();
        for (int k = 0; k < 3; ++k) out.world[k] = out.d * out.right[k];
        return true;
    }

    // CameraFrom="RelativeViewProjection" (Far Cry 3): the relative view-projection every vertex shader positions with,
    // rows x = p00 * right, y = p11 * up, w = forward.
    template<typename AtAspect>
    static bool FromRelativeViewProjection(const unsigned char* data, size_t size, AtAspect atAspect, CameraMove& out)
    {
        const int vp = s_layout.relVP.empty() ? -1 : (int)s_layout.relVP[0];
        const float* a[4] = { Row(data, size, vp, 0), Row(data, size, vp, 1), Row(data, size, vp, 2), Row(data, size, vp, 3) };
        auto len = [](const float* v) { return v ? sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) : 0.f; };
        const float wn = len(a[3]), x = wn > 1e-6f ? len(a[0]) / wn : 0.f, y = wn > 1e-6f ? len(a[1]) / wn : 0.f;
        if (!a[2] || !atAspect(x, y)) return false;
        out.p00 = x;
        for (int k = 0; k < 3; ++k) out.right[k] = a[0][k] / (x * wn);
        return true;
    }

    // The game's (left) bytes become the right camera's: origin kept, the projection family takes x += b*w - p00*d;
    // origin moved (eye-ray passes), positions += d*R and projections take the skew x += b*w only.
    bool Transform(Context11Proxy& ctx, unsigned char* data, size_t size, float eyeShift, bool movedOrigin)
    {
        auto row = [&](int off, int r) -> float* { return const_cast<float*>(Row(data, size, off, r)); };
        CameraMove cam;
        if (!Camera(ctx, data, size, cam)) { ++m_skipped; return false; }
        PerP00& seen = m_byP00[(int)(cam.p00 * 1000.f + 0.5f)];
        ++seen.count;
        seen.latest = cam;
        const float p00 = cam.p00, d = cam.d, b = eyeShift * p00;
        // The translation in clip x: none where the origin moves with the camera, -p00*d where it stays.
        const float t = movedOrigin ? 0.f : -p00 * d;
        double P[4][4] = {}, IP[4][4] = {};
        const bool haveP = row(s_layout.proj, 0) && row(s_layout.proj, 3);
        if (haveP)
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) P[r][c] = row(s_layout.proj, r)[c];
        if (movedOrigin)
        {
            for (UINT p : s_layout.positions)
                if (p + 12 <= size) { float* f = (float*)(data + p); for (int k = 0; k < 3; ++k) f[k] += cam.world[k]; }
            for (int r = 0; r < 3; ++r) if (float* iv = row(s_layout.invView, r)) iv[3] += cam.world[r];
            if (float* v0 = row(s_layout.view, 0)) v0[3] -= d;
            // The skew moves a near-plane quad by b in NDC; |b| wider, it still spans [-1, 1].
            for (UINT n : s_layout.nearPlanes) if (n + 4 <= size) *(float*)(data + n) *= 1.f + fabsf(b);
        }
        // FocalRays: the ray (depth * ndc / focal) has no skew term, so the view takes it: x_view = x_ray + skew*z, with
        // w = P3.z * z; InvView * S and S^-1 * View, the projection left as the game's.
        const bool skewInView = movedOrigin && m_focalRays && haveP;
        if (skewInView)
        {
            const float skew = -eyeShift * (float)P[3][2];
            for (int r = 0; r < 3; ++r) if (float* iv = row(s_layout.invView, r)) iv[2] += skew * iv[0];
            if (float* v0 = row(s_layout.view, 0)) if (float* v2 = row(s_layout.view, 2)) for (int k = 0; k < 4; ++k) v0[k] -= skew * v2[k];
        }
        if (!skewInView)
            if (float* x = row(s_layout.proj, 0)) if (float* w = row(s_layout.proj, 3)) { for (int k = 0; k < 4; ++k) x[k] += b * w[k]; x[3] += t; }
        if (movedOrigin)
        {
            if (!skewInView) for (int r = 0; r < 4; ++r) if (float* ip = row(s_layout.invProj, r)) ip[3] -= b * ip[0];
        }
        else if (haveP && row(s_layout.invProj, 3))
        {
            for (int k = 0; k < 4; ++k) P[0][k] += b * P[3][k];
            P[0][3] += t;
            if (Invert4(P, IP))
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) row(s_layout.invProj, r)[c] = (float)IP[r][c];
        }
        if (float* x = row(s_layout.viewProj, 0))
            if (float* w = row(s_layout.viewProj, 3)) { for (int k = 0; k < 4; ++k) x[k] += b * w[k]; x[3] -= p00 * d; }
        for (UINT m : s_layout.relVP)
            if (float* x = row((int)m, 0)) if (float* w = row((int)m, 3)) { for (int k = 0; k < 4; ++k) x[k] += b * w[k]; x[3] += t; }
        return true;
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeRightCamera(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Buffer").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new RightCameraFeature(*e));
}

} // namespace wiz3d
