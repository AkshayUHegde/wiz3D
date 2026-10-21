/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <RightCamera Buffer Position InvView View Projection InvProjection ViewProjection RelativeViewProjection MoveOriginFor
// CameraFrom NearPlane FocalRays SvPosition/>: the right eye's copy of a per-view block describes the right camera.

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
    bool              registers = false;   // Buffer="*": found by its bytes, fields checked against them
    bool              rows = false;        // Layout="Rows": a matrix row per register
    std::vector<UINT> positions, relVP, nearPlanes, screens, svPositions;
    int               invView = -1, view = -1, proj = -1, invProj = -1, viewProj = -1, invViewProj = -1;
    int               prevViewProj = -1, prevProj = -1, clipToPrevClip = -1;   // last frame's camera (TAA, velocity)
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

// "r46" -> its byte offset; -1 for anything else.
int RegisterOffset(const std::string& s)
{
    if (s.size() < 2 || s[0] != 'r') return -1;
    for (size_t i = 1; i < s.size(); ++i) if (s[i] < '0' || s[i] > '9') return -1;
    return atoi(s.c_str() + 1) * 16;
}

// A 4x4 at off read into m (as stored); false when it is out of the block.
bool LoadMatrix(const unsigned char* data, size_t size, int off, double m[4][4])
{
    if (off < 0 || (size_t)off + 64 > size) return false;
    const float* f = (const float*)(data + off);
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) m[r][c] = f[r * 4 + c];
    return true;
}

void TransposeAt(unsigned char* data, size_t size, int off)
{
    if (off < 0 || (size_t)off + 64 > size) return;
    float* f = (float*)(data + off);
    for (int r = 0; r < 4; ++r) for (int c = r + 1; c < 4; ++c) { const float t = f[r * 4 + c]; f[r * 4 + c] = f[c * 4 + r]; f[c * 4 + r] = t; }
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

// The stored 4x4 at off is m's inverse, within 1% of its largest element: a register field the game laid out otherwise
// is left alone.
bool IsInverseOf(const unsigned char* data, size_t size, int off, const double m[4][4])
{
    double stored[4][4], inv[4][4];
    if (!LoadMatrix(data, size, off, stored) || !Invert4(m, inv)) return false;
    double big = 0.0, err = 0.0;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            if (fabs(inv[r][c]) > big) big = fabs(inv[r][c]);
            if (fabs(stored[r][c] - inv[r][c]) > err) err = fabs(stored[r][c] - inv[r][c]);
        }
    return big > 0.0 && err <= 0.01 * big;
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
          m_focalRays(EntryValue(e, "FocalRays") == "1"), m_invViewProj(EntryValue(e, "InvViewProjection")),
          m_screens(SplitNames(EntryValue(e, "Screen"))), m_svPositions(SplitNames(EntryValue(e, "SvPosition"))),
          m_rows(EntryValue(e, "Layout") == "Rows"),
          m_moveOrigin(EntryValue(e, "MoveOrigin") == "1"), m_anySlot(m_buffer == "*"),
          m_prevViewProj(EntryValue(e, "PrevViewProjection")), m_prevProj(EntryValue(e, "PrevProjection")),
          m_clipToPrevClip(EntryValue(e, "ClipToPrevClip"))
    {
        if (m_anySlot) LayoutFromRegisters();
    }

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        // Buffer="*", reflection stripped: each slot whose code reads one of the block's registers may hold it; the edit
        // tells by the bytes.
        if (m_anySlot)
        {
            if (rdef.Present() || !info.bytecode) return;
            std::vector<std::pair<int, UINT>> reads;
            CodeConstantReads(info.bytecode->data(), info.bytecode->size(), reads);
            for (const auto& r : reads)
                if (r.first >= 0 && r.first < 14 && (r.second == ~0u || ReadsBlock(r.second * 16u)))
                    info.NoteRightCB("cb" + std::to_string(r.first), r.first);
            return;
        }
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
        s_layout.invViewProj = m_invViewProj.empty() ? -1 : off(m_invViewProj);
        for (const std::string& n : m_screens) if (off(n) >= 0) s_layout.screens.push_back((UINT)off(n));
        for (const std::string& n : m_svPositions) if (off(n) >= 0) s_layout.svPositions.push_back((UINT)off(n));
        s_layout.rows = m_rows;
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
        if (!s_layout.known || (m_anySlot ? cbuffer.compare(0, 2, "cb") != 0 : cbuffer != m_buffer)) return false;
        // FocalRays: screen passes rebuild from their own eye's depth with the game's projection; position chains are the core's.
        if (m_focalRays && !ctx.MovedOrigin()) return false;
        if (!Transform(ctx, data, size, wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign(), m_moveOrigin || ctx.MovedOrigin()))
            return false;
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
        if (report && (s_layout.registers || !s_layout.screens.empty() || s_layout.invViewProj >= 0))
            DDILog("  [frame %u] right camera fields (this frame): screen matrices moved %u, not screen-shaped %u | inverse"
                   " fields written %u, left (not the inverse of their forward) %u | last-frame fields moved %u, left (not"
                   " shaped as declared) %u\n", frame, m_screened, m_screenOdd, m_inverses, m_inverseOdd, m_prevMoved, m_prevOdd);
        if (report && !s_layout.svPositions.empty())
            DDILog("  [frame %u] right camera SV_Position matrices (this frame): moved %u, not pixel-to-clip %u\n", frame,
                   m_svMoved, m_svOdd);
        m_atDraw = m_atDispatch = m_moved = m_skipped = 0;
        m_screened = m_screenOdd = m_inverses = m_inverseOdd = 0;
        m_svMoved = m_svOdd = 0;
        m_prevMoved = m_prevOdd = 0;
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
    std::string              m_invViewProj;
    std::vector<std::string> m_screens, m_svPositions;
    bool                     m_rows = false, m_moveOrigin = false, m_anySlot = false;
    std::string              m_prevViewProj, m_prevProj, m_clipToPrevClip;
    unsigned                 m_prevMoved = 0, m_prevOdd = 0;
    std::map<int, PerP00>    m_byP00;   // this frame's moved cameras, by p00 in thousandths
    CameraMove               m_scene;
    bool                     m_sceneValid = false;
    unsigned                 m_atDraw = 0, m_atDispatch = 0, m_moved = 0, m_skipped = 0;
    unsigned                 m_screened = 0, m_screenOdd = 0, m_inverses = 0, m_inverseOdd = 0;
    unsigned                 m_svMoved = 0, m_svOdd = 0;

    // Buffer="*" (reflection stripped, UE4): the block is found by its bytes and fields are registers ("r16"); Layout="Rows",
    // Screen, InvViewProjection, PrevViewProjection, PrevProjection, ClipToPrevClip and MoveOrigin="1" serve UE4's View block.
    void LayoutFromRegisters()
    {
        std::lock_guard<std::mutex> g(s_layoutLock);
        if (s_layout.known) return;
        for (const std::string& n : m_positions) if (RegisterOffset(n) >= 0) s_layout.positions.push_back((UINT)RegisterOffset(n));
        for (const std::string& n : m_relVP) if (RegisterOffset(n) >= 0) s_layout.relVP.push_back((UINT)RegisterOffset(n));
        for (const std::string& n : m_nearPlanes) if (RegisterOffset(n) >= 0) s_layout.nearPlanes.push_back((UINT)RegisterOffset(n));
        for (const std::string& n : m_screens) if (RegisterOffset(n) >= 0) s_layout.screens.push_back((UINT)RegisterOffset(n));
        for (const std::string& n : m_svPositions) if (RegisterOffset(n) >= 0) s_layout.svPositions.push_back((UINT)RegisterOffset(n));
        s_layout.invView = RegisterOffset(m_invView);
        s_layout.view = RegisterOffset(m_view);
        s_layout.proj = RegisterOffset(m_proj);
        s_layout.invProj = RegisterOffset(m_invProj);
        s_layout.viewProj = RegisterOffset(m_viewProj);
        s_layout.invViewProj = RegisterOffset(m_invViewProj);
        s_layout.prevViewProj = RegisterOffset(m_prevViewProj);
        s_layout.prevProj = RegisterOffset(m_prevProj);
        s_layout.clipToPrevClip = RegisterOffset(m_clipToPrevClip);
        s_layout.rows = m_rows;
        s_layout.registers = true;
        s_layout.known = true;
        FeatureLog("  RightCamera: block found by its bytes in shaders without reflection, %s: positions %u, InvView +%d,"
                   " View +%d, Projection +%d, InvProjection +%d, ViewProjection +%d, InvViewProjection +%d, screen"
                   " matrices %u%s\n", m_rows ? "a row per register" : "a column per register",
                   (unsigned)s_layout.positions.size(), s_layout.invView, s_layout.view, s_layout.proj, s_layout.invProj,
                   s_layout.viewProj, s_layout.invViewProj, (unsigned)s_layout.screens.size(),
                   m_moveOrigin ? "; origin at the eye for every pass" : "");
    }

    // A register (byte offset) inside one of the block's fields.
    bool ReadsBlock(UINT off) const
    {
        auto in = [off](int at, UINT n) { return at >= 0 && off >= (UINT)at && off < (UINT)at + n; };
        if (in(s_layout.invView, 64) || in(s_layout.view, 64) || in(s_layout.proj, 64) || in(s_layout.invProj, 64) ||
            in(s_layout.viewProj, 64) || in(s_layout.invViewProj, 64) || in(s_layout.prevViewProj, 64) ||
            in(s_layout.prevProj, 64) || in(s_layout.clipToPrevClip, 64))
            return true;
        for (UINT s : s_layout.screens) if (in((int)s, 64)) return true;
        for (UINT s : s_layout.svPositions) if (in((int)s, 64)) return true;
        for (UINT p : s_layout.positions) if (in((int)p, 16)) return true;
        return false;
    }

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
            if (s_layout.registers && s_layout.invView < 0)
            {
                // No InvView: the view's x output, its row 0 stored a column per register, its column 0 a row per register.
                for (int r = 0; r < 3; ++r)
                    if (s_layout.rows) { if (const float* v = Row(data, size, s_layout.view, r)) R[r] = v[0]; }
                    else if (const float* v = Row(data, size, s_layout.view, 0)) R[r] = v[r];
            }
            else if (s_layout.rows)
            {
                if (const float* iv = Row(data, size, s_layout.invView, 0)) for (int r = 0; r < 3; ++r) R[r] = iv[r];
            }
            else
                for (int r = 0; r < 3; ++r) if (const float* iv = Row(data, size, s_layout.invView, r)) R[r] = iv[0];
            const float rn = sqrtf(R[0] * R[0] + R[1] * R[1] + R[2] * R[2]);
            if (!(rn > 0.5f && rn < 2.f)) return false;
            out.p00 = p00;
            for (int k = 0; k < 3; ++k) out.right[k] = R[k] / rn;
        }
        // A menu's own 3D view takes the gameplay camera's parallax (UiProjectionParallax SceneNear), as its geometry does.
        out.d = eyeShift * stereo::UiParallaxScale(out.p00) * stereo::Convergence();
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
        if (!Camera(ctx, data, size, cam)) { if (!s_layout.registers) ++m_skipped; return false; }
        // Found by its bytes in any slot: the view-projection's x axis has the projection's p00 too.
        if (s_layout.registers && !AxisAgrees(data, size, s_layout.viewProj, cam.p00)) return false;
        eyeShift *= stereo::UiParallaxScale(cam.p00);
        PerP00& seen = m_byP00[(int)(cam.p00 * 1000.f + 0.5f)];
        ++seen.count;
        seen.latest = cam;
        const float p00 = cam.p00, d = cam.d, b = eyeShift * p00;
        // The translation in clip x: none where the origin moves with the camera, -p00*d where it stays.
        const float t = movedOrigin ? 0.f : -p00 * d;
        // Layout="Rows": each matrix is edited a column per register, as the rest assumes, and stored back as rows.
        if (s_layout.rows) TransposeFields(data, size);
        double P[4][4] = {}, IP[4][4] = {};
        const bool haveP = row(s_layout.proj, 0) && row(s_layout.proj, 3);
        if (haveP)
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) P[r][c] = row(s_layout.proj, r)[c];
        // SV_Position matrices, before the view-projection moves: the right pixel stands for the left one.
        for (UINT s : s_layout.svPositions) ++(SvPositionToRight(data, size, (int)s, P, haveP, b) ? m_svMoved : m_svOdd);
        // Register fields are checked against the bytes: an inverse is edited only where the game's is one.
        const bool invProjOk = !s_layout.registers || (haveP && IsInverseOf(data, size, s_layout.invProj, P));
        if (s_layout.registers && s_layout.invProj >= 0) ++(invProjOk ? m_inverses : m_inverseOdd);
        double VP[4][4] = {};
        const bool invVPOk = s_layout.invViewProj >= 0 && LoadMatrix(data, size, s_layout.viewProj, VP) &&
                             IsInverseOf(data, size, s_layout.invViewProj, VP);
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
            if (!skewInView && invProjOk) for (int r = 0; r < 4; ++r) if (float* ip = row(s_layout.invProj, r)) ip[3] -= b * ip[0];
        }
        else if (haveP && row(s_layout.invProj, 3) && invProjOk)
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
        // Last frame's camera (velocity): its view-projection and projection take this eye's terms as the current ones do,
        // only both together (a shifted previous position with an unskewed jitter row gains a constant -b of velocity).
        if (s_layout.prevProj >= 0 || s_layout.prevViewProj >= 0)
        {
            double PP[4][4] = {}, PV[4][4] = {};
            const bool prevProjOk = haveP && LoadMatrix(data, size, s_layout.prevProj, PP) &&
                                    fabs(PP[0][0] - P[0][0]) <= 0.02 * fabs(P[0][0]) && fabs(PP[1][1] - P[1][1]) <= 0.02 * fabs(P[1][1]) &&
                                    fabs(PP[3][0]) < 1e-4 && fabs(PP[3][1]) < 1e-4 && fabs(PP[3][2] - 1.0) < 1e-4 && fabs(PP[3][3]) < 1e-4;
            const bool prevVPOk = LoadMatrix(data, size, s_layout.prevViewProj, PV) &&
                                  fabs(sqrt(PV[0][0] * PV[0][0] + PV[0][1] * PV[0][1] + PV[0][2] * PV[0][2]) - p00) <= 0.03 * p00;
            if (prevProjOk && prevVPOk)
            {
                if (float* x = row(s_layout.prevProj, 0)) if (float* w = row(s_layout.prevProj, 3)) { for (int k = 0; k < 4; ++k) x[k] += b * w[k]; x[3] += t; }
                if (float* x = row(s_layout.prevViewProj, 0)) if (float* w = row(s_layout.prevViewProj, 3)) { for (int k = 0; k < 4; ++k) x[k] += b * w[k]; x[3] -= p00 * d; }
                ++m_prevMoved;
            }
            else ++m_prevOdd;
        }
        // ClipToPrevClip takes (ndc x, y, device z, 1) to last frame's clip: back to the left camera, the game's matrix, then
        // this eye's previous clip. Device z = near / w carries the b*C/w term: a reversed infinite projection only.
        if (s_layout.clipToPrevClip >= 0)
        {
            double CP[4][4];
            const double nearZ = haveP ? P[2][3] : 0.0;
            const bool infinite = haveP && fabs(P[2][2]) < 1e-6 && nearZ > 0.0 && fabs(P[3][2] - 1.0) < 1e-4;
            if (infinite && LoadMatrix(data, size, s_layout.clipToPrevClip, CP))
            {
                const double kz = (double)b * stereo::Convergence() / nearZ;
                double Bt[4][4] = { { 1, 0, -kz, b }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
                double At[4][4] = { { 1, 0, kz, -b }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 }, { 0, 0, 0, 1 } };
                double T[4][4] = {}, R[4][4] = {};
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) for (int k = 0; k < 4; ++k) T[r][c] += Bt[r][k] * CP[k][c];
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) for (int k = 0; k < 4; ++k) R[r][c] += T[r][k] * At[k][c];
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) row(s_layout.clipToPrevClip, r)[c] = (float)R[r][c];
                ++m_prevMoved;
            }
            else ++m_prevOdd;
        }
        // InvViewProjection: the inverse of the right view-projection just built.
        if (s_layout.invViewProj >= 0)
        {
            double R[4][4], IR[4][4];
            if (invVPOk && LoadMatrix(data, size, s_layout.viewProj, R) && Invert4(R, IR))
            {
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) row(s_layout.invViewProj, r)[c] = (float)IR[r][c];
                ++m_inverses;
            }
            else ++m_inverseOdd;
        }
        // Screen matrices take (x*W, y*W, W, 1), the right pixel's x the left's minus b*W plus b*C: input W -= b * input x,
        // input 1 += b*C * input x (element [r][k] is input k's part of output r here).
        const float conv = stereo::Convergence();
        for (UINT s : s_layout.screens)
        {
            if (!ScreenShaped(data, size, (int)s, p00)) { ++m_screenOdd; continue; }
            for (int r = 0; r < 4; ++r) if (float* m = row((int)s, r)) { m[2] -= b * m[0]; m[3] += b * conv * m[0]; }
            ++m_screened;
        }
        if (s_layout.rows) TransposeFields(data, size);
        return true;
    }

    // An SV_Position matrix (pixel x, y, device z, 1; a column per register), pixel-to-clip with the left view-projection:
    // input x becomes px + (W/2)*b*(C/w - 1), 1/w from device z by the projection's z row.
    static bool SvPositionToRight(unsigned char* data, size_t size, int off, const double P[4][4], bool haveP, float b)
    {
        double VP[4][4], S[4][4], M[4][4] = {};
        if (!haveP || !LoadMatrix(data, size, s_layout.viewProj, VP) || !LoadMatrix(data, size, off, S)) return false;
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) for (int k = 0; k < 4; ++k) M[r][c] += VP[r][k] * S[k][c];
        const double w = M[3][3], sx = w != 0.0 ? M[0][0] / w : 0.0;
        if (!(sx > 0.0)) return false;
        auto negligible = [w](double v, double scale) { return fabs(v / w) <= 1e-3 * scale; };
        if (!negligible(M[0][1], sx) || !negligible(M[0][2], 1.0) || !negligible(M[3][0], sx) || !negligible(M[3][1], sx) ||
            !negligible(M[3][2], 1.0) || !negligible(M[2][0], sx) || !negligible(M[2][1], sx) || !negligible(M[2][3], 1.0) ||
            fabs(M[2][2] / w - 1.0) > 1e-2)
            return false;
        const double m22 = P[2][2], m23 = P[2][3];
        if (fabs(P[3][2] - 1.0) > 1e-3 || fabs(P[3][3]) > 1e-4 || m23 == 0.0) return false;
        const double half = 1.0 / sx, conv = stereo::Convergence();
        const double kz = half * b * conv / m23, k1 = -half * b * (1.0 + conv * m22 / m23);
        float* f = (float*)(data + off);
        for (int r = 0; r < 4; ++r) { f[r * 4 + 2] += (float)(kz * S[r][0]); f[r * 4 + 3] += (float)(k1 * S[r][0]); }
        return true;
    }

    // Every matrix field between a row per register and a column per register (its own inverse).
    static void TransposeFields(unsigned char* data, size_t size)
    {
        const int mats[] = { s_layout.invView, s_layout.view, s_layout.proj, s_layout.invProj, s_layout.viewProj, s_layout.invViewProj,
                             s_layout.prevViewProj, s_layout.prevProj, s_layout.clipToPrevClip };
        for (int m : mats) TransposeAt(data, size, m);
        for (UINT m : s_layout.relVP) TransposeAt(data, size, (int)m);
        for (UINT s : s_layout.screens) TransposeAt(data, size, (int)s);
        for (UINT s : s_layout.svPositions) TransposeAt(data, size, (int)s);
    }

    // The view-projection's x axis (as stored) is p00 long, within 2%; true when the block declares none.
    static bool AxisAgrees(const unsigned char* data, size_t size, int vp, float p00)
    {
        if (vp < 0) return true;
        float x[3] = {};
        for (int k = 0; k < 3; ++k)
        {
            const float* r = Row(data, size, vp, s_layout.rows ? k : 0);
            if (!r) return false;
            x[k] = s_layout.rows ? r[0] : r[k];
        }
        return fabsf(sqrtf(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) - p00) <= 0.02f * p00;
    }

    // A screen matrix (a column per register): input x is the right axis over p00 and input W the unit forward, within 5%.
    static bool ScreenShaped(const unsigned char* data, size_t size, int off, float p00)
    {
        float x2 = 0.f, w2 = 0.f;
        for (int r = 0; r < 3; ++r)
        {
            const float* m = Row(data, size, off, r);
            if (!m) return false;
            x2 += m[0] * m[0];
            w2 += m[2] * m[2];
        }
        return fabsf(sqrtf(x2) * p00 - 1.f) <= 0.05f && fabsf(sqrtf(w2) - 1.f) <= 0.05f;
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeRightCamera(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Buffer").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new RightCameraFeature(*e));
}

} // namespace wiz3d
