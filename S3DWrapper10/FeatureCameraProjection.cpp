/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <CameraProjection|InverseCameraProjection Value="cbuffer.matrix ..." Space="View|World"/>: copies of the camera's
// projection at their own scale s, moved for the right eye as F0 += s*b*F3, F0.w -= s*p00*d (inverses: invert, move, invert).

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

namespace wiz3d
{
namespace
{

std::mutex                              s_projLock;
std::map<std::string, std::vector<int>> s_projOffsets;   // per element and shader: byte offsets of the named matrices
std::set<std::string>                   s_projLogged;    // element+offset pairs already logged

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
        if (fabs(a[p][c]) < 1e-20) return false;
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

class CameraProjectionFeature : public ProfileFeature
{
public:
    // Value lists matrices as "cbuffer.name", or a bare name in the Buffer attribute's cbuffer.
    CameraProjectionFeature(const ProfileEntry& e, bool inverse)
        : m_inverse(inverse), m_world(EntryValue(e, "Space") == "World")
    {
        std::istringstream in(EntryValue(e, "Value"));
        std::string n;
        while (in >> n)
        {
            const size_t dot = n.find('.');
            Target t;
            t.buffer = dot == std::string::npos ? EntryValue(e, "Buffer") : n.substr(0, dot);
            t.name = dot == std::string::npos ? n : n.substr(dot + 1);
            if (!t.buffer.empty() && !t.name.empty()) m_targets.push_back(t);
        }
        for (const Target& t : m_targets) if (m_buffers.find(t.buffer) == std::string::npos) m_buffers += t.buffer + " ";
    }

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        for (const Target& t : m_targets)
        {
            const int slot = rdef.CBufferSlot(t.buffer);
            if (slot < 0) continue;
            // Per shader: only one that declares the matrix notes the cbuffer, at its own offset.
            std::vector<int> offs;
            for (const auto& v : rdef.Variables(t.name, false)) if (v.first == slot) offs.push_back((int)v.second);
            if (offs.empty()) continue;
            info.NoteRightCB(t.buffer, slot);
            std::lock_guard<std::mutex> g(s_projLock);
            s_projOffsets[ShaderKey(t, info.crc32)] = offs;
            for (int off : offs)
                if (s_projLogged.insert(Key(t) + "+" + std::to_string(off)).second)
                    FeatureLog("  %sCameraProjection: %s.%s at +%d, %s space (cb%d in CRC=0x%08lX)\n", m_inverse ? "Inverse" : "",
                           t.buffer.c_str(), t.name.c_str(), off, m_world ? "world" : "view", slot, info.crc32);
        }
    }

    bool EditRightConstants(Context11Proxy& ctx, const std::string& cbuffer, unsigned char* data, size_t size) override
    {
        const ShaderAnalysis11Result* shader = ctx.EditingShader();
        if (!shader) return false;
        std::vector<int> offs;
        {
            std::lock_guard<std::mutex> g(s_projLock);
            for (const Target& t : m_targets)
            {
                if (t.buffer != cbuffer) continue;
                auto it = s_projOffsets.find(ShaderKey(t, shader->crc32));
                if (it != s_projOffsets.end()) offs.insert(offs.end(), it->second.begin(), it->second.end());
            }
        }
        if (offs.empty()) return false;
        CameraMove cam;
        if (!ctx.BoundCameraMove(cam)) { ++m_skipped; ctx.RetryEdit(); return false; }
        const double b = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign() * cam.p00;
        const double t = ctx.MovedOrigin() ? 0.0 : -(double)cam.p00 * cam.d;
        const double axis[3] = { m_world ? cam.right[0] : 1.0, m_world ? cam.right[1] : 0.0, m_world ? cam.right[2] : 0.0 };
        bool edited = false;
        for (int off : offs)
        {
            if ((size_t)off + 64 > size) continue;
            float* m = (float*)(data + off);
            double F[4][4], G[4][4];
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) G[r][c] = m[r * 4 + c];
            if (m_inverse) { if (!Invert4(G, F)) { ++m_rejected; continue; } }
            else for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) F[r][c] = G[r][c];
            double wn = sqrt(F[3][0] * F[3][0] + F[3][1] * F[3][1] + F[3][2] * F[3][2]);
            // An inverse is the projection up to scale (the shader divides by w): make its depth axis unit first.
            if (m_inverse && wn > 1e-12)
            {
                for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) F[r][c] /= wn;
                wn = 1.0;
            }
            const double s = (F[0][0] * axis[0] + F[0][1] * axis[1] + F[0][2] * axis[2]) / cam.p00;
            // The w row of a perspective is the unit depth axis; s must be a scale (texture 0.5, pixels W/2).
            if (!(fabs(s) > 0.05 && fabs(s) < 1e5) || fabs(wn - 1.0) > 1e-2) { ++m_rejected; continue; }
            for (int k = 0; k < 4; ++k) F[0][k] += s * b * F[3][k];
            F[0][3] += s * t;
            if (m_inverse) { if (!Invert4(F, G)) { ++m_rejected; continue; } }
            else for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) G[r][c] = F[r][c];
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) m[r * 4 + c] = (float)G[r][c];
            m_lastScale = (float)s;
            edited = true;
        }
        if (edited) ++m_edited;
        return edited;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] %scamera projection copies %s(this frame): moved %u, rejected %u (not a perspective),"
                   " skipped %u (no camera yet), scale %.4g\n", frame, m_inverse ? "inverse " : "", m_buffers.c_str(),
                   m_edited, m_rejected, m_skipped, m_lastScale);
        m_edited = m_rejected = m_skipped = 0;
    }

private:
    struct Target { std::string buffer, name; };
    bool                     m_inverse, m_world;
    std::vector<Target>      m_targets;
    std::string              m_buffers;   // for the row
    unsigned                 m_edited = 0, m_rejected = 0, m_skipped = 0;
    float                    m_lastScale = 0.f;

    std::string Key(const Target& t) const { return (m_inverse ? "inv:" : "fwd:") + t.buffer + "." + t.name; }
    std::string ShaderKey(const Target& t, DWORD crc) const { return Key(t) + "@" + std::to_string(crc); }
};

std::unique_ptr<ProfileFeature> Make(const ProfileEntry* e, bool inverse)
{
    if (!e || EntryValue(*e, "Value").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new CameraProjectionFeature(*e, inverse));
}

} // namespace

std::unique_ptr<ProfileFeature> MakeCameraProjection(const ProfileEntry* e) { return Make(e, false); }
std::unique_ptr<ProfileFeature> MakeInverseCameraProjection(const ProfileEntry* e) { return Make(e, true); }

} // namespace wiz3d
