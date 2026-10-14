/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <ScreenRayMatrix Value Screen Consumer Positions/> ("cbuffer.name" lists): a screen-to-world ray (registers 0-2 times screen x, y, w)
// scaled by scene depth, and matrices taking (x*W, y*W, W, 1); the right eye's get its camera, the ray's users the eye offset.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Buffer11Proxy.h"
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

std::mutex            s_logLock;
std::set<std::string> s_logged;   // key+offset pairs already logged

struct Named { std::string buffer, name, key; int slot = -1; UINT offset = 0; };   // slot >= 0: a register "cbN[R]"

// "cbuffer.name", or a bare name in the Buffer attribute's cbuffer.
std::vector<Named> ParseNames(const ProfileEntry& e, const char* attr, const char* prefix)
{
    std::vector<Named> out;
    std::istringstream in(EntryValue(e, attr));
    std::string n;
    while (in >> n)
    {
        unsigned slot = 0, reg = 0;
        char tail = 0;
        if (sscanf_s(n.c_str(), "cb%u[%u%c", &slot, &reg, &tail, 1) == 3 && tail == ']' && slot < 14)
        {
            Named t;
            t.buffer = "cb" + std::to_string(slot);
            t.name = n.substr(t.buffer.size());
            t.key = prefix + n;
            t.slot = (int)slot;
            t.offset = reg * 16u;
            out.push_back(t);
            continue;
        }
        const size_t dot = n.find('.');
        Named t;
        t.buffer = dot == std::string::npos ? EntryValue(e, "Buffer") : n.substr(0, dot);
        t.name = dot == std::string::npos ? n : n.substr(dot + 1);
        t.key = prefix + t.buffer + "." + t.name;
        if (!t.buffer.empty() && !t.name.empty()) out.push_back(t);
    }
    return out;
}

// The right eye's clip-x shear b of the camera the ray belongs to: r0 = right/p00, r2 = the unit forward at depth 1.
bool RayShear(const float* r, float eyeShift, float& b)
{
    const float n0 = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    const float n2 = sqrtf(r[8] * r[8] + r[9] * r[9] + r[10] * r[10]);
    if (!(n0 > 1e-6f) || fabsf(n2 - 1.f) > 0.05f) return false;
    if (fabsf(r[0] * r[8] + r[1] * r[9] + r[2] * r[10]) > 0.05f * n0 * n2) return false;
    b = eyeShift / n0;
    return true;
}

class ScreenRayMatrixFeature : public ProfileFeature
{
public:
    explicit ScreenRayMatrixFeature(const ProfileEntry& e)
        : m_rays(ParseNames(e, "Value", "ray:")), m_screens(ParseNames(e, "Screen", "scr:")),
          m_consumers(ParseNames(e, "Consumer", "use:")), m_positions(ParseNames(e, "Positions", "pos:")) {}

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        Note(rdef, info, m_rays, "ray");
        Note(rdef, info, m_screens, "screen matrix");
        Note(rdef, info, m_consumers, "consumer");
        Note(rdef, info, m_positions, "position");
        // A register in Screen, reflection stripped: only where the code reads it as (x*W, y*W, W, 1).
        bool registers = false;
        for (const Named& t : m_screens) registers |= t.slot >= 0;
        if (!registers || rdef.Present() || !info.bytecode) return;
        std::vector<std::pair<int, UINT>> found;
        CodeScreenMatrices(info.bytecode->data(), info.bytecode->size(), found);
        for (const Named& t : m_screens)
            for (const auto& f : found)
            {
                if (t.slot < 0 || f.first != t.slot || f.second != t.offset) continue;
                info.NoteRightCB(t.buffer, t.slot);
                info.featureOffsets.emplace_back(t.key, std::vector<int>(1, (int)t.offset));
                std::lock_guard<std::mutex> g(s_logLock);
                if (s_logged.insert(t.key).second)
                    FeatureLog("  ScreenRayMatrix: screen matrix %s read as (x*W, y*W, W, 1) in CRC=0x%08lX (first of its"
                               " shaders)\n", t.key.c_str() + 4, info.crc32);
            }
    }

    bool EditRightConstants(Context11Proxy& ctx, const std::string& cbuffer, unsigned char* data, size_t size) override
    {
        const ShaderAnalysis11Result* shader = ctx.EditingShader();
        if (!shader || ctx.EditAtDispatch()) return false;
        const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign();
        bool edited = false;
        // ndc_left = ndc_right - b + b*C/w: the -b is r2 -= b*r0.
        for (int off : Offsets(*shader, m_rays, cbuffer))
        {
            if ((size_t)off + 64 > size) continue;
            float* r = (float*)(data + off);
            float b = 0.f;
            if (!RayShear(r, eyeShift, b)) { ++m_rejected; continue; }
            for (int k = 0; k < 4; ++k) r[8 + k] -= b * r[k];
            ++m_skewed;
            edited = true;
        }
        const std::vector<int> screens = Offsets(*shader, m_screens, cbuffer), uses = Offsets(*shader, m_consumers, cbuffer),
                               positions = Offsets(*shader, m_positions, cbuffer);
        if (screens.empty() && uses.empty() && positions.empty()) return edited;
        ctx.RetryEdit();   // these follow the vertex stage's ray, not only these bytes
        float b = 0.f;
        const float* ray = BoundRay(ctx, eyeShift, b);
        // Without rays the scene camera's shear serves the Screen matrices; consumers and positions need a ray's axis.
        if (!ray && m_rays.empty() && !screens.empty()) { b = stereo::FallbackB(eyeShift); ++m_fromCamera; }
        else if (!ray)
        {
            if (!screens.empty() || !uses.empty()) ++m_noRay;
            return edited;
        }
        // The b*C/w term is the fixed position offset b*C*r0 of the ray's camera.
        const float c = stereo::Convergence();
        for (int off : screens)
        {
            if ((size_t)off + 64 > size) continue;
            float* m = (float*)(data + off);
            for (int k = 0; k < 4; ++k) { m[8 + k] -= b * m[k]; m[12 + k] += b * c * m[k]; }
            ++m_screened;
            edited = true;
        }
        if (!ray) return edited;
        const float offset[3] = { b * c * ray[0], b * c * ray[1], b * c * ray[2] };
        // Applied to the position: r3 += sum offset_i * r_i.
        for (int off : uses)
        {
            if ((size_t)off + 64 > size) continue;
            float* t = (float*)(data + off);
            for (int k = 0; k < 4; ++k) t[12 + k] += offset[0] * t[k] + offset[1] * t[4 + k] + offset[2] * t[8 + k];
            ++m_moved;
            edited = true;
        }
        // Compared with the position (light minus position): the light takes -offset.
        for (int off : positions)
        {
            if ((size_t)off + 12 > size) continue;
            float* p = (float*)(data + off);
            for (int k = 0; k < 3; ++k) p[k] -= offset[k];
            ++m_positionsMoved;
            edited = true;
        }
        return edited;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] screen rays (this frame): rays skewed %u, screen matrices moved %u (by the scene camera %u),"
                   " consumers given the eye offset %u, positions moved %u, rejected %u (not a camera ray), matrices without a"
                   " bound ray %u\n", frame, m_skewed, m_screened, m_fromCamera, m_moved, m_positionsMoved, m_rejected, m_noRay);
        m_skewed = m_screened = m_fromCamera = m_moved = m_positionsMoved = m_rejected = m_noRay = 0;
    }

private:
    std::vector<Named> m_rays, m_screens, m_consumers, m_positions;
    unsigned           m_skewed = 0, m_screened = 0, m_fromCamera = 0, m_moved = 0, m_positionsMoved = 0, m_rejected = 0;
    unsigned           m_noRay = 0;

    static void Note(const RdefView& rdef, ShaderAnalysis11Result& info, const std::vector<Named>& names, const char* what)
    {
        for (const Named& t : names)
        {
            if (t.slot >= 0) continue;   // a register: OnShaderCreated reads the code for it
            const int slot = rdef.CBufferSlot(t.buffer);
            if (slot < 0) continue;
            std::vector<int> offs;
            for (const auto& v : rdef.Variables(t.name, false)) if (v.first == slot) offs.push_back((int)v.second);
            if (offs.empty()) continue;
            info.NoteRightCB(t.buffer, slot);
            info.featureOffsets.emplace_back(t.key, offs);
            std::lock_guard<std::mutex> g(s_logLock);
            for (int off : offs)
                if (s_logged.insert(t.key + "+" + std::to_string(off)).second)
                    FeatureLog("  ScreenRayMatrix: %s %s.%s at +%d (cb%d in CRC=0x%08lX)\n", what, t.buffer.c_str(),
                               t.name.c_str(), off, slot, info.crc32);
        }
    }

    static std::vector<int> Offsets(const ShaderAnalysis11Result& shader, const std::vector<Named>& names,
                                    const std::string& cbuffer)
    {
        std::vector<int> offs;
        for (const Named& t : names)
        {
            if (t.buffer != cbuffer) continue;
            for (const auto& f : shader.featureOffsets)
                if (f.first == t.key) offs.insert(offs.end(), f.second.begin(), f.second.end());
        }
        return offs;
    }

    // The game's bytes of the bound vertex shader's ray, and its camera's shear b; null when none is bound.
    const float* BoundRay(Context11Proxy& ctx, float eyeShift, float& b) const
    {
        const ShaderAnalysis11Result* vs = ctx.BoundShader(false);
        if (!vs) return nullptr;
        for (const Named& t : m_rays)
            for (const auto& f : vs->featureOffsets)
            {
                if (f.first != t.key || f.second.empty()) continue;
                for (const auto& named : vs->rightCBs)
                {
                    if (named.first != t.buffer) continue;
                    Buffer11Proxy* buf = ctx.BoundCB(false, (UINT)named.second);
                    if (!buf || buf->Shadow().size() < (size_t)f.second[0] + 64) continue;
                    const float* r = (const float*)(buf->Shadow().data() + f.second[0]);
                    return RayShear(r, eyeShift, b) ? r : nullptr;
                }
            }
        return nullptr;
    }
};

} // namespace

// No Value (a game without rays, UE3's forward renderer): the Screen matrices take the scene camera's shear.
std::unique_ptr<ProfileFeature> MakeScreenRayMatrix(const ProfileEntry* e)
{
    if (!e || (EntryValue(*e, "Value").empty() && EntryValue(*e, "Screen").empty())) return nullptr;
    return std::unique_ptr<ProfileFeature>(new ScreenRayMatrixFeature(*e));
}

} // namespace wiz3d
