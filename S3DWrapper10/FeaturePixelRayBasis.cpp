/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <PixelRayBasis Buffer Value="X Y Z" Origin Consumer ShadowOrigin Magnitudes Corners CornerOrigin/>: CryEngine rebuilds
// position from depth with the left camera's rays (pixel basis or frustum corners); the right eye gets its camera's.

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

const char* const kKey = "pixelbasis";
enum { kX, kY, kZ, kOrigin, kConsumer, kShadowOrigin, kMagnitudes, kTL, kTR, kBL, kCornerOrigin, kCount };

std::mutex         s_logLock;
std::set<unsigned> s_logged;   // shader CRCs already logged

// The width of the bound render target 0 at its mip, 0 when none.
UINT TargetWidth(Context11Proxy& ctx)
{
    ID3D11RenderTargetView* rtv = ctx.LeftTarget(0);
    if (!rtv) return 0;
    D3D11_RENDER_TARGET_VIEW_DESC vd = {};
    rtv->GetDesc(&vd);
    ID3D11Resource* r = nullptr;
    rtv->GetResource(&r);
    if (!r) return 0;
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    r->GetType(&dim);
    UINT w = 0;
    if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D)
    {
        D3D11_TEXTURE2D_DESC td = {};
        static_cast<ID3D11Texture2D*>(r)->GetDesc(&td);
        const UINT mip = vd.ViewDimension == D3D11_RTV_DIMENSION_TEXTURE2D ? vd.Texture2D.MipSlice : 0;
        w = (td.Width >> mip) ? (td.Width >> mip) : 1;
    }
    r->Release();
    return w;
}

class PixelRayBasisFeature : public ProfileFeature
{
public:
    explicit PixelRayBasisFeature(const ProfileEntry& e) : m_buffer(EntryValue(e, "Buffer"))
    {
        std::istringstream rows(EntryValue(e, "Value"));
        std::string n;
        for (int i = kX; i <= kZ && (rows >> n); ++i) m_names[i] = n;
        std::istringstream corners(EntryValue(e, "Corners"));
        for (int i = kTL; i <= kBL && (corners >> n); ++i) m_names[i] = n;
        m_names[kOrigin]       = EntryValue(e, "Origin");
        m_names[kConsumer]     = EntryValue(e, "Consumer");
        m_names[kShadowOrigin] = EntryValue(e, "ShadowOrigin");
        m_names[kMagnitudes]   = EntryValue(e, "Magnitudes");
        m_names[kCornerOrigin] = EntryValue(e, "CornerOrigin");
    }
    bool Valid() const { return !m_buffer.empty() && ((!m_names[kX].empty() && !m_names[kY].empty() && !m_names[kZ].empty()) ||
                                                      (!m_names[kTL].empty() && !m_names[kCornerOrigin].empty())); }

    // A shader that READS the rows, the corners and their origin, or the consumer rebuilds position from them; a declared,
    // unread variable holds stale bytes.
    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        const int slot = rdef.CBufferSlot(m_buffer);
        if (slot < 0) return;
        std::vector<int> off(kCount, -1);
        for (int i = 0; i < kCount; ++i)
            if (!m_names[i].empty())
                for (const auto& v : rdef.Variables(m_names[i], true))
                    if (v.first == slot) off[i] = (int)v.second;
        const bool rows = off[kX] >= 0 && off[kY] >= 0 && off[kZ] >= 0;
        const bool corners = off[kTL] >= 0 && off[kTR] >= 0 && off[kBL] >= 0 && off[kCornerOrigin] >= 0;
        if (!rows && !corners && off[kConsumer] < 0) return;
        info.NoteRightCB(m_buffer, slot);
        info.featureOffsets.emplace_back(kKey, off);
        std::lock_guard<std::mutex> g(s_logLock);
        if (s_logged.insert(info.crc32).second)
            FeatureLog("  PixelRayBasis: CRC=0x%08lX %s (cb%d) rows +%d +%d +%d origin %d consumer %d shadow origin %d"
                       " magnitudes %d | corners +%d +%d +%d origin %d\n", info.crc32, m_buffer.c_str(), slot, off[kX], off[kY],
                       off[kZ], off[kOrigin], off[kConsumer], off[kShadowOrigin], off[kMagnitudes], off[kTL], off[kTR], off[kBL],
                       off[kCornerOrigin]);
    }

    // ndc_left = ndc_right - b + b*C/w with b = eyeShift*p00: -b shifts the rays by b/2 of the screen's width (a pixel basis
    // X spans 2/W of it), and b*C/w moves the camera by eyeShift*C along its right axis.
    bool EditRightConstants(Context11Proxy& ctx, const std::string& cbuffer, unsigned char* data, size_t size) override
    {
        const ShaderAnalysis11Result* shader = ctx.EditingShader();
        if (cbuffer != m_buffer || !shader) return false;
        const std::vector<int>* off = nullptr;
        for (const auto& f : shader->featureOffsets) if (f.first == kKey) off = &f.second;
        if (!off) return false;
        const int sizes[kCount] = { 16, 16, 16, 12, 64, 16, 12, 12, 12, 12, 12 };
        for (int i = 0; i < kCount; ++i)
            if ((*off)[i] >= 0 && (size_t)(*off)[i] + sizes[i] > size) return false;
        const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign();
        if (eyeShift == 0.f) return false;
        const float conv = stereo::Convergence();
        const float b = stereo::FallbackB(eyeShift);
        if (b == 0.f) { ++m_noCamera; ctx.RetryEdit(); return false; }

        // Frustum corners (the tiled-lighting dispatch): ray = TL + (TR-TL)*u + (BL-TL)*v, u over the screen's width.
        if ((*off)[kTL] >= 0 && (*off)[kTR] >= 0 && (*off)[kBL] >= 0 && (*off)[kCornerOrigin] >= 0)
        {
            float* tl = (float*)(data + (*off)[kTL]);
            float* tr = (float*)(data + (*off)[kTR]);
            float* bl = (float*)(data + (*off)[kBL]);
            float* o = (float*)(data + (*off)[kCornerOrigin]);
            const float d[3] = { tr[0] - tl[0], tr[1] - tl[1], tr[2] - tl[2] };
            const float dv[3] = { bl[0] - tl[0], bl[1] - tl[1], bl[2] - tl[2] };
            const float nd = sqrtf(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            const float nv = sqrtf(dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]);
            if (!(nd > 1e-12f) || !(nv > 1e-12f) || fabsf(d[0] * dv[0] + d[1] * dv[1] + d[2] * dv[2]) > 0.05f * nd * nv)
            {
                ++m_rejected;
                return false;
            }
            for (int k = 0; k < 3; ++k)
            {
                const float s = 0.5f * b * d[k];
                tl[k] -= s; tr[k] -= s; bl[k] -= s;
                o[k] += eyeShift * conv * d[k] / nd;
                m_right[k] = d[k] / nd;
            }
            m_haveRight = true;
            ++m_corners;
            return true;
        }

        if (ctx.EditAtDispatch() || ctx.LightViewPass()) return false;
        // A camera-relative position from the vertex stage's ray (the light shadow masks read no rows): only the
        // consumer's translation moves, along this frame's camera right axis from the bases and corners.
        if ((*off)[kX] < 0 || (*off)[kY] < 0 || (*off)[kZ] < 0)
        {
            if ((*off)[kConsumer] < 0) return false;
            if (!m_haveRight) { ++m_noCamera; ctx.RetryEdit(); return false; }
            const float e[3] = { eyeShift * conv * m_right[0], eyeShift * conv * m_right[1], eyeShift * conv * m_right[2] };
            float* t = (float*)(data + (*off)[kConsumer]);
            for (int row = 0; row < 4; ++row) t[row * 4 + 3] += t[row * 4] * e[0] + t[row * 4 + 1] * e[1] + t[row * 4 + 2] * e[2];
            ++m_rayConsumers;
            return true;
        }

        const UINT w = TargetWidth(ctx);
        if (!w) { ++m_noCamera; ctx.RetryEdit(); return false; }
        const float halfW = 0.5f * (float)w;
        float* X = (float*)(data + (*off)[kX]);
        float* Y = (float*)(data + (*off)[kY]);
        float* Z = (float*)(data + (*off)[kZ]);

        // Shadow space (the sun's resolve): the rows times Magnitudes are the world basis' images and ShadowOrigin the
        // camera's; the right axis' image is X*mx*W*p00/(2*far), far taken from the world bases (|X|*W*p00/2).
        if ((*off)[kShadowOrigin] >= 0 && (*off)[kMagnitudes] >= 0)
        {
            const float* m = (const float*)(data + (*off)[kMagnitudes]);
            float* o = (float*)(data + (*off)[kShadowOrigin]);
            if (m[2] == 0.f) { ++m_rejected; return false; }
            const float corner = b * halfW * m[0] / m[2];
            for (int k = 0; k < 4; ++k) Z[k] -= corner * X[k];
            if (m_far > 0.f)
            {
                const float origin = b * conv * halfW * m[0] / m_far;
                for (int k = 0; k < 4; ++k) o[k] += origin * X[k];
            }
            else ++m_noFar;
            ++m_shadowBases;
            return true;
        }

        // World or camera-relative: X is along the camera's right axis and Y along its up.
        const float nx = sqrtf(X[0] * X[0] + X[1] * X[1] + X[2] * X[2]);
        const float ny = sqrtf(Y[0] * Y[0] + Y[1] * Y[1] + Y[2] * Y[2]);
        if (!(nx > 1e-12f) || !(ny > 1e-12f) || fabsf(X[0] * Y[0] + X[1] * Y[1] + X[2] * Y[2]) > 0.05f * nx * ny)
        {
            ++m_rejected;
            return false;
        }
        const float e[3] = { eyeShift * conv * X[0] / nx, eyeShift * conv * X[1] / nx, eyeShift * conv * X[2] / nx };
        for (int k = 0; k < 3; ++k) { Z[k] -= b * halfW * X[k]; m_right[k] = X[k] / nx; }
        m_haveRight = true;
        if ((*off)[kOrigin] >= 0)
        {
            float* o = (float*)(data + (*off)[kOrigin]);
            for (int k = 0; k < 3; ++k) o[k] += e[k];
            ++m_origins;
        }
        else if ((*off)[kConsumer] >= 0)
        {
            // A camera-relative position into a matrix of dp4 rows: each row's translation takes the camera's move.
            float* t = (float*)(data + (*off)[kConsumer]);
            for (int row = 0; row < 4; ++row) t[row * 4 + 3] += t[row * 4] * e[0] + t[row * 4 + 1] * e[1] + t[row * 4 + 2] * e[2];
            ++m_consumers;
        }
        ++m_bases;
        m_far = nx * halfW * (b / eyeShift);   // the camera's far plane, for the shadow-space origin
        return true;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] pixel ray bases (this frame): world and camera-relative %u (origins moved %u, consumers moved %u),"
                   " consumers of a vertex ray %u, shadow-space %u (no far plane yet %u), frustum corners %u, rejected %u, without"
                   " a camera or target %u\n", frame, m_bases, m_origins, m_consumers, m_rayConsumers, m_shadowBases, m_noFar,
                   m_corners, m_rejected, m_noCamera);
        m_bases = m_origins = m_consumers = m_rayConsumers = m_shadowBases = m_rejected = m_noCamera = m_noFar = m_corners = 0;
    }

private:
    std::string m_buffer;
    std::string m_names[kCount];
    unsigned    m_bases = 0, m_origins = 0, m_consumers = 0, m_rayConsumers = 0, m_shadowBases = 0, m_rejected = 0;
    unsigned    m_noCamera = 0, m_noFar = 0, m_corners = 0;
    float       m_right[3] = {};       // this frame's camera right axis (world), from the last basis or corners
    bool        m_haveRight = false;
    float       m_far = 0.f;           // the last world basis' far plane, for the shadow-space origin
};

} // namespace

std::unique_ptr<ProfileFeature> MakePixelRayBasis(const ProfileEntry* e)
{
    if (!e) return nullptr;
    std::unique_ptr<PixelRayBasisFeature> f(new PixelRayBasisFeature(*e));
    if (!f->Valid()) return nullptr;
    return std::unique_ptr<ProfileFeature>(f.release());
}

} // namespace wiz3d
