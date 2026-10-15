/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <DecalMatrices Value="1"/>: instanced decals read per-instance matrices by index, which the analyzer cannot see: the
// vertex stage's box-to-clip array and the pixel stage's SV_Position-to-decal array get the right camera.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <d3dcompiler.h>
#include <cmath>
#include <map>
#include <mutex>
#include <set>
#include <sstream>

namespace wiz3d
{
namespace
{

const char* const kKey = "decalmatrices";

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

// An array read as cbN[rX.c + K .. K + 3] by instance: the vertex stage's feeding o0 (with an instance id), the pixel
// stage's scaling SV_Position x. Entries run to the next indexed array (decal-to-world) or the declared end.
bool FindDecalArray(const std::vector<BYTE>& code, bool& vertex, int& slot, int& base, int& entries)
{
    ID3DBlob* text = nullptr;
    if (code.empty() || FAILED(D3DDisassemble(code.data(), code.size(), 0, nullptr, &text)) || !text) return false;
    const std::string s((const char*)text->GetBufferPointer(), text->GetBufferSize());
    text->Release();
    const bool vs = s.find("\nvs_") != std::string::npos;
    const bool ps = s.find("\nps_") != std::string::npos;
    if ((!vs && !ps) || (vs && s.find("instance_id") == std::string::npos) || (ps && s.find(", position") == std::string::npos))
        return false;
    std::set<std::pair<int, int>> seen, qualifying;
    std::istringstream in(s);
    std::string line;
    while (std::getline(in, line))
    {
        for (size_t p = line.find("cb"); p != std::string::npos; p = line.find("cb", p + 2))
        {
            int sl = 0, rn = 0, k = 0;
            char c = 0;
            if (sscanf_s(line.c_str() + p, "cb%d[r%d.%c + %d]", &sl, &rn, &c, 1, &k) != 4) continue;
            seen.insert(std::make_pair(sl, k));
            if (vs ? line.find(" o0.") != std::string::npos : line.find("v0.xxxx") != std::string::npos)
                qualifying.insert(std::make_pair(sl, vs ? k - 3 : k));
        }
    }
    for (const auto& q : qualifying)
    {
        const int sl = q.first, k = q.second;
        bool all = true;
        for (int i = 0; i < 4; ++i) all &= seen.count(std::make_pair(sl, k + i)) != 0;
        if (!all) continue;
        char key[32];
        sprintf_s(key, "dcl_constantbuffer CB%d[", sl);
        const size_t d = s.find(key);
        if (d == std::string::npos) continue;
        int end = atoi(s.c_str() + d + strlen(key));
        for (const auto& r : seen) if (r.first == sl && r.second >= k + 4 && r.second < end) end = r.second;
        if (end < k + 4) continue;
        vertex = vs;
        slot = sl;
        base = k;
        entries = (end - k) / 4;
        return true;
    }
    return false;
}

class DecalMatricesFeature : public ProfileFeature
{
public:
    void OnShaderCreated(const RdefView&, ShaderAnalysis11Result& info) const override
    {
        if (!info.bytecode) return;
        bool vertex = false;
        int slot = -1, base = 0, entries = 0;
        if (!FindDecalArray(*info.bytecode, vertex, slot, base, entries) || slot < 0 || slot >= 14) return;
        info.NoteRightCB("cb" + std::to_string(slot), slot);
        info.featureOffsets.emplace_back(kKey, std::vector<int>{ vertex ? 1 : 0, slot, base, entries });
        std::lock_guard<std::mutex> g(s_logLock);
        if (s_logged.insert(info.crc32).second)
            FeatureLog("  DecalMatrices: CRC=0x%08lX %s array cb%d[%d], %d entries\n", info.crc32,
                       vertex ? "box-to-clip" : "SV_Position-to-decal", slot, base, entries);
    }

    // Vertex: each box-to-clip takes the camera's clip shift, x += b*w - p00*d. Pixel: the right pixel stands for the
    // left one at px + (W/2)*(-b + b*C/w), 1/w from device z by the scene projection's z row.
    bool EditRightConstants(Context11Proxy& ctx, const std::string& cbuffer, unsigned char* data, size_t size) override
    {
        const ShaderAnalysis11Result* shader = ctx.EditingShader();
        if (!shader) return false;
        const std::vector<int>* a = nullptr;
        for (const auto& f : shader->featureOffsets) if (f.first == kKey) a = &f.second;
        if (!a || a->size() < 4 || cbuffer != "cb" + std::to_string((*a)[1])) return false;
        const bool vertex = (*a)[0] != 0;
        const int base = (*a)[2], entries = (*a)[3];
        const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign();
        const float p00 = ctx.CameraP00() > 0.f ? ctx.CameraP00() : stereo::SceneP00();
        if (eyeShift == 0.f) return false;
        if (!(p00 > 0.f)) { ++m_noCamera; ctx.RetryEdit(); return false; }
        const float conv = stereo::Convergence(), b = eyeShift * p00, d = eyeShift * conv;
        float k1 = 0.f, kz = 0.f;
        if (!vertex)
        {
            float m22 = 0.f, m23 = 0.f;
            const UINT w = TargetWidth(ctx);
            if (!w || !stereo::SceneZRow(m22, m23) || m23 == 0.f) { ++m_noCamera; ctx.RetryEdit(); return false; }
            const float half = 0.5f * (float)w;
            kz = half * b * conv / m23;
            k1 = -half * b * (1.f + conv * m22 / m23);
        }
        unsigned moved = 0;
        for (int i = 0; i < entries; ++i)
        {
            const size_t off = (size_t)(base + 4 * i) * 16;
            if (off + 64 > size) break;
            float* r = (float*)(data + off);
            bool any = false;
            for (int k = 0; k < 16; ++k) any |= r[k] != 0.f;
            if (!any) continue;
            if (vertex)
            {
                for (int k = 0; k < 4; ++k) r[k * 4] += b * r[k * 4 + 3];
                r[12] -= p00 * d;
            }
            else
                for (int c = 0; c < 4; ++c) { r[8 + c] += kz * r[c]; r[12 + c] += k1 * r[c]; }
            ++moved;
        }
        if (!moved) return false;
        ++(vertex ? m_vertexEdits : m_pixelEdits);
        m_entries += moved;
        return true;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] decal matrices (this frame): box-to-clip copies %u, SV_Position-to-decal copies %u, entries"
                   " moved %u, without a camera or target %u\n", frame, m_vertexEdits, m_pixelEdits, m_entries, m_noCamera);
        m_vertexEdits = m_pixelEdits = m_entries = m_noCamera = 0;
    }

private:
    unsigned m_vertexEdits = 0, m_pixelEdits = 0, m_entries = 0, m_noCamera = 0;
};

} // namespace

std::unique_ptr<ProfileFeature> MakeDecalMatrices(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value") == "0") return nullptr;
    return std::unique_ptr<ProfileFeature>(new DecalMatricesFeature());
}

} // namespace wiz3d
