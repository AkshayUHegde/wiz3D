/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <HudSceneDepth Crosshair="r" Band="y0 y1" NearQuads="k"/>: a HUD piece, or a camera-facing quad parked within
// k near planes (a world marker, a corona), takes the parallax of the left eye's scene depth at its centre.

#include "StdAfx.h"
#include <stdlib.h>
#include <chrono>
#include <future>
#include <map>
#include <sstream>
#include <string>
#include <d3dcompiler.h>
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Device11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog

namespace wiz3d
{
namespace
{

// The shader view of a depth target's typeless format; UNKNOWN = not readable.
DXGI_FORMAT DepthViewFormat(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_R24G8_TYPELESS:    return DXGI_FORMAT_R24_UNORM_X8_TYPELESS;
    case DXGI_FORMAT_R32_TYPELESS:      return DXGI_FORMAT_R32_FLOAT;
    case DXGI_FORMAT_R16_TYPELESS:      return DXGI_FORMAT_R16_UNORM;
    case DXGI_FORMAT_R32G8X24_TYPELESS: return DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS;
    default:                            return DXGI_FORMAT_UNKNOWN;
    }
}

// A geometry shader compiled on a worker (bytecode, result, errors); no device call happens there.
struct Compiled
{
    ID3DBlob*   code = nullptr;
    HRESULT     hr = E_FAIL;
    std::string err;
};

// One output layout's shader: its compile, then the shader created on the render thread.
struct Layout
{
    std::shared_future<Compiled> job;
    bool                  created = false;
    HRESULT               hr = E_FAIL;
    ID3D11GeometryShader* gs = nullptr;
};

class HudSceneDepthFeature : public ProfileFeature
{
public:
    HudSceneDepthFeature(float crosshair, const float rect[4], unsigned minScene, float nearQuads)
        : m_crosshair(crosshair), m_minScene(minScene), m_nearQuads(nearQuads)
    { for (int i = 0; i < 4; ++i) m_rect[i] = rect[i]; }
    ~HudSceneDepthFeature()
    {
        for (auto& kv : m_bySource)
        {
            Layout& l = kv.second;
            if (!l.created && l.job.valid()) { const Compiled& c = l.job.get(); if (c.code) c.code->Release(); }   // waits
            if (l.gs) l.gs->Release();
        }
        if (m_cb) m_cb->Release();
        if (m_srv) m_srv->Release();
        if (m_copy) m_copy->Release();
        if (m_scene) m_scene->Release();
    }

    // The scene's depth is the last left depth target of the presented size: the HUD itself binds none.
    void OnTargetsSet(Context11Proxy& ctx) override
    {
        ID3D11DepthStencilView* dsv = ctx.LeftDepth();
        Device11Proxy* dev = ctx.Parent();
        if (!dsv || !dev || !dev->GetPresentWidth()) return;
        ID3D11Resource* res = nullptr;
        dsv->GetResource(&res);
        ID3D11Texture2D* tex = nullptr;
        if (res) { res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex); res->Release(); }
        if (!tex) return;
        D3D11_TEXTURE2D_DESC d = {};
        tex->GetDesc(&d);
        if (d.Width == dev->GetPresentWidth() && d.Height == dev->GetPresentHeight() && d.SampleDesc.Count == 1 &&
            DepthViewFormat(d.Format) != DXGI_FORMAT_UNKNOWN)
        {
            m_sceneFrame = m_frame;   // a scene was drawn this frame (a menu without one leaves the HUD flat)
            if (tex != m_scene)
            {
                if (m_scene) m_scene->Release();
                m_scene = tex;   // the reference moves to m_scene
                tex = nullptr;
            }
        }
        if (tex) tex->Release();
    }

    // Menus draw a few scene passes under the HUD, a game frame thousands: the scene draws so far tell them apart.
    void OnLeftDraw(Context11Proxy& ctx) override { if (!ctx.OverlayPass()) ++m_sceneDraws; }

    void OnRightDraw(Context11Proxy& ctx) override
    {
        m_bound = false;
        const bool overlay = ctx.OverlayPass();
        const ShaderAnalysis11Result* vs = ctx.BoundShader(false);
        if (overlay)
        {
            ++m_hudDraws;
            if (m_sceneDraws < m_minScene) { ++m_menuDraws; return; }
        }
        else
        {
            // A one-quad draw of a camera-matrix shader; the geometry shader tests each triangle for the near plane.
            if (!(m_nearQuads > 0.f) || ctx.DrawVertices() > 6 || !vs || !vs->parsed || vs->projection.matrixData.cb.empty())
                return;
            ++m_quadDraws;
        }
        const D3D11_PRIMITIVE_TOPOLOGY topo = ctx.Topology();
        float m22 = 0.f, m23 = 0.f;
        const float p00 = stereo::SceneP00();
        if (!vs || !m_scene || m_sceneFrame != m_frame || ctx.GeometryStageBound() || !(p00 > 0.f) ||
            !stereo::SceneZRow(m22, m23) ||
            (topo != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST && topo != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP))
        { ++(overlay ? m_skipped : m_quadSkipped); return; }
        // The near plane, where z = 0 (z = 1 reversed): z = m22 + m23 / w.
        const float zn = m23 < 0.f ? -m23 / m22 : m23 / (1.f - m22);
        if (!overlay && !(zn > 0.f && zn < 1e30f)) { ++m_quadSkipped; return; }
        ID3D11GeometryShader* gs = Shader(ctx, *vs);
        if (!gs || !DepthCopy(ctx)) { ++(overlay ? m_skipped : m_quadSkipped); return; }
        // shift(z) = b + m41 / w with w = m23 / (z - m22): linear in the stored depth z, bounded by 0.25 units and infinity.
        const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign();
        const float b = eyeShift * p00, m41 = -b * stereo::Convergence();
        const float nearShift = b + m41 / 0.25f;
        // A near quad's right vertices carry the camera's own clip-space shift b * w + m41, taken off before its new one.
        const float k[20] = { b - m41 * m22 / m23, m41 / m23, (std::min)(nearShift, b), (std::max)(nearShift, b),
                              (float)m_w, (float)m_h, b < 0.f ? -1.f : 1.f, kFullScreen,
                              m_crosshair, (float)m_w / (float)m_h, overlay ? 0.f : 1.f, 0.f,
                              m_rect[0], m_rect[1], m_rect[2], m_rect[3],
                              b, m41, zn * 0.99f, zn * m_nearQuads };
        D3D11_MAPPED_SUBRESOURCE map = {};
        if (FAILED(ctx.Real()->Map(m_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &map)) || !map.pData) { ++m_skipped; return; }
        memcpy(map.pData, k, sizeof k);
        ctx.Real()->Unmap(m_cb, 0);
        ctx.Real()->GSGetConstantBuffers(0, 1, &m_savedCB);
        ctx.Real()->GSGetShaderResources(0, 1, &m_savedSRV);
        ctx.Real()->GSSetShader(gs, nullptr, 0);
        ctx.Real()->GSSetConstantBuffers(0, 1, &m_cb);
        ctx.Real()->GSSetShaderResources(0, 1, &m_srv);
        m_bound = true;
        ++(overlay ? m_shifted : m_quadTested);
    }

    // The game had no geometry shader bound: it goes back to none, its slot 0 bindings as they were.
    void OnRightDrawDone(Context11Proxy& ctx) override
    {
        if (!m_bound) return;
        m_bound = false;
        ctx.Real()->GSSetShader(nullptr, nullptr, 0);
        ctx.Real()->GSSetConstantBuffers(0, 1, &m_savedCB);
        ctx.Real()->GSSetShaderResources(0, 1, &m_savedSRV);
        if (m_savedCB) { m_savedCB->Release(); m_savedCB = nullptr; }
        if (m_savedSRV) { m_savedSRV->Release(); m_savedSRV = nullptr; }
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        m_frame = frame;
        if (report)
            DDILog("  [frame %u] hud scene depth (this frame): HUD draws %u, given the scene depth %u, flat in a menu %u (scene"
                   " draws %u), left as drawn %u (no scene depth, camera or usable shader) | one-quad camera draws %u, tested"
                   " for the near plane %u, left as drawn %u\n", frame, m_hudDraws, m_shifted, m_menuDraws, m_sceneDraws,
                   m_skipped, m_quadDraws, m_quadTested, m_quadSkipped);
        m_hudDraws = m_shifted = m_skipped = m_menuDraws = m_sceneDraws = 0;
        m_quadDraws = m_quadTested = m_quadSkipped = 0;
    }

private:
    float    m_crosshair, m_rect[4];              // radius, band (NDC x0 x1 y0 y1, x unbounded)
    // A HUD piece this wide AND tall (NDC) covers the screen: the scene's own composite pass, which must not move.
    static constexpr float kFullScreen = 1.9f;
    unsigned m_minScene;                          // fewer scene draws before the HUD = a menu, left flat
    float    m_nearQuads;                         // near-plane multiples a quad may sit within, 0 = off
    unsigned m_quadDraws = 0, m_quadTested = 0, m_quadSkipped = 0;
    unsigned m_sceneFrame = ~0u;                  // the frame the scene depth was last bound
    unsigned m_sceneDraws = 0, m_menuDraws = 0;
    std::map<DWORD, ID3D11GeometryShader*> m_gs;   // per HUD vertex shader, null = unbuildable (m_bySource holds them)
    std::map<std::string, Layout> m_bySource;      // per generated source
    ID3D11Buffer*             m_cb = nullptr;
    ID3D11Texture2D*          m_scene = nullptr;   // the game's left scene depth (held)
    ID3D11Texture2D*          m_copy = nullptr;    // its copy the geometry shader reads
    ID3D11ShaderResourceView* m_srv = nullptr;
    UINT        m_w = 0, m_h = 0;
    DXGI_FORMAT m_fmt = DXGI_FORMAT_UNKNOWN;
    unsigned    m_frame = 0, m_copiedFrame = ~0u;
    ID3D11Buffer*             m_savedCB = nullptr;
    ID3D11ShaderResourceView* m_savedSRV = nullptr;
    bool        m_bound = false;
    unsigned    m_hudDraws = 0, m_shifted = 0, m_skipped = 0;

    // The scene depth copied once a frame, at its first HUD draw (after every scene pass wrote it).
    bool DepthCopy(Context11Proxy& ctx)
    {
        ID3D11Device* dev = ctx.Parent()->GetReal();
        D3D11_TEXTURE2D_DESC d = {};
        m_scene->GetDesc(&d);
        if (!m_copy || m_w != d.Width || m_h != d.Height || m_fmt != d.Format)
        {
            if (m_srv) { m_srv->Release(); m_srv = nullptr; }
            if (m_copy) { m_copy->Release(); m_copy = nullptr; }
            D3D11_TEXTURE2D_DESC cd = d;
            cd.BindFlags = D3D11_BIND_SHADER_RESOURCE; cd.MiscFlags = 0; cd.CPUAccessFlags = 0; cd.Usage = D3D11_USAGE_DEFAULT;
            HRESULT hr = dev->CreateTexture2D(&cd, nullptr, &m_copy);
            if (SUCCEEDED(hr))
            {
                D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
                sd.Format = DepthViewFormat(d.Format); sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                sd.Texture2D.MostDetailedMip = 0; sd.Texture2D.MipLevels = 1;
                hr = dev->CreateShaderResourceView(m_copy, &sd, &m_srv);
                if (FAILED(hr)) { m_copy->Release(); m_copy = nullptr; m_srv = nullptr; }
            }
            m_w = d.Width; m_h = d.Height; m_fmt = d.Format;
            m_copiedFrame = ~0u;
            FeatureLog("  HudSceneDepth: scene depth %ux%u fmt %u -> copy hr=0x%08lX\n", d.Width, d.Height, (unsigned)d.Format,
                       (unsigned long)hr);
        }
        if (!m_cb)
        {
            D3D11_BUFFER_DESC bd = {};
            bd.ByteWidth = 80; bd.Usage = D3D11_USAGE_DYNAMIC; bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (FAILED(dev->CreateBuffer(&bd, nullptr, &m_cb))) m_cb = nullptr;
        }
        if (!m_srv || !m_cb) return false;
        if (m_copiedFrame != m_frame) { ctx.Real()->CopyResource(m_copy, m_scene); m_copiedFrame = m_frame; }
        return true;
    }

    // The geometry shader over a HUD vertex shader's outputs as declared; null when one is a system value beside the position.
    ID3D11GeometryShader* Shader(Context11Proxy& ctx, const ShaderAnalysis11Result& vs)
    {
        auto it = m_gs.find(vs.crc32);
        if (it != m_gs.end()) return it->second;
        std::string src = "cbuffer wiz3D_hud : register(b0) { float4 wiz3D_k; float4 wiz3D_size; float4 wiz3D_sel; float4 wiz3D_rect;"
                          " float4 wiz3D_near; };\n"
                          "Texture2D<float> wiz3D_depth : register(t0);\nstruct V {\n";
        int pos = -1;
        for (size_t i = 0; i < vs.outputs.size(); ++i)
        {
            const ShaderAnalysis11Result::OutputElement& e = vs.outputs[i];
            if (e.sysValue == 1) pos = (int)i;
            else if (e.sysValue != 0) return m_gs[vs.crc32] = nullptr;
            unsigned n = 0;
            for (unsigned m = e.mask; m; m >>= 1) n += m & 1;
            if (!n) n = 4;
            char line[192];
            _snprintf_s(line, sizeof line, _TRUNCATE, "    %s%u m%u : %s%u;\n",
                        e.type == 1 ? "uint" : e.type == 2 ? "int" : "float", n, (unsigned)i, e.name.c_str(), e.index);
            src += line;
        }
        if (pos < 0) return m_gs[vs.crc32] = nullptr;
        // A piece's centre and extent are its longest edge's in pixels (a quad's diagonal, shared by both triangles). A HUD
        // piece covering the screen (the scene's own composite) stays as drawn; the nearest of five taps is the nearest.
        char body[4096];
        _snprintf_s(body, sizeof body, _TRUNCATE,
            "};\n#define wiz3D_POS m%d\n"
            "float wiz3D_shift(float2 ndc)\n{\n"
            "    int2 px = clamp(int2((ndc.x * 0.5 + 0.5) * wiz3D_size.x, (0.5 - ndc.y * 0.5) * wiz3D_size.y), int2(0, 0), int2(wiz3D_size.xy) - 1);\n"
            "    return clamp(wiz3D_k.x + wiz3D_k.y * wiz3D_depth.Load(int3(px, 0)), wiz3D_k.z, wiz3D_k.w);\n}\n"
            "float wiz3D_nearest(float2 c)\n{\n"
            "    const float d = 0.004, g = wiz3D_size.z;\n"
            "    float n0 = g * wiz3D_shift(c), n1 = g * wiz3D_shift(c + float2(d, 0)), n2 = g * wiz3D_shift(c - float2(d, 0));\n"
            "    float n3 = g * wiz3D_shift(c + float2(0, d)), n4 = g * wiz3D_shift(c - float2(0, d));\n"
            "    return g * min(min(min(n0, n1), min(n2, n3)), n4);\n}\n"
            "[maxvertexcount(3)]\nvoid main(triangle V i[3], inout TriangleStream<V> o)\n{\n"
            "    float3 w = float3(i[0].wiz3D_POS.w, i[1].wiz3D_POS.w, i[2].wiz3D_POS.w), u = 0;\n"
            "    bool use = true;\n"
            "    if (wiz3D_sel.z > 0)\n    {\n"
            "        float wl = min(min(w.x, w.y), w.z), wh = max(max(w.x, w.y), w.z);\n"
            "        use = wl >= wiz3D_near.z && wh <= wiz3D_near.w && wh - wl <= 0.01 * wh;\n"
            "        u = wiz3D_near.x * w + wiz3D_near.y;\n    }\n"
            "    float2 p[3] = { float2(i[0].wiz3D_POS.x - u.x, i[0].wiz3D_POS.y) / w.x, float2(i[1].wiz3D_POS.x - u.y, i[1].wiz3D_POS.y) / w.y,\n"
            "                    float2(i[2].wiz3D_POS.x - u.z, i[2].wiz3D_POS.y) / w.z };\n"
            "    float2 a = float2(wiz3D_sel.y, 1), e0 = p[0], e1 = p[1];\n"
            "    float l0 = length((p[1] - p[0]) * a), l1 = length((p[2] - p[1]) * a), l2 = length((p[0] - p[2]) * a);\n"
            "    if (l1 >= l0 && l1 >= l2) { e0 = p[1]; e1 = p[2]; } else if (l2 >= l0) { e0 = p[2]; e1 = p[0]; }\n"
            "    float2 c = (e0 + e1) * 0.5, ext = abs(e1 - e0);\n"
            "    float s = 0;\n"
            "    if (!use || (wiz3D_sel.z == 0 && min(ext.x, ext.y) >= wiz3D_size.w)) u = 0;\n"
            "    else if (wiz3D_sel.z > 0) s = wiz3D_nearest(c);\n"
            "    else if (wiz3D_sel.x > 0 && length(c * a) < wiz3D_sel.x) s = wiz3D_shift(float2(0, 0));\n"
            "    else if (c.x > wiz3D_rect.x && c.x < wiz3D_rect.y && c.y > wiz3D_rect.z && c.y < wiz3D_rect.w) s = wiz3D_nearest(c);\n"
            "    [unroll] for (int n = 0; n < 3; ++n)\n    {\n        V v = i[n];\n"
            "        v.wiz3D_POS.x += s * v.wiz3D_POS.w - u[n];\n        o.Append(v);\n    }\n}\n",
            pos);
        src += body;
        // The source depends only on the vertex shader's outputs, which many materials share: one build per layout, compiled
        // on a worker; the shader is created here, as the game's device may be single-threaded.
        ID3D11Device* dev = ctx.Parent()->GetReal();
        auto b = m_bySource.find(src);
        if (b == m_bySource.end())
        {
            const std::string target = dev->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0 ? "gs_5_0" : "gs_4_0";
            b = m_bySource.emplace(src, Layout()).first;
            b->second.job = std::async(std::launch::async, [src, target]() {
                    Compiled r;
                    ID3DBlob* err = nullptr;
                    r.hr = D3DCompile(src.c_str(), src.size(), "wiz3D_hud", nullptr, nullptr, "main", target.c_str(),
                                      D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &r.code, &err);
                    if (err) { r.err = (const char*)err->GetBufferPointer(); err->Release(); }
                    return r;
                }).share();
        }
        Layout& l = b->second;
        if (!l.created)
        {
            // Still compiling: this draw stays as drawn.
            if (l.job.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return nullptr;
            const Compiled& c = l.job.get();
            l.created = true;
            l.hr = c.hr;
            if (SUCCEEDED(l.hr) && c.code)
                l.hr = dev->CreateGeometryShader(c.code->GetBufferPointer(), c.code->GetBufferSize(), nullptr, &l.gs);
            if (FAILED(l.hr) && l.gs) { l.gs->Release(); l.gs = nullptr; }
            if (c.code) c.code->Release();
            FeatureLog("  HudSceneDepth: vs %08lX, %u outputs -> geometry shader hr=0x%08lX%s%s\n", vs.crc32,
                       (unsigned)vs.outputs.size(), (unsigned long)l.hr, c.err.empty() ? "" : ": ", c.err.c_str());
        }
        return m_gs[vs.crc32] = l.gs;
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeHudSceneDepth(const ProfileEntry* e)
{
    if (!e) return nullptr;
    const float crosshair = (float)atof(EntryValue(*e, "Crosshair").c_str());
    // Band: "y0 y1" in NDC, or "y" half-height about the centre; absent = none. Full width: a side edge would cut a line of
    // text, a top or bottom edge never does.
    std::istringstream in(EntryValue(*e, "Band"));
    float v[2] = {}, rect[4] = { -1e30f, 1e30f, 0.f, 0.f };
    int n = 0;
    while (n < 2 && (in >> v[n])) ++n;
    if (n == 2) { rect[2] = v[0]; rect[3] = v[1]; }
    else if (n == 1) { rect[2] = -v[0]; rect[3] = v[0]; }
    const bool region = rect[3] > rect[2];
    const float nearQuads = (float)atof(EntryValue(*e, "NearQuads").c_str());
    if (crosshair <= 0.f && !region && !(nearQuads > 0.f)) return nullptr;
    const std::string minScene = EntryValue(*e, "MinSceneDraws");
    return std::unique_ptr<ProfileFeature>(new HudSceneDepthFeature(crosshair, rect,
        minScene.empty() ? 200u : (unsigned)atoi(minScene.c_str()), nearQuads));
}

} // namespace wiz3d
