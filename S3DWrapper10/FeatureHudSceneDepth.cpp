/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <HudSceneDepth Band="..."/>: a HUD piece, or a camera-facing quad near the camera, takes the parallax of the scene depth
// under it; child <Flat>, <GameDepth> and <SceneDepth> elements (first match) say which pieces stay or bring their own depth.

#include "StdAfx.h"
#include <stdlib.h>
#include <chrono>
#include <future>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <d3dcompiler.h>
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Device11Proxy.h"
#include "Buffer11Proxy.h"
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

// DepthCopy="any": a typed depth format's typeless family, which a copy of it can be made in and read from.
DXGI_FORMAT DepthTypeless(DXGI_FORMAT f)
{
    switch (f)
    {
    case DXGI_FORMAT_D24_UNORM_S8_UINT:    return DXGI_FORMAT_R24G8_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT:            return DXGI_FORMAT_R32_TYPELESS;
    case DXGI_FORMAT_D16_UNORM:            return DXGI_FORMAT_R16_TYPELESS;
    case DXGI_FORMAT_D32_FLOAT_S8X24_UINT: return DXGI_FORMAT_R32G8X24_TYPELESS;
    default:                               return f;
    }
}

const char* const kHudTag = "wiz3D.hud";   // ShaderAnalysis11Result::tags: a vertex shader declaring a HudConstants cbuffer
const char* const kAnchorTag = "wiz3D.anchor ";   // + "slot offset slot offset" of its Anchor matrices
const std::string kTextureTag = "wiz3D.tex ";     // + "name slot": a pixel shader's texture a rule names
enum { kSearched, kFlat, kGiven };                // a HUD piece: its depth searched, as drawn, or the game's own

// One child element of <HudSceneDepth>: <Flat>, <GameDepth> or <SceneDepth> and the conditions a HUD piece must meet.
struct Rule
{
    int         kind = kSearched;
    std::string name;                          // the element as written, for the log
    std::string texture;                       // Texture: the pixel shader's texture of that name...
    bool        boundTarget = false;           // ...Bound="RenderTarget": with a render target bound there
    DWORD       vs = 0, ps = 0;                // VertexShader, PixelShader: CRCs
    bool        tilted = false;                // Tilted="1"
    float       zIn[2] = { 1.f, -1.f };        // ZInside="a b" (none while a > b)
    float       zOut[2] = { 1.f, -1.f };       // ZOutside="a b"
};

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
    HudSceneDepthFeature(float crosshair, const float rect[4], unsigned minScene, float nearQuads, float menuLayers,
                         const std::vector<std::string>& hudConstants, bool anyDepth, float below,
                         const std::vector<std::string>& anchor, float anchorGroup, const std::vector<Rule>& rules)
        : m_crosshair(crosshair), m_minScene(minScene), m_nearQuads(nearQuads), m_menuLayers(menuLayers),
          m_hudConstants(hudConstants), m_anyDepth(anyDepth), m_below(below), m_anchor(anchor), m_group(anchorGroup),
          m_rules(rules), m_ruleHits(rules.size(), 0u)
    { for (int i = 0; i < 4; ++i) m_rect[i] = rect[i]; }

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        // A texture a rule names: its slot in this shader, if it declares it.
        for (const Rule& r : m_rules)
        {
            const int slot = r.texture.empty() ? -1 : rdef.TextureSlot(r.texture);
            if (slot >= 0) info.tags.push_back(kTextureTag + r.texture + " " + std::to_string(slot));
        }
        for (const std::string& n : m_hudConstants)
            if (rdef.CBufferSlot(n) >= 0)
            {
                info.tags.push_back(kHudTag);
                int ws = -1, vs = -1;
                UINT wo = 0, vo = 0;
                if (m_anchor.size() == 2 && rdef.Variable(m_anchor[0], true, ws, wo) && rdef.Variable(m_anchor[1], true, vs, vo))
                {
                    char tag[96];
                    _snprintf_s(tag, sizeof tag, _TRUNCATE, "%s%d %u %d %u", kAnchorTag, ws, wo, vs, vo);
                    info.tags.push_back(tag);
                }
                return;
            }
    }
    ~HudSceneDepthFeature()
    {
        for (auto& kv : m_bySource)
        {
            Layout& l = kv.second;
            if (!l.created && l.job.valid()) { const Compiled& c = l.job.get(); if (c.code) c.code->Release(); }   // waits
            if (l.gs) l.gs->Release();
        }
        if (m_cb) m_cb->Release();
        if (!m_csDone && m_csJob.valid()) { const Compiled& c = m_csJob.get(); if (c.code) c.code->Release(); }   // waits
        if (m_cs) m_cs->Release();
        if (m_trackCB) m_trackCB->Release();
        if (m_smoothUAV) m_smoothUAV->Release();
        if (m_smoothSRV) m_smoothSRV->Release();
        if (m_smooth) m_smooth->Release();
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
        const bool readable = m_anyDepth ? DepthViewFormat(DepthTypeless(d.Format)) != DXGI_FORMAT_UNKNOWN
                                         : d.SampleDesc.Count == 1 && DepthViewFormat(d.Format) != DXGI_FORMAT_UNKNOWN;
        if (d.Width == dev->GetPresentWidth() && d.Height == dev->GetPresentHeight() && readable)
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
    void OnLeftDraw(Context11Proxy& ctx) override
    {
        if (!ctx.OverlayPass() && !HudShader(ctx)) { ++m_sceneDraws; m_groupOpen = false; }   // a scene draw ends an element
    }

    void OnRightDraw(Context11Proxy& ctx) override
    {
        m_bound = false;
        const bool overlay = ctx.OverlayPass() || HudShader(ctx);
        const ShaderAnalysis11Result* vs = ctx.BoundShader(false);
        bool menuLayer = false;
        // Anchor + UiProjectionParallax SceneNear: a frame whose scene camera is another camera's (a pause map) is that
        // view's own screen, not a menu: its HUD (the map's icons) takes its depth.
        const bool otherView = overlay && !m_anchor.empty() && UiProjectionReferenceNear() > 0.f &&
                               stereo::UiParallaxScale(stereo::SceneP00()) != 1.f;
        if (overlay)
        {
            if (m_crosshair <= 0.f && !(m_rect[3] > m_rect[2])) return;   // no crosshair or band: the HUD stays as drawn
            ++m_hudDraws;
            if (m_sceneDraws < m_minScene && !otherView) { ++m_menuDraws; return; }
        }
        else
        {
            // A one-quad draw of a camera-matrix shader; the geometry shader tests each triangle for the near plane. In a menu
            // (last frame drew fewer scene draws than MinSceneDraws) such a quad is the menu's own layer (NieR's title logo).
            menuLayer = m_menuLayers > 0.f && m_lastSceneDraws < m_minScene;
            if (!(menuLayer || m_nearQuads > 0.f) || ctx.DrawVertices() > 6 || !vs || !vs->parsed ||
                vs->projection.matrixData.cb.empty())
                return;
            ++m_quadDraws;
        }
        const D3D11_PRIMITIVE_TOPOLOGY topo = ctx.Topology();
        float m22 = 0.f, m23 = 0.f;
        // A menu's own 3D view takes the gameplay camera's parallax (UiProjectionParallax SceneNear), so its layers do.
        const float p00 = stereo::SceneP00() * stereo::UiParallaxScale(stereo::SceneP00());
        if (!vs || (!menuLayer && (!m_scene || m_sceneFrame != m_frame)) || ctx.GeometryStageBound() || !(p00 > 0.f) ||
            !stereo::SceneZRow(m22, m23) ||
            (topo != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST && topo != D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP))
        { ++(overlay ? m_skipped : m_quadSkipped); return; }
        // The near plane, where z = 0 (z = 1 reversed): z = m22 + m23 / w.
        const float zn = m23 < 0.f ? -m23 / m22 : m23 / (1.f - m22);
        if (!overlay && !(zn > 0.f && zn < 1e30f)) { ++m_quadSkipped; return; }
        // UiProjectionParallax SceneNear: a menu layer lies within k of the gameplay camera's near planes, not of the drawing
        // camera's (a map camera's near plane lies far out: its billboard markers sit on the map, not on the screen).
        const float znRef = UiProjectionReferenceNear();
        const float zl = menuLayer && znRef > 0.f ? znRef : zn;
        // Anchor: origin and reach; the tracked slot and whether it is ready; the game's own depth (z) when it gives one.
        float anchor[8] = {}, given = 0.f;
        int kind = kSearched;
        if (overlay && (!m_anchor.empty() || !m_rules.empty()))
        {
            kind = Anchor(ctx, *vs, anchor, given);
            if (kind == kFlat) { ++m_flat; return; }   // a <Flat> rule: as drawn
        }
        ID3D11GeometryShader* gs = Shader(ctx, *vs);
        // A menu layer only loses the camera's shift: no scene depth is read.
        if (!gs || !(menuLayer ? ConstantBuffer(ctx) : DepthCopy(ctx))) { ++(overlay ? m_skipped : m_quadSkipped); return; }
        // shift(z) = b + m41 / w with w = m23 / (z - m22): linear in the stored depth z, bounded by 0.25 units and infinity.
        const float eyeShift = wiz3D_GetEffectiveEyeShift() * ctx.EyeShiftSign();
        const float b = eyeShift * p00, m41 = -b * stereo::Convergence();
        const float nearShift = b + m41 / 0.25f;
        // A near quad's right vertices carry the camera's own clip-space shift b * w + m41, taken off before its new one.
        // A depth the game gives is a stored depth like the scene's: the same linear map, bounded alike.
        if (kind == kGiven)
        {
            anchor[6] = (std::min)((std::max)(b - m41 * m22 / m23 + m41 / m23 * given, (std::min)(nearShift, b)),
                                   (std::max)(nearShift, b));
            anchor[7] = 1.f;
            ++m_given;
        }
        const float k[28] = { b - m41 * m22 / m23, m41 / m23, (std::min)(nearShift, b), (std::max)(nearShift, b),
                              (float)m_w, (float)m_h, b < 0.f ? -1.f : 1.f, kFullScreen,
                              m_crosshair, (float)m_w / (float)m_h, overlay ? 0.f : 1.f,
                              otherView ? -1.f : menuLayer ? 1.f : 0.f,
                              m_rect[0], m_rect[1], m_rect[2], m_rect[3],
                              b, m41, zl * 0.99f, zl * (menuLayer ? m_menuLayers : m_nearQuads),
                              anchor[0], anchor[1], anchor[2], anchor[3], anchor[4], anchor[5], anchor[6], anchor[7] };
        // The frame's depth-to-shift terms, for the tracked elements' pass at the frame's end.
        if (overlay) for (int i = 0; i < 4; ++i) { m_trackK[i] = k[i]; m_trackB = b; }
        D3D11_MAPPED_SUBRESOURCE map = {};
        if (FAILED(ctx.Real()->Map(m_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &map)) || !map.pData) { ++m_skipped; return; }
        memcpy(map.pData, k, sizeof k);
        ctx.Real()->Unmap(m_cb, 0);
        if (ctx.Recording())
        {
            // BatchRightEye: the recorder reads nothing back; the proxy knows what the right-eye draw bound there.
            if ((m_savedCB = ctx.GSConstantBufferNow(0)) != nullptr) m_savedCB->AddRef();
            if ((m_savedSRV = ctx.GSResourceNow(0)) != nullptr) m_savedSRV->AddRef();
            if (m_smoothSRV && (m_savedSRV1 = ctx.GSResourceNow(1)) != nullptr) m_savedSRV1->AddRef();
        }
        else
        {
            ctx.Real()->GSGetConstantBuffers(0, 1, &m_savedCB);
            ctx.Real()->GSGetShaderResources(0, 1, &m_savedSRV);
            if (m_smoothSRV) ctx.Real()->GSGetShaderResources(1, 1, &m_savedSRV1);
        }
        ctx.Real()->GSSetShader(gs, nullptr, 0);
        ctx.Real()->GSSetConstantBuffers(0, 1, &m_cb);
        ctx.Real()->GSSetShaderResources(0, 1, &m_srv);
        if (m_smoothSRV) ctx.Real()->GSSetShaderResources(1, 1, &m_smoothSRV);
        m_boundSmooth = m_smoothSRV != nullptr;
        m_bound = true;
        ++(overlay ? m_shifted : menuLayer ? m_menuLayerTested : m_quadTested);
    }

    // The game had no geometry shader bound: it goes back to none, its slot 0 bindings as they were.
    void OnRightDrawDone(Context11Proxy& ctx) override
    {
        if (!m_bound) return;
        m_bound = false;
        ctx.Real()->GSSetShader(nullptr, nullptr, 0);
        ctx.Real()->GSSetConstantBuffers(0, 1, &m_savedCB);
        ctx.Real()->GSSetShaderResources(0, 1, &m_savedSRV);
        if (m_boundSmooth) ctx.Real()->GSSetShaderResources(1, 1, &m_savedSRV1);
        m_boundSmooth = false;
        if (m_savedCB) { m_savedCB->Release(); m_savedCB = nullptr; }
        if (m_savedSRV) { m_savedSRV->Release(); m_savedSRV = nullptr; }
        if (m_savedSRV1) { m_savedSRV1->Release(); m_savedSRV1 = nullptr; }
    }

    void OnFrameEnd(Context11Proxy& ctx, unsigned frame, bool report) override
    {
        if (!m_anchor.empty()) Track(ctx);
        m_frame = frame;
        if (report)
            DDILog("  [frame %u] hud scene depth (this frame): HUD draws %u, given the scene depth %u, flat in a menu %u (scene"
                   " draws %u), left as drawn %u (no scene depth, camera or usable shader) | one-quad camera draws %u, tested"
                   " for the near plane %u, as menu layers %u, left as drawn %u\n", frame, m_hudDraws, m_shifted, m_menuDraws,
                   m_sceneDraws, m_skipped, m_quadDraws, m_quadTested, m_menuLayerTested, m_quadSkipped);
        if (report && !m_anchor.empty())
            DDILog("  [frame %u] hud anchors (this frame): HUD draws at their element's origin %u, elements %u (tracked from"
                   " last frame %u, new %u), tracking pass %s | at the game's own depth %u, as drawn %u\n", frame, m_anchored,
                   m_elements, m_tracked, m_newTracks, m_trackRan ? "ran" : m_cs ? "idle" : "not built", m_given, m_flat);
        if (report && !m_rules.empty())
        {
            std::string hits;
            for (size_t i = 0; i < m_rules.size(); ++i)
                hits += (i ? ", " : "") + m_rules[i].name + " " + std::to_string(m_ruleHits[i]);
            DDILog("  [frame %u] hud rules (this frame, HUD pieces each took): %s\n", frame, hits.c_str());
        }
        for (unsigned& h : m_ruleHits) h = 0;
        m_tracked = m_newTracks = 0;
        m_flat = m_given = 0;
        m_trackRan = false;
        m_lastSceneDraws = m_sceneDraws;
        m_groupOpen = false;
        m_anchored = m_elements = 0;
        m_hudDraws = m_shifted = m_skipped = m_menuDraws = m_sceneDraws = 0;
        m_quadDraws = m_quadTested = m_quadSkipped = m_menuLayerTested = 0;
    }

private:
    float    m_crosshair, m_rect[4];              // radius, band (NDC x0 x1 y0 y1, x unbounded)
    // A HUD piece this wide AND tall (NDC) covers the screen: the scene's own composite pass, which must not move.
    static constexpr float kFullScreen = 1.9f;
    unsigned m_minScene;                          // fewer scene draws before the HUD = a menu, left flat
    float    m_nearQuads;                         // near-plane multiples a quad may sit within, 0 = off
    float    m_menuLayers;                        // the same for a menu's own layers, which sit at the screen; 0 = off
    std::vector<std::string> m_hudConstants;      // HudConstants: cbuffers whose vertex shaders draw HUD pieces
    bool     m_anyDepth;                          // DepthCopy="any": a typed or multisampled scene depth is read too
    float    m_below;                             // Below: piece heights the depth search reaches down, 0 = the centre
    std::vector<std::string> m_anchor;            // Anchor: world matrix and HUD view-projection names, empty = per triangle
    float    m_group;                             // AnchorGroup: NDC reach of one element's origins
    std::vector<Rule> m_rules;                    // the child elements, first match decides
    std::vector<unsigned> m_ruleHits;             // HUD pieces each rule took (this frame)
    unsigned m_flat = 0, m_given = 0;
    bool     m_groupOpen = false;
    float    m_groupAt[2] = {};                   // the open element's origin (NDC)
    float    m_groupReach = 0.f;                  // its reach, AnchorGroup times its scale
    float    m_groupTrack[2] = {};                // its tracked slot and whether that slot holds last frame's depth
    // Tracking: each element keeps a slot of wiz3D_smooth across frames (matched by origin); a compute pass at the frame's
    // end searches its depth, keeps the depth it had while that is still under it, and eases toward a new one.
    static const int kSlots = 64;
    struct Element { float x, y, rx, ry; int slot; bool fresh, claimed; };
    std::vector<Element> m_cur, m_prev;
    ID3D11Buffer*              m_smooth = nullptr;
    ID3D11ShaderResourceView*  m_smoothSRV = nullptr;
    ID3D11UnorderedAccessView* m_smoothUAV = nullptr;
    ID3D11Buffer*              m_trackCB = nullptr;
    ID3D11ComputeShader*       m_cs = nullptr;
    std::shared_future<Compiled> m_csJob;
    bool     m_csMs = false, m_csDone = false;
    float    m_trackK[4] = {}, m_trackB = 0.f;
    unsigned m_tracked = 0, m_newTracks = 0;
    bool     m_trackRan = false, m_boundSmooth = false, m_trackFailed = false;
    ID3D11ShaderResourceView* m_savedSRV1 = nullptr;
    unsigned m_anchored = 0, m_elements = 0;
    unsigned m_quadDraws = 0, m_quadTested = 0, m_quadSkipped = 0, m_menuLayerTested = 0;
    unsigned m_sceneFrame = ~0u;                  // the frame the scene depth was last bound
    unsigned m_sceneDraws = 0, m_menuDraws = 0;
    unsigned m_lastSceneDraws = ~0u;              // the previous frame's scene draws (unknown = not a menu)
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
            // DepthCopy="any": the copy is the typeless family of a typed depth, multisampled as the scene's is.
            if (m_anyDepth) cd.Format = DepthTypeless(d.Format);
            HRESULT hr = dev->CreateTexture2D(&cd, nullptr, &m_copy);
            if (SUCCEEDED(hr))
            {
                D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
                sd.Format = DepthViewFormat(cd.Format); sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
                sd.Texture2D.MostDetailedMip = 0; sd.Texture2D.MipLevels = 1;
                if (cd.SampleDesc.Count > 1) sd.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2DMS;
                hr = dev->CreateShaderResourceView(m_copy, &sd, &m_srv);
                if (FAILED(hr)) { m_copy->Release(); m_copy = nullptr; m_srv = nullptr; }
            }
            m_w = d.Width; m_h = d.Height; m_fmt = d.Format;
            m_copiedFrame = ~0u;
            FeatureLog("  HudSceneDepth: scene depth %ux%u fmt %u%s -> copy hr=0x%08lX\n", d.Width, d.Height, (unsigned)d.Format,
                       d.SampleDesc.Count > 1 ? " multisampled" : "", (unsigned long)hr);
        }
        const bool cb = ConstantBuffer(ctx);
        if (!m_srv || !cb) return false;
        if (m_copiedFrame != m_frame) { ctx.Real()->CopyResource(m_copy, m_scene); m_copiedFrame = m_frame; }
        return true;
    }

    // The geometry shader's constants, made on first use.
    bool ConstantBuffer(Context11Proxy& ctx)
    {
        if (!m_cb)
        {
            D3D11_BUFFER_DESC bd = {};
            bd.ByteWidth = 112; bd.Usage = D3D11_USAGE_DYNAMIC; bd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
            bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            if (FAILED(ctx.Parent()->GetReal()->CreateBuffer(&bd, nullptr, &m_cb))) m_cb = nullptr;
        }
        return m_cb != nullptr;
    }

    // The geometry shader over a HUD vertex shader's outputs as declared; null when one is a system value beside the position.
    ID3D11GeometryShader* Shader(Context11Proxy& ctx, const ShaderAnalysis11Result& vs)
    {
        // DepthCopy="any": a multisampled scene depth is read as such (sample 0), a build of its own per vertex shader.
        bool ms = false;
        if (m_anyDepth && m_scene)
        {
            D3D11_TEXTURE2D_DESC d = {};
            m_scene->GetDesc(&d);
            ms = d.SampleDesc.Count > 1;
        }
        const DWORD key = ms ? ~vs.crc32 : vs.crc32;
        auto it = m_gs.find(key);
        if (it != m_gs.end()) return it->second;
        std::string src = "cbuffer wiz3D_hud : register(b0) { float4 wiz3D_k; float4 wiz3D_size; float4 wiz3D_sel; float4 wiz3D_rect;"
                          " float4 wiz3D_near; };\n";
        // Anchor: the element's origin and search extent (z > 0 = given), else the triangle's own; its tracked slot in
        // wiz3D_smooth (slot, value ready) and the shift of a depth the game gives (shift, given).
        if (!m_anchor.empty()) src.insert(src.size() - 4, " float4 wiz3D_anchor; float4 wiz3D_track;");
        src += ms ? "Texture2DMS<float> wiz3D_depth : register(t0);\n" : "Texture2D<float> wiz3D_depth : register(t0);\n";
        if (!m_anchor.empty()) src += "Buffer<float> wiz3D_smooth : register(t1);\n";
        src += "struct V {\n";
        int pos = -1;
        for (size_t i = 0; i < vs.outputs.size(); ++i)
        {
            const ShaderAnalysis11Result::OutputElement& e = vs.outputs[i];
            if (e.sysValue == 1) pos = (int)i;
            else if (e.sysValue != 0) return m_gs[key] = nullptr;
            unsigned n = 0;
            for (unsigned m = e.mask; m; m >>= 1) n += m & 1;
            if (!n) n = 4;
            char line[192];
            _snprintf_s(line, sizeof line, _TRUNCATE, "    %s%u m%u : %s%u;\n",
                        e.type == 1 ? "uint" : e.type == 2 ? "int" : "float", n, (unsigned)i, e.name.c_str(), e.index);
            src += line;
        }
        if (pos < 0) return m_gs[key] = nullptr;
        // Below: the nearest over a 5x6 grid across the piece's width, from its centre down m_below of its heights.
        char below[512] = "";
        if (m_below > 0.f)
            _snprintf_s(below, sizeof below, _TRUNCATE,
                "float wiz3D_below(float2 c, float2 ext)\n{\n"
                "    const float g = wiz3D_size.z;\n"
                "    float n = 1e30;\n"
                "    [unroll] for (int i = 0; i < 5; ++i)\n"
                "        [unroll] for (int j = 0; j < 6; ++j)\n"
                "            n = min(n, g * wiz3D_shift(c + float2((i / 4.0 - 0.5) * ext.x, -(j / 5.0) * %.3f * ext.y)));\n"
                "    return g * n;\n}\n", m_below);
        // Anchor: down the column under the element's origin, the first step nearer than the origin's own depth (a far
        // marker's NPC, not the nearer ground in front of it); none = the origin's depth.
        std::string helpers = below;
        if (!m_anchor.empty())
        {
            char column[1024];
            _snprintf_s(column, sizeof column, _TRUNCATE,
                "float wiz3D_column(float2 c, float2 ext)\n{\n"
                "    const float g = wiz3D_size.z, m = 0.05 * abs(wiz3D_near.x);\n"
                "    float s0 = g * wiz3D_shift(c), r = s0;\n"
                "    bool hit = false;\n"
                "    [unroll] for (int j = 1; j <= 6; ++j)\n    {\n"
                "        float y = c.y - (j / 6.0) * %.3f * ext.y;\n"
                "        float n = min(min(g * wiz3D_shift(float2(c.x - 0.5 * ext.x, y)), g * wiz3D_shift(float2(c.x, y))),\n"
                "                      g * wiz3D_shift(float2(c.x + 0.5 * ext.x, y)));\n"
                "        if (!hit && n < s0 - m) { r = n; hit = true; }\n    }\n"
                "    return g * r;\n}\n", m_below > 0.f ? m_below : 1.f);
            helpers += column;
        }
        // A piece's centre and extent are its longest edge's in pixels (a quad's diagonal, shared by both triangles). A HUD
        // piece covering the screen (the scene's own composite) stays as drawn; the nearest of five taps is the nearest.
        char body[4096];
        _snprintf_s(body, sizeof body, _TRUNCATE,
            "};\n#define wiz3D_POS m%d\n"
            "float wiz3D_shift(float2 ndc)\n{\n"
            "    int2 px = clamp(int2((ndc.x * 0.5 + 0.5) * wiz3D_size.x, (0.5 - ndc.y * 0.5) * wiz3D_size.y), int2(0, 0), int2(wiz3D_size.xy) - 1);\n"
            "    return clamp(wiz3D_k.x + wiz3D_k.y * wiz3D_depth.Load(%s), wiz3D_k.z, wiz3D_k.w);\n}\n"
            "float wiz3D_nearest(float2 c)\n{\n"
            "    const float d = 0.004, g = wiz3D_size.z;\n"
            "    float n0 = g * wiz3D_shift(c), n1 = g * wiz3D_shift(c + float2(d, 0)), n2 = g * wiz3D_shift(c - float2(d, 0));\n"
            "    float n3 = g * wiz3D_shift(c + float2(0, d)), n4 = g * wiz3D_shift(c - float2(0, d));\n"
            "    return g * min(min(min(n0, n1), min(n2, n3)), n4);\n}\n"
            "%s"
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
            "%s"
            "    float s = 0;\n"
            "    if (!use || (wiz3D_sel.z == 0 && min(ext.x, ext.y) >= wiz3D_size.w)) u = 0;\n"
            "    else if (wiz3D_sel.z > 0) s = %s;\n"
            "%s"
            "    else if (wiz3D_sel.x > 0 && length(c * a) < wiz3D_sel.x) s = wiz3D_shift(float2(0, 0));\n"
            "    else if (c.x > wiz3D_rect.x && c.x < wiz3D_rect.y && c.y > wiz3D_rect.z && c.y < wiz3D_rect.w) s = %s;\n"
            "    [unroll] for (int n = 0; n < 3; ++n)\n    {\n        V v = i[n];\n"
            "        v.wiz3D_POS.x += s * v.wiz3D_POS.w - u[n];\n        o.Append(v);\n    }\n}\n",
            pos, ms ? "px, 0" : "int3(px, 0)", helpers.c_str(),
            // The band tests the triangle's own centre; in the map's own view (sel.w < 0) the element's origin.
            m_anchor.empty() ? "" : "    bool an = wiz3D_anchor.z > 0;\n"
                                    "    if (an && wiz3D_sel.w < 0) c = wiz3D_anchor.xy;\n",
            m_menuLayers > 0.f ? "wiz3D_sel.w > 0 ? 0 : wiz3D_nearest(c)" : "wiz3D_nearest(c)",
            // The game's own depth for the element applies wherever it is drawn.
            m_anchor.empty() ? "" : "    else if (wiz3D_track.w > 0) s = wiz3D_track.z;\n",
            !m_anchor.empty() ? (m_below > 0.f ? "an ? (wiz3D_track.y > 0 ? wiz3D_smooth[(uint)wiz3D_track.x] : "
                                                 "wiz3D_column(wiz3D_anchor.xy, wiz3D_anchor.zw)) : wiz3D_below(c, ext)"
                                               : "an ? (wiz3D_track.y > 0 ? wiz3D_smooth[(uint)wiz3D_track.x] : "
                                                 "wiz3D_column(wiz3D_anchor.xy, wiz3D_anchor.zw)) : wiz3D_nearest(c)")
                              : m_below > 0.f ? "wiz3D_below(c, ext)" : "wiz3D_nearest(c)");
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
        return m_gs[key] = l.gs;
    }

    // The first rule this HUD piece matches decides (kFlat, kGiven, kSearched); a searched piece gets its element's origin,
    // (x, y, width, height) in NDC, from the Anchor matrices (rows, as the shader's dp4s read them).
    int Anchor(Context11Proxy& ctx, const ShaderAnalysis11Result& vs, float out[8], float& given)
    {
        int ws = -1, vps = -1;
        unsigned wo = 0, vpo = 0;
        bool found = false;
        for (const std::string& t : vs.tags)
            if (!t.compare(0, strlen(kAnchorTag), kAnchorTag))
                found = sscanf_s(t.c_str() + strlen(kAnchorTag), "%d %u %d %u", &ws, &wo, &vps, &vpo) == 4;
        Buffer11Proxy* wb = found && ws >= 0 && ws < (int)Context11Proxy::kMaxCBs ? ctx.BoundCB(false, (UINT)ws) : nullptr;
        Buffer11Proxy* vb = found && vps >= 0 && vps < (int)Context11Proxy::kMaxCBs ? ctx.BoundCB(false, (UINT)vps) : nullptr;
        const bool haveW = wb && vb && wb->Shadow().size() >= wo + 64 && vb->Shadow().size() >= vpo + 64;
        float w[16] = {}, v[16] = {};
        if (haveW) { memcpy(w, wb->Shadow().data() + wo, 64); memcpy(v, vb->Shadow().data() + vpo, 64); }
        for (size_t i = 0; i < m_rules.size(); ++i)
        {
            if (!Matches(ctx, vs, m_rules[i], haveW ? w : nullptr)) continue;
            ++m_ruleHits[i];
            if (m_rules[i].kind == kGiven) given = w[11];
            if (m_rules[i].kind != kSearched) return m_rules[i].kind;
            break;
        }
        if (!haveW) return kSearched;
        const float o[4] = { w[3], w[7], w[11], w[15] };   // the world matrix's image of the origin
        const float cx = v[0] * o[0] + v[1] * o[1] + v[2] * o[2] + v[3] * o[3];
        const float cy = v[4] * o[0] + v[5] * o[1] + v[6] * o[2] + v[7] * o[3];
        const float cw = v[12] * o[0] + v[13] * o[1] + v[14] * o[2] + v[15] * o[3];
        if (!(cw > 1e-6f)) return kSearched;
        const float x = cx / cw, y = cy / cw;
        // The element's own scale sizes its reach: a game shrinking distant markers keeps them apart.
        const float scale = sqrtf(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);
        // A piece within the open element's reach is part of it: the same origin, so the same depth to the bit.
        if (!(m_groupOpen && fabsf(x - m_groupAt[0]) <= m_groupReach && fabsf(y - m_groupAt[1]) <= m_groupReach))
        {
            m_groupAt[0] = x; m_groupAt[1] = y;
            m_groupReach = m_group * (scale > 0.f && scale < 1.f ? scale : 1.f);
            m_groupOpen = true;
            ++m_elements;
            NewTrack(ctx, x, y);
        }
        // The search spans the element's reach across and a third of it per Below step down (a marker's height).
        out[0] = m_groupAt[0]; out[1] = m_groupAt[1]; out[2] = m_groupReach; out[3] = m_groupReach / 3.f;
        out[4] = m_groupTrack[0]; out[5] = m_groupTrack[1];
        ++m_anchored;
        return kSearched;
    }

    // One rule's conditions, all of which must hold; w = the Anchor world matrix (null: unknown, its conditions fail).
    bool Matches(Context11Proxy& ctx, const ShaderAnalysis11Result& vs, const Rule& r, const float* w) const
    {
        if (r.vs && r.vs != vs.crc32) return false;
        const ShaderAnalysis11Result* ps = ctx.BoundShader(true);
        if (r.ps && (!ps || r.ps != ps->crc32)) return false;
        if (!r.texture.empty())
        {
            // Texture: the pixel shader declares it; Bound="RenderTarget": what is bound there is one (a mask the game drew).
            int slot = -1;
            const std::string tag = kTextureTag + r.texture + " ";
            if (ps) for (const std::string& t : ps->tags) if (!t.compare(0, tag.size(), tag)) slot = atoi(t.c_str() + tag.size());
            if (slot < 0) return false;
            if (r.boundTarget)
            {
                ID3D11ShaderResourceView* srv = ctx.LeftSRV(true, (UINT)slot);
                ID3D11Resource* res = nullptr;
                if (srv) srv->GetResource(&res);
                D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
                D3D11_TEXTURE2D_DESC td = {};
                if (res)
                {
                    res->GetType(&dim);
                    if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) static_cast<ID3D11Texture2D*>(res)->GetDesc(&td);
                    res->Release();
                }
                if (!(td.BindFlags & D3D11_BIND_RENDER_TARGET)) return false;
            }
        }
        const bool needsW = r.tilted || r.zIn[0] <= r.zIn[1] || r.zOut[0] <= r.zOut[1] || r.kind == kGiven;
        if (needsW && !w) return false;
        // Tilted: the world matrix turns the piece out of the canvas plane.
        if (r.tilted && !(fabsf(w[8]) > 1e-4f || fabsf(w[9]) > 1e-4f)) return false;
        // ZInside="a b": a < z < b; ZOutside="a b": z <= a or z >= b (z = the world matrix's z translation).
        if (r.zIn[0] <= r.zIn[1] && !(w[11] > r.zIn[0] && w[11] < r.zIn[1])) return false;
        if (r.zOut[0] <= r.zOut[1] && !(w[11] <= r.zOut[0] || w[11] >= r.zOut[1])) return false;
        return true;
    }

    // A new element takes the slot of last frame's element at its place (its depth ready), or a free one.
    void NewTrack(Context11Proxy& ctx, float x, float y)
    {
        m_groupTrack[0] = m_groupTrack[1] = 0.f;
        // Only an origin inside the band can take a depth (the minimap's tiles have theirs far off the screen).
        if (!(x > m_rect[0] && x < m_rect[1] && y > m_rect[2] && y < m_rect[3])) return;
        if (!TrackBuffers(ctx) || m_cur.size() >= (size_t)kSlots) return;
        int best = -1;
        float bd = 1e30f;
        // A marker may bob up and down: a third of its reach across, its whole reach up and down.
        for (size_t i = 0; i < m_prev.size(); ++i)
        {
            const Element& p = m_prev[i];
            const float dx = fabsf(p.x - x), dy = fabsf(p.y - y);
            if (p.claimed || dx > m_groupReach / 3.f || dy > m_groupReach) continue;
            if (dx + dy < bd) { bd = dx + dy; best = (int)i; }
        }
        int slot = -1;
        if (best >= 0) { m_prev[best].claimed = true; slot = m_prev[best].slot; ++m_tracked; }
        else
        {
            bool used[kSlots] = {};
            for (const Element& e : m_prev) used[e.slot] = true;
            for (const Element& e : m_cur) used[e.slot] = true;
            for (int s = 0; s < kSlots && slot < 0; ++s) if (!used[s]) slot = s;
            if (slot < 0) return;
            ++m_newTracks;
        }
        m_cur.push_back({ x, y, m_groupReach, m_groupReach / 3.f, slot, best < 0, false });
        m_groupTrack[0] = (float)slot;
        m_groupTrack[1] = best >= 0 ? 1.f : 0.f;
    }

    // The tracked depths (one float per slot) and the compute pass's constants, made on first use.
    bool TrackBuffers(Context11Proxy& ctx)
    {
        if (m_smoothSRV) return true;
        if (m_trackFailed) return false;
        ID3D11Device* dev = ctx.Parent()->GetReal();
        D3D11_BUFFER_DESC bd = {};
        bd.ByteWidth = kSlots * 4; bd.Usage = D3D11_USAGE_DEFAULT;
        bd.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
        HRESULT hr = dev->CreateBuffer(&bd, nullptr, &m_smooth);
        if (SUCCEEDED(hr))
        {
            D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
            sd.Format = DXGI_FORMAT_R32_FLOAT; sd.ViewDimension = D3D11_SRV_DIMENSION_BUFFER; sd.Buffer.NumElements = kSlots;
            hr = dev->CreateShaderResourceView(m_smooth, &sd, &m_smoothSRV);
        }
        if (SUCCEEDED(hr))
        {
            D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
            ud.Format = DXGI_FORMAT_R32_FLOAT; ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER; ud.Buffer.NumElements = kSlots;
            hr = dev->CreateUnorderedAccessView(m_smooth, &ud, &m_smoothUAV);
        }
        if (SUCCEEDED(hr))
        {
            D3D11_BUFFER_DESC cd = {};
            cd.ByteWidth = (3 + 2 * kSlots) * 16; cd.Usage = D3D11_USAGE_DYNAMIC;
            cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER; cd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
            hr = dev->CreateBuffer(&cd, nullptr, &m_trackCB);
        }
        FeatureLog("  HudSceneDepth: element tracking buffers hr=0x%08lX\n", (unsigned long)hr);
        if (FAILED(hr))
        {
            if (m_smoothUAV) { m_smoothUAV->Release(); m_smoothUAV = nullptr; }
            if (m_smoothSRV) { m_smoothSRV->Release(); m_smoothSRV = nullptr; }
            if (m_smooth) { m_smooth->Release(); m_smooth = nullptr; }
            m_trackFailed = true;
            return false;
        }
        return true;
    }

    // The frame's end: one thread per element searches its depth in this frame's copy and updates its slot.
    void Track(Context11Proxy& ctx)
    {
        const bool copied = m_copiedFrame == m_frame && m_srv;
        // Its state reads must reach the real context, never BatchRightEye's recorder.
        if (copied && !m_cur.empty() && !ctx.Recording() && TrackShader(ctx))
        {
            D3D11_MAPPED_SUBRESOURCE map = {};
            if (SUCCEEDED(ctx.Real()->Map(m_trackCB, 0, D3D11_MAP_WRITE_DISCARD, 0, &map)) && map.pData)
            {
                float* f = (float*)map.pData;
                memset(f, 0, (3 + 2 * kSlots) * 16);
                memcpy(f, m_trackK, 16);
                f[4] = (float)m_w; f[5] = (float)m_h; f[6] = m_trackB < 0.f ? -1.f : 1.f; f[7] = (float)m_cur.size();
                f[8] = m_trackB; f[9] = 0.25f; f[10] = 0.05f * fabsf(m_trackB);   // b, easing per frame, the margin
                for (size_t i = 0; i < m_cur.size(); ++i)
                {
                    float* e = f + 12 + i * 4;
                    e[0] = m_cur[i].x; e[1] = m_cur[i].y; e[2] = m_cur[i].rx; e[3] = m_cur[i].ry;
                    float* g = f + 12 + (kSlots + i) * 4;
                    g[0] = (float)m_cur[i].slot; g[1] = m_cur[i].fresh ? 1.f : 0.f;
                }
                ctx.Real()->Unmap(m_trackCB, 0);
                // The game's compute bindings at our slots, put back after.
                ID3D11ComputeShader* cs = nullptr;
                ID3D11ClassInstance* ci[256] = {};
                UINT nci = 256;
                ID3D11Buffer* cb = nullptr;
                ID3D11ShaderResourceView* srv = nullptr;
                ID3D11UnorderedAccessView* uav = nullptr;
                ctx.Real()->CSGetShader(&cs, ci, &nci);
                ctx.Real()->CSGetConstantBuffers(0, 1, &cb);
                ctx.Real()->CSGetShaderResources(0, 1, &srv);
                ctx.Real()->CSGetUnorderedAccessViews(0, 1, &uav);
                ctx.Real()->CSSetShader(m_cs, nullptr, 0);
                ctx.Real()->CSSetConstantBuffers(0, 1, &m_trackCB);
                ctx.Real()->CSSetShaderResources(0, 1, &m_srv);
                ctx.Real()->CSSetUnorderedAccessViews(0, 1, &m_smoothUAV, nullptr);
                ctx.Real()->Dispatch(1, 1, 1);
                ctx.Real()->CSSetShader(cs, ci, nci);
                ctx.Real()->CSSetConstantBuffers(0, 1, &cb);
                ctx.Real()->CSSetShaderResources(0, 1, &srv);
                ctx.Real()->CSSetUnorderedAccessViews(0, 1, &uav, nullptr);
                if (cs) cs->Release();
                for (UINT i = 0; i < nci; ++i) if (ci[i]) ci[i]->Release();
                if (cb) cb->Release();
                if (srv) srv->Release();
                if (uav) uav->Release();
                m_trackRan = true;
            }
        }
        // Slots the pass did not fill this frame hand no depth on: next frame's elements start fresh.
        m_prev.swap(m_cur);
        if (!m_trackRan) m_prev.clear();
        m_cur.clear();
    }

    // The tracking pass, compiled on a worker for the scene depth's kind; the source repeats the geometry shader's search.
    bool TrackShader(Context11Proxy& ctx)
    {
        D3D11_TEXTURE2D_DESC d = {};
        if (m_scene) m_scene->GetDesc(&d);
        const bool ms = m_anyDepth && d.SampleDesc.Count > 1;
        if (m_cs && m_csMs == ms) return true;
        if (m_cs) { m_cs->Release(); m_cs = nullptr; m_csJob = std::shared_future<Compiled>(); m_csDone = false; }
        if (!m_csJob.valid())
        {
            m_csMs = ms;
            char src[4096];
            _snprintf_s(src, sizeof src, _TRUNCATE,
                "cbuffer wiz3D_track : register(b0) { float4 wiz3D_k; float4 wiz3D_size; float4 wiz3D_b; float4 wiz3D_e[%d];"
                " float4 wiz3D_f[%d]; };\n"
                "%s wiz3D_depth : register(t0);\n"
                "RWBuffer<float> wiz3D_smooth : register(u0);\n"
                "float wiz3D_shift(float2 ndc)\n{\n"
                "    int2 px = clamp(int2((ndc.x * 0.5 + 0.5) * wiz3D_size.x, (0.5 - ndc.y * 0.5) * wiz3D_size.y), int2(0, 0), int2(wiz3D_size.xy) - 1);\n"
                "    return clamp(wiz3D_k.x + wiz3D_k.y * wiz3D_depth.Load(%s), wiz3D_k.z, wiz3D_k.w);\n}\n"
                "[numthreads(%d, 1, 1)]\nvoid main(uint i : SV_DispatchThreadID)\n{\n"
                "    if (i >= (uint)wiz3D_size.w) return;\n"
                "    float4 e = wiz3D_e[i], f = wiz3D_f[i];\n"
                "    const float g = wiz3D_size.z, m = wiz3D_b.z;\n"
                "    uint slot = (uint)f.x;\n"
                "    float prev = g * wiz3D_smooth[slot];\n"
                "    float s0 = g * wiz3D_shift(e.xy), r = s0, keep = abs(s0 - prev), kv = s0;\n"
                "    bool hit = false;\n"
                "    [unroll] for (int j = 1; j <= 6; ++j)\n    {\n"
                "        float y = e.y - (j / 6.0) * %.3f * e.w;\n"
                "        [unroll] for (int t = -1; t <= 1; ++t)\n        {\n"
                "            float n = g * wiz3D_shift(float2(e.x + 0.5 * t * e.z, y));\n"
                "            if (abs(n - prev) < keep) { keep = abs(n - prev); kv = n; }\n"
                "            if (!hit && n < s0 - m) { r = n; hit = true; }\n        }\n    }\n"
                "    if (f.y == 0 && keep < m) r = kv;\n"
                "    wiz3D_smooth[slot] = g * (f.y > 0 ? r : lerp(prev, r, wiz3D_b.y));\n}\n",
                kSlots, kSlots, ms ? "Texture2DMS<float>" : "Texture2D<float>", ms ? "px, 0" : "int3(px, 0)", kSlots,
                m_below > 0.f ? m_below : 1.f);
            const std::string s = src;
            const std::string target = ctx.Parent()->GetReal()->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_0 ? "cs_5_0" : "cs_4_0";
            m_csJob = std::async(std::launch::async, [s, target]() {
                    Compiled r;
                    ID3DBlob* err = nullptr;
                    r.hr = D3DCompile(s.c_str(), s.size(), "wiz3D_hudtrack", nullptr, nullptr, "main", target.c_str(),
                                      D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &r.code, &err);
                    if (err) { r.err = (const char*)err->GetBufferPointer(); err->Release(); }
                    return r;
                }).share();
        }
        if (m_csDone || m_csJob.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
        const Compiled& c = m_csJob.get();
        m_csDone = true;
        HRESULT hr = c.hr;
        if (SUCCEEDED(hr) && c.code)
            hr = ctx.Parent()->GetReal()->CreateComputeShader(c.code->GetBufferPointer(), c.code->GetBufferSize(), nullptr, &m_cs);
        if (c.code) c.code->Release();
        FeatureLog("  HudSceneDepth: element tracking pass%s hr=0x%08lX%s%s\n", ms ? " (multisampled)" : "", (unsigned long)hr,
                   c.err.empty() ? "" : ": ", c.err.c_str());
        return m_cs != nullptr;
    }

    // HudConstants: the bound vertex shader declares one of the named cbuffers.
    bool HudShader(Context11Proxy& ctx) const
    {
        if (m_hudConstants.empty()) return false;
        const ShaderAnalysis11Result* vs = ctx.BoundShader(false);
        if (!vs) return false;
        for (const std::string& t : vs->tags) if (t == kHudTag) return true;
        return false;
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeHudSceneDepth(const ProfileEntry* e)
{
    if (!e) return nullptr;
    const float crosshair = (float)atof(EntryValue(*e, "Crosshair").c_str());
    // Band: "y0 y1" in NDC, or "y" half-height about the centre; absent = none. Full width: a side edge would cut a line of
    // text, a top or bottom edge never does. "x0 x1 y0 y1" bounds x as well (edge HUD beside world markers).
    std::istringstream in(EntryValue(*e, "Band"));
    float v[4] = {}, rect[4] = { -1e30f, 1e30f, 0.f, 0.f };
    int n = 0;
    while (n < 4 && (in >> v[n])) ++n;
    if (n == 4) { rect[0] = v[0]; rect[1] = v[1]; rect[2] = v[2]; rect[3] = v[3]; }
    else if (n >= 2) { rect[2] = v[0]; rect[3] = v[1]; }
    else if (n == 1) { rect[2] = -v[0]; rect[3] = v[0]; }
    const bool region = rect[3] > rect[2];
    const float nearQuads = (float)atof(EntryValue(*e, "NearQuads").c_str());
    const float menuLayers = (float)atof(EntryValue(*e, "MenuLayers").c_str());
    if (crosshair <= 0.f && !region && !(nearQuads > 0.f) && !(menuLayers > 0.f)) return nullptr;
    const std::string minScene = EntryValue(*e, "MinSceneDraws");
    std::vector<std::string> hudConstants;
    std::istringstream names(EntryValue(*e, "HudConstants"));
    for (std::string s; names >> s; ) hudConstants.push_back(s);
    std::vector<std::string> anchor;
    std::istringstream an(EntryValue(*e, "Anchor"));
    for (std::string s; an >> s; ) anchor.push_back(s);
    const float group = (float)atof(EntryValue(*e, "AnchorGroup").c_str());
    if (anchor.size() != 2 || !(group > 0.f)) anchor.clear();
    std::vector<Rule> rules;
    for (const ProfileEntry* c : ChildEntries("HudSceneDepth"))
    {
        Rule r;
        r.name = EntryValue(*c, "@");
        r.kind = r.name == "Flat" ? kFlat : r.name == "GameDepth" ? kGiven : r.name == "SceneDepth" ? kSearched : -1;
        if (r.kind < 0) { FeatureLog("  HudSceneDepth: <%s> is no rule (Flat, GameDepth, SceneDepth); ignored\n", r.name.c_str()); continue; }
        r.texture = EntryValue(*c, "Texture");
        r.boundTarget = EntryValue(*c, "Bound") == "RenderTarget";
        r.vs = strtoul(EntryValue(*c, "VertexShader").c_str(), nullptr, 0);
        r.ps = strtoul(EntryValue(*c, "PixelShader").c_str(), nullptr, 0);
        r.tilted = EntryValue(*c, "Tilted") == "1";
        sscanf_s(EntryValue(*c, "ZInside").c_str(), "%f %f", &r.zIn[0], &r.zIn[1]);
        sscanf_s(EntryValue(*c, "ZOutside").c_str(), "%f %f", &r.zOut[0], &r.zOut[1]);
        rules.push_back(r);
    }
    return std::unique_ptr<ProfileFeature>(new HudSceneDepthFeature(crosshair, rect,
        minScene.empty() ? 200u : (unsigned)atoi(minScene.c_str()), nearQuads, menuLayers, hudConstants,
        EntryValue(*e, "DepthCopy") == "any", (float)atof(EntryValue(*e, "Below").c_str()), anchor, group, rules));
}

} // namespace wiz3d
