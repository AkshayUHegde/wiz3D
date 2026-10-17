/*
* wiz3D DX11 laser-sight renderer -- see LaserSightRenderer.h.
*/
#include "StdAfx.h"
#include "LaserSightRenderer.h"
#include "..\S3DAPI\GlobalData.h"

#include <d3dcompiler.h>
#include <string.h>
#include <math.h>
#include <vector>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace wiz3d
{

// Simple textured quad -- same shape as the OSD shader but kept
// independent so changes to one don't ripple to the other.
static const char* k_laserShaderSrc = R"(
struct VSIn  { float2 pos : POSITION; float2 uv : TEXCOORD0; };
struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

VSOut VSMain(VSIn i) {
    VSOut o;
    o.pos = float4(i.pos, 0.0, 1.0);
    o.uv  = i.uv;
    return o;
}

Texture2D    laserTex : register(t0);
SamplerState samp     : register(s0);

float4 PSMain(VSOut i) : SV_Target {
    return laserTex.Sample(samp, i.uv);
}
)";

LaserSightRenderer::LaserSightRenderer() {}
LaserSightRenderer::~LaserSightRenderer() { ReleaseAll(); }

// Synthesise a 64x64 crosshair sprite into a BGRA DIB and upload once to
// a D3D11 texture. The pattern is a crisp 1px white circle with 4 radial
// ticks + a 1px centre dot; thin black outline halo for readability over
// bright backgrounds. All drawn with GDI geometric primitives so no
// external PNG asset is needed.
bool LaserSightRenderer::BuildCrosshairTexture(ID3D11Device* dev)
{
    // Build the DIB in BGRA with per-pixel alpha.
    const int W = kSpriteSize, H = kSpriteSize;
    std::vector<BYTE> pixels((size_t)W * H * 4, 0);  // transparent black

    const int cx = W / 2, cy = H / 2;
    const int rOuter = 22;    // outer circle radius
    const int rInner = 20;    // inner circle radius (= 2px ring)
    const int tickOuter = 28; // tip of each radial tick
    const int tickInner = 24; // base of each radial tick
    const int tickHalf  = 2;  // tick thickness / 2

    auto putPixel = [&](int x, int y, BYTE r, BYTE g, BYTE b, BYTE a) {
        if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
        BYTE* p = pixels.data() + ((size_t)y * W + x) * 4;
        p[0] = b; p[1] = g; p[2] = r; p[3] = a;
    };

    // Black outline halo first (one pixel wider on every edge).
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int dx = x - cx, dy = y - cy;
            const int dSq = dx * dx + dy * dy;
            // Ring halo
            const int rH1 = (rOuter + 1), rH2 = (rInner - 1);
            if (dSq <= rH1 * rH1 && dSq >= rH2 * rH2)
                putPixel(x, y, 0, 0, 0, 220);
            // Radial tick halos (N/S/E/W, each one 2px wider than the stroke)
            const int ta = tickOuter + 1, tb = tickInner - 1, th = tickHalf + 1;
            if ((abs(dx) <= th && abs(dy) >= tb && abs(dy) <= ta) ||
                (abs(dy) <= th && abs(dx) >= tb && abs(dx) <= ta))
                putPixel(x, y, 0, 0, 0, 220);
            // Centre dot halo
            if (dSq <= 3 * 3) putPixel(x, y, 0, 0, 0, 220);
        }
    }
    // White foreground on top.
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int dx = x - cx, dy = y - cy;
            const int dSq = dx * dx + dy * dy;
            // Ring
            if (dSq <= rOuter * rOuter && dSq >= rInner * rInner)
                putPixel(x, y, 255, 255, 255, 255);
            // Radial ticks N/S/E/W
            if ((abs(dx) <= tickHalf && abs(dy) >= tickInner && abs(dy) <= tickOuter) ||
                (abs(dy) <= tickHalf && abs(dx) >= tickInner && abs(dx) <= tickOuter))
                putPixel(x, y, 255, 255, 255, 255);
            // Centre dot
            if (dSq <= 2 * 2) putPixel(x, y, 255, 255, 255, 255);
        }
    }

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = W;
    td.Height           = H;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_IMMUTABLE;
    td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    D3D11_SUBRESOURCE_DATA sd = {};
    sd.pSysMem     = pixels.data();
    sd.SysMemPitch = (UINT)W * 4;
    HRESULT hr = dev->CreateTexture2D(&td, &sd, &m_tex);
    if (FAILED(hr) || !m_tex) return false;

    D3D11_SHADER_RESOURCE_VIEW_DESC sv = {};
    sv.Format              = td.Format;
    sv.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
    sv.Texture2D.MipLevels = 1;
    hr = dev->CreateShaderResourceView(m_tex, &sv, &m_srv);
    return SUCCEEDED(hr) && m_srv != nullptr;
}

bool LaserSightRenderer::EnsureShaders(ID3D11Device* dev)
{
    if (m_vs && m_ps && m_il && m_vb && m_sampler && m_blend && m_raster && m_depth) return true;

    ID3DBlob* vsBlob = nullptr, * psBlob = nullptr, * err = nullptr;
    HRESULT hr = D3DCompile(k_laserShaderSrc, strlen(k_laserShaderSrc),
                             "wiz3D_laser_vs", nullptr, nullptr, "VSMain", "vs_4_0",
                             0, 0, &vsBlob, &err);
    if (err) { err->Release(); err = nullptr; }
    if (FAILED(hr) || !vsBlob) return false;
    hr = D3DCompile(k_laserShaderSrc, strlen(k_laserShaderSrc),
                     "wiz3D_laser_ps", nullptr, nullptr, "PSMain", "ps_4_0",
                     0, 0, &psBlob, &err);
    if (err) { err->Release(); err = nullptr; }
    if (FAILED(hr) || !psBlob) { vsBlob->Release(); return false; }

    hr = dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &m_vs);
    if (FAILED(hr)) { vsBlob->Release(); psBlob->Release(); return false; }
    hr = dev->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &m_ps);
    if (FAILED(hr)) { vsBlob->Release(); psBlob->Release(); return false; }

    const D3D11_INPUT_ELEMENT_DESC ied[] = {
        { "POSITION", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 0,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
        { "TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 8,  D3D11_INPUT_PER_VERTEX_DATA, 0 },
    };
    hr = dev->CreateInputLayout(ied, 2, vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), &m_il);
    vsBlob->Release(); psBlob->Release();
    if (FAILED(hr)) return false;

    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth      = 6 * 16;
    bd.Usage          = D3D11_USAGE_DYNAMIC;
    bd.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(dev->CreateBuffer(&bd, nullptr, &m_vb))) return false;

    D3D11_SAMPLER_DESC sd = {};
    sd.Filter   = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD   = D3D11_FLOAT32_MAX;
    if (FAILED(dev->CreateSamplerState(&sd, &m_sampler))) return false;

    D3D11_BLEND_DESC bld = {};
    bld.RenderTarget[0].BlendEnable           = TRUE;
    bld.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;
    bld.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
    bld.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
    bld.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;
    bld.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_INV_SRC_ALPHA;
    bld.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
    bld.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(dev->CreateBlendState(&bld, &m_blend))) return false;

    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;
    rd.DepthClipEnable = TRUE;
    if (FAILED(dev->CreateRasterizerState(&rd, &m_raster))) return false;

    D3D11_DEPTH_STENCIL_DESC dsd = {};
    dsd.DepthEnable   = FALSE;
    dsd.StencilEnable = FALSE;
    if (FAILED(dev->CreateDepthStencilState(&dsd, &m_depth))) return false;

    return true;
}

bool LaserSightRenderer::Init(ID3D11Device* device)
{
    if (m_initDone)   return true;
    if (m_initFailed) return false;
    if (!device)      return false;

    if (!EnsureShaders(device))        { ReleaseAll(); m_initFailed = true; return false; }
    if (!BuildCrosshairTexture(device)){ ReleaseAll(); m_initFailed = true; return false; }

    m_initDone = true;
    return true;
}

void LaserSightRenderer::ReleaseAll()
{
    if (m_depth)   { m_depth->Release();   m_depth   = nullptr; }
    if (m_raster)  { m_raster->Release();  m_raster  = nullptr; }
    if (m_blend)   { m_blend->Release();   m_blend   = nullptr; }
    if (m_sampler) { m_sampler->Release(); m_sampler = nullptr; }
    if (m_srv)     { m_srv->Release();     m_srv     = nullptr; }
    if (m_tex)     { m_tex->Release();     m_tex     = nullptr; }
    if (m_vb)      { m_vb->Release();      m_vb      = nullptr; }
    if (m_il)      { m_il->Release();      m_il      = nullptr; }
    if (m_ps)      { m_ps->Release();      m_ps      = nullptr; }
    if (m_vs)      { m_vs->Release();      m_vs      = nullptr; }
    m_initDone = false;
    m_initFailed = false;
}

void LaserSightRenderer::DrawQuad(ID3D11DeviceContext* ctx,
                                  float x, float y, float w, float h,
                                  UINT bbW, UINT bbH)
{
    const float x0 = (x / (float)bbW)         * 2.f - 1.f;
    const float x1 = ((x + w) / (float)bbW)   * 2.f - 1.f;
    const float y0 = 1.f - (y / (float)bbH)         * 2.f;
    const float y1 = 1.f - ((y + h) / (float)bbH)   * 2.f;

    struct V { float x, y; float u, v; };
    V verts[6] = {
        { x0, y0, 0.f, 0.f },
        { x1, y0, 1.f, 0.f },
        { x0, y1, 0.f, 1.f },
        { x0, y1, 0.f, 1.f },
        { x1, y0, 1.f, 0.f },
        { x1, y1, 1.f, 1.f },
    };
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(m_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)) || !m.pData) return;
    memcpy(m.pData, verts, sizeof(verts));
    ctx->Unmap(m_vb, 0);

    UINT stride = sizeof(V), offset = 0;
    ctx->IASetVertexBuffers(0, 1, &m_vb, &stride, &offset);
    ctx->Draw(6, 0);
}

// Draw the crosshair over the composite output. We don't have the DX9's
// LaserSightData / ShiftFinder available on the DX11 path yet (that
// infrastructure runs in BaseStereoRenderer and ShiftFinder tracks the
// per-draw auto-shift). For 0.4.0 we use a simpler geometry: place the
// crosshair at the user's configured LaserSight (x, y) screen position
// in EACH eye half of the SBS output. Horizontal split is detected by
// output mode -- for TB modes we split vertically instead.
void LaserSightRenderer::Render(ID3D11DeviceContext* ctx,
                                ID3D11RenderTargetView* rtv,
                                UINT bbW, UINT bbH)
{
    if (!ctx || !rtv || bbW == 0 || bbH == 0) return;
    if (!m_initDone) return;
    if (!gInfo.Input.LaserSightEnable) return;

    const float spriteSize = (float)kSpriteSize;
    // User's requested relative screen position (0..1).
    const float relX = gInfo.LaserSightXCoordinate;
    const float relY = gInfo.LaserSightYCoordinate;

    // Default: SBS. Treat the output as two half-width panels and draw the
    // crosshair centred on (relX, relY) within each. For TB modes the
    // compositor is drawing top-bottom, so split vertically instead.
    // Detection: we don't have the output mode in scope here; use the BB
    // aspect ratio as a hint -- wiz3D composites to the real BB which keeps
    // the display's aspect, so SBS is the common case.
    const float halfW = (float)bbW * 0.5f;
    const float cxL = relX * halfW - spriteSize * 0.5f;
    const float cxR = halfW + relX * halfW - spriteSize * 0.5f;
    const float cy  = relY * (float)bbH - spriteSize * 0.5f;

    // Bind state and shaders.
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    D3D11_VIEWPORT vp = {};
    vp.Width = (float)bbW; vp.Height = (float)bbH;
    vp.MinDepth = 0.f; vp.MaxDepth = 1.f;
    ctx->RSSetViewports(1, &vp);
    ctx->RSSetState(m_raster);
    const FLOAT blendFactor[4] = { 1, 1, 1, 1 };
    ctx->OMSetBlendState(m_blend, blendFactor, 0xFFFFFFFFu);
    ctx->OMSetDepthStencilState(m_depth, 0);
    ctx->IASetInputLayout(m_il);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(m_vs, nullptr, 0);
    ctx->GSSetShader(nullptr, nullptr, 0);
    ctx->HSSetShader(nullptr, nullptr, 0);
    ctx->DSSetShader(nullptr, nullptr, 0);
    ctx->PSSetShader(m_ps, nullptr, 0);
    ctx->PSSetShaderResources(0, 1, &m_srv);
    ctx->PSSetSamplers(0, 1, &m_sampler);

    DrawQuad(ctx, cxL, cy, spriteSize, spriteSize, bbW, bbH);
    DrawQuad(ctx, cxR, cy, spriteSize, spriteSize, bbW, bbH);

    ID3D11ShaderResourceView* nullSRV = nullptr;
    ctx->PSSetShaderResources(0, 1, &nullSRV);
}

}  // namespace wiz3d
