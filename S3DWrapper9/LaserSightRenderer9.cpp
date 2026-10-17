/*
* wiz3D DX9 laser-sight renderer -- see LaserSightRenderer9.h.
*/
#include "stdafx.h"
#include "LaserSightRenderer9.h"
#include "..\S3DAPI\GlobalData.h"

#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <vector>

namespace wiz3d
{

LaserSightRenderer9::LaserSightRenderer9() {}
LaserSightRenderer9::~LaserSightRenderer9() { ReleaseAll(); }

// Same crosshair pattern as the DX11 LaserSightRenderer -- 1px white
// ring + 4 radial ticks + centre dot, 2px semi-opaque black halo.
bool LaserSightRenderer9::BuildCrosshairTexture(IDirect3DDevice9* dev)
{
    const int W = kSpriteSize, H = kSpriteSize;
    std::vector<BYTE> pixels((size_t)W * H * 4, 0);

    const int cx = W / 2, cy = H / 2;
    const int rOuter = 22, rInner = 20;
    const int tickOuter = 28, tickInner = 24, tickHalf = 2;

    auto putPixel = [&](int x, int y, BYTE r, BYTE g, BYTE b, BYTE a) {
        if ((unsigned)x >= (unsigned)W || (unsigned)y >= (unsigned)H) return;
        BYTE* p = pixels.data() + ((size_t)y * W + x) * 4;
        p[0] = b; p[1] = g; p[2] = r; p[3] = a;
    };

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int dx = x - cx, dy = y - cy;
            const int dSq = dx * dx + dy * dy;
            const int rH1 = (rOuter + 1), rH2 = (rInner - 1);
            if (dSq <= rH1 * rH1 && dSq >= rH2 * rH2)
                putPixel(x, y, 0, 0, 0, 220);
            const int ta = tickOuter + 1, tb = tickInner - 1, th = tickHalf + 1;
            if ((abs(dx) <= th && abs(dy) >= tb && abs(dy) <= ta) ||
                (abs(dy) <= th && abs(dx) >= tb && abs(dx) <= ta))
                putPixel(x, y, 0, 0, 0, 220);
            if (dSq <= 3 * 3) putPixel(x, y, 0, 0, 0, 220);
        }
    }
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            const int dx = x - cx, dy = y - cy;
            const int dSq = dx * dx + dy * dy;
            if (dSq <= rOuter * rOuter && dSq >= rInner * rInner)
                putPixel(x, y, 255, 255, 255, 255);
            if ((abs(dx) <= tickHalf && abs(dy) >= tickInner && abs(dy) <= tickOuter) ||
                (abs(dy) <= tickHalf && abs(dx) >= tickInner && abs(dx) <= tickOuter))
                putPixel(x, y, 255, 255, 255, 255);
            if (dSq <= 2 * 2) putPixel(x, y, 255, 255, 255, 255);
        }
    }

    // DX9 D3DFMT_A8R8G8B8 is stored as (B, G, R, A) in memory, which
    // matches our DIB-style BGRA buffer above directly.
    HRESULT hr = dev->CreateTexture(W, H, 1, 0, D3DFMT_A8R8G8B8, D3DPOOL_DEFAULT,
                                    &m_tex, nullptr);
    if (FAILED(hr) || !m_tex) return false;

    D3DLOCKED_RECT lr = {};
    if (FAILED(m_tex->LockRect(0, &lr, nullptr, 0))) return false;
    for (int y = 0; y < H; ++y) {
        memcpy((BYTE*)lr.pBits + (size_t)y * lr.Pitch,
               pixels.data() + (size_t)y * W * 4,
               (size_t)W * 4);
    }
    m_tex->UnlockRect(0);
    return true;
}

bool LaserSightRenderer9::Init(IDirect3DDevice9* device)
{
    if (m_initDone)   return true;
    if (m_initFailed) return false;
    if (!device)      return false;
    if (!BuildCrosshairTexture(device)) { ReleaseAll(); m_initFailed = true; return false; }
    m_initDone = true;
    return true;
}

void LaserSightRenderer9::OnDeviceLost()
{
    if (m_tex) { m_tex->Release(); m_tex = nullptr; }
    m_initDone = false;
}

void LaserSightRenderer9::ReleaseAll()
{
    if (m_tex) { m_tex->Release(); m_tex = nullptr; }
    m_initDone = false;
    m_initFailed = false;
}

void LaserSightRenderer9::DrawQuad(IDirect3DDevice9* device, float x, float y, float w, float h)
{
    struct TLVert { float x, y, z, rhw; float u, v; };
    const TLVert verts[6] = {
        { x,     y,     0.f, 1.f, 0.f, 0.f },
        { x + w, y,     0.f, 1.f, 1.f, 0.f },
        { x,     y + h, 0.f, 1.f, 0.f, 1.f },
        { x,     y + h, 0.f, 1.f, 0.f, 1.f },
        { x + w, y,     0.f, 1.f, 1.f, 0.f },
        { x + w, y + h, 0.f, 1.f, 1.f, 1.f },
    };
    device->DrawPrimitiveUP(D3DPT_TRIANGLELIST, 2, verts, sizeof(TLVert));
}

void LaserSightRenderer9::Render(IDirect3DDevice9* device, UINT bbW, UINT bbH)
{
    if (!device || bbW == 0 || bbH == 0) return;
    if (m_initFailed) return;
    if (!m_initDone && !Init(device)) return;
    if (!gInfo.Input.LaserSightEnable) return;

    if (!m_tex) {
        // Re-create after device Reset().
        if (!BuildCrosshairTexture(device)) return;
    }

    const float spriteSize = (float)kSpriteSize;
    const float relX = gInfo.LaserSightXCoordinate;
    const float relY = gInfo.LaserSightYCoordinate;
    const float halfW = (float)bbW * 0.5f;
    const float cxL = relX * halfW - spriteSize * 0.5f;
    const float cxR = halfW + relX * halfW - spriteSize * 0.5f;
    const float cy  = relY * (float)bbH - spriteSize * 0.5f;

    IDirect3DStateBlock9* sb = nullptr;
    if (FAILED(device->CreateStateBlock(D3DSBT_ALL, &sb)) || !sb) return;

    device->SetVertexShader(nullptr);
    device->SetPixelShader(nullptr);
    device->SetFVF(D3DFVF_XYZRHW | D3DFVF_TEX1);
    device->SetTexture(0, m_tex);
    device->SetRenderState(D3DRS_LIGHTING, FALSE);
    device->SetRenderState(D3DRS_ZENABLE, FALSE);
    device->SetRenderState(D3DRS_CULLMODE, D3DCULL_NONE);
    device->SetRenderState(D3DRS_ALPHABLENDENABLE, TRUE);
    device->SetRenderState(D3DRS_SRCBLEND, D3DBLEND_SRCALPHA);
    device->SetRenderState(D3DRS_DESTBLEND, D3DBLEND_INVSRCALPHA);
    device->SetRenderState(D3DRS_BLENDOP, D3DBLENDOP_ADD);
    device->SetRenderState(D3DRS_COLORWRITEENABLE, 0xF);
    device->SetTextureStageState(0, D3DTSS_COLOROP,   D3DTOP_SELECTARG1);
    device->SetTextureStageState(0, D3DTSS_COLORARG1, D3DTA_TEXTURE);
    device->SetTextureStageState(0, D3DTSS_ALPHAOP,   D3DTOP_SELECTARG1);
    device->SetTextureStageState(0, D3DTSS_ALPHAARG1, D3DTA_TEXTURE);
    device->SetSamplerState(0, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(0, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
    device->SetSamplerState(0, D3DSAMP_ADDRESSU,  D3DTADDRESS_CLAMP);
    device->SetSamplerState(0, D3DSAMP_ADDRESSV,  D3DTADDRESS_CLAMP);

    DrawQuad(device, cxL, cy, spriteSize, spriteSize);
    DrawQuad(device, cxR, cy, spriteSize, spriteSize);

    device->SetTexture(0, nullptr);
    sb->Apply();
    sb->Release();
}

}  // namespace wiz3d
