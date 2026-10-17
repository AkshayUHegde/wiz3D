/*
* wiz3D OSD renderer -- see OSDRenderer.h for the design notes.
*/
#include "StdAfx.h"
#include "OSDRenderer.h"

#include "..\S3DAPI\GlobalData.h"

#include <d3dcompiler.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <algorithm>

// Win32 defines DrawText as DrawTextW/DrawTextA macros; our TexSlot method
// has the same base name. Undef locally so it doesn't collide (we still use
// ::DrawTextW explicitly inside TexSlot::DrawTextLine below).
#ifdef DrawText
#undef DrawText
#endif

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace wiz3d
{

// -----------------------------------------------------------------------
// Shader source. Textured quad with alpha blend. Uses SV_VertexID so the
// vertex buffer only carries (pos.xy, uv.xy). The pos is in normalised
// device coordinates (NDC) built per-draw on the CPU side.
// -----------------------------------------------------------------------
static const char* k_osdShaderSrc = R"(
struct VSIn  { float2 pos : POSITION; float2 uv : TEXCOORD0; };
struct VSOut { float4 pos : SV_Position; float2 uv : TEXCOORD0; };

VSOut VSMain(VSIn i) {
    VSOut o;
    o.pos = float4(i.pos, 0.0, 1.0);
    o.uv  = i.uv;
    return o;
}

Texture2D    osdTex : register(t0);
SamplerState samp   : register(s0);

float4 PSMain(VSOut i) : SV_Target {
    return osdTex.Sample(samp, i.uv);
}
)";

// -----------------------------------------------------------------------
// GDIFont
// -----------------------------------------------------------------------
bool OSDRenderer::GDIFont::Create(int ptSize, bool bold)
{
    // Pt->px at 96 DPI: px = pt * 96 / 72. GDI expects negative height for
    // "character height in logical units" (not cell height).
    heightPx = -MulDiv(ptSize, 96, 72);
    font = CreateFontW(
        heightPx, 0, 0, 0,
        bold ? FW_BOLD : FW_NORMAL,
        FALSE /*italic*/, FALSE /*underline*/, FALSE /*strikeout*/,
        DEFAULT_CHARSET,
        OUT_TT_PRECIS,
        CLIP_DEFAULT_PRECIS,
        CLEARTYPE_QUALITY,
        VARIABLE_PITCH | FF_SWISS,
        L"Segoe UI");
    return font != nullptr;
}

void OSDRenderer::GDIFont::Destroy()
{
    if (font) { DeleteObject(font); font = nullptr; }
    heightPx = 0;
}

// -----------------------------------------------------------------------
// TexSlot — the GDI DIB + D3D11 texture pair backing one overlay element.
// -----------------------------------------------------------------------
bool OSDRenderer::TexSlot::Create(ID3D11Device* dev, int width, int height)
{
    w = width; h = height;

    // 1) Create the GDI memory DC + DIB section. BITMAPINFO is top-down
    // (negative height) so the GDI pixel layout matches D3D11's
    // top-left origin without a flip. 32-bit BGRA with the per-pixel
    // alpha preserved for alpha-blending the overlay.
    HDC screen = GetDC(NULL);
    dc = CreateCompatibleDC(screen);
    ReleaseDC(NULL, screen);
    if (!dc) return false;

    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth       = w;
    bmi.bmiHeader.biHeight      = -h;      // top-down
    bmi.bmiHeader.biPlanes      = 1;
    bmi.bmiHeader.biBitCount    = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    void* pBits = nullptr;
    dib = CreateDIBSection(dc, &bmi, DIB_RGB_COLORS, &pBits, NULL, 0);
    if (!dib || !pBits) { Destroy(); return false; }
    bits = (BYTE*)pBits;

    prev = (HBITMAP)SelectObject(dc, dib);
    SetBkMode(dc, TRANSPARENT);     // DrawTextW shouldn't fill behind glyphs

    // 2) Create the matching D3D11 dynamic texture + SRV. BGRA matches
    // the DIB's byte layout: dib[0]=B, dib[1]=G, dib[2]=R, dib[3]=A.
    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = w;
    td.Height           = h;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_B8G8R8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DYNAMIC;
    td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;
    HRESULT hr = dev->CreateTexture2D(&td, nullptr, &tex);
    if (FAILED(hr) || !tex) { Destroy(); return false; }

    D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format              = td.Format;
    sd.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels = 1;
    hr = dev->CreateShaderResourceView(tex, &sd, &srv);
    if (FAILED(hr) || !srv) { Destroy(); return false; }

    Clear();
    dirty = true;
    return true;
}

void OSDRenderer::TexSlot::Destroy()
{
    if (srv)  { srv->Release();  srv  = nullptr; }
    if (tex)  { tex->Release();  tex  = nullptr; }
    if (dc && prev) { SelectObject(dc, prev); prev = nullptr; }
    if (dib)  { DeleteObject(dib); dib = nullptr; }
    if (dc)   { DeleteDC(dc);     dc  = nullptr; }
    bits = nullptr; w = h = 0; dirty = false;
    visibleW = visibleH = 0;
}

void OSDRenderer::TexSlot::Clear()
{
    if (!bits) return;
    memset(bits, 0, (size_t)w * h * 4);   // fully transparent black
    dirty = true;
    visibleW = visibleH = 0;
}

// GDI AlphaBlend-style rect fill with explicit alpha (DIB is 32-bit).
// Writes BGRA with (B, G, R, alpha) ordering. Overwrites existing pixels
// in the rect (not accumulated) -- callers that want blending should Clear
// first.
void OSDRenderer::TexSlot::DrawRect(const RECT& r, COLORREF rgb, BYTE alpha)
{
    if (!bits) return;
    const BYTE R = GetRValue(rgb), G = GetGValue(rgb), B = GetBValue(rgb);
    const int x0 = (std::max)(0L, (long)r.left),   x1 = (std::min)((long)w, (long)r.right);
    const int y0 = (std::max)(0L, (long)r.top),    y1 = (std::min)((long)h, (long)r.bottom);
    for (int y = y0; y < y1; ++y) {
        BYTE* row = bits + (size_t)y * w * 4;
        for (int x = x0; x < x1; ++x) {
            BYTE* p = row + x * 4;
            p[0] = B; p[1] = G; p[2] = R; p[3] = alpha;
        }
    }
    dirty = true;
    if (x1 > visibleW) visibleW = x1;
    if (y1 > visibleH) visibleH = y1;
}

// DrawText via GDI, then stamp alpha=255 for all pixels GDI touched. GDI
// DrawTextW on a DIB doesn't set the alpha byte (per-pixel alpha is zero
// on initial glyph draw), so we post-process: any pixel in the clip rect
// whose RGB is non-zero gets alpha=255. For ClearType AA transitions GDI
// renders anti-aliased grey values -- the alpha stamp picks them up at
// their luminance, so the quad composites smoothly over game content.
void OSDRenderer::TexSlot::DrawTextLine(const wchar_t* text, const RECT& r, HFONT font,
                                        COLORREF rgb, UINT flags)
{
    if (!dc || !bits || !font || !text) return;

    HFONT prevFont = (HFONT)SelectObject(dc, font);
    SetTextColor(dc, rgb);

    RECT rr = r;
    ::DrawTextW(dc, text, -1, &rr, flags);

    // Post-process: synthesise alpha from luminance for pixels GDI touched.
    const int x0 = (std::max)(0L, (long)r.left),   x1 = (std::min)((long)w, (long)r.right);
    const int y0 = (std::max)(0L, (long)r.top),    y1 = (std::min)((long)h, (long)r.bottom);
    const BYTE Rt = GetRValue(rgb), Gt = GetGValue(rgb), Bt = GetBValue(rgb);
    for (int y = y0; y < y1; ++y) {
        BYTE* row = bits + (size_t)y * w * 4;
        for (int x = x0; x < x1; ++x) {
            BYTE* p = row + x * 4;
            // GDI wrote (B, G, R, 0) on glyph pixels. The max channel is
            // the ClearType luminance -- use that as alpha for a smooth
            // premultiplied-ish blend. Pixels GDI didn't touch stay fully
            // transparent (and whatever Clear or DrawRect put down).
            BYTE lum = (std::max)((std::max)(p[0], p[1]), p[2]);
            if (lum > 0 && p[3] == 0) {
                // Fill the colour at full strength and use luminance as alpha.
                p[0] = Bt; p[1] = Gt; p[2] = Rt; p[3] = lum;
            }
        }
    }

    SelectObject(dc, prevFont);
    dirty = true;
    if (x1 > visibleW) visibleW = x1;
    if (y1 > visibleH) visibleH = y1;
}

void OSDRenderer::TexSlot::Upload(ID3D11DeviceContext* ctx)
{
    if (!dirty || !tex || !bits) return;
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)) || !m.pData) return;
    const size_t rowBytes = (size_t)w * 4;
    for (int y = 0; y < h; ++y) {
        memcpy((BYTE*)m.pData + (size_t)y * m.RowPitch, bits + (size_t)y * rowBytes, rowBytes);
    }
    ctx->Unmap(tex, 0);
    dirty = false;
}

// -----------------------------------------------------------------------
// OSDRenderer
// -----------------------------------------------------------------------
OSDRenderer::OSDRenderer() {}
OSDRenderer::~OSDRenderer() { ReleaseAll(); }

bool OSDRenderer::EnsureShaders(ID3D11Device* dev)
{
    if (m_vs && m_ps && m_il && m_vb && m_sampler && m_blend && m_raster && m_depth) return true;

    // Compile VS + PS.
    ID3DBlob* vsBlob = nullptr, * psBlob = nullptr, * err = nullptr;
    HRESULT hr = D3DCompile(k_osdShaderSrc, strlen(k_osdShaderSrc),
                             "wiz3D_osd_vs", nullptr, nullptr, "VSMain", "vs_4_0",
                             0, 0, &vsBlob, &err);
    if (err) { err->Release(); err = nullptr; }
    if (FAILED(hr) || !vsBlob) return false;

    hr = D3DCompile(k_osdShaderSrc, strlen(k_osdShaderSrc),
                     "wiz3D_osd_ps", nullptr, nullptr, "PSMain", "ps_4_0",
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

    // Dynamic vertex buffer: 6 verts (two triangles), 16 bytes each.
    D3D11_BUFFER_DESC bd = {};
    bd.ByteWidth      = 6 * 16;
    bd.Usage          = D3D11_USAGE_DYNAMIC;
    bd.BindFlags      = D3D11_BIND_VERTEX_BUFFER;
    bd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    if (FAILED(dev->CreateBuffer(&bd, nullptr, &m_vb))) return false;

    // Linear sampler, clamp. Point would make ClearType AA transitions jaggy.
    D3D11_SAMPLER_DESC sd = {};
    sd.Filter   = D3D11_FILTER_MIN_MAG_LINEAR_MIP_POINT;
    sd.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.MaxLOD   = D3D11_FLOAT32_MAX;
    if (FAILED(dev->CreateSamplerState(&sd, &m_sampler))) return false;

    // Standard "over" alpha blend: Src.rgb * Src.a + Dst.rgb * (1 - Src.a).
    D3D11_BLEND_DESC blendDesc = {};
    blendDesc.RenderTarget[0].BlendEnable           = TRUE;
    blendDesc.RenderTarget[0].SrcBlend              = D3D11_BLEND_SRC_ALPHA;
    blendDesc.RenderTarget[0].DestBlend             = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOp               = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].SrcBlendAlpha         = D3D11_BLEND_ONE;
    blendDesc.RenderTarget[0].DestBlendAlpha        = D3D11_BLEND_INV_SRC_ALPHA;
    blendDesc.RenderTarget[0].BlendOpAlpha          = D3D11_BLEND_OP_ADD;
    blendDesc.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    if (FAILED(dev->CreateBlendState(&blendDesc, &m_blend))) return false;

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

bool OSDRenderer::Init(ID3D11Device* device)
{
    if (m_initDone)   return true;
    if (m_initFailed) return false;
    if (!device)      return false;

    m_dev = device;

    if (!EnsureShaders(device)) { ReleaseAll(); m_initFailed = true; return false; }

    if (!m_fontStatus.Create(20, true))    { ReleaseAll(); m_initFailed = true; return false; }
    if (!m_fontFps.Create(16, true))       { ReleaseAll(); m_initFailed = true; return false; }
    if (!m_fontHelp.Create(22, true))      { ReleaseAll(); m_initFailed = true; return false; }
    if (!m_fontHelpBody.Create(15, false)) { ReleaseAll(); m_initFailed = true; return false; }

    // Fixed slot sizes picked to fit the maximum content.
    if (!m_slots[(int)Slot::Status].Create(device, 640, 48))      { ReleaseAll(); m_initFailed = true; return false; }
    if (!m_slots[(int)Slot::Fps].Create(device, 180, 36))         { ReleaseAll(); m_initFailed = true; return false; }
    if (!m_slots[(int)Slot::HotkeyHelp].Create(device, 720, 520)) { ReleaseAll(); m_initFailed = true; return false; }

    m_initDone = true;
    return true;
}

void OSDRenderer::ReleaseAll()
{
    for (int i = 0; i < (int)Slot::_Count; ++i) m_slots[i].Destroy();
    m_fontHelpBody.Destroy();
    m_fontHelp.Destroy();
    m_fontFps.Destroy();
    m_fontStatus.Destroy();
    if (m_depth)   { m_depth->Release();   m_depth   = nullptr; }
    if (m_raster)  { m_raster->Release();  m_raster  = nullptr; }
    if (m_blend)   { m_blend->Release();   m_blend   = nullptr; }
    if (m_sampler) { m_sampler->Release(); m_sampler = nullptr; }
    if (m_vb)      { m_vb->Release();      m_vb      = nullptr; }
    if (m_il)      { m_il->Release();      m_il      = nullptr; }
    if (m_ps)      { m_ps->Release();      m_ps      = nullptr; }
    if (m_vs)      { m_vs->Release();      m_vs      = nullptr; }
    m_initDone = false;
    m_initFailed = false;
    m_lastStatus[0] = m_lastFps[0] = L'\0';
    m_helpTextRendered = false;
    m_dev = nullptr;
}

void OSDRenderer::NoteSettingChanged()
{
    // 2 seconds like DX9 BaseStereoRenderer_Main.cpp m_nOSDShowTimeLeft.
    m_statusFadeDeadline = GetTickCount64() + 2000;
}

void OSDRenderer::TickFrame()
{
    ULONGLONG now = GetTickCount64();
    ++m_frameCountSinceSample;
    if (m_lastFpsSample == 0) m_lastFpsSample = now;
    if (now - m_lastFpsSample >= 250) {
        const double seconds = (double)(now - m_lastFpsSample) / 1000.0;
        const double instantFps = seconds > 0 ? (double)m_frameCountSinceSample / seconds : 0.0;
        // Simple EMA so the number stops flickering.
        m_fpsEMA = (m_fpsEMA == 0.f) ? (float)instantFps
                                     : (0.6f * (float)instantFps + 0.4f * m_fpsEMA);
        m_frameCountSinceSample = 0;
        m_lastFpsSample = now;
    }

    // Detect hotkey-driven setting changes vs the last-frame snapshot and
    // auto-trigger the 2s status pop. Avoids wiring a callback through
    // the shared KeyboardHook thread for every hotkey.
    const CameraPreset* p = gInfo.Input.GetActivePreset();
    const float stereoBase  = p ? p->StereoBase   : 0.f;
    const float oneDivZPS   = p ? p->One_div_ZPS  : 0.f;
    const bool  swapEyes    = gInfo.Input.SwapEyes != 0;
    const unsigned preset   = gInfo.Input.ActivePreset;
    if (m_lastActivePreset == (unsigned)-1) {
        // Prime the snapshot on first tick without firing the pop.
        m_lastActivePreset = preset;
        m_lastStereoBase   = stereoBase;
        m_lastOneDivZPS    = oneDivZPS;
        m_lastSwapEyes     = swapEyes;
    } else if (preset != m_lastActivePreset ||
               stereoBase != m_lastStereoBase ||
               oneDivZPS  != m_lastOneDivZPS  ||
               swapEyes   != m_lastSwapEyes) {
        NoteSettingChanged();
        m_lastActivePreset = preset;
        m_lastStereoBase   = stereoBase;
        m_lastOneDivZPS    = oneDivZPS;
        m_lastSwapEyes     = swapEyes;
    }
}

bool OSDRenderer::WantStatus() const
{
    // DX9's DataInput has no ShowOSD toggle; the status overlay is a
    // transient 2s pop after a hotkey changes a stereo value. gInfo.ShowOSD
    // (on GlobalInfo, not Input) is the "always-on" variant used by some
    // test configs -- honour it too.
    if (gInfo.ShowOSD) return true;
    return GetTickCount64() < m_statusFadeDeadline;
}

bool OSDRenderer::WantFps() const
{
    return gInfo.Input.ShowFPS != 0;
}

bool OSDRenderer::WantHotkeyHelp() const
{
    return gInfo.Input.ShowHotKeyOSD != 0;
}

// Status: "wiz3D | Preset N | Sep X% | Conv Yunits [EyesSwapped]"
void OSDRenderer::BuildStatusText(wchar_t* out, size_t cap) const
{
    const CameraPreset* p = gInfo.Input.GetActivePreset();
    const int presetIdx = (int)gInfo.Input.ActivePreset + 1;  // 1-based for user
    const float sepPercent = p ? (p->StereoBase / 0.008f) : 0.f;  // StereoBase = percent * 0.008
    const float convUnits  = (p && p->One_div_ZPS != 0.f) ? (1.f / p->One_div_ZPS) : 0.f;
    if (gInfo.Input.SwapEyes) {
        _snwprintf_s(out, cap, _TRUNCATE,
                     L"wiz3D   |   Preset %d   |   Sep %.0f%%   |   Conv %.1f   |   Eyes Swapped",
                     presetIdx, sepPercent, convUnits);
    } else {
        _snwprintf_s(out, cap, _TRUNCATE,
                     L"wiz3D   |   Preset %d   |   Sep %.0f%%   |   Conv %.1f",
                     presetIdx, sepPercent, convUnits);
    }
}

void OSDRenderer::BuildFpsText(wchar_t* out, size_t cap) const
{
    _snwprintf_s(out, cap, _TRUNCATE, L"%.0f fps", m_fpsEMA);
}

// Draw a textured quad. (x, y) and (w, h) are in pixels in the real BB
// coordinate system; converted to NDC per draw. Samples the full texture
// (uv 0..1) regardless of the slot's visible extent -- the slot's
// transparent pixels alpha-blend to a no-op over the game content.
void OSDRenderer::DrawQuad(ID3D11DeviceContext* ctx,
                           ID3D11ShaderResourceView* srv,
                           float x, float y, float w, float h,
                           UINT bbW, UINT bbH)
{
    if (!srv || !m_vb) return;

    // Pixel -> NDC. Pixel origin top-left, NDC origin centre, Y up.
    const float x0 = (x / (float)bbW)         * 2.f - 1.f;
    const float x1 = ((x + w) / (float)bbW)    * 2.f - 1.f;
    const float y0 = 1.f - (y / (float)bbH)         * 2.f;        // top edge
    const float y1 = 1.f - ((y + h) / (float)bbH)   * 2.f;        // bottom edge

    struct V { float x, y; float u, v; };
    V verts[6] = {
        { x0, y0, 0.f, 0.f },   // TL
        { x1, y0, 1.f, 0.f },   // TR
        { x0, y1, 0.f, 1.f },   // BL
        { x0, y1, 0.f, 1.f },   // BL
        { x1, y0, 1.f, 0.f },   // TR
        { x1, y1, 1.f, 1.f },   // BR
    };
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (FAILED(ctx->Map(m_vb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)) || !m.pData) return;
    memcpy(m.pData, verts, sizeof(verts));
    ctx->Unmap(m_vb, 0);

    UINT stride = sizeof(V), offset = 0;
    ctx->IASetVertexBuffers(0, 1, &m_vb, &stride, &offset);
    ctx->IASetInputLayout(m_il);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->VSSetShader(m_vs, nullptr, 0);
    ctx->GSSetShader(nullptr, nullptr, 0);
    ctx->HSSetShader(nullptr, nullptr, 0);
    ctx->DSSetShader(nullptr, nullptr, 0);
    ctx->PSSetShader(m_ps, nullptr, 0);
    ctx->PSSetShaderResources(0, 1, &srv);
    ctx->PSSetSamplers(0, 1, &m_sampler);
    ctx->Draw(6, 0);
    // Clear the SRV binding so the caller's next PSSetShaderResources on
    // the same slot doesn't trip a D3D11 read/write hazard warning.
    ID3D11ShaderResourceView* nullSRV = nullptr;
    ctx->PSSetShaderResources(0, 1, &nullSRV);
}

void OSDRenderer::Render(ID3D11DeviceContext* ctx,
                         ID3D11RenderTargetView* rtv,
                         UINT bbW, UINT bbH)
{
    if (!ctx || !rtv || bbW == 0 || bbH == 0) return;
    if (!m_initDone) return;      // caller must have called Init() with a real device first

    const bool wantStatus = WantStatus();
    const bool wantFps    = WantFps();
    const bool wantHelp   = WantHotkeyHelp();
    if (!wantStatus && !wantFps && !wantHelp) return;

    // Update DIBs where content changed.
    if (wantStatus) {
        wchar_t text[128];
        BuildStatusText(text, _countof(text));
        if (wcscmp(text, m_lastStatus) != 0) {
            TexSlot& s = m_slots[(int)Slot::Status];
            s.Clear();
            // Semi-opaque dark pill behind the text for readability over bright scenes.
            RECT bg = { 0, 0, s.w, s.h };
            s.DrawRect(bg, RGB(0, 0, 0), 160);
            RECT textRect = { 12, 4, s.w - 12, s.h - 4 };
            s.DrawTextLine(text, textRect, m_fontStatus.font, RGB(240, 240, 240),
                       DT_LEFT | DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX);
            wcscpy_s(m_lastStatus, text);
        }
    }

    if (wantFps) {
        wchar_t text[32];
        BuildFpsText(text, _countof(text));
        if (wcscmp(text, m_lastFps) != 0) {
            TexSlot& s = m_slots[(int)Slot::Fps];
            s.Clear();
            RECT bg = { 0, 0, s.w, s.h };
            s.DrawRect(bg, RGB(0, 0, 0), 160);
            RECT textRect = { 8, 2, s.w - 8, s.h - 2 };
            s.DrawTextLine(text, textRect, m_fontFps.font, RGB(240, 240, 240),
                       DT_RIGHT | DT_SINGLELINE | DT_VCENTER | DT_NOPREFIX);
            wcscpy_s(m_lastFps, text);
        }
    }

    if (wantHelp && !m_helpTextRendered) {
        TexSlot& s = m_slots[(int)Slot::HotkeyHelp];
        s.Clear();
        // Semi-opaque dark panel. ~65% opacity so game content shows faintly through.
        RECT bg = { 0, 0, s.w, s.h };
        s.DrawRect(bg, RGB(10, 10, 20), 180);
        // Title.
        RECT title = { 20, 20, s.w - 20, 60 };
        s.DrawTextLine(L"wiz3D  —  Hotkeys", title, m_fontHelp.font, RGB(255, 255, 255),
                   DT_LEFT | DT_SINGLELINE | DT_NOPREFIX);
        // Body: two columns. Keep it tight so it fits in the 720x520 slot.
        static const wchar_t* lines[] = {
            L"Numpad *",             L"Toggle stereo on / off",
            L"Numpad +  /  -",       L"Separation (depth strength)",
            L"Shift + Numpad +/-",   L"Convergence (where screen plane sits)",
            L"Numpad 7 / 8 / 9",     L"Preset Low / Default / Strong",
            L"Ctrl + F8",            L"Swap left / right eyes",
            L"Numpad /",             L"Toggle auto-focus",
            L"Ctrl + Numpad /",      L"Toggle laser-sight crosshair",
            L"PrtScn",               L"Stereo screenshot",
            L"Shift + F1",           L"Toggle this hotkey list",
            L"Shift + Numpad *",     L"Open the wizard overlay",
        };
        int row = 72;
        const int rowH = 32;
        const int keyX = 32, valueX = 240;
        for (int i = 0; i < _countof(lines); i += 2) {
            RECT k = { keyX,   row, valueX - 8,  row + rowH };
            RECT v = { valueX, row, s.w - 20,    row + rowH };
            s.DrawTextLine(lines[i],     k, m_fontHelpBody.font, RGB(180, 220, 255),
                       DT_LEFT | DT_SINGLELINE | DT_NOPREFIX);
            s.DrawTextLine(lines[i + 1], v, m_fontHelpBody.font, RGB(230, 230, 230),
                       DT_LEFT | DT_SINGLELINE | DT_NOPREFIX);
            row += rowH;
        }
        m_helpTextRendered = true;
    }
    if (!wantHelp && m_helpTextRendered) m_helpTextRendered = false;    // next show re-renders

    // Upload any dirty DIBs.
    if (wantStatus) m_slots[(int)Slot::Status].Upload(ctx);
    if (wantFps)    m_slots[(int)Slot::Fps].Upload(ctx);
    if (wantHelp)   m_slots[(int)Slot::HotkeyHelp].Upload(ctx);

    // State setup once for all draws.
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    D3D11_VIEWPORT vp = {};
    vp.Width = (float)bbW; vp.Height = (float)bbH;
    vp.MinDepth = 0.f; vp.MaxDepth = 1.f;
    ctx->RSSetViewports(1, &vp);
    ctx->RSSetState(m_raster);
    const FLOAT blendFactor[4] = { 1, 1, 1, 1 };
    ctx->OMSetBlendState(m_blend, blendFactor, 0xFFFFFFFFu);
    ctx->OMSetDepthStencilState(m_depth, 0);

    // Draw order: hotkey help underneath (full-screen dim), then status, then FPS.
    if (wantHelp) {
        TexSlot& s = m_slots[(int)Slot::HotkeyHelp];
        // Centred, max 720x520 pixels. Scale down proportionally only if BB
        // is smaller than the slot (rare -- 480p-era displays). Otherwise
        // 1:1 pixel mapping.
        float dw = (float)(std::min<UINT>)(bbW > 40 ? bbW - 40 : (UINT)s.w, (UINT)s.w);
        float dh = (float)(std::min<UINT>)(bbH > 40 ? bbH - 40 : (UINT)s.h, (UINT)s.h);
        float dx = ((float)bbW - dw) * 0.5f;
        float dy = ((float)bbH - dh) * 0.5f;
        DrawQuad(ctx, s.srv, dx, dy, dw, dh, bbW, bbH);
    }
    if (wantStatus) {
        TexSlot& s = m_slots[(int)Slot::Status];
        DrawQuad(ctx, s.srv, 20.f, 20.f, (float)s.w, (float)s.h, bbW, bbH);
    }
    if (wantFps) {
        TexSlot& s = m_slots[(int)Slot::Fps];
        DrawQuad(ctx, s.srv, (float)bbW - 20.f - (float)s.w, 20.f, (float)s.w, (float)s.h, bbW, bbH);
    }
}

}  // namespace wiz3d
