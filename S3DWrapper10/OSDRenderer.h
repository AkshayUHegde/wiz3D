/*
* wiz3D OSD renderer — on-screen display for the DX11 StereoInjector path.
*
* Self-contained text overlay drawn after the main stereo composite. Uses
* GDI (DrawTextW with the OS-shipped Segoe UI font) to render text into a
* 32-bit BGRA DIB section, uploads to a dynamic D3D11 texture, and draws
* alpha-blended quads over the real backbuffer.
*
* No asset-file dependencies -- avoids the asset-pipeline requirement the
* legacy DX10 Font library had (OSDFont.fnt + font.fx were loaded from
* %ALLUSERSPROFILE%\Font and the game cwd respectively, neither of which
* ships with wiz3D any more). All glyph work goes through Win32 GDI.
*
* Three overlay elements, each independently gated:
*   - Status line (top-left): shown when gInfo.Input.ShowOSD == 1 OR
*     during the ~2 second fade after a hotkey changed a stereo setting.
*     Content: "wiz3D | Preset N | Sep X% | Conv Yunits".
*   - FPS counter (top-right): shown when gInfo.Input.ShowFPS == 1. One
*     frame-time EMA, updates every ~250 ms.
*   - Hotkey help overlay (full-screen dim + centred list): shown when
*     gInfo.Input.ShowHotKeyOSD == 1 (hotkey Shift+F1 toggles it).
*
* Lifecycle: lazy init on first Render() call (so we attach to the real
* ID3D11Device at a point the game has finished creating it). Released
* in dtor + before ResizeBuffers.
*/
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>

namespace wiz3d
{

class OSDRenderer
{
public:
    OSDRenderer();
    ~OSDRenderer();

    // Lazy init / release. ReleaseAll is safe to call even if Init never
    // succeeded. Init can be re-attempted after a release.
    bool Init(ID3D11Device* device);
    void ReleaseAll();

    // Render all active OSD elements over the given RTV at bbW x bbH.
    // No-op if nothing to show. Caller is responsible for having bound
    // the real BB RTV; this method saves+restores only the state it
    // mutates (shaders, VB, IL, blend, RS, DS, viewport, PS resources).
    void Render(ID3D11DeviceContext* ctx,
                ID3D11RenderTargetView* rtv,
                UINT bbW, UINT bbH);

    // Called from the hotkey thread when the user adjusts a stereo
    // setting (preset switch, separation / convergence step). Flips a
    // sticky "show the status line for 2 seconds" flag so the user sees
    // their change land even with ShowOSD off.
    void NoteSettingChanged();

    // Call once per Present to advance the FPS counter and fade timers.
    void TickFrame();

private:
    enum class Slot {
        Status = 0,    // top-left status line
        Fps,           // top-right FPS counter
        HotkeyHelp,    // centred, full-screen dim behind
        _Count
    };

    struct GDIFont {
        HFONT  font    = nullptr;
        int    heightPx = 0;
        bool Create(int ptSize, bool bold);
        void Destroy();
    };

    struct TexSlot {
        int               w = 0, h = 0;
        HDC               dc = nullptr;
        HBITMAP           dib = nullptr;
        HBITMAP           prev = nullptr;      // GDI's pre-select bitmap, for cleanup
        BYTE*             bits = nullptr;      // DIB pixel bytes (BGRA, w*h*4)
        ID3D11Texture2D*  tex = nullptr;       // DYNAMIC, DXGI_FORMAT_B8G8R8A8_UNORM
        ID3D11ShaderResourceView* srv = nullptr;
        bool              dirty = false;        // DIB bits changed, needs upload
        int               visibleW = 0, visibleH = 0; // extent of rendered content

        bool  Create(ID3D11Device* dev, int width, int height);
        void  Destroy();
        void  Clear();                           // zero the DIB (fully transparent)
        void  DrawRect(const RECT& r, COLORREF rgb, BYTE alpha);
        void  DrawTextLine(const wchar_t* text, const RECT& r, HFONT font,
                           COLORREF rgb, UINT flags);
        void  Upload(ID3D11DeviceContext* ctx);
    };

    bool  EnsureShaders(ID3D11Device* dev);
    void  BuildStatusText(wchar_t* out, size_t cap) const;
    void  BuildFpsText(wchar_t* out, size_t cap) const;
    void  DrawQuad(ID3D11DeviceContext* ctx,
                   ID3D11ShaderResourceView* srv,
                   float x, float y, float w, float h,
                   UINT bbW, UINT bbH);
    bool  WantStatus() const;
    bool  WantFps() const;
    bool  WantHotkeyHelp() const;

    // Shared D3D11 state. One VS, one PS, one blend, one sampler.
    ID3D11VertexShader*       m_vs      = nullptr;
    ID3D11PixelShader*        m_ps      = nullptr;
    ID3D11InputLayout*        m_il      = nullptr;
    ID3D11Buffer*             m_vb      = nullptr;    // 6 verts, dynamic
    ID3D11SamplerState*       m_sampler = nullptr;
    ID3D11BlendState*         m_blend   = nullptr;
    ID3D11RasterizerState*    m_raster  = nullptr;
    ID3D11DepthStencilState*  m_depth   = nullptr;

    // GDI fonts.
    GDIFont   m_fontStatus;    // 20pt bold Segoe UI
    GDIFont   m_fontFps;       // 16pt bold
    GDIFont   m_fontHelp;      // 18pt bold for titles
    GDIFont   m_fontHelpBody;  // 15pt regular for body lines

    // Per-slot DIBs + textures.
    TexSlot   m_slots[(int)Slot::_Count];

    // FPS tracking.
    ULONGLONG m_lastFpsSample = 0;
    unsigned  m_frameCountSinceSample = 0;
    float     m_fpsEMA = 0.f;

    // Settings-changed fade (2s like DX9).
    ULONGLONG m_statusFadeDeadline = 0;

    // Cached rendered text — skip re-blitting the DIB if content unchanged.
    wchar_t   m_lastStatus[128] = {};
    wchar_t   m_lastFps[32]     = {};
    bool      m_helpTextRendered = false;

    bool      m_initDone   = false;
    bool      m_initFailed = false;
    ID3D11Device* m_dev    = nullptr;         // not AddRef'd
};

}  // namespace wiz3d
