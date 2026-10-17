/*
* wiz3D laser-sight renderer -- DX11 StereoInjector variant.
*
* Draws a crosshair sprite at the stereo convergence depth in both eye
* halves of the composited SBS backbuffer. Replaces the legacy
* S3DWrapper10/LaserSight.cpp (and S3DWrapper9/DX9LaserSight.cpp) path
* which depends on %ALLUSERSPROFILE%\resources.zip -> Driver/LaserSight.png
* -- an iZ3D-era InnoSetup installer asset that doesn't ship with the
* current wiz3D zip-release pipeline, so the legacy laser sight has been
* dead at runtime like the OSD was.
*
* Synthesises the crosshair sprite in-code via GDI (circle + 4 radial
* ticks), uploads once to a D3D11 texture, and draws two alpha-blended
* quads per Present -- one centred in the left half of the composite,
* one in the right half with a horizontal shift derived from the
* stereo convergence value.
*
* Gated by gInfo.Input.LaserSightEnable (Ctrl+Numpad / hotkey).
*/
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>

namespace wiz3d
{

class LaserSightRenderer
{
public:
    LaserSightRenderer();
    ~LaserSightRenderer();

    bool Init(ID3D11Device* device);
    void ReleaseAll();

    // Draws the crosshair in both eye halves of the composite BB.
    // Must be called AFTER DrawCompositePass has blitted the eyes into rtv.
    // bbW / bbH are the real BB dims (= composite output dims).
    void Render(ID3D11DeviceContext* ctx,
                ID3D11RenderTargetView* rtv,
                UINT bbW, UINT bbH);

private:
    bool  EnsureShaders(ID3D11Device* dev);
    bool  BuildCrosshairTexture(ID3D11Device* dev);
    void  DrawQuad(ID3D11DeviceContext* ctx,
                   float x, float y, float w, float h,
                   UINT bbW, UINT bbH);

    static const int kSpriteSize = 64;  // 64x64 crosshair sprite

    ID3D11VertexShader*       m_vs      = nullptr;
    ID3D11PixelShader*        m_ps      = nullptr;
    ID3D11InputLayout*        m_il      = nullptr;
    ID3D11Buffer*             m_vb      = nullptr;
    ID3D11Texture2D*          m_tex     = nullptr;
    ID3D11ShaderResourceView* m_srv     = nullptr;
    ID3D11SamplerState*       m_sampler = nullptr;
    ID3D11BlendState*         m_blend   = nullptr;
    ID3D11RasterizerState*    m_raster  = nullptr;
    ID3D11DepthStencilState*  m_depth   = nullptr;

    bool                      m_initDone   = false;
    bool                      m_initFailed = false;
};

}  // namespace wiz3d
