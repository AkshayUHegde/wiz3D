/*
* wiz3D laser-sight renderer -- DX9 StereoInjector variant.
*
* Mirror of S3DWrapper10/LaserSightRenderer.h. Replaces the dead
* S3DWrapper9/DX9LaserSight.cpp path which depends on the never-shipped
* %ALLUSERSPROFILE%\resources.zip asset (same iZ3D installer problem
* that killed the legacy OSD).
*
* Synthesises the crosshair sprite in-code and draws two alpha-blended
* quads per Present (one per eye half of the SBS composite) using
* fixed-function pipeline with D3DFVF_XYZRHW textured quads.
*/
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d9.h>

namespace wiz3d
{

class LaserSightRenderer9
{
public:
    LaserSightRenderer9();
    ~LaserSightRenderer9();

    bool Init(IDirect3DDevice9* device);
    void ReleaseAll();
    void OnDeviceLost();   // drop DEFAULT-pool texture at game Reset()
    void Render(IDirect3DDevice9* device, UINT bbW, UINT bbH);

private:
    bool BuildCrosshairTexture(IDirect3DDevice9* dev);
    void DrawQuad(IDirect3DDevice9* device, float x, float y, float w, float h);

    static const int kSpriteSize = 64;

    IDirect3DTexture9* m_tex        = nullptr;
    bool               m_initDone   = false;
    bool               m_initFailed = false;
};

}  // namespace wiz3d
