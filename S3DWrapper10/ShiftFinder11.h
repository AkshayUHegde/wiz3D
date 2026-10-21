/* wiz3D - DX11 analogue of ShiftFinder10 (AutoFocus / LaserSight image correlation)
 *
 * 0.4.1: the legacy DX10 ShiftFinder10 (AutoFocus.cpp/.h) ran on the DDI
 * interfaces (D3D10DDI_HRESOURCE / pfnResourceCopyRegion / pfnStagingResourceMap)
 * from the DDI-era wrapper path. Akshay's DX11 path (Context11Proxy /
 * Device11Proxy / SwapChain11Proxy) talks modern COM, so AutoFocus silently
 * dropped off the DX11 build.
 *
 * ShiftFinder11 keeps the shared S3DAPI ShiftFinder base (correlator, SSE
 * luminance converters, worker thread) and only re-implements the GPU staging
 * copy + CPU readback in DX11 COM:
 *   - CreateTexture2D(D3D11_USAGE_STAGING + CPU_ACCESS_READ) for the staging
 *     interleaved L/R strip pack
 *   - CopySubresourceRegion for the per-eye BB -> staging copy
 *   - Map(D3D11_MAP_READ) for the readback into the correlator buffers
 *
 * One instance per SwapChain11Proxy (the staging size follows the BB dims).
 */

#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include "..\S3DAPI\ImageProcessing.h"

namespace wiz3d
{

class ShiftFinder11 : public ShiftFinder
{
public:
    ShiftFinder11();
    ~ShiftFinder11() { Clear(); }

    // dev / ctx are not AddRef'd; the SwapChain11Proxy owns them for the
    // staging texture's lifetime. leftRect / rightRect are the per-eye BB
    // rects to sample from (full BB usually).
    HRESULT Initialize(ID3D11Device* dev,
                       ID3D11DeviceContext* ctx,
                       const RECT* pLeftRect,
                       const RECT* pRightRect,
                       DXGI_FORMAT bbFormat);
    void    Clear();

    // Kick a per-eye staging copy + queue the worker correlator. Safe to call
    // every frame; internal state tracks whether the correlator is idle.
    void    FindShift(ID3D11Texture2D* leftBB,
                      ID3D11Texture2D* rightBB,
                      UINT currentFrame);

protected:
    virtual bool GetRTData() override;

private:
    HRESULT CreateMemoryBufferTexture(int width, int height);

    ID3D11Device*         m_dev      = nullptr;
    ID3D11DeviceContext*  m_ctx      = nullptr;
    ID3D11Texture2D*      m_staging  = nullptr;   // AddRef'd (released in Clear)
    DXGI_FORMAT           m_bbFormat = DXGI_FORMAT_UNKNOWN;

    // Source copy boxes derived from Initialize's RECTs.
    D3D11_BOX             m_srcBoxLeft  = {};
    D3D11_BOX             m_srcBoxRight = {};
};

} // namespace wiz3d
