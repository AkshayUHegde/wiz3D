/* wiz3D - stereo-doubling heuristic helpers (Option B Stage 3)
 *
 * Mirrors the logic in S3DWrapper10/ResourceWrapper.cpp's
 * IsStereoRenderTargetTexture / IsStereoRenderTargetSurface /
 * IsStereoDepthStencilTexture / IsStereoDepthStencilSurface, but exposed
 * as plain free functions so the new COM-layer Resource11Proxy /
 * Texture2D11Proxy / etc. can decide whether to create a right-eye
 * sibling at construction time. The legacy DDI ResourceWrapper keeps
 * its own copies for the UseCOMWrap=0 fallback path; we don't modify
 * it. When Stage 4 ports per-eye routing onto the proxies, the
 * config knobs (gInfo.MonoRenderTargetTextures, CreateSquareRTInMono,
 * etc.) become the single source of truth for both paths.
 */

#pragma once

#include <dxgi.h>
#include <d3d10.h>
#include "..\S3DAPI\GlobalData.h"

namespace wiz3d
{

inline bool IsBlockCompressedFormat(DXGI_FORMAT f)
{
    return (f >= DXGI_FORMAT_BC1_TYPELESS && f <= DXGI_FORMAT_BC5_SNORM) ||
           (f >= DXGI_FORMAT_BC6H_TYPELESS && f <= DXGI_FORMAT_BC7_UNORM_SRGB);
}

inline bool IsSquareSize_Stage3(UINT Width, UINT Height, BOOL CreateBigInStereo, const SIZE* pBBSize)
{
    if (Width != Height) return false;
    if (pBBSize)
    {
        if (pBBSize->cx == pBBSize->cy && Width == (UINT)pBBSize->cx) return false;
        if (CreateBigInStereo && (Width >= (UINT)pBBSize->cx && Height >= (UINT)pBBSize->cy)) return false;
    }
    return true;
}

inline bool IsLessThanBB_Stage3(UINT Width, UINT Height, const SIZE* pBBSize)
{
    if (!pBBSize) return false;
    if (Width == Height) return false;
    if ((Width + 5) >= (UINT)pBBSize->cx || (Height + 5) >= (UINT)pBBSize->cy) return false;
    float aRT = (1.0f * Width / Height);
    float aBB = (1.0f * pBBSize->cx / pBBSize->cy);
    float f   = aRT > aBB ? aRT - aBB : aBB - aRT;
    return f >= 0.01f;
}

inline bool IsShadowFormat_Stage3(DXGI_FORMAT Format)
{
    switch (Format)
    {
        case DXGI_FORMAT_R32_TYPELESS:
        case DXGI_FORMAT_R32_FLOAT:
        case DXGI_FORMAT_R32_UINT:
        case DXGI_FORMAT_R32_SINT:
            return gInfo.CreateOneComponentRTInMono != 0;
        default:
            return false;
    }
}

// Render-target Texture2D: decides whether to double the resource for stereo.
// Mirrors ResourceWrapper::IsStereoRenderTargetTexture (S3DWrapper10/
// ResourceWrapper.cpp:207) at the COM layer.
// outReason names the first heuristic that vetoed doubling (diagnostics only).
inline bool ShouldDoubleRenderTargetTexture(DXGI_FORMAT Format, UINT Width, UINT Height, const SIZE* pBBSize, const char** outReason = nullptr)
{
    const char* reason = "";
    bool result = true;
    switch (gInfo.RenderTargetCreationMode)
    {
        case 0: reason = "mode0";  result = false; break;
        case 1: result = true; break;
        default:
            if      (IsShadowFormat_Stage3(Format))                                                                            { reason = "oneComponentFmt"; result = false; }
            else if (gInfo.MonoRenderTargetTextures)                                                                            { reason = "monoRTFlag";      result = false; }
            else if (gInfo.CreateSquareRTInMono && IsSquareSize_Stage3(Width, Height, gInfo.CreateBigSquareRTInStereo, pBBSize)) { reason = "squareRT";        result = false; }
            else if (gInfo.CreateRTThatLessThanBBInMono && IsLessThanBB_Stage3(Width, Height, pBBSize))                          { reason = "lessThanBB";      result = false; }
            else if (Width == 1 || Height == 1)                                                                                 { reason = "1pixel";          result = false; }
            break;
    }
    if (outReason) *outReason = reason;
    return result;
}

// Depth-stencil Texture2D: mirrors ResourceWrapper::IsStereoDepthStencilTexture.
inline bool ShouldDoubleDepthStencilTexture(UINT Width, UINT Height, const SIZE* pBBSize, const char** outReason = nullptr)
{
    const char* reason = "";
    bool result = true;
    switch (gInfo.RenderTargetCreationMode)
    {
        case 0: reason = "mode0"; result = false; break;
        case 1: result = true; break;
        default:
            if      (gInfo.MonoDepthStencilTextures)                                                                            { reason = "monoDSFlag"; result = false; }
            else if (gInfo.CreateSquareDSInMono && IsSquareSize_Stage3(Width, Height, gInfo.CreateBigSquareDSInStereo, pBBSize)) { reason = "squareDS";   result = false; }
            else if (gInfo.CreateDSThatLessThanBBInMono && IsLessThanBB_Stage3(Width, Height, pBBSize))                          { reason = "lessThanBB"; result = false; }
            else if (Width == 1 || Height == 1)                                                                                 { reason = "1pixel";     result = false; }
            break;
    }
    if (outReason) *outReason = reason;
    return result;
}

// Helper: given a Texture2D desc, classify it as a render-target / depth-
// stencil / neither, then dispatch to the matching heuristic. Returns true
// if a right-eye sibling should be created. hasInitialData separates
// resolve/copy destinations (never seeded) from ordinary material textures.
inline bool ShouldDoubleTexture2D(const D3D11_TEXTURE2D_DESC* pDesc, const SIZE* pBBSize, bool hasInitialData = false, const char** outReason = nullptr)
{
    if (outReason) *outReason = "";
    if (!pDesc) return false;
    if ((pDesc->BindFlags & D3D11_BIND_DEPTH_STENCIL) != 0)
        return ShouldDoubleDepthStencilTexture(pDesc->Width, pDesc->Height, pBBSize, outReason);
    if ((pDesc->BindFlags & D3D11_BIND_RENDER_TARGET) != 0)
        return ShouldDoubleRenderTargetTexture(pDesc->Format, pDesc->Width, pDesc->Height, pBBSize, outReason);
    // SRV-only textures created empty, MSAA ones included, are resolve/copy destinations: a mono one flattens both eyes
    // to the left image downstream (BC2's MSAA resolve, BF3's G-buffer depth copy). The Resolve/Copy mirrors fill both.
    // A block-compressed one is a streamed asset (never a render or resolve destination) and stays single.
    if ((pDesc->BindFlags & D3D11_BIND_SHADER_RESOURCE) != 0 &&
        (pDesc->BindFlags & D3D11_BIND_UNORDERED_ACCESS) == 0 &&
        pDesc->Usage == D3D11_USAGE_DEFAULT && !hasInitialData &&
        pDesc->Width >= 640 && pDesc->Height >= 360 && !IsBlockCompressedFormat(pDesc->Format))
        return true;
    // UAV textures stay single, as NVIDIA's heuristics have it: a compute output written outside a duplicated
    // dispatch would leave its sibling uninitialised.
    if (outReason) *outReason = "noRTBind";
    return false;
}

// D3D10 overload — same flag values as D3D11 (D3D10_BIND_RENDER_TARGET ==
// D3D11_BIND_RENDER_TARGET == 0x20, D3D10_BIND_DEPTH_STENCIL == 0x40) so we
// share the underlying decisions; only the desc type changes.
inline bool ShouldDoubleTexture2D(const D3D10_TEXTURE2D_DESC* pDesc, const SIZE* pBBSize)
{
    if (!pDesc) return false;
    if ((pDesc->BindFlags & D3D10_BIND_DEPTH_STENCIL) != 0)
        return ShouldDoubleDepthStencilTexture(pDesc->Width, pDesc->Height, pBBSize);
    if ((pDesc->BindFlags & D3D10_BIND_RENDER_TARGET) != 0)
        return ShouldDoubleRenderTargetTexture(pDesc->Format, pDesc->Width, pDesc->Height, pBBSize);
    return false;
}

} // namespace wiz3d
