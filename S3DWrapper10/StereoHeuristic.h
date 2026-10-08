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

// A shadow atlas is a depth target whose shape is unrelated to the screen. The
// old backbuffer-AREA test answered differently at 1080p and 4K for one atlas.
inline bool IsShadowAtlasShape(UINT Width, UINT Height, const SIZE* pBBSize)
{
    if (!pBBSize || pBBSize->cx <= 0 || pBBSize->cy <= 0) return false;
    if (Height == 0) return false;
    if (Width == (UINT)pBBSize->cx && Height == (UINT)pBBSize->cy) return false;
    const float aRT = (float)Width / (float)Height;
    const float aBB = (float)pBBSize->cx / (float)pBBSize->cy;
    const float ratio = (aRT > aBB) ? aRT / aBB : aBB / aRT;
    // 1.5 clears a square page against any widescreen backbuffer (16:9 gives
    // 1.78) while leaving scene-shaped targets -- full, half or quarter res --
    // well inside.
    return ratio >= 1.5f;
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
            // Shadow atlases: light-camera constants never shift, so a sibling
            // is identical (GTA V: ~450 draws/frame). ShadowAtlasByShape is off
            // until measured -- it makes strictly MORE depth targets mono, and a
            // wrong call there flattens a real per-eye buffer.
            if (gInfo.ShadowAtlasByShape
                    ? IsShadowAtlasShape(Width, Height, pBBSize)
                    : (pBBSize && (ULONGLONG)Width * Height >=
                                  4ull * (ULONGLONG)pBBSize->cx * (ULONGLONG)pBBSize->cy))
                { reason = "shadowAtlas"; result = false; }
            else if (gInfo.MonoDepthStencilTextures)                                                                            { reason = "monoDSFlag"; result = false; }
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
    // SRV-only scene-sized textures with no seed data are resolve/copy
    // destinations. BC2's MSAA resolve funnelled both eyes through a mono
    // one — everything downstream sampled the left scene and the output
    // went byte-identical. Doubling the dst lets the existing Resolve/Copy
    // mirroring route each eye's content through automatically.
    // MSAA ones too: only a copy can fill them (BF3 copies its MSAA G-buffer depth into one, resolves it for both eyes,
    // and the right eye lit and ambient-occluded the left eye's depth).
    if (gInfo.DoubleResolveTargets &&
        (pDesc->BindFlags & D3D11_BIND_SHADER_RESOURCE) != 0 &&
        (pDesc->BindFlags & D3D11_BIND_UNORDERED_ACCESS) == 0 &&
        pDesc->Usage == D3D11_USAGE_DEFAULT && !hasInitialData &&
        pDesc->Width >= 640 && pDesc->Height >= 360)
        return true;
    // UAV-only textures are deliberately not doubled: NVIDIA's 3D Vision
    // heuristics cover render targets and depth only, and a compute output
    // written once outside a duplicated dispatch would leave the sibling
    // uninitialised. GTA V has none of these anyway.
    if (outReason) *outReason = "noRTBind";
    return false;
}

// D3D10 overload — same flag values as D3D11 (D3D10_BIND_RENDER_TARGET ==
// D3D11_BIND_RENDER_TARGET == 0x20, D3D10_BIND_DEPTH_STENCIL == 0x40) so we
// share the underlying decisions; only the desc type changes.
inline bool ShouldDoubleTexture2D(const D3D10_TEXTURE2D_DESC* pDesc, const SIZE* pBBSize, bool hasInitialData = false)
{
    if (!pDesc) return false;
    if ((pDesc->BindFlags & D3D10_BIND_DEPTH_STENCIL) != 0)
        return ShouldDoubleDepthStencilTexture(pDesc->Width, pDesc->Height, pBBSize);
    if ((pDesc->BindFlags & D3D10_BIND_RENDER_TARGET) != 0)
        return ShouldDoubleRenderTargetTexture(pDesc->Format, pDesc->Width, pDesc->Height, pBBSize);
    // Same resolve/copy-destination rule as the D3D11 overload above.
    if (gInfo.DoubleResolveTargets &&
        (pDesc->BindFlags & D3D10_BIND_SHADER_RESOURCE) != 0 &&
        pDesc->Usage == D3D10_USAGE_DEFAULT && !hasInitialData &&
        pDesc->Width >= 640 && pDesc->Height >= 360)
        return true;
    return false;
}

} // namespace wiz3d
