/* wiz3D - DX11 ShiftFinder implementation (see ShiftFinder11.h) */

#include "StdAfx.h"
#include "ShiftFinder11.h"

#define INTERLACE_STEP (4)

namespace wiz3d
{

ShiftFinder11::ShiftFinder11()
{
}

void ShiftFinder11::Clear()
{
    // Stop the worker thread + free the correlator buffers BEFORE releasing
    // the staging texture: the worker may be mid-Map on it (base::Clear
    // joins the thread).
    ShiftFinder::Clear();

    if (m_staging) { m_staging->Release(); m_staging = nullptr; }
    m_dev      = nullptr;
    m_ctx      = nullptr;
    m_bbFormat = DXGI_FORMAT_UNKNOWN;
}

HRESULT ShiftFinder11::Initialize(ID3D11Device* dev,
                                  ID3D11DeviceContext* ctx,
                                  const RECT* pLeftRect,
                                  const RECT* pRightRect,
                                  DXGI_FORMAT bbFormat)
{
    if (m_bInitializationSuccessful) return S_OK;
    if (!dev || !ctx || !pLeftRect || !pRightRect) return E_INVALIDARG;

    // base::Initialize spins the worker thread + allocates the correlator
    // image buffers against the given RECTs. Needs non-const RECT* in the
    // base signature, so cast through -- base does not mutate.
    HRESULT hr = ShiftFinder::Initialize(const_cast<RECT*>(pLeftRect),
                                         const_cast<RECT*>(pRightRect));
    if (FAILED(hr) || !m_bInitializationSuccessful) return hr;

    m_dev      = dev;
    m_ctx      = ctx;
    m_bbFormat = bbFormat;

    // DX11 immediate context is single-threaded; the worker only runs the SSE
    // correlator over CPU-side buffers, never touches the context. Matches
    // the legacy DX10 toggle at AutoFocus.cpp:51.
    m_bDeviceIsMultithreaded = false;

    // Convert the base-class RECTs into DX11 copy boxes.
    m_srcBoxLeft.left   = (UINT)m_srcRectLeft.left;
    m_srcBoxLeft.top    = (UINT)m_srcRectLeft.top;
    m_srcBoxLeft.right  = (UINT)m_srcRectLeft.right;
    m_srcBoxLeft.bottom = (UINT)m_srcRectLeft.bottom;
    m_srcBoxLeft.front  = 0;
    m_srcBoxLeft.back   = 1;

    m_srcBoxRight.left   = (UINT)m_srcRectRight.left;
    m_srcBoxRight.top    = (UINT)m_srcRectRight.top;
    m_srcBoxRight.right  = (UINT)m_srcRectRight.right;
    m_srcBoxRight.bottom = (UINT)m_srcRectRight.bottom;
    m_srcBoxRight.front  = 0;
    m_srcBoxRight.back   = 1;

    hr = CreateMemoryBufferTexture(m_SearchAreaSizeX, 2 * m_FrameSizeY * INTERLACE_STEP);
    if (FAILED(hr))
    {
        m_bInitializationSuccessful = false;
        return hr;
    }
    return S_OK;
}

HRESULT ShiftFinder11::CreateMemoryBufferTexture(int width, int height)
{
    if (!m_dev) return E_FAIL;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = (UINT)width;
    td.Height           = (UINT)height;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_R8G8B8A8_UNORM;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_STAGING;
    td.BindFlags        = 0;
    td.CPUAccessFlags   = D3D11_CPU_ACCESS_READ;
    td.MiscFlags        = 0;

    return m_dev->CreateTexture2D(&td, nullptr, &m_staging);
}

void ShiftFinder11::FindShift(ID3D11Texture2D* leftBB,
                              ID3D11Texture2D* rightBB,
                              UINT currentFrame)
{
    if (!m_bInitializationSuccessful) return;
    if (!leftBB || !rightBB || !m_staging || !m_ctx) return;
    // base::FindShift gates on m_bThreadReady: the worker is idle and we may
    // queue a new sample. Mirror the legacy DX10 guard order.
    if (!m_bThreadReady) return;

    m_ctx->CopySubresourceRegion(m_staging, 0,
                                 (UINT)m_dstPointLeft.x, (UINT)m_dstPointLeft.y, 0,
                                 leftBB, 0, &m_srcBoxLeft);
    m_ctx->CopySubresourceRegion(m_staging, 0,
                                 (UINT)m_dstPointRight.x, (UINT)m_dstPointRight.y, 0,
                                 rightBB, 0, &m_srcBoxRight);

    ShiftFinder::FindShift(currentFrame);
}

bool ShiftFinder11::GetRTData()
{
    if (!m_staging || !m_ctx) return false;

    D3D11_MAPPED_SUBRESOURCE mapped = {};
    HRESULT hr = m_ctx->Map(m_staging, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr) || !mapped.pData) return false;

    const DWORD interlacedPitch = mapped.RowPitch * INTERLACE_STEP;
    BYTE* srcL = (BYTE*)mapped.pData;
    BYTE* srcR = (BYTE*)mapped.pData + m_FrameSizeY * interlacedPitch;

    // Format-specific luminance conversion. R8G8B8A8_UNORM (+ _SRGB) is the
    // common swap-chain format; B8G8R8A8 variants map to the X8R8G8B8 (BGRX)
    // converter -- numerically fine ignoring sRGB gamma for luminance. HDR
    // float BBs (R16G16B16A16_FLOAT) are rare on game BBs and skipped --
    // AutoFocus/LaserSight no-op on those until a float converter is added.
    switch (m_bbFormat)
    {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_R8G8B8A8_TYPELESS:
        convert_R8G8B8A8_ToMonochrome(srcL, interlacedPitch, m_ImageLeft,  m_FrameSizeX);
        convert_R8G8B8A8_ToMonochrome(srcR, interlacedPitch, m_ImageRight, m_SearchAreaSizeX);
        break;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_TYPELESS:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_TYPELESS:
        convert_X8R8G8B8_ToMonochrome(srcL, interlacedPitch, m_ImageLeft,  m_FrameSizeX);
        convert_X8R8G8B8_ToMonochrome(srcR, interlacedPitch, m_ImageRight, m_SearchAreaSizeX);
        break;
    default:
        // Unknown format: unmap and bail without a result. The correlator
        // will see empty buffers and leave m_bCurrentShiftValid=false.
        m_ctx->Unmap(m_staging, 0);
        return false;
    }

    m_ctx->Unmap(m_staging, 0);
    return true;
}

} // namespace wiz3d
