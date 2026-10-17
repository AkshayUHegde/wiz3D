/* wiz3D - IDXGISwapChain proxy implementation (Option B Stage 4b.2) */

#include "StdAfx.h"
#include "SwapChain11Proxy.h"
#include "Device11Proxy.h"
#include "Context11Proxy.h"
#include "Texture2D11Proxy.h"
#include "proxy_factory.h"     // IID_wiz3D_SwapChain11Proxy (declared in 4b.2 update)
#include "AdapterFunctions.h"  // DDILog
#include "..\S3DAPI\ReadData.h"      // WriteInputData — persist hotkey-tuned stereo
#include "..\S3DAPI\KeyboardHook.h"  // gKbdHook — the DX9 paths' hotkey thread
#include "..\S3DAPI\ConvertImages.h"  // GetBitWidthOfDXGIFormat
#include "Utils.h"                    // SaveImageToFile, screenshot extensions
#include "ProductNames.h"
#include <Shlobj.h>
#include <d3dcompiler.h>
#include "SR.hpp"                    // Simulated Reality weave (delay-loaded runtime)
#include "SRDisplayInfo.h"           // LeiaSR Displays runtime: recommended views-texture dimensions

#pragma comment(lib, "dxguid.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace wiz3d
{

SwapChain11Proxy::SwapChain11Proxy(IDXGISwapChain* real, IDXGISwapChain1* real1, Device11Proxy* parent)
    : m_real(real)
    , m_real1(real1)
    , m_real2(nullptr)
    , m_real3(nullptr)
    , m_real4(nullptr)
    , m_parent(parent)
    , m_refs(1)
    , m_leftBB(nullptr)
    , m_rightBB(nullptr)
    , m_wrappedBB(nullptr)
    , m_leftSRV(nullptr)
    , m_rightSRV(nullptr)
    , m_realBBRTV(nullptr)
    , m_bbWidth(0)
    , m_bbHeight(0)
    , m_bbFormat(DXGI_FORMAT_UNKNOWN)
    , m_compositeVS(nullptr)
    , m_compositeSampler(nullptr)
    , m_compositeRaster(nullptr)
    , m_compositeBlend(nullptr)
    , m_compositeDepthStencil(nullptr)
{
    // Stage 4b.8 fix: AddRef the parent device. Without this, if the game
    // releases its Device11Proxy reference before the swap chain proxy,
    // m_parent dangles and the next Present hook reads garbage (BioShock
    // crashed at OnPresentBoundaryPre line 92 with ECX=1 — Device11Proxy
    // memory had been freed and m_ctxProxy was reading reused heap data).
    if (m_parent) m_parent->AddRef();
    // The DX9 paths' KeyboardHook: refcounted, own polling thread, writes
    // gInfo.Input — which is what the COM-wrap stereo math reads.
    gKbdHook.initialize(&gInfo.Input);
    gKbdHook.m_Access.Enter();
    m_savedInput = gInfo.Input;
    gKbdHook.m_Access.Leave();
    // Cache the higher-version chains so QI can return `this` for them.
    if (m_real)
    {
        if (FAILED(m_real->QueryInterface(__uuidof(IDXGISwapChain2), reinterpret_cast<void**>(&m_real2)))) m_real2 = nullptr;
        if (FAILED(m_real->QueryInterface(__uuidof(IDXGISwapChain3), reinterpret_cast<void**>(&m_real3)))) m_real3 = nullptr;
        if (FAILED(m_real->QueryInterface(__uuidof(IDXGISwapChain4), reinterpret_cast<void**>(&m_real4)))) m_real4 = nullptr;
    }
    DDILog("SwapChain11Proxy ctor: real=%p real1=%p real2=%p real3=%p real4=%p parent=%p\n",
           real, real1, m_real2, m_real3, m_real4, parent);
    // SwapEffect >= 3 is flip-model, where the game rotates through buffers
    // by index instead of always drawing to buffer 0.
    DXGI_SWAP_CHAIN_DESC scd = {};
    if (m_real && SUCCEEDED(m_real->GetDesc(&scd)))
        DDILog("  SwapChain desc: BufferCount=%u %ux%u fmt=%d SwapEffect=%d Flags=0x%X\n",
               scd.BufferCount, scd.BufferDesc.Width, scd.BufferDesc.Height,
               (int)scd.BufferDesc.Format, (int)scd.SwapEffect, scd.Flags);
}

SwapChain11Proxy::~SwapChain11Proxy()
{
    // Persist hotkey-tuned stereo values: the legacy DDI wrapper saves them on
    // teardown, but the COM-wrap path never did, so tuning died with the run.
    WriteInputData(&gInfo.Input);
    gKbdHook.clear();
    for (ID3D11Query*& q : m_frameEvents) if (q) { q->Release(); q = nullptr; }
    ReleaseSRWeave(true);
    ReleaseStereoBackBuffer();
    ReleaseComposite();
    if (m_real4)  { m_real4->Release();  m_real4  = nullptr; }
    if (m_real3)  { m_real3->Release();  m_real3  = nullptr; }
    if (m_real2)  { m_real2->Release();  m_real2  = nullptr; }
    if (m_real1)  { m_real1->Release();  m_real1  = nullptr; }
    if (m_real)   { m_real->Release();   m_real   = nullptr; }
    if (m_parent) { m_parent->Release(); m_parent = nullptr; }
}

ULONG STDMETHODCALLTYPE SwapChain11Proxy::Release()
{
    LONG r = InterlockedDecrement(&m_refs);
    if (r == 0) delete this;
    return (ULONG)r;
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::QueryInterface(REFIID riid, void** ppvObj)
{
    if (!ppvObj) return E_POINTER;
    if (riid == IID_IUnknown          ||
        riid == IID_IDXGIObject       ||
        riid == IID_IDXGIDeviceSubObject ||
        riid == IID_IDXGISwapChain)
    {
        *ppvObj = static_cast<IDXGISwapChain*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == IID_IDXGISwapChain1 && m_real1)
    {
        *ppvObj = static_cast<IDXGISwapChain1*>(this);
        AddRef();
        return S_OK;
    }
    // Only claim a version the real chain actually supports, so a game can
    // still feature-detect correctly on older runtimes.
    if (riid == __uuidof(IDXGISwapChain2) && m_real2)
    {
        *ppvObj = static_cast<IDXGISwapChain2*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == __uuidof(IDXGISwapChain3) && m_real3)
    {
        *ppvObj = static_cast<IDXGISwapChain3*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == __uuidof(IDXGISwapChain4) && m_real4)
    {
        *ppvObj = static_cast<IDXGISwapChain4*>(this);
        AddRef();
        return S_OK;
    }
    // Stage 4b.2: private identity IID — used by the dxgi.dll-side factory
    // hook (4b.3) to detect "is this swap chain one of ours?" cross-DLL.
    if (riid == IID_wiz3D_SwapChain11Proxy)
    {
        *ppvObj = static_cast<IUnknown*>(static_cast<IDXGISwapChain*>(this));
        AddRef();
        return S_OK;
    }
    // Anything still unhandled goes to the real chain unwrapped, which lets
    // the game drive it behind our back. Logged so those cases stay visible.
    HRESULT hr = m_real->QueryInterface(riid, ppvObj);
    static int s_logged = 0;
    if (s_logged < 8)
    {
        char iidName[64];
        FormatGUID(riid, iidName, sizeof(iidName));
        DDILog("  SwapChain11Proxy::QI(%s) hr=0x%08lX -- bypass risk\n", iidName, hr);
        ++s_logged;
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::GetDevice(REFIID riid, void** ppDevice)
{
    // COM identity: route ID3D11Device QI through our wrapped device, so a
    // game that fetches the device via sc->GetDevice keeps using our proxy.
    if (!ppDevice) return E_POINTER;
    if (riid == __uuidof(ID3D11Device) && m_parent)
    {
        return m_parent->QueryInterface(riid, ppDevice);
    }
    return m_real->GetDevice(riid, ppDevice);
}

// Stage 4d: HLSL source for the SBS composite. VS emits a fullscreen triangle
// covering NDC [-1,1]^2; PS samples the left-eye texture into the left half of
// the BB and the right-eye texture into the right half (each eye horizontally
// scaled by 2x to fit). Two separate sampler calls keep the branch out of the
// flow — both sides issue one read regardless of texcoord.
// All composite shaders share this VS and the t0/t1/s0 bindings. Each
// pixel-shader entry point (PSMain_SBS, PSMain_TB, ...) is compiled
// separately by EnsureComposite into its own ID3D11PixelShader, with
// one PS per OutputMode group that uses the same sampling pattern.
//
// SBS and TB both branchlessly sample both eyes via lerp so the branch
// doesn't appear in generated code. Half vs Full is just a user config
// choice: at Half the game renders at display width/height and each
// eye gets squeezed; at Full the game renders at half width/height so
// each eye lands at native resolution in its half of the output.
static const char* k_compositeShaderSrc = R"(
struct VSOut {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut VSMain(uint id : SV_VertexID) {
    VSOut o;
    o.uv  = float2((id << 1) & 2, id & 2);
    o.pos = float4(o.uv * float2(2, -2) + float2(-1, 1), 0, 1);
    return o;
}

Texture2D    leftTex  : register(t0);
Texture2D    rightTex : register(t1);
SamplerState samp     : register(s0);

// Modes 1 (Half SBS, default) and 2 (Full SBS). Left eye in left half,
// right eye in right half; each eye UV scaled 2x horizontally.
float4 PSMain_SBS(VSOut input) : SV_Target {
    float2 uv = input.uv;
    float  side = step(0.5, uv.x);   // 0 = left half, 1 = right half
    float2 leftSample  = float2(uv.x * 2.0,            uv.y);
    float2 rightSample = float2((uv.x - 0.5) * 2.0,    uv.y);
    float4 colL = leftTex.Sample (samp, leftSample);
    float4 colR = rightTex.Sample(samp, rightSample);
    return lerp(colL, colR, side);
}

// Modes 3 (Half TB) and 4 (Full TB). Left eye on top, right eye on
// bottom; each eye UV scaled 2x vertically. SBS logic rotated 90 deg.
float4 PSMain_TB(VSOut input) : SV_Target {
    float2 uv = input.uv;
    float  side = step(0.5, uv.y);   // 0 = top half, 1 = bottom half
    float2 leftSample  = float2(uv.x, uv.y * 2.0);
    float2 rightSample = float2(uv.x, (uv.y - 0.5) * 2.0);
    float4 colL = leftTex.Sample (samp, leftSample);
    float4 colR = rightTex.Sample(samp, rightSample);
    return lerp(colL, colR, side);
}

// Mode 5: Anaglyph Red/Cyan (Dubois colour-optimized). Operates in
// linear-light space (gamma-correct), re-encodes to sRGB. Matches the
// OGL path's shipped anaglyph choice -- earlier grayscale and naive
// full-colour variants were retired for worse retinal rivalry /
// ghosting. The Dubois coefficients are embedded as literals so no
// constant buffer is needed at Present time.
float4 PSMain_Anaglyph(VSOut input) : SV_Target {
    float4 cL = leftTex.Sample (samp, input.uv);
    float4 cR = rightTex.Sample(samp, input.uv);
    float3 L = pow(cL.rgb, 2.2);
    float3 R = pow(cR.rgb, 2.2);
    float r = dot(L, float3( 0.4561, 0.5008, 0.0526))
            + dot(R, float3(-0.0434,-0.0083, 0.0003));
    float g = dot(L, float3(-0.0565,-0.0034, 0.0017))
            + dot(R, float3( 0.3784, 0.7334,-0.0162));
    float b = dot(L, float3(-0.0103,-0.0087, 0.0009))
            + dot(R, float3(-0.0258,-0.0450, 0.9741));
    float3 c = pow(max(float3(r, g, b), 0.0), 0.4545);
    return float4(c, 1.0);
}

// Mode 6: Row Interleaved (passive 3D TVs, row-polarised monitors).
// Even rows carry the left eye, odd rows the right. Screen-space row
// parity via SV_Position.y so the pattern is independent of texture
// sampling; the caller binds a point sampler at s0 so eye boundaries
// aren't smeared by linear filtering.
float4 PSMain_RowInterleaved(VSOut input) : SV_Target {
    float4 cL = leftTex.Sample (samp, input.uv);
    float4 cR = rightTex.Sample(samp, input.uv);
    bool isOddRow = (uint(input.pos.y) & 1u) != 0u;
    return isOddRow ? cR : cL;
}

// Mode 7: Column Interleaved. Like row-interleaved but on vertical
// stripes; used by rare column-polarised displays.
float4 PSMain_ColumnInterleaved(VSOut input) : SV_Target {
    float4 cL = leftTex.Sample (samp, input.uv);
    float4 cR = rightTex.Sample(samp, input.uv);
    bool isOddCol = (uint(input.pos.x) & 1u) != 0u;
    return isOddCol ? cR : cL;
}

// Mode 8: Checkerboard. XOR of x+y pixel parity; DLP-Link 3D projectors
// decode this directly.
float4 PSMain_Checkerboard(VSOut input) : SV_Target {
    float4 cL = leftTex.Sample (samp, input.uv);
    float4 cR = rightTex.Sample(samp, input.uv);
    bool white = ((uint(input.pos.x) + uint(input.pos.y)) & 1u) != 0u;
    return white ? cR : cL;
}
)";

HRESULT SwapChain11Proxy::EnsureStereoBackBuffer()
{
    // Stage 4d: allocate the wiz3D LEFT and RIGHT BB sibling textures the
    // first time GetBuffer(0) is called. Done lazily (not in ctor) so we
    // pick up the real swap chain's current desc — for one-call factory
    // creation the desc is known but for ResizeBuffers we have to wait
    // for the next GetBuffer call after the resize.
    if (m_leftBB && m_rightBB) return S_OK;
    if (!m_real || !m_parent) return E_FAIL;

    DXGI_SWAP_CHAIN_DESC desc = {};
    HRESULT hr = m_real->GetDesc(&desc);
    if (FAILED(hr)) return hr;

    m_bbWidth  = desc.BufferDesc.Width;
    m_bbHeight = desc.BufferDesc.Height;
    m_bbFormat = desc.BufferDesc.Format;
    m_realBBW  = m_bbWidth;
    m_realBBH  = m_bbHeight;

    // SR Weave output needs the real BB at panel-native pixel pitch so the
    // interlace pattern lines up with the lenslets. The LeiaSR Displays
    // runtime reports the recommended SBS view-texture dimensions (equal
    // to 2*panelW x panelH on the physical display); resize the real swap
    // chain to half of that. m_bbWidth/m_bbHeight stay at the game's
    // requested dims so GetDesc/m_leftBB/m_rightBB keep the game-side
    // view intact; the compositor + weave write at m_realBBW/m_realBBH.
    if (SelectedOutputMode() == OM_SR_WEAVE)
    {
        int srRecW = 0, srRecH = 0;
        if (wiz3d::QueryRecommendedSRViewsSize(srRecW, srRecH))
        {
            const UINT panelW = static_cast<UINT>(srRecW / 2);
            const UINT panelH = static_cast<UINT>(srRecH);
            if (panelW > 0 && panelH > 0 &&
                (panelW != m_bbWidth || panelH != m_bbHeight))
            {
                DDILog("SwapChain11Proxy::EnsureStereoBackBuffer: SR Weave selected; resizing real BB %ux%u -> panel-native %ux%u\n",
                       m_bbWidth, m_bbHeight, panelW, panelH);
                HRESULT rhr = m_real->ResizeBuffers(0, panelW, panelH, DXGI_FORMAT_UNKNOWN, desc.Flags);
                if (SUCCEEDED(rhr))
                {
                    m_realBBW = panelW;
                    m_realBBH = panelH;
                }
                else
                {
                    DDILog("SwapChain11Proxy::EnsureStereoBackBuffer: real BB resize failed 0x%08lX; staying at game dims (weave will be scaled downstream)\n", rhr);
                }
            }
        }
    }

    // The ZPS hotkey step is expressed in pixels and scaled by this width.
    gKbdHook.setBackBufferWidth(m_bbWidth);
    m_parent->SetPresentSize(m_bbWidth, m_bbHeight);

    ID3D11Device* dev = m_parent->GetReal();
    if (!dev) return E_FAIL;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = m_bbWidth;
    td.Height           = m_bbHeight;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = m_bbFormat;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DEFAULT;
    // RT so both eyes draw into them, SRV so the output pass samples them, UAV when the game asked the swap chain for
    // one (Frostbite 3 writes its final pass through a UAV on the back buffer; Andromeda stops without one).
    td.BindFlags        = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    if (desc.BufferUsage & DXGI_USAGE_UNORDERED_ACCESS) td.BindFlags |= D3D11_BIND_UNORDERED_ACCESS;

    HRESULT hrL = dev->CreateTexture2D(&td, nullptr, &m_leftBB);
    HRESULT hrR = dev->CreateTexture2D(&td, nullptr, &m_rightBB);
    if (FAILED(hrL) || FAILED(hrR))
    {
        DDILog("SwapChain11Proxy::EnsureStereoBackBuffer: CreateTexture2D failed hrL=0x%08lX hrR=0x%08lX\n", hrL, hrR);
        if (m_leftBB)  { m_leftBB->Release();  m_leftBB  = nullptr; }
        if (m_rightBB) { m_rightBB->Release(); m_rightBB = nullptr; }
        return FAILED(hrL) ? hrL : hrR;
    }

    HRESULT hrSL = dev->CreateShaderResourceView(m_leftBB,  nullptr, &m_leftSRV);
    HRESULT hrSR = dev->CreateShaderResourceView(m_rightBB, nullptr, &m_rightSRV);
    if (FAILED(hrSL) || FAILED(hrSR))
    {
        DDILog("SwapChain11Proxy::EnsureStereoBackBuffer: CreateSRV failed hrSL=0x%08lX hrSR=0x%08lX\n", hrSL, hrSR);
        ReleaseStereoBackBuffer();
        return FAILED(hrSL) ? hrSL : hrSR;
    }

    // Wrap the pair as a Texture2D11Proxy so the rest of the wiz3D wrap
    // machinery (CreateRenderTargetView unwrap + DoOMSet eye routing) sees
    // it as a normal stereo-doubled texture. m_wrappedBB is a strong ref
    // we hold for the lifetime of the swap chain; GetBuffer just AddRefs.
    //
    // Texture2D11Proxy's ctor takes ownership of the refs passed in (its
    // dtor calls Release on m_realLeft/m_realRight) — so before handing
    // ours over we AddRef once more, so SwapChain still holds an
    // independent ref via m_leftBB / m_rightBB. Without this AddRef both
    // SwapChain and the proxy think they own the same single ref, and
    // ReleaseStereoBackBuffer's wrappedBB->Release() + leftBB->Release()
    // chain ends up double-releasing the texture (BioShock + DA2 crashed
    // on the second GetBuffer call when EnsureStereoBackBuffer
    // re-entered after a ResizeBuffers — the cached m_leftBB pointer
    // pointed at freed memory and the d3d11 runtime crashed inside
    // CreateShaderResourceView / OMSetRenderTargets).
    m_leftBB->AddRef();
    m_rightBB->AddRef();
    m_wrappedBB = new Texture2D11Proxy(m_leftBB, m_rightBB, m_parent);
    m_wrappedBB->MarkBackBuffer();
    m_wrappedBB->MakeRefInternal();   // PrivateBackBufferRefs: ours is DXGI's own reference, not one the game counts
    // Views on the pair resolve to this proxy like any other texture's (the overlay rule asks by the real left).
    if (m_parent) m_parent->RegisterRealToProxy(m_leftBB, m_wrappedBB);

    // RTV on the REAL swap-chain back buffer — used as the composite
    // destination. Acquired now so we don't pay the cost mid-frame.
    ID3D11Texture2D* realBBTex = nullptr;
    hr = m_real->GetBuffer(0, __uuidof(ID3D11Texture2D), reinterpret_cast<void**>(&realBBTex));
    if (FAILED(hr) || !realBBTex)
    {
        DDILog("SwapChain11Proxy::EnsureStereoBackBuffer: real GetBuffer failed 0x%08lX\n", hr);
        ReleaseStereoBackBuffer();
        return hr;
    }
    hr = dev->CreateRenderTargetView(realBBTex, nullptr, &m_realBBRTV);
    realBBTex->Release();
    if (FAILED(hr))
    {
        DDILog("SwapChain11Proxy::EnsureStereoBackBuffer: real BB RTV create failed 0x%08lX\n", hr);
        ReleaseStereoBackBuffer();
        return hr;
    }

    DDILog("SwapChain11Proxy::EnsureStereoBackBuffer: leftBB=%p rightBB=%p wrappedBB=%p (%ux%u fmt=%d)\n",
           m_leftBB, m_rightBB, m_wrappedBB, m_bbWidth, m_bbHeight, (int)m_bbFormat);
    return S_OK;
}

void SwapChain11Proxy::ReleaseStereoBackBuffer()
{
    if (m_realBBRTV) { m_realBBRTV->Release(); m_realBBRTV = nullptr; }
    if (m_leftSRV)   { m_leftSRV->Release();   m_leftSRV   = nullptr; }
    if (m_rightSRV)  { m_rightSRV->Release();  m_rightSRV  = nullptr; }
    if (m_wrappedBB && m_parent && m_leftBB) m_parent->UnregisterRealToProxy(m_leftBB);
    if (m_wrappedBB) { m_wrappedBB->ReleaseInternal(); m_wrappedBB = nullptr; }
    if (m_leftBB)    { m_leftBB->Release();    m_leftBB    = nullptr; }
    if (m_rightBB)   { m_rightBB->Release();   m_rightBB   = nullptr; }
    m_bbWidth = m_bbHeight = 0;
    m_realBBW = m_realBBH = 0;
    m_bbFormat = DXGI_FORMAT_UNKNOWN;
}

// Local helper: compile one PSMain entry from k_compositeShaderSrc and
// turn the blob into an ID3D11PixelShader. Keeps EnsureComposite flat as
// the number of modes grows.
static HRESULT CompileCompositePS(ID3D11Device* dev, const char* entry,
                                  ID3D11PixelShader** out)
{
    ID3DBlob* blob = nullptr;
    ID3DBlob* err  = nullptr;
    const UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
    HRESULT hr = D3DCompile(k_compositeShaderSrc, strlen(k_compositeShaderSrc),
                            "wiz3D_composite", nullptr, nullptr,
                            entry, "ps_4_0", flags, 0, &blob, &err);
    if (FAILED(hr))
    {
        DDILog("SwapChain11Proxy::EnsureComposite: PS compile failed for %s 0x%08lX: %s\n",
               entry, hr, err ? (const char*)err->GetBufferPointer() : "(no log)");
        if (err) err->Release();
        return hr;
    }
    if (err) err->Release();
    hr = dev->CreatePixelShader(blob->GetBufferPointer(), blob->GetBufferSize(), nullptr, out);
    blob->Release();
    return hr;
}

HRESULT SwapChain11Proxy::EnsureComposite()
{
    // Stage 4d: lazily compile the composite shaders + pipeline state
    // on first Present where the composite actually runs. D3DCompile is a
    // one-time cost; the shaders are independent of BB format/size so we
    // don't have to re-compile across ResizeBuffers.
    if (m_compositeVS && m_compositePS_SBS && m_compositePS_TB &&
        m_compositePS_Anaglyph && m_compositePS_RowInterleaved &&
        m_compositePS_ColInterleaved && m_compositePS_Checkerboard &&
        m_compositeSampler && m_compositeSamplerPoint) return S_OK;
    if (!m_parent) return E_FAIL;
    ID3D11Device* dev = m_parent->GetReal();
    if (!dev) return E_FAIL;

    ID3DBlob* vsBlob = nullptr; ID3DBlob* err = nullptr;
    const UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;

    HRESULT hrVS = D3DCompile(
        k_compositeShaderSrc, strlen(k_compositeShaderSrc),
        "wiz3D_composite", nullptr, nullptr,
        "VSMain", "vs_4_0", flags, 0, &vsBlob, &err);
    if (FAILED(hrVS))
    {
        DDILog("SwapChain11Proxy::EnsureComposite: VS compile failed 0x%08lX: %s\n",
               hrVS, err ? (const char*)err->GetBufferPointer() : "(no log)");
        if (err) err->Release();
        return hrVS;
    }
    if (err) { err->Release(); err = nullptr; }

    HRESULT hr = dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                          nullptr, &m_compositeVS);
    vsBlob->Release();
    if (FAILED(hr)) return hr;

    hr = CompileCompositePS(dev, "PSMain_SBS",               &m_compositePS_SBS);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }
    hr = CompileCompositePS(dev, "PSMain_TB",                &m_compositePS_TB);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }
    hr = CompileCompositePS(dev, "PSMain_Anaglyph",          &m_compositePS_Anaglyph);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }
    hr = CompileCompositePS(dev, "PSMain_RowInterleaved",   &m_compositePS_RowInterleaved);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }
    hr = CompileCompositePS(dev, "PSMain_ColumnInterleaved", &m_compositePS_ColInterleaved);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }
    hr = CompileCompositePS(dev, "PSMain_Checkerboard",      &m_compositePS_Checkerboard);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }

    D3D11_SAMPLER_DESC sd = {};
    sd.Filter         = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sd.AddressU       = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressV       = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.AddressW       = D3D11_TEXTURE_ADDRESS_CLAMP;
    sd.ComparisonFunc = D3D11_COMPARISON_NEVER;
    sd.MinLOD         = 0.f;
    sd.MaxLOD         = D3D11_FLOAT32_MAX;
    hr = dev->CreateSamplerState(&sd, &m_compositeSampler);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }

    // Point sampler for the interleaved + checkerboard modes: screen-space
    // pixel parity selects the eye, and linear filtering would smear eye
    // boundaries at the half-pixel between stripes.
    D3D11_SAMPLER_DESC sdPoint = sd;
    sdPoint.Filter = D3D11_FILTER_MIN_MAG_MIP_POINT;
    hr = dev->CreateSamplerState(&sdPoint, &m_compositeSamplerPoint);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }

    D3D11_RASTERIZER_DESC rd = {};
    rd.FillMode = D3D11_FILL_SOLID;
    rd.CullMode = D3D11_CULL_NONE;          // fullscreen tri, no cull
    rd.DepthClipEnable = TRUE;
    hr = dev->CreateRasterizerState(&rd, &m_compositeRaster);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }

    D3D11_BLEND_DESC bd = {};
    bd.RenderTarget[0].RenderTargetWriteMask = D3D11_COLOR_WRITE_ENABLE_ALL;
    hr = dev->CreateBlendState(&bd, &m_compositeBlend);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }

    D3D11_DEPTH_STENCIL_DESC dsd = {};
    dsd.DepthEnable    = FALSE;
    dsd.StencilEnable  = FALSE;
    dsd.DepthWriteMask = D3D11_DEPTH_WRITE_MASK_ZERO;
    hr = dev->CreateDepthStencilState(&dsd, &m_compositeDepthStencil);
    if (FAILED(hr)) { ReleaseComposite(); return hr; }

    DDILog("SwapChain11Proxy::EnsureComposite: VS=%p PS(SBS=%p TB=%p Anaglyph=%p Line=%p Col=%p Checker=%p) sampler(linear=%p point=%p)\n",
           m_compositeVS, m_compositePS_SBS, m_compositePS_TB, m_compositePS_Anaglyph,
           m_compositePS_RowInterleaved, m_compositePS_ColInterleaved, m_compositePS_Checkerboard,
           m_compositeSampler, m_compositeSamplerPoint);
    return S_OK;
}

void SwapChain11Proxy::ReleaseComposite()
{
    if (m_compositeDepthStencil) { m_compositeDepthStencil->Release(); m_compositeDepthStencil = nullptr; }
    if (m_compositeBlend)        { m_compositeBlend->Release();        m_compositeBlend        = nullptr; }
    if (m_compositeRaster)       { m_compositeRaster->Release();       m_compositeRaster       = nullptr; }
    if (m_compositeSamplerPoint)      { m_compositeSamplerPoint->Release();      m_compositeSamplerPoint      = nullptr; }
    if (m_compositeSampler)           { m_compositeSampler->Release();           m_compositeSampler           = nullptr; }
    if (m_compositePS_Checkerboard)   { m_compositePS_Checkerboard->Release();   m_compositePS_Checkerboard   = nullptr; }
    if (m_compositePS_ColInterleaved) { m_compositePS_ColInterleaved->Release(); m_compositePS_ColInterleaved = nullptr; }
    if (m_compositePS_RowInterleaved){ m_compositePS_RowInterleaved->Release();m_compositePS_RowInterleaved= nullptr; }
    if (m_compositePS_Anaglyph)       { m_compositePS_Anaglyph->Release();       m_compositePS_Anaglyph       = nullptr; }
    if (m_compositePS_TB)             { m_compositePS_TB->Release();             m_compositePS_TB             = nullptr; }
    if (m_compositePS_SBS)            { m_compositePS_SBS->Release();            m_compositePS_SBS            = nullptr; }
    if (m_compositeVS)           { m_compositeVS->Release();           m_compositeVS           = nullptr; }
    if (m_compositeState)        { m_compositeState->Release();        m_compositeState        = nullptr; }
}

bool SwapChain11Proxy::EnsureCompositeState()
{
    if (m_compositeState) return true;
    ID3D11Device* dev = m_parent ? m_parent->GetReal() : nullptr;
    ID3D11Device1* dev1 = nullptr;
    if (m_compositeStateFailed || !dev ||
        FAILED(dev->QueryInterface(__uuidof(ID3D11Device1), reinterpret_cast<void**>(&dev1))) || !dev1)
    {
        m_compositeStateFailed = true;
        return false;
    }
    const D3D_FEATURE_LEVEL level = dev->GetFeatureLevel();
    const UINT flags = (dev->GetCreationFlags() & D3D11_CREATE_DEVICE_SINGLETHREADED)
                           ? D3D11_1_CREATE_DEVICE_CONTEXT_STATE_SINGLETHREADED : 0;
    const HRESULT hr = dev1->CreateDeviceContextState(flags, &level, 1, D3D11_SDK_VERSION, __uuidof(ID3D11Device),
                                                      nullptr, &m_compositeState);
    dev1->Release();
    if (FAILED(hr) || !m_compositeState)
    {
        DDILog("SwapChain11Proxy::EnsureCompositeState: CreateDeviceContextState failed 0x%08lX -- topology-only restore\n", hr);
        m_compositeState = nullptr;
        m_compositeStateFailed = true;
        return false;
    }
    return true;
}

void SwapChain11Proxy::DoComposite()
{
    // The output pass runs on the real context in a context state of its own, so the game's state comes back
    // exactly as it was (the proxy's tracking stays true).
    if (!m_parent || !m_realBBRTV || !m_leftSRV || !m_rightSRV) return;
    if (FAILED(EnsureComposite())) return;

    ID3D11DeviceContext* ctx = nullptr;
    m_parent->GetReal()->GetImmediateContext(&ctx);
    if (!ctx) return;

    ID3D11DeviceContext1* ctx1 = nullptr;
    ID3DDeviceContextState* gameState = nullptr;
    if (EnsureCompositeState() &&
        SUCCEEDED(ctx->QueryInterface(__uuidof(ID3D11DeviceContext1), reinterpret_cast<void**>(&ctx1))) && ctx1)
        ctx1->SwapDeviceContextState(m_compositeState, &gameState);
    // Without a state of its own, only the topology goes back (MP3's loading screen draws a strip without re-setting it).
    D3D11_PRIMITIVE_TOPOLOGY gameTopology = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED;
    if (!gameState) ctx->IAGetPrimitiveTopology(&gameTopology);
    const OutputMode mode = SelectedOutputMode();
    if (!(mode == OM_SR_WEAVE && RunSRWeave(ctx)))
        DrawCompositePass(ctx, m_realBBRTV, m_realBBW, m_realBBH, mode);
    if (gameState)
    {
        ctx->ClearState();   // our state holds no back-buffer view across a ResizeBuffers
        ctx1->SwapDeviceContextState(gameState, nullptr);
        gameState->Release();
    }
    else if (gameTopology != D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED) ctx->IASetPrimitiveTopology(gameTopology);
    if (ctx1) ctx1->Release();
    ctx->Release();
}

void SwapChain11Proxy::DrawCompositePass(ID3D11DeviceContext* ctx,
                                         ID3D11RenderTargetView* rtv,
                                         UINT width, UINT height,
                                         OutputMode mode)
{
    // Unbind any SRVs that might collide with t0/t1 (the game's PSSet
    // probably already has shader inputs there). Cheap defensive clear.
    ID3D11ShaderResourceView* nullSRVs[2] = { nullptr, nullptr };
    ctx->PSSetShaderResources(0, 2, nullSRVs);

    // Composite pass setup.
    ctx->OMSetRenderTargets(1, &rtv, nullptr);
    D3D11_VIEWPORT vp = {};
    vp.Width    = (float)width;
    vp.Height   = (float)height;
    vp.MinDepth = 0.f;
    vp.MaxDepth = 1.f;
    ctx->RSSetViewports(1, &vp);
    ctx->RSSetState(m_compositeRaster);

    const FLOAT blendFactor[4] = { 1.f, 1.f, 1.f, 1.f };
    ctx->OMSetBlendState(m_compositeBlend, blendFactor, 0xFFFFFFFF);
    ctx->OMSetDepthStencilState(m_compositeDepthStencil, 0);

    // Stereo toggle presents the left eye to both sides; SwapEyes crosses them.
    ID3D11ShaderResourceView* eyeL = m_leftSRV;
    ID3D11ShaderResourceView* eyeR = gInfo.Input.StereoActive ? m_rightSRV : m_leftSRV;
    if (gInfo.Input.SwapEyes && gInfo.Input.StereoActive)
    {
        ID3D11ShaderResourceView* t = eyeL;
        eyeL = eyeR;
        eyeR = t;
    }
    // Pick the pixel shader + sampler for the active OutputMode. The
    // null-checks guard against a shader not yet compiled for a follow-up
    // mode so a mis-configured OutputMode falls back to the SBS path
    // instead of crashing. Modes 6-8 swap in a point sampler so the
    // screen-space pixel parity isn't smeared by linear filtering.
    ID3D11PixelShader*  ps      = m_compositePS_SBS;
    ID3D11SamplerState* sampler = m_compositeSampler;
    switch (mode)
    {
    case OM_HALF_TB:
    case OM_FULL_TB:
        if (m_compositePS_TB) ps = m_compositePS_TB;
        break;
    case OM_ANAGLYPH:
        if (m_compositePS_Anaglyph) ps = m_compositePS_Anaglyph;
        break;
    case OM_ROW_INTERLEAVED:
        if (m_compositePS_RowInterleaved)  ps = m_compositePS_RowInterleaved;
        if (m_compositeSamplerPoint)        sampler = m_compositeSamplerPoint;
        break;
    case OM_COLUMN_INTERLEAVED:
        if (m_compositePS_ColInterleaved)   ps = m_compositePS_ColInterleaved;
        if (m_compositeSamplerPoint)        sampler = m_compositeSamplerPoint;
        break;
    case OM_CHECKERBOARD:
        if (m_compositePS_Checkerboard)     ps = m_compositePS_Checkerboard;
        if (m_compositeSamplerPoint)        sampler = m_compositeSamplerPoint;
        break;
    case OM_HALF_SBS:
    case OM_FULL_SBS:
    default:
        break;
    }

    ID3D11ShaderResourceView* srvs[2] = { eyeL, eyeR };
    ctx->PSSetShaderResources(0, 2, srvs);
    ctx->PSSetSamplers(0, 1, &sampler);

    ctx->VSSetShader(m_compositeVS, nullptr, 0);
    ctx->PSSetShader(ps, nullptr, 0);
    ctx->GSSetShader(nullptr, nullptr, 0);
    ctx->HSSetShader(nullptr, nullptr, 0);
    ctx->DSSetShader(nullptr, nullptr, 0);

    ctx->IASetInputLayout(nullptr);
    ctx->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    ctx->IASetVertexBuffers(0, 0, nullptr, nullptr, nullptr);
    ctx->IASetIndexBuffer(nullptr, DXGI_FORMAT_UNKNOWN, 0);

    ctx->Draw(3, 0);

    // Unbind our SRVs so the game's next OMSetRenderTargets binding the
    // wrapped BB-RTV doesn't trip a D3D11 read/write hazard warning.
    ctx->PSSetShaderResources(0, 2, nullSRVs);
}

// Same config key the DX9 wrapper uses to pick its output plugin.
bool SwapChain11Proxy::WeaveSelected() const
{
    return _tcsicmp(gInfo.OutputMethodDll, _T("SimulatedRealityWeaveOutput")) == 0;
}

// Primary selector: gInfo.OutputMode (matches OGL / NvDirectMode / HD3D
// convention). Backward-compat: configs that only set OutputMethodDll =
// "SimulatedRealityWeaveOutput" still get OM_SR_WEAVE regardless of the
// numeric mode. Mode 0 (iZ3D legacy dual-panel) and out-of-range values
// fall back to OM_HALF_SBS.
SwapChain11Proxy::OutputMode SwapChain11Proxy::SelectedOutputMode() const
{
    if (WeaveSelected()) return OM_SR_WEAVE;
    const DWORD m = gInfo.OutputMode;
    if (m >= OM_HALF_SBS && m <= OM_SR_WEAVE) return static_cast<OutputMode>(m);
    return OM_HALF_SBS;
}

bool SwapChain11Proxy::EnsureSRWeaver()
{
    if (m_srFailed) return false;
    if (m_srInterface) return true;
    if (!m_parent || !m_real) return false;

    DXGI_SWAP_CHAIN_DESC scDesc = {};
    if (FAILED(m_real->GetDesc(&scDesc))) { m_srFailed = true; return false; }

    ID3D11Device* dev = m_parent->GetReal();
    if (!dev) { m_srFailed = true; return false; }
    ID3D11DeviceContext* immCtx = nullptr;
    dev->GetImmediateContext(&immCtx);
    if (!immCtx) { m_srFailed = true; return false; }

    // E_NOINTERFACE when the delay-loaded SR runtime is absent.
    SimulatedReality::SRInterfaceDX11* sr = nullptr;
    HRESULT hr = SimulatedReality::CreateSRInterfaceDX11(immCtx, scDesc.OutputWindow, &sr);
    immCtx->Release();
    if (FAILED(hr) || !sr)
    {
        DDILog("SwapChain11Proxy::EnsureSRWeaver: CreateSRInterfaceDX11 failed"
               " 0x%08lX (hWnd=%p) -- falling back to SBS for this session\n",
               hr, (void*)scDesc.OutputWindow);
        m_srFailed = true;
        return false;
    }
    m_srInterface = sr;
    DDILog("SwapChain11Proxy::EnsureSRWeaver: ready (hWnd=%p sr=%p)\n",
           (void*)scDesc.OutputWindow, (void*)sr);

    // Query the LeiaSR Displays runtime for the panel-recommended SBS views
    // texture dimensions. Log-only for now; task #32 will consume these to
    // size the SBS input and output RT to panel-native, so sub-native games
    // on 4K SR displays stop losing lenticular alignment through downstream
    // scaling. False just means the SR Displays runtime isn't present on
    // this host; the existing BB-sized weave path keeps working.
    int srRecW = 0, srRecH = 0;
    if (wiz3d::QueryRecommendedSRViewsSize(srRecW, srRecH))
    {
        DDILog("SwapChain11Proxy::EnsureSRWeaver: LeiaSR recommended views %dx%d; current game BB %ux%u -> SBS %ux%u\n",
               srRecW, srRecH, m_bbWidth, m_bbHeight, m_bbWidth * 2, m_bbHeight);
    }
    else
    {
        DDILog("SwapChain11Proxy::EnsureSRWeaver: LeiaSR Displays API unavailable; continuing with BB-sized weave\n");
    }
    return true;
}

HRESULT SwapChain11Proxy::EnsureSRSBSTexture()
{
    // Size the SBS input to the real BB's dims x 2 width. When the real BB
    // has been resized to panel-native (SR_Weave path), this gives us the
    // SR-recommended SBS size directly; otherwise it stays at game BB dims
    // and we fall back to the pre-panel-native behaviour.
    const UINT        wantW   = m_realBBW * 2;
    const UINT        wantH   = m_realBBH;
    const DXGI_FORMAT wantFmt = m_bbFormat;
    if (!wantW || !wantH || wantFmt == DXGI_FORMAT_UNKNOWN) return E_FAIL;
    if (m_srSBSTex && m_srSBSW == wantW && m_srSBSH == wantH && m_srSBSFmt == wantFmt)
        return S_OK;

    ReleaseSRWeave(false);

    ID3D11Device* dev = m_parent ? m_parent->GetReal() : nullptr;
    if (!dev) return E_FAIL;

    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = wantW;
    td.Height           = wantH;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = wantFmt;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DEFAULT;
    td.BindFlags        = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    HRESULT hr = dev->CreateTexture2D(&td, nullptr, &m_srSBSTex);
    if (FAILED(hr)) return hr;
    hr = dev->CreateRenderTargetView(m_srSBSTex, nullptr, &m_srSBSRTV);
    if (FAILED(hr)) { ReleaseSRWeave(false); return hr; }
    hr = dev->CreateShaderResourceView(m_srSBSTex, nullptr, &m_srSBSSRV);
    if (FAILED(hr)) { ReleaseSRWeave(false); return hr; }

    m_srSBSW = wantW; m_srSBSH = wantH; m_srSBSFmt = wantFmt;

    // Bind the weave input once per texture creation, not per frame.
    if (m_srInterface) m_srInterface->SetInputTexture(m_srSBSSRV);

    DDILog("SwapChain11Proxy::EnsureSRSBSTexture: %ux%u fmt=%d tex=%p\n",
           wantW, wantH, (int)wantFmt, (void*)m_srSBSTex);
    return S_OK;
}

bool SwapChain11Proxy::RunSRWeave(ID3D11DeviceContext* ctx)
{
    if (!EnsureSRWeaver()) return false;
    if (FAILED(EnsureSRSBSTexture())) return false;

    // SR Weave reads a Half-SBS input texture; the compositor fills it in
    // the format SR-Lib's Weave() expects, regardless of the user's
    // configured output mode (which is OM_SR_WEAVE here anyway).
    DrawCompositePass(ctx, m_srSBSRTV, m_srSBSW, m_srSBSH, OM_HALF_SBS);

    // Weave() reads the registered input SRV and writes into the bound RT.
    ctx->OMSetRenderTargets(1, &m_realBBRTV, nullptr);
    D3D11_VIEWPORT vp = {};
    vp.Width    = (float)m_realBBW;
    vp.Height   = (float)m_realBBH;
    vp.MinDepth = 0.f;
    vp.MaxDepth = 1.f;
    ctx->RSSetViewports(1, &vp);
    m_srInterface->Weave();

    static int s_logged = 0;
    if (s_logged < 3)
    {
        ++s_logged;
        DDILog("SwapChain11Proxy::RunSRWeave: weave OK (SBS %ux%u -> real BB %ux%u; game BB %ux%u)\n",
               m_srSBSW, m_srSBSH, m_realBBW, m_realBBH, m_bbWidth, m_bbHeight);
    }
    return true;
}

void SwapChain11Proxy::ReleaseSRWeave(bool releaseWeaver)
{
    if (m_srSBSSRV) { m_srSBSSRV->Release(); m_srSBSSRV = nullptr; }
    if (m_srSBSRTV) { m_srSBSRTV->Release(); m_srSBSRTV = nullptr; }
    if (m_srSBSTex) { m_srSBSTex->Release(); m_srSBSTex = nullptr; }
    m_srSBSW = m_srSBSH = 0;
    m_srSBSFmt = DXGI_FORMAT_UNKNOWN;
    if (releaseWeaver && m_srInterface)
    {
        m_srInterface->Delete();
        m_srInterface = nullptr;
    }
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::GetBuffer(UINT Buffer, REFIID riid, void** ppSurface)
{
    // Stage 4d: index-0 buffer = our wrapped stereo BB. Other indices fall
    // through to the real swap chain (multi-buffer / shared-surface use is
    // rare for the games we wrap; first-pass leaves them unwrapped).
    if (Buffer != 0 || !ppSurface)
    {
        // Logged separately: a flip-model game asking for a non-zero index
        // gets a real buffer while index 0 is wrapped, mixing the two.
        static unsigned s_nonZero = 0;
        if (s_nonZero++ < 16)
            DDILog("SwapChain11Proxy::GetBuffer(%u): UNWRAPPED real buffer returned\n", Buffer);
        return m_real->GetBuffer(Buffer, riid, ppSurface);
    }

    if (FAILED(EnsureStereoBackBuffer()))
        return m_real->GetBuffer(Buffer, riid, ppSurface);

    if (!m_wrappedBB)
        return m_real->GetBuffer(Buffer, riid, ppSurface);

    // Honour the requested interface. Texture2D11Proxy::QueryInterface
    // covers the common IIDs (IUnknown, Resource, Texture2D); for anything
    // outside that surface area fall back to the real BB so the game's
    // sub-interface query doesn't get an E_NOINTERFACE here.
    HRESULT hr = m_wrappedBB->QueryInterface(riid, ppSurface);
    if (SUCCEEDED(hr))
    {
        // The pointer the game got is registered: QI may return another address than m_wrappedBB, and CreateRTV
        // compares against what the game passes back.
        if (m_parent && *ppSurface)
            m_parent->RegisterBackBufferTexture(*ppSurface);
        DDILog("SwapChain11Proxy::GetBuffer(0): returned wrapped BB=%p (as %p) registered on dev=%p\n",
               m_wrappedBB, *ppSurface, m_parent);
        return hr;
    }
    return m_real->GetBuffer(Buffer, riid, ppSurface);
}

static int s_FramePresentCounter = 0;

void SwapChain11Proxy::OnPresentBoundaryPre()
{
    if (FrameTraceActive())
        FrameTrace("\n=== Present #%d PRE ===\n", s_FramePresentCounter);
    // BatchRightEye: the right eye's last recorded pass runs before both eyes are composited.
    if (Context11Proxy* c = m_parent ? m_parent->GetContextProxy() : nullptr)
        c->FlushRightBatch(Context11Proxy::kFlushPresent);
    // Before Present, under UseCOMWrapReplay (on by default): a recorded frame replays for the right eye when recording
    // is armed (never under DuplicateDraws), then both eyes are composited into the back buffer.
    if (!gInfo.UseCOMWrapReplay) return;
    if (!m_parent) return;
    Context11Proxy* ctx = m_parent->GetContextProxy();
    if (!ctx) return;
    if (ctx->IsPresentHookActive())
        ctx->ReplayFrameCommands(Context11Proxy::Eye::Right);

    if (FrameTraceActive())
        FrameTrace("--- composite start ---\n");

    if (gKbdHook.TakeScreenshotRequest() || gInfo.MakeScreenshot) WriteScreenshot();

    // Both eyes into the real back buffer; skipped if the wrapped back buffer was never allocated.
    const UINT g = ctx->GpuOutputBegin();
    DoComposite();
    ctx->GpuOutputEnd(g);
    if (FrameTraceActive())
        FrameTrace("--- composite end ---\n");
}

// A game that exits without releasing its swap chain never reaches the destructor's write, so changes are saved here.
void SwapChain11Proxy::PersistSettings()
{
    gKbdHook.m_Access.Enter();
    DataInput now = gInfo.Input;
    gKbdHook.m_Access.Leave();
    const ULONGLONG t = GetTickCount64();
    if (m_savedInput.IsChanged(now))
    {
        m_savedInput = now;
        m_writeSettingsAt = t + 5000;
    }
    else if (m_writeSettingsAt && t >= m_writeSettingsAt)
    {
        WriteInputData(&m_savedInput);
        m_writeSettingsAt = 0;
    }
}

// The autotests' MakeScreenshotPath, else My Pictures\<product> Screenshots\<profile>\<profile> #NNNN<ext>.
static bool ScreenshotPath(bool stereo, TCHAR (&out)[MAX_PATH])
{
    if (gInfo.MakeScreenshot)
    {
        _tcscpy_s(out, gInfo.MakeScreenshotPath);
        return true;
    }
    TCHAR dir[MAX_PATH] = {}, name[MAX_PATH] = {};
    if (FAILED(SHGetFolderPath(nullptr, CSIDL_MYPICTURES | CSIDL_FLAG_CREATE, nullptr, 0, dir))) return false;
    PathAppend(dir, _T(PRODUCT_NAME) _T(" Screenshots"));
    CreateDirectory(dir, nullptr);
    _tcscpy_s(name, gInfo.ProfileName);
    for (TCHAR* c = name; *c; ++c) if (_tcschr(_T("\\/:*?\"<>|"), *c)) *c = _T('_');
    PathAppend(dir, name);
    CreateDirectory(dir, nullptr);
    static unsigned s_count = 0;
    do _stprintf_s(out, _T("%s\\%s #%04u%s"), dir, name, s_count++, stereo ? GetStereoImageFileExtension() : GetImageFileExtension());
    while (PathFileExists(out));
    return true;
}

// The right eye on the left half and the left eye on the right, as JPS and the DX9/DX10 wrappers lay them out; the
// left eye alone with stereo off.
void SwapChain11Proxy::WriteScreenshot()
{
    if (!m_parent || !m_leftBB) return;
    ID3D11Device* dev = m_parent->GetReal();
    ID3D11DeviceContext* ctx = nullptr;
    dev->GetImmediateContext(&ctx);
    if (!ctx) return;
    const bool stereo = gInfo.Input.StereoActive && m_rightBB;
    const UINT views = stereo ? 2 : 1;
    D3D11_TEXTURE2D_DESC td = {};
    m_leftBB->GetDesc(&td);
    const UINT rowBytes = td.Width * (UINT)(GetBitWidthOfDXGIFormat(td.Format) / 8), pitch = rowBytes * views;
    std::vector<char> pixels((size_t)pitch * td.Height);

    D3D11_TEXTURE2D_DESC sd = td;
    sd.MipLevels = 1; sd.ArraySize = 1; sd.SampleDesc.Count = 1; sd.SampleDesc.Quality = 0;
    sd.MiscFlags = 0; sd.BindFlags = 0; sd.CPUAccessFlags = 0; sd.Usage = D3D11_USAGE_DEFAULT;
    ID3D11Texture2D* resolved = nullptr;
    ID3D11Texture2D* stg = nullptr;
    if (td.SampleDesc.Count > 1 && FAILED(dev->CreateTexture2D(&sd, nullptr, &resolved))) resolved = nullptr;
    sd.Usage = D3D11_USAGE_STAGING; sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    if (FAILED(dev->CreateTexture2D(&sd, nullptr, &stg))) stg = nullptr;
    bool ok = rowBytes && stg && (td.SampleDesc.Count == 1 || resolved);
    ID3D11Texture2D* eyes[2] = { stereo ? m_rightBB : m_leftBB, m_leftBB };
    for (UINT v = 0; ok && v < views; ++v)
    {
        if (resolved)
        {
            ctx->ResolveSubresource(resolved, 0, eyes[v], 0, td.Format);
            ctx->CopyResource(stg, resolved);
        }
        else ctx->CopyResource(stg, eyes[v]);
        D3D11_MAPPED_SUBRESOURCE m = {};
        if (FAILED(ctx->Map(stg, 0, D3D11_MAP_READ, 0, &m)) || !m.pData) { ok = false; break; }
        for (UINT y = 0; y < td.Height; ++y)
            memcpy(&pixels[(size_t)y * pitch + v * rowBytes], (const char*)m.pData + (size_t)y * m.RowPitch, rowBytes);
        ctx->Unmap(stg, 0);
    }
    if (stg) stg->Release();
    if (resolved) resolved->Release();
    ctx->Release();

    TCHAR path[MAX_PATH] = {};
    if (ok && ScreenshotPath(stereo, path))
    {
        SaveImageToFile(path, td.Format, td.Width * views, td.Height, pixels.data(), pitch, true);
        DDILog("Screenshot: %ls (%ux%u, format %d) %s\n", path, td.Width * views, td.Height, (int)td.Format,
               PathFileExists(path) ? "written" : "NOT written");
    }
    else DDILog("Screenshot: FAILED (format %d, %u samples)\n", (int)td.Format, td.SampleDesc.Count);
    if (gInfo.MakeScreenshot)
    {
        gInfo.MakeScreenshot = FALSE;
        if (gInfo.MakeScreenshotCallback) gInfo.MakeScreenshotCallback();
    }
}

void SwapChain11Proxy::OnPresentBoundaryPost()
{
    if (FrameTraceActive())
        FrameTrace("=== Present #%d POST ===\n", s_FramePresentCounter++);
    FrameTraceTickFrame();
    PersistSettings();
    // Stage 4b.8: POST-PRESENT housekeeping. Real Present has flipped the
    // BB; clear the recording vector and arm it for frame N+1. We arm
    // unconditionally (not just on the first call) so a context that's
    // been ClearState()'d still gets recording.
    if (!m_parent) return;
    Context11Proxy* ctx = m_parent->GetContextProxy();
    if (!ctx) return;
    // Before ClearFrameCommands(), so the frame report counts the recording of the frame that just presented.
    ctx->EndFrame();
    ctx->ClearFrameCommands();
    ctx->SetPresentHookActive(true);
}

// GetDesc / GetDesc1: when the real BB has been resized away from the game's
// requested dims (SR_Weave panel-native path), patch the returned description
// so the game continues to see the dims it asked for. Everything downstream
// in the game -- viewport, scissor, RTVs sized to its own cached BB dims --
// keeps working at the game's resolution; the compositor and weave operate
// on the real (panel-native) BB behind the scenes.
HRESULT STDMETHODCALLTYPE SwapChain11Proxy::GetDesc(DXGI_SWAP_CHAIN_DESC* pDesc)
{
    HRESULT hr = m_real->GetDesc(pDesc);
    if (SUCCEEDED(hr) && pDesc && m_bbWidth && m_bbHeight &&
        (m_realBBW != m_bbWidth || m_realBBH != m_bbHeight))
    {
        pDesc->BufferDesc.Width  = m_bbWidth;
        pDesc->BufferDesc.Height = m_bbHeight;
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::GetDesc1(DXGI_SWAP_CHAIN_DESC1* pDesc)
{
    if (!m_real1) return E_NOINTERFACE;
    HRESULT hr = m_real1->GetDesc1(pDesc);
    if (SUCCEEDED(hr) && pDesc && m_bbWidth && m_bbHeight &&
        (m_realBBW != m_bbWidth || m_realBBH != m_bbHeight))
    {
        pDesc->Width  = m_bbWidth;
        pDesc->Height = m_bbHeight;
    }
    return hr;
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::ResizeBuffers(
    UINT BufferCount, UINT Width, UINT Height, DXGI_FORMAT NewFormat, UINT SwapChainFlags)
{
    // Stage 4b.8 fix: ResizeBuffers invalidates every BB-derived RTV/DSV
    // the game (and our recorded closures) might hold. The recorded
    // command stream becomes unsafe to replay — Max Payne 3's fullscreen
    // toggle hit this and crashed inside d3d11.dll when replay tried to
    // re-bind dead handles. Clear the recording before forwarding.
    if (m_parent)
    {
        if (Context11Proxy* ctx = m_parent->GetContextProxy())
            ctx->ClearFrameCommands();
    }
    // Stage 4d: tear down the wiz3D stereo BB siblings + the RTV we hold
    // on the real BB. DXGI requires every reference to back-buffer-derived
    // resources be released before ResizeBuffers, including ours.
    // Composite shaders/sampler are format-independent so they stay alive.
    // The wrapped Texture2D11Proxy from GetBuffer is dropped — game will
    // call GetBuffer again post-resize and we'll allocate new siblings
    // matching the new BB dims/format.
    DDILog("SwapChain11Proxy::ResizeBuffers: count=%u %ux%u fmt=%d flags=0x%X\n",
           BufferCount, Width, Height, (int)NewFormat, SwapChainFlags);
    ReleaseSRWeave(false);
    ReleaseStereoBackBuffer();
    HRESULT hr = m_real->ResizeBuffers(BufferCount, Width, Height, NewFormat, SwapChainFlags);
    DDILog("SwapChain11Proxy::ResizeBuffers -> hr=0x%08lX\n", hr);
    return hr;
}

// Same invalidation as ResizeBuffers: it reallocates the same buffers, so our
// stereo siblings and the recorded command stream must go first.
HRESULT STDMETHODCALLTYPE SwapChain11Proxy::ResizeBuffers1(
    UINT BufferCount, UINT Width, UINT Height, DXGI_FORMAT Format, UINT SwapChainFlags,
    const UINT* pCreationNodeMask, IUnknown* const* ppPresentQueue)
{
    if (!m_real3) return E_NOINTERFACE;
    if (m_parent)
    {
        if (Context11Proxy* ctx = m_parent->GetContextProxy())
            ctx->ClearFrameCommands();
    }
    DDILog("SwapChain11Proxy::ResizeBuffers1: count=%u %ux%u fmt=%d flags=0x%X\n",
           BufferCount, Width, Height, (int)Format, SwapChainFlags);
    ReleaseSRWeave(false);
    ReleaseStereoBackBuffer();
    HRESULT hr = m_real3->ResizeBuffers1(BufferCount, Width, Height, Format,
                                         SwapChainFlags, pCreationNodeMask, ppPresentQueue);
    DDILog("SwapChain11Proxy::ResizeBuffers1 -> hr=0x%08lX\n", hr);
    return hr;
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::SetFullscreenState(BOOL Fullscreen, IDXGIOutput* pTarget)
{
    // Stage 4b.8 fix: same logic as ResizeBuffers — fullscreen transitions
    // can implicitly resize the swap chain and invalidate BB-derived
    // handles. Clear the recording so replay can't re-issue stale state.
    DDILog("SwapChain11Proxy::SetFullscreenState: fullscreen=%d target=%p\n",
           (int)Fullscreen, pTarget);
    if (m_parent)
    {
        if (Context11Proxy* ctx = m_parent->GetContextProxy())
            ctx->ClearFrameCommands();
    }
    HRESULT hr = m_real->SetFullscreenState(Fullscreen, pTarget);
    DDILog("SwapChain11Proxy::SetFullscreenState -> hr=0x%08lX\n", hr);
    return hr;
}

// The frames whose end the GPU has not reached, newest first; this frame's event is issued first, so 0 means it had
// finished the previous frame. Polled without a flush.
unsigned SwapChain11Proxy::GpuFramesBehind()
{
    if (!m_parent) return 0;
    ID3D11Device* dev = m_parent->GetReal();
    ID3D11DeviceContext* ctx = nullptr;
    dev->GetImmediateContext(&ctx);
    if (!ctx) return 0;
    ID3D11Query*& now = m_frameEvents[m_frameEventNext % kFrameEvents];
    if (!now)
    {
        D3D11_QUERY_DESC qd = { D3D11_QUERY_EVENT, 0 };
        if (FAILED(dev->CreateQuery(&qd, &now))) now = nullptr;
    }
    if (now) ctx->End(now);
    ++m_frameEventNext;
    unsigned behind = 0;
    for (unsigned back = 2; back <= kFrameEvents && back <= m_frameEventNext; ++back)
    {
        ID3D11Query* q = m_frameEvents[(m_frameEventNext - back) % kFrameEvents];
        BOOL done = FALSE;
        if (!q || ctx->GetData(q, &done, sizeof(done), D3D11_ASYNC_GETDATA_DONOTFLUSH) == S_OK) break;
        ++behind;
    }
    ctx->Release();
    return behind;
}

HRESULT SwapChain11Proxy::PresentFrame(bool present1, UINT SyncInterval, UINT Flags, const DXGI_PRESENT_PARAMETERS* pParams)
{
    const bool timed = gInfo.CollectDebugInformation;
    LARGE_INTEGER t0 = {}, t1 = {}, t2 = {};
    if (timed) QueryPerformanceCounter(&t0);
    OnPresentBoundaryPre();
    const unsigned behind = timed ? GpuFramesBehind() : 0;
    if (timed) QueryPerformanceCounter(&t1);
    HRESULT hr = present1 ? (m_real1 ? m_real1->Present1(SyncInterval, Flags, pParams) : E_NOINTERFACE)
                          : m_real->Present(SyncInterval, Flags);
    if (timed)
    {
        QueryPerformanceCounter(&t2);
        if (Context11Proxy* ctx = m_parent ? m_parent->GetContextProxy() : nullptr)
            ctx->NoteFrameTiming(t1.QuadPart - t0.QuadPart, t2.QuadPart - t1.QuadPart, behind);
    }
    OnPresentBoundaryPost();
    if (m_parent && RecycleRightCopiesOn()) m_parent->TickRecycle();
    return hr;
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::Present(UINT SyncInterval, UINT Flags)
{
    return PresentFrame(false, SyncInterval, Flags, nullptr);
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::Present1(
    UINT SyncInterval, UINT PresentFlags, const DXGI_PRESENT_PARAMETERS* pPresentParameters)
{
    return PresentFrame(true, SyncInterval, PresentFlags, pPresentParameters);
}

} // namespace wiz3d
