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
#include <d3dcompiler.h>
#include "SR.hpp"                    // Simulated Reality weave (delay-loaded runtime)

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
    , m_compositePS(nullptr)
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

float4 PSMain(VSOut input) : SV_Target {
    float2 uv = input.uv;
    float  side = step(0.5, uv.x);   // 0 = left half, 1 = right half
    float2 leftSample  = float2(uv.x * 2.0,            uv.y);
    float2 rightSample = float2((uv.x - 0.5) * 2.0,    uv.y);
    float4 colL = leftTex.Sample (samp, leftSample);
    float4 colR = rightTex.Sample(samp, rightSample);
    return lerp(colL, colR, side);
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
    if (m_wrappedBB) { m_wrappedBB->Release(); m_wrappedBB = nullptr; }
    if (m_leftBB)    { m_leftBB->Release();    m_leftBB    = nullptr; }
    if (m_rightBB)   { m_rightBB->Release();   m_rightBB   = nullptr; }
    m_bbWidth = m_bbHeight = 0;
    m_bbFormat = DXGI_FORMAT_UNKNOWN;
}

HRESULT SwapChain11Proxy::EnsureComposite()
{
    // Stage 4d: lazily compile the SBS composite shaders + pipeline state
    // on first Present where the composite actually runs. D3DCompile is a
    // one-time cost; the shader is independent of BB format/size so we
    // don't have to re-compile across ResizeBuffers.
    if (m_compositeVS && m_compositePS && m_compositeSampler) return S_OK;
    if (!m_parent) return E_FAIL;
    ID3D11Device* dev = m_parent->GetReal();
    if (!dev) return E_FAIL;

    ID3DBlob* vsBlob = nullptr; ID3DBlob* psBlob = nullptr; ID3DBlob* err = nullptr;
    UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;

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

    HRESULT hrPS = D3DCompile(
        k_compositeShaderSrc, strlen(k_compositeShaderSrc),
        "wiz3D_composite", nullptr, nullptr,
        "PSMain", "ps_4_0", flags, 0, &psBlob, &err);
    if (FAILED(hrPS))
    {
        DDILog("SwapChain11Proxy::EnsureComposite: PS compile failed 0x%08lX: %s\n",
               hrPS, err ? (const char*)err->GetBufferPointer() : "(no log)");
        if (err) err->Release();
        vsBlob->Release();
        return hrPS;
    }
    if (err) { err->Release(); err = nullptr; }

    HRESULT hr = dev->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(),
                                          nullptr, &m_compositeVS);
    vsBlob->Release();
    if (FAILED(hr)) { psBlob->Release(); return hr; }

    hr = dev->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(),
                                 nullptr, &m_compositePS);
    psBlob->Release();
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

    DDILog("SwapChain11Proxy::EnsureComposite: VS=%p PS=%p sampler=%p\n",
           m_compositeVS, m_compositePS, m_compositeSampler);
    return S_OK;
}

void SwapChain11Proxy::ReleaseComposite()
{
    if (m_compositeDepthStencil) { m_compositeDepthStencil->Release(); m_compositeDepthStencil = nullptr; }
    if (m_compositeBlend)        { m_compositeBlend->Release();        m_compositeBlend        = nullptr; }
    if (m_compositeRaster)       { m_compositeRaster->Release();       m_compositeRaster       = nullptr; }
    if (m_compositeSampler)      { m_compositeSampler->Release();      m_compositeSampler      = nullptr; }
    if (m_compositePS)           { m_compositePS->Release();           m_compositePS           = nullptr; }
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
    if (!(WeaveSelected() && RunSRWeave(ctx)))
        DrawCompositePass(ctx, m_realBBRTV, m_bbWidth, m_bbHeight);
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
                                         UINT width, UINT height)
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
    ID3D11ShaderResourceView* srvs[2] = { eyeL, eyeR };
    ctx->PSSetShaderResources(0, 2, srvs);
    ctx->PSSetSamplers(0, 1, &m_compositeSampler);

    ctx->VSSetShader(m_compositeVS, nullptr, 0);
    ctx->PSSetShader(m_compositePS, nullptr, 0);
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
    return true;
}

HRESULT SwapChain11Proxy::EnsureSRSBSTexture()
{
    const UINT        wantW   = m_bbWidth * 2;
    const UINT        wantH   = m_bbHeight;
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

    DrawCompositePass(ctx, m_srSBSRTV, m_srSBSW, m_srSBSH);

    // Weave() reads the registered input SRV and writes into the bound RT.
    ctx->OMSetRenderTargets(1, &m_realBBRTV, nullptr);
    D3D11_VIEWPORT vp = {};
    vp.Width    = (float)m_bbWidth;
    vp.Height   = (float)m_bbHeight;
    vp.MinDepth = 0.f;
    vp.MaxDepth = 1.f;
    ctx->RSSetViewports(1, &vp);
    m_srInterface->Weave();

    static int s_logged = 0;
    if (s_logged < 3)
    {
        ++s_logged;
        DDILog("SwapChain11Proxy::RunSRWeave: weave OK (%ux%u -> %ux%u)\n",
               m_srSBSW, m_srSBSH, m_bbWidth, m_bbHeight);
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

    // Both eyes into the real back buffer; skipped if the wrapped back buffer was never allocated.
    DoComposite();
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

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::Present(UINT SyncInterval, UINT Flags)
{
    OnPresentBoundaryPre();
    HRESULT hr = m_real->Present(SyncInterval, Flags);
    OnPresentBoundaryPost();
    return hr;
}

HRESULT STDMETHODCALLTYPE SwapChain11Proxy::Present1(
    UINT SyncInterval, UINT PresentFlags, const DXGI_PRESENT_PARAMETERS* pPresentParameters)
{
    OnPresentBoundaryPre();
    HRESULT hr = m_real1 ? m_real1->Present1(SyncInterval, PresentFlags, pPresentParameters)
                          : E_NOINTERFACE;
    OnPresentBoundaryPost();
    return hr;
}

} // namespace wiz3d
