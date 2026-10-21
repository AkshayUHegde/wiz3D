// -----------------------------------------------------------------------------
// wiz3D stereo-params texture for DX11 (0.4.1). See header for the contract.
// -----------------------------------------------------------------------------
#include "StdAfx.h"
#include "StereoParams11.h"
#include "AdapterFunctions.h"           // DDILog

#include <mutex>
#include <unordered_map>

namespace wiz3d
{

namespace
{

struct Entry
{
    ID3D11Texture2D*          tex = nullptr;
    ID3D11ShaderResourceView* srv = nullptr;
};

std::mutex                                       s_lock;
std::unordered_map<ID3D11Device*, Entry>         s_byDevice;

bool CreateFor(ID3D11Device* dev, Entry& out)
{
    D3D11_TEXTURE2D_DESC td = {};
    td.Width            = 1;
    td.Height           = 1;
    td.MipLevels        = 1;
    td.ArraySize        = 1;
    td.Format           = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc.Count = 1;
    td.Usage            = D3D11_USAGE_DYNAMIC;
    td.BindFlags        = D3D11_BIND_SHADER_RESOURCE;
    td.CPUAccessFlags   = D3D11_CPU_ACCESS_WRITE;

    // Seed with (0,0,0,0) so a stage that reads before BindStereoParamsForEye
    // ever ran (shouldn't happen, but defensive) gets a safe mono baseline
    // rather than uninitialised memory.
    const float seed[4] = { 0, 0, 0, 0 };
    D3D11_SUBRESOURCE_DATA init = { seed, sizeof(seed), 0 };

    HRESULT hr = dev->CreateTexture2D(&td, &init, &out.tex);
    if (FAILED(hr))
    {
        DDILog("[stereoparams] CreateTexture2D failed hr=0x%08X\n", (unsigned)hr);
        return false;
    }
    D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
    sd.Format              = DXGI_FORMAT_R32G32B32A32_FLOAT;
    sd.ViewDimension       = D3D11_SRV_DIMENSION_TEXTURE2D;
    sd.Texture2D.MipLevels = 1;
    hr = dev->CreateShaderResourceView(out.tex, &sd, &out.srv);
    if (FAILED(hr))
    {
        DDILog("[stereoparams] CreateShaderResourceView failed hr=0x%08X\n", (unsigned)hr);
        out.tex->Release();
        out.tex = nullptr;
        return false;
    }
    DDILog("[stereoparams] created 1x1 RGBA32F texture + SRV for device=%p "
           "(bound at t%u on every eye switch)\n", dev, kStereoParamsSlot);
    return true;
}

}  // anon namespace

bool EnsureStereoParams(ID3D11Device* dev)
{
    if (!dev) return false;
    std::lock_guard<std::mutex> g(s_lock);
    auto it = s_byDevice.find(dev);
    if (it != s_byDevice.end()) return it->second.srv != nullptr;
    Entry e;
    if (!CreateFor(dev, e)) return false;
    s_byDevice[dev] = e;
    return true;
}

void UpdateStereoParamsBytes(ID3D11DeviceContext* immediateCtx, ID3D11Device* dev,
                             bool right, float separation, float convergence)
{
    if (!immediateCtx || !dev) return;
    ID3D11Texture2D* tex = nullptr;
    {
        std::lock_guard<std::mutex> g(s_lock);
        auto it = s_byDevice.find(dev);
        if (it == s_byDevice.end()) return;
        tex = it->second.tex;
    }
    if (!tex) return;

    // MAP_WRITE_DISCARD renames the GPU-side storage, so an in-flight draw
    // reading the old eye's bytes isn't disturbed. Must be the immediate
    // context — RightRecorder11::Map rejects textures (BUFFER-only), which
    // was the batch-killer bug in 0.4.1 before this split.
    const float params[4] = {
        separation,
        convergence,
        right ? +1.f : -1.f,
        0.f
    };
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (SUCCEEDED(immediateCtx->Map(tex, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)) && m.pData)
    {
        memcpy(m.pData, params, sizeof(params));
        immediateCtx->Unmap(tex, 0);
    }
}

void BindStereoParamsSRV(ID3D11DeviceContext* ctx, ID3D11Device* dev)
{
    if (!ctx || !dev) return;
    ID3D11ShaderResourceView* srv = nullptr;
    {
        std::lock_guard<std::mutex> g(s_lock);
        auto it = s_byDevice.find(dev);
        if (it == s_byDevice.end()) return;
        srv = it->second.srv;
    }
    if (!srv) return;

    // SSetShaderResources is a first-class recorded op (RightRecorder11
    // forwards it), so this is safe on either the immediate context or the
    // right-eye recorder. The game is unlikely to bind its own SRV at t125
    // but if it did, our Bind clobbers it. Documented as a known limitation
    // if any game turns out to use this slot (none observed so far).
    ctx->VSSetShaderResources(kStereoParamsSlot, 1, &srv);
    ctx->HSSetShaderResources(kStereoParamsSlot, 1, &srv);
    ctx->DSSetShaderResources(kStereoParamsSlot, 1, &srv);
    ctx->GSSetShaderResources(kStereoParamsSlot, 1, &srv);
    ctx->PSSetShaderResources(kStereoParamsSlot, 1, &srv);
    ctx->CSSetShaderResources(kStereoParamsSlot, 1, &srv);
}

void ReleaseStereoParams(ID3D11Device* dev)
{
    if (!dev) return;
    Entry e;
    {
        std::lock_guard<std::mutex> g(s_lock);
        auto it = s_byDevice.find(dev);
        if (it == s_byDevice.end()) return;
        e = it->second;
        s_byDevice.erase(it);
    }
    if (e.srv) e.srv->Release();
    if (e.tex) e.tex->Release();
}

}  // namespace wiz3d
