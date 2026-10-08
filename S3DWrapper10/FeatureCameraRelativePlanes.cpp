/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <CameraRelativePlanes Value="structured buffer" Stride="bytes" Planes="offsets"/>: camera-relative planes (n, w) for a
// moved-origin right dispatch, copied on the GPU with every plane's w -= n.(d*R) and bound in the game's place.

#include "StdAfx.h"
#include "ProfileFeature.h"
#include "Context11Proxy.h"
#include "Device11Proxy.h"
#include "ShaderAnalyzer11.h"
#include "AdapterFunctions.h"   // DDILog
#include <d3dcompiler.h>
#include <map>
#include <sstream>

namespace wiz3d
{
namespace
{

class CameraRelativePlanesFeature : public ProfileFeature
{
public:
    explicit CameraRelativePlanesFeature(const ProfileEntry& e)
        : m_name(EntryValue(e, "Value")), m_stride((UINT)atoi(EntryValue(e, "Stride").c_str()))
    {
        std::istringstream in(EntryValue(e, "Planes"));
        int off;
        while (in >> off) if (off >= 0 && off % 16 == 0 && off / 16 < 32) m_planeMask |= 1u << (off / 16);
    }

    ~CameraRelativePlanesFeature() override
    {
        for (auto& kv : m_copies) kv.second.Release();
        if (m_cs) m_cs->Release();
        if (m_cb) m_cb->Release();
    }

    void OnShaderCreated(const RdefView& rdef, ShaderAnalysis11Result& info) const override
    {
        const int slot = rdef.SrvSlot(m_name);
        if (slot >= 0) info.namedSRVs.emplace_back(m_name, slot);
    }

    void OnRightDispatch(Context11Proxy& ctx) override
    {
        if (!ctx.MovedOrigin() || !m_planeMask || m_stride < 16 || m_stride % 16) return;
        const ShaderAnalysis11Result* info = ctx.BoundCompute();
        int slot = -1;
        if (info) for (const auto& n : info->namedSRVs) if (n.first == m_name) slot = n.second;
        if (slot < 0) return;
        ID3D11ShaderResourceView* src = ctx.ComputeSRVForRight((UINT)slot);
        CameraMove cam;
        if (!src || !ctx.BoundCameraMove(cam)) { ++m_skipped; return; }
        ID3D11Device* dev = ctx.Parent() ? ctx.Parent()->GetReal() : nullptr;
        ID3D11DeviceContext* dc = ctx.Real();
        if (!dev || !dc || !EnsureShader(dev)) { ++m_skipped; return; }

        ID3D11Resource* res = nullptr;
        src->GetResource(&res);
        if (!res) return;
        D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
        res->GetType(&dim);
        D3D11_BUFFER_DESC bd = {};
        if (dim == D3D11_RESOURCE_DIMENSION_BUFFER) static_cast<ID3D11Buffer*>(res)->GetDesc(&bd);
        if (dim != D3D11_RESOURCE_DIMENSION_BUFFER || bd.StructureByteStride != m_stride || bd.ByteWidth < m_stride)
        {
            res->Release();
            ++m_skipped;
            return;
        }
        Copy& c = m_copies[res];
        if (c.bytes != bd.ByteWidth) { c.Release(); c = Copy(); }
        if (!c.srv)
        {
            D3D11_BUFFER_DESC d = {};
            d.ByteWidth = bd.ByteWidth;
            d.Usage = D3D11_USAGE_DEFAULT;
            d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS;
            d.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
            d.StructureByteStride = m_stride;
            D3D11_SHADER_RESOURCE_VIEW_DESC sd = {};
            src->GetDesc(&sd);
            D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
            ud.Format = DXGI_FORMAT_UNKNOWN;
            ud.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            ud.Buffer.NumElements = bd.ByteWidth / m_stride;
            if (FAILED(dev->CreateBuffer(&d, nullptr, &c.buf)) || FAILED(dev->CreateUnorderedAccessView(c.buf, &ud, &c.uav)) ||
                FAILED(dev->CreateShaderResourceView(c.buf, &sd, &c.srv)))
            {
                c.Release();
                m_copies.erase(res);
                res->Release();
                ++m_skipped;
                return;
            }
            c.bytes = bd.ByteWidth;
        }
        res->Release();

        // Parameters: the camera move, the element count, the plane mask.
        const UINT count = bd.ByteWidth / m_stride;
        D3D11_MAPPED_SUBRESOURCE m = {};
        if (FAILED(dc->Map(m_cb, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)) || !m.pData) { ++m_skipped; return; }
        float* f = (float*)m.pData;
        f[0] = cam.world[0]; f[1] = cam.world[1]; f[2] = cam.world[2]; f[3] = 0.f;
        UINT* u = (UINT*)(f + 4);
        u[0] = count; u[1] = m_planeMask; u[2] = 0; u[3] = 0;
        dc->Unmap(m_cb, 0);

        // Borrow slot 0 of each binding for the patch, then give the game's back.
        ID3D11ComputeShader* oldCS = nullptr;
        ID3D11ClassInstance* oldInst[256] = {};
        UINT oldInstCount = 256;
        dc->CSGetShader(&oldCS, oldInst, &oldInstCount);
        ID3D11ShaderResourceView* oldSRV = nullptr;
        ID3D11UnorderedAccessView* oldUAV = nullptr;
        ID3D11Buffer* oldCB = nullptr;
        dc->CSGetShaderResources(0, 1, &oldSRV);
        dc->CSGetUnorderedAccessViews(0, 1, &oldUAV);
        dc->CSGetConstantBuffers(0, 1, &oldCB);

        dc->CSSetShader(m_cs, nullptr, 0);
        dc->CSSetConstantBuffers(0, 1, &m_cb);
        dc->CSSetShaderResources(0, 1, &src);
        const UINT keep = ~0u;
        dc->CSSetUnorderedAccessViews(0, 1, &c.uav, &keep);
        dc->Dispatch((count + 63) / 64, 1, 1);

        ID3D11UnorderedAccessView* nullUAV = nullptr;
        dc->CSSetUnorderedAccessViews(0, 1, oldUAV ? &oldUAV : &nullUAV, &keep);
        dc->CSSetShaderResources(0, 1, &oldSRV);
        dc->CSSetConstantBuffers(0, 1, &oldCB);
        dc->CSSetShader(oldCS, oldInstCount ? oldInst : nullptr, oldInstCount);
        if (oldCS) oldCS->Release();
        for (UINT i = 0; i < oldInstCount; ++i) if (oldInst[i]) oldInst[i]->Release();
        if (oldSRV) oldSRV->Release();
        if (oldUAV) oldUAV->Release();
        if (oldCB) oldCB->Release();

        ctx.SubstituteComputeSRV((UINT)slot, c.srv);
        ++m_patched;
    }

    void OnFrameEnd(Context11Proxy&, unsigned frame, bool report) override
    {
        if (report)
            DDILog("  [frame %u] camera-relative planes %s (this frame): patched %u, skipped %u, copies held %u%s\n", frame,
                   m_name.c_str(), m_patched, m_skipped, (unsigned)m_copies.size(), m_compileFailed ? " (SHADER COMPILE FAILED)" : "");
        m_patched = m_skipped = 0;
    }

private:
    struct Copy
    {
        ID3D11Buffer* buf = nullptr;
        ID3D11UnorderedAccessView* uav = nullptr;
        ID3D11ShaderResourceView* srv = nullptr;
        UINT bytes = 0;
        void Release()
        {
            if (srv) srv->Release();
            if (uav) uav->Release();
            if (buf) buf->Release();
            srv = nullptr; uav = nullptr; buf = nullptr;
        }
    };
    std::string                        m_name;
    UINT                               m_stride;
    UINT                               m_planeMask = 0;
    std::map<ID3D11Resource*, Copy>    m_copies;   // by the game's buffer; a reused address is re-made on a size change
    ID3D11ComputeShader*               m_cs = nullptr;
    ID3D11Buffer*                      m_cb = nullptr;
    bool                               m_compileFailed = false;
    unsigned                           m_patched = 0, m_skipped = 0;

    // Each element is m_stride/16 uint4s; the masked ones are planes: w -= dot(xyz, move).
    bool EnsureShader(ID3D11Device* dev)
    {
        if (m_cs && m_cb) return true;
        if (m_compileFailed) return false;
        char src[1024];
        sprintf_s(src,
                  "struct E { uint4 d[%u]; };\n"
                  "StructuredBuffer<E> Src : register(t0);\n"
                  "RWStructuredBuffer<E> Dst : register(u0);\n"
                  "cbuffer P : register(b0) { float4 Move; uint4 Info; };\n"
                  "[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID)\n"
                  "{\n"
                  "    if (id.x >= Info.x) return;\n"
                  "    E e = Src[id.x];\n"
                  "    [unroll] for (uint k = 0; k < %u; ++k)\n"
                  "        if (Info.y & (1u << k)) { float4 p = asfloat(e.d[k]); p.w -= dot(p.xyz, Move.xyz); e.d[k] = asuint(p); }\n"
                  "    Dst[id.x] = e;\n"
                  "}\n", m_stride / 16, m_stride / 16);
        ID3DBlob* code = nullptr;
        ID3DBlob* errors = nullptr;
        const HRESULT hr = D3DCompile(src, strlen(src), "wiz3D_planes", nullptr, nullptr, "main", "cs_5_0",
                                      D3DCOMPILE_OPTIMIZATION_LEVEL3, 0, &code, &errors);
        if (FAILED(hr) || !code)
        {
            DDILog("  CameraRelativePlanes: patch shader compile failed hr=0x%08lX %s\n", hr,
                   errors ? (const char*)errors->GetBufferPointer() : "");
            if (errors) errors->Release();
            m_compileFailed = true;
            return false;
        }
        if (errors) errors->Release();
        const bool ok = SUCCEEDED(dev->CreateComputeShader(code->GetBufferPointer(), code->GetBufferSize(), nullptr, &m_cs));
        code->Release();
        D3D11_BUFFER_DESC cd = {};
        cd.ByteWidth = 32;
        cd.Usage = D3D11_USAGE_DYNAMIC;
        cd.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        cd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
        if (!ok || FAILED(dev->CreateBuffer(&cd, nullptr, &m_cb)))
        {
            m_compileFailed = true;
            return false;
        }
        FeatureLog("  CameraRelativePlanes: %s, stride %u, plane mask 0x%X, patch shader ready\n", m_name.c_str(), m_stride,
                   m_planeMask);
        return true;
    }
};

} // namespace

std::unique_ptr<ProfileFeature> MakeCameraRelativePlanes(const ProfileEntry* e)
{
    if (!e || EntryValue(*e, "Value").empty() || EntryValue(*e, "Stride").empty()) return nullptr;
    return std::unique_ptr<ProfileFeature>(new CameraRelativePlanesFeature(*e));
}

} // namespace wiz3d
