/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The right eye's recording (RightRecorder11.h): each call is a record of 8-byte fields holding a reference to every
// object it names; the replay issues the same calls with the same arguments on the immediate context.

#include "StdAfx.h"
#include "RightRecorder11.h"
#include "AdapterFunctions.h"   // DDILog
#include <string.h>

namespace wiz3d
{
namespace
{

enum Op : UINT
{
    kSRVs, kSamplers, kCBs, kShader, kLayout, kVBs, kIB, kTopology, kRSState, kViewports, kScissors, kBlend, kDepth,
    kTargets, kTargetsUAVs, kCSUAVs, kPredication, kDraw, kDrawIndexed, kDrawInstanced, kDrawIndexedInstanced, kDrawAuto,
    kDrawInstancedIndirect, kDrawIndexedInstancedIndirect, kWrite, kUpdate, kClearState, kWriteRange, kCopyRegion, kCopy
};

struct Head { UINT op, size; };

inline size_t Pad8(size_t n) { return (n + 7) & ~size_t(7); }

// Writes and reads one record's fields: scalars as 8 bytes, arrays padded to 8.
struct Out
{
    unsigned char* p;
    void u(UINT64 v) { memcpy(p, &v, 8); p += 8; }
    void ptr(const void* v) { u((UINT64)(UINT_PTR)v); }
    void bytes(const void* src, size_t n) { if (n) memcpy(p, src, n); p += Pad8(n); }
};
struct In
{
    const unsigned char* p;
    UINT64 u() { UINT64 v; memcpy(&v, p, 8); p += 8; return v; }
    template<class T> T* ptr() { return reinterpret_cast<T*>((UINT_PTR)u()); }
    template<class T> const T* arr(size_t n) { const T* r = reinterpret_cast<const T*>(p); p += Pad8(n * sizeof(T)); return r; }
};

void SetSRVs(ID3D11DeviceContext* c, UINT st, UINT s, UINT n, ID3D11ShaderResourceView* const* v)
{
    switch (st)
    {
    case 0: c->VSSetShaderResources(s, n, v); break;
    case 1: c->HSSetShaderResources(s, n, v); break;
    case 2: c->DSSetShaderResources(s, n, v); break;
    case 3: c->GSSetShaderResources(s, n, v); break;
    case 4: c->PSSetShaderResources(s, n, v); break;
    case 5: c->CSSetShaderResources(s, n, v); break;
    }
}
void SetSamplers(ID3D11DeviceContext* c, UINT st, UINT s, UINT n, ID3D11SamplerState* const* v)
{
    switch (st)
    {
    case 0: c->VSSetSamplers(s, n, v); break;
    case 1: c->HSSetSamplers(s, n, v); break;
    case 2: c->DSSetSamplers(s, n, v); break;
    case 3: c->GSSetSamplers(s, n, v); break;
    case 4: c->PSSetSamplers(s, n, v); break;
    case 5: c->CSSetSamplers(s, n, v); break;
    }
}
void SetCBs(ID3D11DeviceContext* c, UINT st, UINT s, UINT n, ID3D11Buffer* const* v)
{
    switch (st)
    {
    case 0: c->VSSetConstantBuffers(s, n, v); break;
    case 1: c->HSSetConstantBuffers(s, n, v); break;
    case 2: c->DSSetConstantBuffers(s, n, v); break;
    case 3: c->GSSetConstantBuffers(s, n, v); break;
    case 4: c->PSSetConstantBuffers(s, n, v); break;
    case 5: c->CSSetConstantBuffers(s, n, v); break;
    }
}
void SetCBs1(ID3D11DeviceContext1* c, UINT st, UINT s, UINT n, ID3D11Buffer* const* v, const UINT* f, const UINT* k)
{
    switch (st)
    {
    case 0: c->VSSetConstantBuffers1(s, n, v, f, k); break;
    case 1: c->HSSetConstantBuffers1(s, n, v, f, k); break;
    case 2: c->DSSetConstantBuffers1(s, n, v, f, k); break;
    case 3: c->GSSetConstantBuffers1(s, n, v, f, k); break;
    case 4: c->PSSetConstantBuffers1(s, n, v, f, k); break;
    case 5: c->CSSetConstantBuffers1(s, n, v, f, k); break;
    }
}
void SetShader(ID3D11DeviceContext* c, UINT st, void* p, ID3D11ClassInstance* const* ci, UINT n)
{
    switch (st)
    {
    case 0: c->VSSetShader(static_cast<ID3D11VertexShader*>(p), ci, n); break;
    case 1: c->HSSetShader(static_cast<ID3D11HullShader*>(p), ci, n); break;
    case 2: c->DSSetShader(static_cast<ID3D11DomainShader*>(p), ci, n); break;
    case 3: c->GSSetShader(static_cast<ID3D11GeometryShader*>(p), ci, n); break;
    case 4: c->PSSetShader(static_cast<ID3D11PixelShader*>(p), ci, n); break;
    case 5: c->CSSetShader(static_cast<ID3D11ComputeShader*>(p), ci, n); break;
    }
}

} // namespace

RightRecorder11::RightRecorder11(ID3D11Device* device)
    : m_device(device)
{
    if (m_device) m_device->AddRef();
    m_uavSlots = (m_device && m_device->GetFeatureLevel() >= D3D_FEATURE_LEVEL_11_1) ? D3D11_1_UAV_SLOT_COUNT
                                                                                      : D3D11_PS_CS_UAV_REGISTER_COUNT;
    m_cbSlots = D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT;
    m_ops.resize(4u << 20);
    m_held.reserve(1u << 16);
}

RightRecorder11::~RightRecorder11()
{
    for (IUnknown* p : m_held) p->Release();
    if (m_active) ReleaseSnapshot();
    if (m_device) m_device->Release();
}

HRESULT STDMETHODCALLTYPE RightRecorder11::QueryInterface(REFIID riid, void** ppv)
{
    if (!ppv) return E_POINTER;
    *ppv = nullptr;
    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D11DeviceChild) || riid == __uuidof(ID3D11DeviceContext) ||
        riid == __uuidof(ID3D11DeviceContext1))
    {
        *ppv = static_cast<ID3D11DeviceContext1*>(this);
        AddRef();
        return S_OK;
    }
    return E_NOINTERFACE;
}

void STDMETHODCALLTYPE RightRecorder11::GetDevice(ID3D11Device** pp)
{
    if (!pp) return;
    *pp = m_device;
    if (m_device) m_device->AddRef();
}
HRESULT STDMETHODCALLTYPE RightRecorder11::GetPrivateData(REFGUID, UINT*, void*)          { Unexpected("GetPrivateData"); return DXGI_ERROR_NOT_FOUND; }
HRESULT STDMETHODCALLTYPE RightRecorder11::SetPrivateData(REFGUID, UINT, const void*)     { Unexpected("SetPrivateData"); return E_NOTIMPL; }
HRESULT STDMETHODCALLTYPE RightRecorder11::SetPrivateDataInterface(REFGUID, const IUnknown*) { Unexpected("SetPrivateDataInterface"); return E_NOTIMPL; }

void RightRecorder11::Unexpected(const char* name)
{
    if (m_unexpected) return;
    m_unexpected = name;
    DDILog("BatchRightEye: %s reached the right-eye recorder; recording stops after this pass\n", name);
}

// The arena grows by doubling and is never cleared, so a record is written once (no zero fill first).
unsigned char* RightRecorder11::Put(UINT op, size_t bytes)
{
    const size_t size = sizeof(Head) + Pad8(bytes);
    if (m_used + size > m_ops.size()) m_ops.resize((m_used + size) * 2);
    Head* h = reinterpret_cast<Head*>(&m_ops[m_used]);
    h->op = op;
    h->size = (UINT)size;
    unsigned char* p = &m_ops[m_used] + sizeof(Head);
    m_used += size;
    return p;
}

void RightRecorder11::Hold(IUnknown* p)
{
    if (!p) return;
    p->AddRef();
    m_held.push_back(p);
}

void RightRecorder11::RecSRVs(UINT st, UINT s, UINT n, ID3D11ShaderResourceView* const* v)
{
    if (!m_active) return;
    if (n > D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT) n = D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT;
    Out o = { Put(kSRVs, 32 + (v ? n * 8 : 0)) };
    o.u(st); o.u(s); o.u(n); o.u(v != nullptr);
    if (v) { o.bytes(v, n * sizeof(void*)); for (UINT i = 0; i < n; ++i) Hold(v[i]); }
}

void RightRecorder11::RecSamplers(UINT st, UINT s, UINT n, ID3D11SamplerState* const* v)
{
    if (!m_active) return;
    if (n > D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT) n = D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT;
    Out o = { Put(kSamplers, 32 + (v ? n * 8 : 0)) };
    o.u(st); o.u(s); o.u(n); o.u(v != nullptr);
    if (v) { o.bytes(v, n * sizeof(void*)); for (UINT i = 0; i < n; ++i) Hold(v[i]); }
}

void RightRecorder11::RecCBs(UINT st, UINT s, UINT n, ID3D11Buffer* const* v, const UINT* first, const UINT* num, bool v1)
{
    if (!m_active) return;
    if (n > D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT + 1) n = D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT + 1;
    const size_t arr = (v ? Pad8(n * 8) : 0) + (first ? Pad8(n * 4) : 0) + (num ? Pad8(n * 4) : 0);
    Out o = { Put(kCBs, 56 + arr) };
    o.u(st); o.u(s); o.u(n); o.u(v != nullptr); o.u(first != nullptr); o.u(num != nullptr); o.u(v1);
    if (v) { o.bytes(v, n * sizeof(void*)); for (UINT i = 0; i < n; ++i) Hold(v[i]); }
    if (first) o.bytes(first, n * sizeof(UINT));
    if (num) o.bytes(num, n * sizeof(UINT));
}

void RightRecorder11::RecShader(UINT st, void* shader, ID3D11ClassInstance* const* ci, UINT n)
{
    if (!m_active) return;
    if (n > D3D11_SHADER_MAX_INTERFACES) n = D3D11_SHADER_MAX_INTERFACES;
    Out o = { Put(kShader, 32 + (ci ? n * 8 : 0)) };
    o.u(st); o.ptr(shader); o.u(n); o.u(ci != nullptr);
    Hold(static_cast<IUnknown*>(shader));
    if (ci) { o.bytes(ci, n * sizeof(void*)); for (UINT i = 0; i < n; ++i) Hold(ci[i]); }
}

void STDMETHODCALLTYPE RightRecorder11::IASetInputLayout(ID3D11InputLayout* p)
{
    if (!m_active) return;
    Out o = { Put(kLayout, 8) };
    o.ptr(p);
    Hold(p);
}

void STDMETHODCALLTYPE RightRecorder11::IASetVertexBuffers(UINT s, UINT n, ID3D11Buffer* const* v, const UINT* strides, const UINT* offsets)
{
    if (!m_active) return;
    if (n > D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT) n = D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT;
    const size_t arr = (v ? Pad8(n * 8) : 0) + (strides ? Pad8(n * 4) : 0) + (offsets ? Pad8(n * 4) : 0);
    Out o = { Put(kVBs, 40 + arr) };
    o.u(s); o.u(n); o.u(v != nullptr); o.u(strides != nullptr); o.u(offsets != nullptr);
    if (v) { o.bytes(v, n * sizeof(void*)); for (UINT i = 0; i < n; ++i) Hold(v[i]); }
    if (strides) o.bytes(strides, n * sizeof(UINT));
    if (offsets) o.bytes(offsets, n * sizeof(UINT));
    for (UINT i = 0; v && i < n; ++i) if (v[i] && !ReadsIABuffer(v[i])) m_iaBuffers.push_back(v[i]);
}

bool RightRecorder11::ReadsIABuffer(ID3D11Buffer* b) const
{
    for (ID3D11Buffer* p : m_iaBuffers) if (p == b) return true;
    return false;
}

void STDMETHODCALLTYPE RightRecorder11::IASetIndexBuffer(ID3D11Buffer* p, DXGI_FORMAT f, UINT off)
{
    if (!m_active) return;
    Out o = { Put(kIB, 24) };
    o.ptr(p); o.u(f); o.u(off);
    Hold(p);
    if (p && !ReadsIABuffer(p)) m_iaBuffers.push_back(p);
}

void STDMETHODCALLTYPE RightRecorder11::IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY t)
{
    if (!m_active) return;
    Out o = { Put(kTopology, 8) };
    o.u(t);
}

void STDMETHODCALLTYPE RightRecorder11::RSSetState(ID3D11RasterizerState* p)
{
    if (!m_active) return;
    Out o = { Put(kRSState, 8) };
    o.ptr(p);
    Hold(p);
}

void STDMETHODCALLTYPE RightRecorder11::RSSetViewports(UINT n, const D3D11_VIEWPORT* v)
{
    if (!m_active) return;
    if (n > D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE) n = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    Out o = { Put(kViewports, 16 + (v ? Pad8(n * sizeof(D3D11_VIEWPORT)) : 0)) };
    o.u(n); o.u(v != nullptr);
    if (v) o.bytes(v, n * sizeof(D3D11_VIEWPORT));
}

void STDMETHODCALLTYPE RightRecorder11::RSSetScissorRects(UINT n, const D3D11_RECT* r)
{
    if (!m_active) return;
    if (n > D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE) n = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    Out o = { Put(kScissors, 16 + (r ? Pad8(n * sizeof(D3D11_RECT)) : 0)) };
    o.u(n); o.u(r != nullptr);
    if (r) o.bytes(r, n * sizeof(D3D11_RECT));
}

void STDMETHODCALLTYPE RightRecorder11::OMSetBlendState(ID3D11BlendState* p, const FLOAT f[4], UINT m)
{
    if (!m_active) return;
    Out o = { Put(kBlend, 24 + 16) };
    o.ptr(p); o.u(f != nullptr); o.u(m);
    const FLOAT zero[4] = {};
    o.bytes(f ? f : zero, 4 * sizeof(FLOAT));
    Hold(p);
}

void STDMETHODCALLTYPE RightRecorder11::OMSetDepthStencilState(ID3D11DepthStencilState* p, UINT ref)
{
    if (!m_active) return;
    Out o = { Put(kDepth, 16) };
    o.ptr(p); o.u(ref);
    Hold(p);
}

void STDMETHODCALLTYPE RightRecorder11::OMSetRenderTargets(UINT n, ID3D11RenderTargetView* const* rtvs, ID3D11DepthStencilView* dsv)
{
    if (!m_active) return;
    if (n > D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT) n = D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT;
    Out o = { Put(kTargets, 24 + (rtvs ? n * 8 : 0)) };
    o.u(n); o.u(rtvs != nullptr); o.ptr(dsv);
    Hold(dsv);
    if (rtvs) { o.bytes(rtvs, n * sizeof(void*)); for (UINT i = 0; i < n; ++i) Hold(rtvs[i]); }
}

void STDMETHODCALLTYPE RightRecorder11::OMSetRenderTargetsAndUnorderedAccessViews(UINT nRTV, ID3D11RenderTargetView* const* rtvs,
    ID3D11DepthStencilView* dsv, UINT uavStart, UINT nUAV, ID3D11UnorderedAccessView* const* uavs, const UINT* counts)
{
    if (!m_active) return;
    const bool keepRT = nRTV == D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL;
    const bool keepUAV = nUAV == D3D11_KEEP_UNORDERED_ACCESS_VIEWS;
    const UINT rtCount = (keepRT || !rtvs) ? 0 : (nRTV > D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT ? D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT : nRTV);
    const UINT uaCount = (keepUAV || !uavs) ? 0 : (nUAV > D3D11_1_UAV_SLOT_COUNT ? D3D11_1_UAV_SLOT_COUNT : nUAV);
    const UINT ctCount = (keepUAV || !counts) ? 0 : uaCount;
    Out o = { Put(kTargetsUAVs, 72 + rtCount * 8 + uaCount * 8 + Pad8(ctCount * 4)) };
    o.u(nRTV); o.u(rtvs != nullptr); o.ptr(dsv); o.u(uavStart); o.u(nUAV); o.u(uavs != nullptr); o.u(counts != nullptr);
    o.u(rtCount); o.u(uaCount);
    if (!keepRT) Hold(dsv);
    if (rtCount) { o.bytes(rtvs, rtCount * sizeof(void*)); for (UINT i = 0; i < rtCount; ++i) Hold(rtvs[i]); }
    if (uaCount) { o.bytes(uavs, uaCount * sizeof(void*)); for (UINT i = 0; i < uaCount; ++i) Hold(uavs[i]); }
    if (ctCount) o.bytes(counts, ctCount * sizeof(UINT));
}

void STDMETHODCALLTYPE RightRecorder11::CSSetUnorderedAccessViews(UINT s, UINT n, ID3D11UnorderedAccessView* const* v, const UINT* counts)
{
    if (!m_active) return;
    if (n > D3D11_1_UAV_SLOT_COUNT) n = D3D11_1_UAV_SLOT_COUNT;
    Out o = { Put(kCSUAVs, 32 + (v ? n * 8 : 0) + (counts ? Pad8(n * 4) : 0)) };
    o.u(s); o.u(n); o.u(v != nullptr); o.u(counts != nullptr);
    if (v) { o.bytes(v, n * sizeof(void*)); for (UINT i = 0; i < n; ++i) Hold(v[i]); }
    if (counts) o.bytes(counts, n * sizeof(UINT));
}

void STDMETHODCALLTYPE RightRecorder11::SetPredication(ID3D11Predicate* p, BOOL v)
{
    if (!m_active) return;
    Out o = { Put(kPredication, 16) };
    o.ptr(p); o.u(v);
    Hold(p);
}

void STDMETHODCALLTYPE RightRecorder11::Draw(UINT c, UINT s)
{
    if (!m_active) { Unexpected("Draw"); return; }
    ++m_draws;
    Out o = { Put(kDraw, 16) };
    o.u(c); o.u(s);
}

void STDMETHODCALLTYPE RightRecorder11::DrawIndexed(UINT c, UINT s, INT b)
{
    if (!m_active) { Unexpected("DrawIndexed"); return; }
    ++m_draws;
    Out o = { Put(kDrawIndexed, 24) };
    o.u(c); o.u(s); o.u((UINT64)(INT64)b);
}

void STDMETHODCALLTYPE RightRecorder11::DrawInstanced(UINT c, UINT n, UINT s, UINT si)
{
    if (!m_active) { Unexpected("DrawInstanced"); return; }
    ++m_draws;
    Out o = { Put(kDrawInstanced, 32) };
    o.u(c); o.u(n); o.u(s); o.u(si);
}

void STDMETHODCALLTYPE RightRecorder11::DrawIndexedInstanced(UINT c, UINT n, UINT s, INT b, UINT si)
{
    if (!m_active) { Unexpected("DrawIndexedInstanced"); return; }
    ++m_draws;
    Out o = { Put(kDrawIndexedInstanced, 40) };
    o.u(c); o.u(n); o.u(s); o.u((UINT64)(INT64)b); o.u(si);
}

void STDMETHODCALLTYPE RightRecorder11::DrawAuto()
{
    if (!m_active) { Unexpected("DrawAuto"); return; }
    ++m_draws;
    Put(kDrawAuto, 0);
}

void STDMETHODCALLTYPE RightRecorder11::DrawInstancedIndirect(ID3D11Buffer* a, UINT off)
{
    if (!m_active) { Unexpected("DrawInstancedIndirect"); return; }
    ++m_draws;
    Out o = { Put(kDrawInstancedIndirect, 16) };
    o.ptr(a); o.u(off);
    Hold(a);
}

void STDMETHODCALLTYPE RightRecorder11::DrawIndexedInstancedIndirect(ID3D11Buffer* a, UINT off)
{
    if (!m_active) { Unexpected("DrawIndexedInstancedIndirect"); return; }
    ++m_draws;
    Out o = { Put(kDrawIndexedInstancedIndirect, 16) };
    o.ptr(a); o.u(off);
    Hold(a);
}

// A whole-buffer DISCARD write: the caller fills our copy, and the replay maps the buffer the same way and copies it.
// Map on a resource that is not a dynamic buffer fails, as Map on a deferred context does.
HRESULT STDMETHODCALLTYPE RightRecorder11::Map(ID3D11Resource* r, UINT sub, D3D11_MAP type, UINT, D3D11_MAPPED_SUBRESOURCE* m)
{
    if (!m_active || !r || !m || m_mapRes) { Unexpected("Map"); return E_INVALIDARG; }
    if (type != D3D11_MAP_WRITE_DISCARD || sub != 0) { Unexpected("Map (not a whole-buffer DISCARD)"); return E_INVALIDARG; }
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    r->GetType(&dim);
    if (dim != D3D11_RESOURCE_DIMENSION_BUFFER) { Unexpected("Map (not a buffer)"); return E_INVALIDARG; }
    D3D11_BUFFER_DESC d = {};
    static_cast<ID3D11Buffer*>(r)->GetDesc(&d);
    if (d.Usage != D3D11_USAGE_DYNAMIC) return E_INVALIDARG;
    // The caller writes straight into the record (nothing else records until Unmap).
    Out o = { Put(kWrite, 16 + Pad8(d.ByteWidth)) };
    o.ptr(r); o.u(d.ByteWidth);
    m_mapRes = r;
    m->pData = o.p;
    m->RowPitch = m->DepthPitch = d.ByteWidth;
    return S_OK;
}

void STDMETHODCALLTYPE RightRecorder11::Unmap(ID3D11Resource* r, UINT)
{
    if (!m_active || r != m_mapRes) { Unexpected("Unmap"); return; }
    Hold(r);
    m_mapRes = nullptr;
}

void STDMETHODCALLTYPE RightRecorder11::UpdateSubresource(ID3D11Resource* r, UINT sub, const D3D11_BOX* box, const void* data, UINT rp, UINT dp)
{
    if (!m_active || !r || !data) { Unexpected("UpdateSubresource"); return; }
    D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
    r->GetType(&dim);
    size_t n = 0;
    if (dim == D3D11_RESOURCE_DIMENSION_BUFFER)
    {
        D3D11_BUFFER_DESC d = {};
        static_cast<ID3D11Buffer*>(r)->GetDesc(&d);
        n = box ? (box->right > box->left ? box->right - box->left : 0) : d.ByteWidth;
    }
    else if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D && !box)
    {
        // A whole uncompressed 2D subresource (StereoTexture's rows): its height in rows of the given pitch.
        D3D11_TEXTURE2D_DESC d = {};
        static_cast<ID3D11Texture2D*>(r)->GetDesc(&d);
        const UINT mip = d.MipLevels ? sub % d.MipLevels : 0;
        const UINT h = (d.Height >> mip) ? (d.Height >> mip) : 1;
        if (d.Format >= DXGI_FORMAT_BC1_TYPELESS && d.Format <= DXGI_FORMAT_BC5_SNORM) { Unexpected("UpdateSubresource (BC)"); return; }
        if (d.Format >= DXGI_FORMAT_BC6H_TYPELESS && d.Format <= DXGI_FORMAT_BC7_UNORM_SRGB) { Unexpected("UpdateSubresource (BC)"); return; }
        n = (size_t)rp * h;
    }
    else { Unexpected("UpdateSubresource (not a buffer or whole 2D texture)"); return; }
    Out o = { Put(kUpdate, 48 + Pad8(sizeof(D3D11_BOX)) + Pad8(n)) };
    o.ptr(r); o.u(sub); o.u(box != nullptr); o.u(rp); o.u(dp); o.u(n);
    const D3D11_BOX none = {};
    o.bytes(box ? box : &none, sizeof(D3D11_BOX));
    o.bytes(data, n);
    Hold(r);
}

void STDMETHODCALLTYPE RightRecorder11::ClearState()
{
    if (!m_active) return;
    Put(kClearState, 0);
}

void RightRecorder11::WriteRange(ID3D11Buffer* b, size_t off, const unsigned char* p, size_t n)
{
    if (!m_active || !b || !p) { Unexpected("WriteRange"); return; }
    Out o = { Put(kWriteRange, 24 + Pad8(n)) };
    o.ptr(b); o.u(off); o.u(n);
    o.bytes(p, n);
    Hold(b);
}

void STDMETHODCALLTYPE RightRecorder11::CopyResource(ID3D11Resource* dst, ID3D11Resource* src)
{
    if (!m_active) { Unexpected("CopyResource"); return; }
    Out o = { Put(kCopy, 16) };
    o.ptr(dst); o.ptr(src);
    Hold(dst);
    Hold(src);
}

void STDMETHODCALLTYPE RightRecorder11::CopySubresourceRegion(ID3D11Resource* dst, UINT ds, UINT x, UINT y, UINT z,
                                                              ID3D11Resource* src, UINT ss, const D3D11_BOX* box)
{
    if (!m_active) { Unexpected("CopySubresourceRegion"); return; }
    Out o = { Put(kCopyRegion, 64 + Pad8(sizeof(D3D11_BOX))) };
    o.ptr(dst); o.u(ds); o.u(x); o.u(y); o.u(z); o.ptr(src); o.u(ss); o.u(box != nullptr);
    const D3D11_BOX none = {};
    o.bytes(box ? box : &none, sizeof(D3D11_BOX));
    Hold(dst);
    Hold(src);
}

// The immediate context's whole state through its Get* calls; each returned pointer carries the reference Get added.
void RightRecorder11::Capture(ID3D11DeviceContext* c, ID3D11DeviceContext1* c1)
{
    Snapshot& s = m_snap;
    c->IAGetInputLayout(&s.layout);
    c->IAGetPrimitiveTopology(&s.topology);
    c->IAGetVertexBuffers(0, D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT, s.vb, s.stride, s.offset);
    c->IAGetIndexBuffer(&s.ib, &s.ibFormat, &s.ibOffset);
    for (UINT st = 0; st < 6; ++st)
    {
        StageState& g = s.stage[st];
        g.nInst = D3D11_SHADER_MAX_INTERFACES;
        switch (st)
        {
        case 0: { ID3D11VertexShader* p = nullptr;   c->VSGetShader(&p, g.inst, &g.nInst); g.shader = p;
                  c->VSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, g.srv); c->VSGetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, g.smp);
                  if (c1) c1->VSGetConstantBuffers1(0, m_cbSlots, g.cb, g.first, g.num); else c->VSGetConstantBuffers(0, m_cbSlots, g.cb); break; }
        case 1: { ID3D11HullShader* p = nullptr;     c->HSGetShader(&p, g.inst, &g.nInst); g.shader = p;
                  c->HSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, g.srv); c->HSGetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, g.smp);
                  if (c1) c1->HSGetConstantBuffers1(0, m_cbSlots, g.cb, g.first, g.num); else c->HSGetConstantBuffers(0, m_cbSlots, g.cb); break; }
        case 2: { ID3D11DomainShader* p = nullptr;   c->DSGetShader(&p, g.inst, &g.nInst); g.shader = p;
                  c->DSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, g.srv); c->DSGetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, g.smp);
                  if (c1) c1->DSGetConstantBuffers1(0, m_cbSlots, g.cb, g.first, g.num); else c->DSGetConstantBuffers(0, m_cbSlots, g.cb); break; }
        case 3: { ID3D11GeometryShader* p = nullptr; c->GSGetShader(&p, g.inst, &g.nInst); g.shader = p;
                  c->GSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, g.srv); c->GSGetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, g.smp);
                  if (c1) c1->GSGetConstantBuffers1(0, m_cbSlots, g.cb, g.first, g.num); else c->GSGetConstantBuffers(0, m_cbSlots, g.cb); break; }
        case 4: { ID3D11PixelShader* p = nullptr;    c->PSGetShader(&p, g.inst, &g.nInst); g.shader = p;
                  c->PSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, g.srv); c->PSGetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, g.smp);
                  if (c1) c1->PSGetConstantBuffers1(0, m_cbSlots, g.cb, g.first, g.num); else c->PSGetConstantBuffers(0, m_cbSlots, g.cb); break; }
        case 5: { ID3D11ComputeShader* p = nullptr;  c->CSGetShader(&p, g.inst, &g.nInst); g.shader = p;
                  c->CSGetShaderResources(0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, g.srv); c->CSGetSamplers(0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, g.smp);
                  if (c1) c1->CSGetConstantBuffers1(0, m_cbSlots, g.cb, g.first, g.num); else c->CSGetConstantBuffers(0, m_cbSlots, g.cb); break; }
        }
        if (!c1) for (UINT i = 0; i < m_cbSlots; ++i) { g.first[i] = 0; g.num[i] = D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT; }
    }
    c->RSGetState(&s.rs);
    s.nVp = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    c->RSGetViewports(&s.nVp, s.vp);
    s.nSc = D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE;
    c->RSGetScissorRects(&s.nSc, s.sc);
    c->OMGetBlendState(&s.blend, s.factor, &s.mask);
    c->OMGetDepthStencilState(&s.dss, &s.stencilRef);
    c->OMGetRenderTargets(D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT, s.rtv, &s.dsv);
    c->OMGetRenderTargetsAndUnorderedAccessViews(0, nullptr, nullptr, 0, m_uavSlots, s.omUav);
    c->CSGetUnorderedAccessViews(0, m_uavSlots, s.csUav);
    c->GetPredication(&s.pred, &s.predValue);
}

// Outputs are unbound first, so no input of the captured state meets an output of the current one; the captured outputs
// go back last, as they were bound with these inputs.
void RightRecorder11::Apply(ID3D11DeviceContext* c, ID3D11DeviceContext1* c1)
{
    Snapshot& s = m_snap;
    ID3D11UnorderedAccessView* noUav[D3D11_1_UAV_SLOT_COUNT] = {};
    c->OMSetRenderTargets(0, nullptr, nullptr);
    c->CSSetUnorderedAccessViews(0, m_uavSlots, noUav, nullptr);
    c->IASetInputLayout(s.layout);
    c->IASetPrimitiveTopology(s.topology);
    c->IASetVertexBuffers(0, D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT, s.vb, s.stride, s.offset);
    c->IASetIndexBuffer(s.ib, s.ibFormat, s.ibOffset);
    for (UINT st = 0; st < 6; ++st)
    {
        StageState& g = s.stage[st];
        SetShader(c, st, g.shader, g.nInst ? g.inst : nullptr, g.nInst);
        SetSRVs(c, st, 0, D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT, g.srv);
        SetSamplers(c, st, 0, D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT, g.smp);
        SetCBs(c, st, 0, m_cbSlots, g.cb);
        // A slot bound by 11.1 range: the range the immediate context reported.
        for (UINT i = 0; c1 && i < m_cbSlots; ++i)
        {
            if (!g.cb[i] || !g.num[i]) continue;
            D3D11_BUFFER_DESC d = {};
            g.cb[i]->GetDesc(&d);
            const UINT whole = d.ByteWidth / 16 < D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT ? d.ByteWidth / 16 : D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT;
            if (g.first[i] == 0 && (g.num[i] == D3D11_REQ_CONSTANT_BUFFER_ELEMENT_COUNT || g.num[i] == whole)) continue;
            SetCBs1(c1, st, i, 1, &g.cb[i], &g.first[i], &g.num[i]);
        }
    }
    c->RSSetState(s.rs);
    c->RSSetViewports(s.nVp, s.vp);
    c->RSSetScissorRects(s.nSc, s.sc);
    c->OMSetBlendState(s.blend, s.factor, s.mask);
    c->OMSetDepthStencilState(s.dss, s.stencilRef);
    c->SetPredication(s.pred, s.predValue);
    UINT nRTV = 0;
    for (UINT i = 0; i < D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT; ++i) if (s.rtv[i]) nRTV = i + 1;
    bool anyUav = false;
    for (UINT i = 0; i < m_uavSlots; ++i) if (s.omUav[i]) anyUav = true;
    UINT keep[D3D11_1_UAV_SLOT_COUNT];
    for (UINT i = 0; i < D3D11_1_UAV_SLOT_COUNT; ++i) keep[i] = (UINT)-1;   // counters stay where the game left them
    if (anyUav && nRTV < m_uavSlots)
        c->OMSetRenderTargetsAndUnorderedAccessViews(nRTV, s.rtv, s.dsv, nRTV, m_uavSlots - nRTV, s.omUav + nRTV, keep);
    else
        c->OMSetRenderTargets(nRTV, s.rtv, s.dsv);
    c->CSSetUnorderedAccessViews(0, m_uavSlots, s.csUav, keep);
}

void RightRecorder11::ReleaseSnapshot()
{
    Snapshot& s = m_snap;
    auto rel = [](IUnknown* p) { if (p) p->Release(); };
    rel(s.layout); rel(s.ib); rel(s.rs); rel(s.blend); rel(s.dss); rel(s.dsv); rel(s.pred);
    for (ID3D11Buffer* p : s.vb) rel(p);
    for (StageState& g : s.stage)
    {
        rel(static_cast<IUnknown*>(g.shader));
        for (UINT i = 0; i < g.nInst && i < D3D11_SHADER_MAX_INTERFACES; ++i) rel(g.inst[i]);
        for (ID3D11ShaderResourceView* p : g.srv) rel(p);
        for (ID3D11SamplerState* p : g.smp) rel(p);
        for (UINT i = 0; i < m_cbSlots; ++i) rel(g.cb[i]);
    }
    for (ID3D11RenderTargetView* p : s.rtv) rel(p);
    for (UINT i = 0; i < m_uavSlots; ++i) { rel(s.omUav[i]); rel(s.csUav[i]); }
    s = Snapshot();
}

void RightRecorder11::Begin(ID3D11DeviceContext* imm, ID3D11DeviceContext1* imm1)
{
    if (m_active) return;
    Capture(imm, imm1);
    m_iaBuffers.clear();
    for (ID3D11Buffer* b : m_snap.vb) if (b && !ReadsIABuffer(b)) m_iaBuffers.push_back(b);
    if (m_snap.ib && !ReadsIABuffer(m_snap.ib)) m_iaBuffers.push_back(m_snap.ib);
    m_active = true;
}

void RightRecorder11::Execute(ID3D11DeviceContext* c, ID3D11DeviceContext1* c1)
{
    const unsigned char* p = m_ops.data();
    const unsigned char* end = p + m_used;
    while (p < end)
    {
        const Head* h = reinterpret_cast<const Head*>(p);
        In in = { p + sizeof(Head) };
        p += h->size;
        switch (h->op)
        {
        case kSRVs: case kSamplers:
        {
            const UINT st = (UINT)in.u(), s = (UINT)in.u(), n = (UINT)in.u();
            const bool has = in.u() != 0;
            if (h->op == kSRVs) SetSRVs(c, st, s, n, has ? in.arr<ID3D11ShaderResourceView*>(n) : nullptr);
            else SetSamplers(c, st, s, n, has ? in.arr<ID3D11SamplerState*>(n) : nullptr);
            break;
        }
        case kCBs:
        {
            const UINT st = (UINT)in.u(), s = (UINT)in.u(), n = (UINT)in.u();
            const bool hasV = in.u() != 0, hasF = in.u() != 0, hasN = in.u() != 0, v1 = in.u() != 0;
            ID3D11Buffer* const* v = hasV ? in.arr<ID3D11Buffer*>(n) : nullptr;
            const UINT* f = hasF ? in.arr<UINT>(n) : nullptr;
            const UINT* k = hasN ? in.arr<UINT>(n) : nullptr;
            if (v1) { if (c1) SetCBs1(c1, st, s, n, v, f, k); }
            else SetCBs(c, st, s, n, v);
            break;
        }
        case kShader:
        {
            const UINT st = (UINT)in.u();
            void* sh = in.ptr<void>();
            const UINT n = (UINT)in.u();
            const bool has = in.u() != 0;
            SetShader(c, st, sh, has ? in.arr<ID3D11ClassInstance*>(n) : nullptr, n);
            break;
        }
        case kLayout:   c->IASetInputLayout(in.ptr<ID3D11InputLayout>()); break;
        case kVBs:
        {
            const UINT s = (UINT)in.u(), n = (UINT)in.u();
            const bool hasV = in.u() != 0, hasS = in.u() != 0, hasO = in.u() != 0;
            ID3D11Buffer* const* v = hasV ? in.arr<ID3D11Buffer*>(n) : nullptr;
            const UINT* strides = hasS ? in.arr<UINT>(n) : nullptr;
            const UINT* offsets = hasO ? in.arr<UINT>(n) : nullptr;
            c->IASetVertexBuffers(s, n, v, strides, offsets);
            break;
        }
        case kIB:
        {
            ID3D11Buffer* b = in.ptr<ID3D11Buffer>();
            const DXGI_FORMAT f = (DXGI_FORMAT)in.u();
            c->IASetIndexBuffer(b, f, (UINT)in.u());
            break;
        }
        case kTopology: c->IASetPrimitiveTopology((D3D11_PRIMITIVE_TOPOLOGY)in.u()); break;
        case kRSState:  c->RSSetState(in.ptr<ID3D11RasterizerState>()); break;
        case kViewports:
        {
            const UINT n = (UINT)in.u();
            const bool has = in.u() != 0;
            c->RSSetViewports(n, has ? in.arr<D3D11_VIEWPORT>(n) : nullptr);
            break;
        }
        case kScissors:
        {
            const UINT n = (UINT)in.u();
            const bool has = in.u() != 0;
            c->RSSetScissorRects(n, has ? in.arr<D3D11_RECT>(n) : nullptr);
            break;
        }
        case kBlend:
        {
            ID3D11BlendState* b = in.ptr<ID3D11BlendState>();
            const bool hasF = in.u() != 0;
            const UINT m = (UINT)in.u();
            const FLOAT* f = in.arr<FLOAT>(4);
            c->OMSetBlendState(b, hasF ? f : nullptr, m);
            break;
        }
        case kDepth:
        {
            ID3D11DepthStencilState* d = in.ptr<ID3D11DepthStencilState>();
            c->OMSetDepthStencilState(d, (UINT)in.u());
            break;
        }
        case kTargets:
        {
            const UINT n = (UINT)in.u();
            const bool has = in.u() != 0;
            ID3D11DepthStencilView* dsv = in.ptr<ID3D11DepthStencilView>();
            c->OMSetRenderTargets(n, has ? in.arr<ID3D11RenderTargetView*>(n) : nullptr, dsv);
            break;
        }
        case kTargetsUAVs:
        {
            const UINT nRTV = (UINT)in.u();
            const bool hasR = in.u() != 0;
            ID3D11DepthStencilView* dsv = in.ptr<ID3D11DepthStencilView>();
            const UINT uavStart = (UINT)in.u(), nUAV = (UINT)in.u();
            const bool hasU = in.u() != 0, hasC = in.u() != 0;
            const UINT rtCount = (UINT)in.u(), uaCount = (UINT)in.u();
            ID3D11RenderTargetView* const* r = rtCount ? in.arr<ID3D11RenderTargetView*>(rtCount) : nullptr;
            ID3D11UnorderedAccessView* const* u = uaCount ? in.arr<ID3D11UnorderedAccessView*>(uaCount) : nullptr;
            const UINT* k = (hasC && uaCount) ? in.arr<UINT>(uaCount) : nullptr;
            ID3D11RenderTargetView* noRtv[1] = {};
            ID3D11UnorderedAccessView* noUav[1] = {};
            c->OMSetRenderTargetsAndUnorderedAccessViews(nRTV, hasR ? (r ? r : noRtv) : nullptr, dsv, uavStart, nUAV,
                                                         hasU ? (u ? u : noUav) : nullptr, k);
            break;
        }
        case kCSUAVs:
        {
            const UINT s = (UINT)in.u(), n = (UINT)in.u();
            const bool hasV = in.u() != 0, hasC = in.u() != 0;
            ID3D11UnorderedAccessView* const* v = hasV ? in.arr<ID3D11UnorderedAccessView*>(n) : nullptr;
            c->CSSetUnorderedAccessViews(s, n, v, hasC ? in.arr<UINT>(n) : nullptr);
            break;
        }
        case kPredication:
        {
            ID3D11Predicate* pr = in.ptr<ID3D11Predicate>();
            c->SetPredication(pr, (BOOL)in.u());
            break;
        }
        case kDraw:          { const UINT a = (UINT)in.u(); c->Draw(a, (UINT)in.u()); break; }
        case kDrawIndexed:   { const UINT a = (UINT)in.u(), b = (UINT)in.u(); c->DrawIndexed(a, b, (INT)(INT64)in.u()); break; }
        case kDrawInstanced: { const UINT a = (UINT)in.u(), b = (UINT)in.u(), d = (UINT)in.u(); c->DrawInstanced(a, b, d, (UINT)in.u()); break; }
        case kDrawIndexedInstanced:
        {
            const UINT a = (UINT)in.u(), b = (UINT)in.u(), d = (UINT)in.u();
            const INT bv = (INT)(INT64)in.u();
            c->DrawIndexedInstanced(a, b, d, bv, (UINT)in.u());
            break;
        }
        case kDrawAuto: c->DrawAuto(); break;
        case kDrawInstancedIndirect:        { ID3D11Buffer* a = in.ptr<ID3D11Buffer>(); c->DrawInstancedIndirect(a, (UINT)in.u()); break; }
        case kDrawIndexedInstancedIndirect: { ID3D11Buffer* a = in.ptr<ID3D11Buffer>(); c->DrawIndexedInstancedIndirect(a, (UINT)in.u()); break; }
        case kWrite:
        {
            ID3D11Resource* r = in.ptr<ID3D11Resource>();
            const size_t n = (size_t)in.u();
            const unsigned char* bytes = in.arr<unsigned char>(n);
            D3D11_MAPPED_SUBRESOURCE m = {};
            if (SUCCEEDED(c->Map(r, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)) && m.pData)
            {
                memcpy(m.pData, bytes, n);
                c->Unmap(r, 0);
            }
            break;
        }
        case kUpdate:
        {
            ID3D11Resource* r = in.ptr<ID3D11Resource>();
            const UINT sub = (UINT)in.u();
            const bool hasBox = in.u() != 0;
            const UINT rp = (UINT)in.u(), dp = (UINT)in.u();
            const size_t n = (size_t)in.u();
            const D3D11_BOX* box = in.arr<D3D11_BOX>(1);
            c->UpdateSubresource(r, sub, hasBox ? box : nullptr, in.arr<unsigned char>(n), rp, dp);
            break;
        }
        case kClearState: c->ClearState(); break;
        case kWriteRange:
        {
            ID3D11Buffer* b = in.ptr<ID3D11Buffer>();
            const size_t off = (size_t)in.u(), n = (size_t)in.u();
            const unsigned char* bytes = in.arr<unsigned char>(n);
            D3D11_MAPPED_SUBRESOURCE m = {};
            if (SUCCEEDED(c->Map(b, 0, D3D11_MAP_WRITE_NO_OVERWRITE, 0, &m)) && m.pData)
            {
                memcpy(static_cast<unsigned char*>(m.pData) + off, bytes, n);
                c->Unmap(b, 0);
            }
            break;
        }
        case kCopyRegion:
        {
            ID3D11Resource* dst = in.ptr<ID3D11Resource>();
            const UINT ds = (UINT)in.u(), x = (UINT)in.u(), y = (UINT)in.u(), z = (UINT)in.u();
            ID3D11Resource* src = in.ptr<ID3D11Resource>();
            const UINT ss = (UINT)in.u();
            const bool hasBox = in.u() != 0;
            const D3D11_BOX* box = in.arr<D3D11_BOX>(1);
            c->CopySubresourceRegion(dst, ds, x, y, z, src, ss, hasBox ? box : nullptr);
            break;
        }
        case kCopy:
        {
            ID3D11Resource* dst = in.ptr<ID3D11Resource>();
            c->CopyResource(dst, in.ptr<ID3D11Resource>());
            break;
        }
        }
    }
}

void RightRecorder11::Replay(ID3D11DeviceContext* imm, ID3D11DeviceContext1* imm1)
{
    if (!m_active) return;
    Apply(imm, imm1);
    Execute(imm, imm1);
    ReleaseSnapshot();
    for (IUnknown* p : m_held) p->Release();
    m_held.clear();
    m_used = 0;
    m_mapRes = nullptr;
    m_draws = 0;
    m_active = false;
}

} // namespace wiz3d
