/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// <BatchRightEye>: the right eye's calls recorded on the CPU and replayed on the immediate context once per pass, as
// DX12's right list. It records what the right-eye path and the state mirror call; any other call is reported.

#pragma once

#include <windows.h>
#include <d3d9types.h>   // D3DCOLORVALUE for the DXGI_RGBA shim below
#ifndef _DXGI_RGBA_DEFINED
#define _DXGI_RGBA_DEFINED
typedef D3DCOLORVALUE DXGI_RGBA;
#endif
#include <d3d11_1.h>
#include <vector>

namespace wiz3d
{

class RightRecorder11 final : public ID3D11DeviceContext1
{
public:
    explicit RightRecorder11(ID3D11Device* device);
    ~RightRecorder11();

    // Takes the immediate context's state (its Get* calls), so the replay starts from it whatever the game set before.
    void Begin(ID3D11DeviceContext* imm, ID3D11DeviceContext1* imm1);
    // The recorded calls on the immediate context from the state Begin took; they end in the immediate context's state
    // now, as every right-eye draw rebinds the left eye's views after it.
    void Replay(ID3D11DeviceContext* imm, ID3D11DeviceContext1* imm1);
    bool Active() const { return m_active; }
    // A range written in place, as Context11Proxy::WriteRingRange does: replayed as a NO_OVERWRITE map at the offset.
    void WriteRange(ID3D11Buffer* b, size_t off, const unsigned char* p, size_t n);
    // The first call the recorder does not record, null if none; the caller stops recording.
    const char* Unexpected() const { return m_unexpected; }
    unsigned Draws() const { return m_draws; }   // draws in the recording
    // A vertex or index buffer the recorded draws can read: bound when recording began or bound since.
    bool ReadsIABuffer(ID3D11Buffer* b) const;

    // IUnknown, ID3D11DeviceChild
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppv) override;
    ULONG   STDMETHODCALLTYPE AddRef() override  { return (ULONG)InterlockedIncrement(&m_refs); }
    ULONG   STDMETHODCALLTYPE Release() override
    {
        const LONG r = InterlockedDecrement(&m_refs);
        if (r == 0) delete this;
        return (ULONG)r;
    }
    void    STDMETHODCALLTYPE GetDevice(ID3D11Device** pp) override;
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID, UINT*, void*) override;
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID, UINT, const void*) override;
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID, const IUnknown*) override;

    // Recorded: state, draws, buffer writes
    void STDMETHODCALLTYPE VSSetShaderResources(UINT s, UINT n, ID3D11ShaderResourceView* const* v) override { RecSRVs(0, s, n, v); }
    void STDMETHODCALLTYPE HSSetShaderResources(UINT s, UINT n, ID3D11ShaderResourceView* const* v) override { RecSRVs(1, s, n, v); }
    void STDMETHODCALLTYPE DSSetShaderResources(UINT s, UINT n, ID3D11ShaderResourceView* const* v) override { RecSRVs(2, s, n, v); }
    void STDMETHODCALLTYPE GSSetShaderResources(UINT s, UINT n, ID3D11ShaderResourceView* const* v) override { RecSRVs(3, s, n, v); }
    void STDMETHODCALLTYPE PSSetShaderResources(UINT s, UINT n, ID3D11ShaderResourceView* const* v) override { RecSRVs(4, s, n, v); }
    void STDMETHODCALLTYPE CSSetShaderResources(UINT s, UINT n, ID3D11ShaderResourceView* const* v) override { RecSRVs(5, s, n, v); }
    void STDMETHODCALLTYPE VSSetSamplers(UINT s, UINT n, ID3D11SamplerState* const* v) override { RecSamplers(0, s, n, v); }
    void STDMETHODCALLTYPE HSSetSamplers(UINT s, UINT n, ID3D11SamplerState* const* v) override { RecSamplers(1, s, n, v); }
    void STDMETHODCALLTYPE DSSetSamplers(UINT s, UINT n, ID3D11SamplerState* const* v) override { RecSamplers(2, s, n, v); }
    void STDMETHODCALLTYPE GSSetSamplers(UINT s, UINT n, ID3D11SamplerState* const* v) override { RecSamplers(3, s, n, v); }
    void STDMETHODCALLTYPE PSSetSamplers(UINT s, UINT n, ID3D11SamplerState* const* v) override { RecSamplers(4, s, n, v); }
    void STDMETHODCALLTYPE CSSetSamplers(UINT s, UINT n, ID3D11SamplerState* const* v) override { RecSamplers(5, s, n, v); }
    void STDMETHODCALLTYPE VSSetConstantBuffers(UINT s, UINT n, ID3D11Buffer* const* v) override { RecCBs(0, s, n, v, nullptr, nullptr, false); }
    void STDMETHODCALLTYPE HSSetConstantBuffers(UINT s, UINT n, ID3D11Buffer* const* v) override { RecCBs(1, s, n, v, nullptr, nullptr, false); }
    void STDMETHODCALLTYPE DSSetConstantBuffers(UINT s, UINT n, ID3D11Buffer* const* v) override { RecCBs(2, s, n, v, nullptr, nullptr, false); }
    void STDMETHODCALLTYPE GSSetConstantBuffers(UINT s, UINT n, ID3D11Buffer* const* v) override { RecCBs(3, s, n, v, nullptr, nullptr, false); }
    void STDMETHODCALLTYPE PSSetConstantBuffers(UINT s, UINT n, ID3D11Buffer* const* v) override { RecCBs(4, s, n, v, nullptr, nullptr, false); }
    void STDMETHODCALLTYPE CSSetConstantBuffers(UINT s, UINT n, ID3D11Buffer* const* v) override { RecCBs(5, s, n, v, nullptr, nullptr, false); }
    void STDMETHODCALLTYPE VSSetConstantBuffers1(UINT s, UINT n, ID3D11Buffer* const* v, const UINT* f, const UINT* c) override { RecCBs(0, s, n, v, f, c, true); }
    void STDMETHODCALLTYPE HSSetConstantBuffers1(UINT s, UINT n, ID3D11Buffer* const* v, const UINT* f, const UINT* c) override { RecCBs(1, s, n, v, f, c, true); }
    void STDMETHODCALLTYPE DSSetConstantBuffers1(UINT s, UINT n, ID3D11Buffer* const* v, const UINT* f, const UINT* c) override { RecCBs(2, s, n, v, f, c, true); }
    void STDMETHODCALLTYPE GSSetConstantBuffers1(UINT s, UINT n, ID3D11Buffer* const* v, const UINT* f, const UINT* c) override { RecCBs(3, s, n, v, f, c, true); }
    void STDMETHODCALLTYPE PSSetConstantBuffers1(UINT s, UINT n, ID3D11Buffer* const* v, const UINT* f, const UINT* c) override { RecCBs(4, s, n, v, f, c, true); }
    void STDMETHODCALLTYPE CSSetConstantBuffers1(UINT s, UINT n, ID3D11Buffer* const* v, const UINT* f, const UINT* c) override { RecCBs(5, s, n, v, f, c, true); }
    void STDMETHODCALLTYPE VSSetShader(ID3D11VertexShader* p, ID3D11ClassInstance* const* ci, UINT n) override   { RecShader(0, p, ci, n); }
    void STDMETHODCALLTYPE HSSetShader(ID3D11HullShader* p, ID3D11ClassInstance* const* ci, UINT n) override     { RecShader(1, p, ci, n); }
    void STDMETHODCALLTYPE DSSetShader(ID3D11DomainShader* p, ID3D11ClassInstance* const* ci, UINT n) override   { RecShader(2, p, ci, n); }
    void STDMETHODCALLTYPE GSSetShader(ID3D11GeometryShader* p, ID3D11ClassInstance* const* ci, UINT n) override { RecShader(3, p, ci, n); }
    void STDMETHODCALLTYPE PSSetShader(ID3D11PixelShader* p, ID3D11ClassInstance* const* ci, UINT n) override    { RecShader(4, p, ci, n); }
    void STDMETHODCALLTYPE CSSetShader(ID3D11ComputeShader* p, ID3D11ClassInstance* const* ci, UINT n) override  { RecShader(5, p, ci, n); }
    void STDMETHODCALLTYPE IASetInputLayout(ID3D11InputLayout* p) override;
    void STDMETHODCALLTYPE IASetVertexBuffers(UINT s, UINT n, ID3D11Buffer* const* v, const UINT* strides, const UINT* offsets) override;
    void STDMETHODCALLTYPE IASetIndexBuffer(ID3D11Buffer* p, DXGI_FORMAT f, UINT o) override;
    void STDMETHODCALLTYPE IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY t) override;
    void STDMETHODCALLTYPE RSSetState(ID3D11RasterizerState* p) override;
    void STDMETHODCALLTYPE RSSetViewports(UINT n, const D3D11_VIEWPORT* v) override;
    void STDMETHODCALLTYPE RSSetScissorRects(UINT n, const D3D11_RECT* r) override;
    void STDMETHODCALLTYPE OMSetBlendState(ID3D11BlendState* p, const FLOAT f[4], UINT m) override;
    void STDMETHODCALLTYPE OMSetDepthStencilState(ID3D11DepthStencilState* p, UINT ref) override;
    void STDMETHODCALLTYPE OMSetRenderTargets(UINT n, ID3D11RenderTargetView* const* rtvs, ID3D11DepthStencilView* dsv) override;
    void STDMETHODCALLTYPE OMSetRenderTargetsAndUnorderedAccessViews(UINT nRTV, ID3D11RenderTargetView* const* rtvs,
        ID3D11DepthStencilView* dsv, UINT uavStart, UINT nUAV, ID3D11UnorderedAccessView* const* uavs, const UINT* counts) override;
    void STDMETHODCALLTYPE CSSetUnorderedAccessViews(UINT s, UINT n, ID3D11UnorderedAccessView* const* v, const UINT* counts) override;
    void STDMETHODCALLTYPE SetPredication(ID3D11Predicate* p, BOOL v) override;
    void STDMETHODCALLTYPE Draw(UINT c, UINT s) override;
    void STDMETHODCALLTYPE DrawIndexed(UINT c, UINT s, INT b) override;
    void STDMETHODCALLTYPE DrawInstanced(UINT c, UINT n, UINT s, UINT si) override;
    void STDMETHODCALLTYPE DrawIndexedInstanced(UINT c, UINT n, UINT s, INT b, UINT si) override;
    void STDMETHODCALLTYPE DrawAuto() override;
    void STDMETHODCALLTYPE DrawInstancedIndirect(ID3D11Buffer* a, UINT o) override;
    void STDMETHODCALLTYPE DrawIndexedInstancedIndirect(ID3D11Buffer* a, UINT o) override;
    HRESULT STDMETHODCALLTYPE Map(ID3D11Resource* r, UINT sub, D3D11_MAP type, UINT flags, D3D11_MAPPED_SUBRESOURCE* m) override;
    void STDMETHODCALLTYPE Unmap(ID3D11Resource* r, UINT sub) override;
    void STDMETHODCALLTYPE UpdateSubresource(ID3D11Resource* r, UINT sub, const D3D11_BOX* box, const void* data, UINT rp, UINT dp) override;
    void STDMETHODCALLTYPE UpdateSubresource1(ID3D11Resource* r, UINT sub, const D3D11_BOX* box, const void* data, UINT rp, UINT dp, UINT) override
    { UpdateSubresource(r, sub, box, data, rp, dp); }
    void STDMETHODCALLTYPE ClearState() override;

    // Not recorded: reported, and the outputs left empty
    void STDMETHODCALLTYPE Begin(ID3D11Asynchronous*) override                                  { Unexpected("Begin"); }
    void STDMETHODCALLTYPE End(ID3D11Asynchronous*) override                                    { Unexpected("End"); }
    HRESULT STDMETHODCALLTYPE GetData(ID3D11Asynchronous*, void*, UINT, UINT) override          { Unexpected("GetData"); return S_FALSE; }
    void STDMETHODCALLTYPE SOSetTargets(UINT, ID3D11Buffer* const*, const UINT*) override       { Unexpected("SOSetTargets"); }
    void STDMETHODCALLTYPE Dispatch(UINT, UINT, UINT) override                                  { Unexpected("Dispatch"); }
    void STDMETHODCALLTYPE DispatchIndirect(ID3D11Buffer*, UINT) override                       { Unexpected("DispatchIndirect"); }
    void STDMETHODCALLTYPE CopySubresourceRegion(ID3D11Resource* dst, UINT ds, UINT x, UINT y, UINT z, ID3D11Resource* src, UINT ss, const D3D11_BOX* box) override;
    void STDMETHODCALLTYPE CopyResource(ID3D11Resource* dst, ID3D11Resource* src) override;
    void STDMETHODCALLTYPE CopyStructureCount(ID3D11Buffer*, UINT, ID3D11UnorderedAccessView*) override { Unexpected("CopyStructureCount"); }
    void STDMETHODCALLTYPE ClearRenderTargetView(ID3D11RenderTargetView*, const FLOAT[4]) override { Unexpected("ClearRenderTargetView"); }
    void STDMETHODCALLTYPE ClearUnorderedAccessViewUint(ID3D11UnorderedAccessView*, const UINT[4]) override { Unexpected("ClearUnorderedAccessViewUint"); }
    void STDMETHODCALLTYPE ClearUnorderedAccessViewFloat(ID3D11UnorderedAccessView*, const FLOAT[4]) override { Unexpected("ClearUnorderedAccessViewFloat"); }
    void STDMETHODCALLTYPE ClearDepthStencilView(ID3D11DepthStencilView*, UINT, FLOAT, UINT8) override { Unexpected("ClearDepthStencilView"); }
    void STDMETHODCALLTYPE GenerateMips(ID3D11ShaderResourceView*) override                     { Unexpected("GenerateMips"); }
    void STDMETHODCALLTYPE SetResourceMinLOD(ID3D11Resource*, FLOAT) override                   { Unexpected("SetResourceMinLOD"); }
    FLOAT STDMETHODCALLTYPE GetResourceMinLOD(ID3D11Resource*) override                         { Unexpected("GetResourceMinLOD"); return 0.f; }
    void STDMETHODCALLTYPE ResolveSubresource(ID3D11Resource*, UINT, ID3D11Resource*, UINT, DXGI_FORMAT) override { Unexpected("ResolveSubresource"); }
    void STDMETHODCALLTYPE ExecuteCommandList(ID3D11CommandList*, BOOL) override                { Unexpected("ExecuteCommandList"); }
    void STDMETHODCALLTYPE Flush() override                                                     { Unexpected("Flush"); }
    D3D11_DEVICE_CONTEXT_TYPE STDMETHODCALLTYPE GetType() override                              { return D3D11_DEVICE_CONTEXT_IMMEDIATE; }
    UINT STDMETHODCALLTYPE GetContextFlags() override                                           { return 0; }
    HRESULT STDMETHODCALLTYPE FinishCommandList(BOOL, ID3D11CommandList** pp) override          { Unexpected("FinishCommandList"); if (pp) *pp = nullptr; return DXGI_ERROR_INVALID_CALL; }
    void STDMETHODCALLTYPE CopySubresourceRegion1(ID3D11Resource*, UINT, UINT, UINT, UINT, ID3D11Resource*, UINT, const D3D11_BOX*, UINT) override { Unexpected("CopySubresourceRegion1"); }
    void STDMETHODCALLTYPE DiscardResource(ID3D11Resource*) override                            { Unexpected("DiscardResource"); }
    void STDMETHODCALLTYPE DiscardView(ID3D11View*) override                                    { Unexpected("DiscardView"); }
    void STDMETHODCALLTYPE SwapDeviceContextState(ID3DDeviceContextState*, ID3DDeviceContextState** pp) override { Unexpected("SwapDeviceContextState"); if (pp) *pp = nullptr; }
    void STDMETHODCALLTYPE ClearView(ID3D11View*, const FLOAT[4], const D3D11_RECT*, UINT) override { Unexpected("ClearView"); }
    void STDMETHODCALLTYPE DiscardView1(ID3D11View*, const D3D11_RECT*, UINT) override          { Unexpected("DiscardView1"); }

    // Getters: not recorded either (the right-eye path of the shipped profiles calls none)
    void STDMETHODCALLTYPE VSGetConstantBuffers(UINT, UINT n, ID3D11Buffer** pp) override               { NoGet("VSGetConstantBuffers", pp, n); }
    void STDMETHODCALLTYPE HSGetConstantBuffers(UINT, UINT n, ID3D11Buffer** pp) override               { NoGet("HSGetConstantBuffers", pp, n); }
    void STDMETHODCALLTYPE DSGetConstantBuffers(UINT, UINT n, ID3D11Buffer** pp) override               { NoGet("DSGetConstantBuffers", pp, n); }
    void STDMETHODCALLTYPE GSGetConstantBuffers(UINT, UINT n, ID3D11Buffer** pp) override               { NoGet("GSGetConstantBuffers", pp, n); }
    void STDMETHODCALLTYPE PSGetConstantBuffers(UINT, UINT n, ID3D11Buffer** pp) override               { NoGet("PSGetConstantBuffers", pp, n); }
    void STDMETHODCALLTYPE CSGetConstantBuffers(UINT, UINT n, ID3D11Buffer** pp) override               { NoGet("CSGetConstantBuffers", pp, n); }
    void STDMETHODCALLTYPE VSGetConstantBuffers1(UINT, UINT n, ID3D11Buffer** pp, UINT*, UINT*) override { NoGet("VSGetConstantBuffers1", pp, n); }
    void STDMETHODCALLTYPE HSGetConstantBuffers1(UINT, UINT n, ID3D11Buffer** pp, UINT*, UINT*) override { NoGet("HSGetConstantBuffers1", pp, n); }
    void STDMETHODCALLTYPE DSGetConstantBuffers1(UINT, UINT n, ID3D11Buffer** pp, UINT*, UINT*) override { NoGet("DSGetConstantBuffers1", pp, n); }
    void STDMETHODCALLTYPE GSGetConstantBuffers1(UINT, UINT n, ID3D11Buffer** pp, UINT*, UINT*) override { NoGet("GSGetConstantBuffers1", pp, n); }
    void STDMETHODCALLTYPE PSGetConstantBuffers1(UINT, UINT n, ID3D11Buffer** pp, UINT*, UINT*) override { NoGet("PSGetConstantBuffers1", pp, n); }
    void STDMETHODCALLTYPE CSGetConstantBuffers1(UINT, UINT n, ID3D11Buffer** pp, UINT*, UINT*) override { NoGet("CSGetConstantBuffers1", pp, n); }
    void STDMETHODCALLTYPE VSGetShaderResources(UINT, UINT n, ID3D11ShaderResourceView** pp) override   { NoGet("VSGetShaderResources", pp, n); }
    void STDMETHODCALLTYPE HSGetShaderResources(UINT, UINT n, ID3D11ShaderResourceView** pp) override   { NoGet("HSGetShaderResources", pp, n); }
    void STDMETHODCALLTYPE DSGetShaderResources(UINT, UINT n, ID3D11ShaderResourceView** pp) override   { NoGet("DSGetShaderResources", pp, n); }
    void STDMETHODCALLTYPE GSGetShaderResources(UINT, UINT n, ID3D11ShaderResourceView** pp) override   { NoGet("GSGetShaderResources", pp, n); }
    void STDMETHODCALLTYPE PSGetShaderResources(UINT, UINT n, ID3D11ShaderResourceView** pp) override   { NoGet("PSGetShaderResources", pp, n); }
    void STDMETHODCALLTYPE CSGetShaderResources(UINT, UINT n, ID3D11ShaderResourceView** pp) override   { NoGet("CSGetShaderResources", pp, n); }
    void STDMETHODCALLTYPE VSGetSamplers(UINT, UINT n, ID3D11SamplerState** pp) override                { NoGet("VSGetSamplers", pp, n); }
    void STDMETHODCALLTYPE HSGetSamplers(UINT, UINT n, ID3D11SamplerState** pp) override                { NoGet("HSGetSamplers", pp, n); }
    void STDMETHODCALLTYPE DSGetSamplers(UINT, UINT n, ID3D11SamplerState** pp) override                { NoGet("DSGetSamplers", pp, n); }
    void STDMETHODCALLTYPE GSGetSamplers(UINT, UINT n, ID3D11SamplerState** pp) override                { NoGet("GSGetSamplers", pp, n); }
    void STDMETHODCALLTYPE PSGetSamplers(UINT, UINT n, ID3D11SamplerState** pp) override                { NoGet("PSGetSamplers", pp, n); }
    void STDMETHODCALLTYPE CSGetSamplers(UINT, UINT n, ID3D11SamplerState** pp) override                { NoGet("CSGetSamplers", pp, n); }
    void STDMETHODCALLTYPE VSGetShader(ID3D11VertexShader** pp, ID3D11ClassInstance**, UINT* n) override   { NoGet("VSGetShader", pp, 1); if (n) *n = 0; }
    void STDMETHODCALLTYPE HSGetShader(ID3D11HullShader** pp, ID3D11ClassInstance**, UINT* n) override     { NoGet("HSGetShader", pp, 1); if (n) *n = 0; }
    void STDMETHODCALLTYPE DSGetShader(ID3D11DomainShader** pp, ID3D11ClassInstance**, UINT* n) override   { NoGet("DSGetShader", pp, 1); if (n) *n = 0; }
    void STDMETHODCALLTYPE GSGetShader(ID3D11GeometryShader** pp, ID3D11ClassInstance**, UINT* n) override { NoGet("GSGetShader", pp, 1); if (n) *n = 0; }
    void STDMETHODCALLTYPE PSGetShader(ID3D11PixelShader** pp, ID3D11ClassInstance**, UINT* n) override    { NoGet("PSGetShader", pp, 1); if (n) *n = 0; }
    void STDMETHODCALLTYPE CSGetShader(ID3D11ComputeShader** pp, ID3D11ClassInstance**, UINT* n) override  { NoGet("CSGetShader", pp, 1); if (n) *n = 0; }
    void STDMETHODCALLTYPE IAGetInputLayout(ID3D11InputLayout** pp) override                             { NoGet("IAGetInputLayout", pp, 1); }
    void STDMETHODCALLTYPE IAGetVertexBuffers(UINT, UINT n, ID3D11Buffer** pp, UINT*, UINT*) override    { NoGet("IAGetVertexBuffers", pp, n); }
    void STDMETHODCALLTYPE IAGetIndexBuffer(ID3D11Buffer** pp, DXGI_FORMAT*, UINT*) override            { NoGet("IAGetIndexBuffer", pp, 1); }
    void STDMETHODCALLTYPE IAGetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY* t) override                  { Unexpected("IAGetPrimitiveTopology"); if (t) *t = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED; }
    void STDMETHODCALLTYPE GetPredication(ID3D11Predicate** pp, BOOL* v) override                       { NoGet("GetPredication", pp, 1); if (v) *v = FALSE; }
    void STDMETHODCALLTYPE OMGetRenderTargets(UINT n, ID3D11RenderTargetView** pp, ID3D11DepthStencilView** d) override
    { NoGet("OMGetRenderTargets", pp, n); NoGet("OMGetRenderTargets", d, 1); }
    void STDMETHODCALLTYPE OMGetRenderTargetsAndUnorderedAccessViews(UINT n, ID3D11RenderTargetView** pp, ID3D11DepthStencilView** d,
        UINT, UINT nu, ID3D11UnorderedAccessView** u) override
    { NoGet("OMGetRenderTargetsAndUnorderedAccessViews", pp, n); NoGet("OMGetRenderTargetsAndUnorderedAccessViews", d, 1); NoGet("OMGetRenderTargetsAndUnorderedAccessViews", u, nu); }
    void STDMETHODCALLTYPE OMGetBlendState(ID3D11BlendState** pp, FLOAT[4], UINT*) override             { NoGet("OMGetBlendState", pp, 1); }
    void STDMETHODCALLTYPE OMGetDepthStencilState(ID3D11DepthStencilState** pp, UINT*) override          { NoGet("OMGetDepthStencilState", pp, 1); }
    void STDMETHODCALLTYPE SOGetTargets(UINT n, ID3D11Buffer** pp) override                              { NoGet("SOGetTargets", pp, n); }
    void STDMETHODCALLTYPE RSGetState(ID3D11RasterizerState** pp) override                               { NoGet("RSGetState", pp, 1); }
    void STDMETHODCALLTYPE RSGetViewports(UINT* n, D3D11_VIEWPORT*) override                              { Unexpected("RSGetViewports"); if (n) *n = 0; }
    void STDMETHODCALLTYPE RSGetScissorRects(UINT* n, D3D11_RECT*) override                               { Unexpected("RSGetScissorRects"); if (n) *n = 0; }
    void STDMETHODCALLTYPE CSGetUnorderedAccessViews(UINT, UINT n, ID3D11UnorderedAccessView** pp) override { NoGet("CSGetUnorderedAccessViews", pp, n); }

private:
    struct StageState
    {
        void* shader;
        ID3D11ClassInstance* inst[D3D11_SHADER_MAX_INTERFACES];
        UINT nInst;
        ID3D11ShaderResourceView* srv[D3D11_COMMONSHADER_INPUT_RESOURCE_SLOT_COUNT];
        ID3D11SamplerState* smp[D3D11_COMMONSHADER_SAMPLER_SLOT_COUNT];
        ID3D11Buffer* cb[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT + 1];
        UINT first[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT + 1], num[D3D11_COMMONSHADER_CONSTANT_BUFFER_API_SLOT_COUNT + 1];
    };
    // The immediate context's state when the recording began, each pointer held by the Get* call that returned it.
    struct Snapshot
    {
        ID3D11InputLayout* layout;
        D3D11_PRIMITIVE_TOPOLOGY topology;
        ID3D11Buffer* vb[D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT];
        UINT stride[D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT], offset[D3D11_IA_VERTEX_INPUT_RESOURCE_SLOT_COUNT];
        ID3D11Buffer* ib; DXGI_FORMAT ibFormat; UINT ibOffset;
        StageState stage[6];
        ID3D11RasterizerState* rs;
        D3D11_VIEWPORT vp[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]; UINT nVp;
        D3D11_RECT sc[D3D11_VIEWPORT_AND_SCISSORRECT_OBJECT_COUNT_PER_PIPELINE]; UINT nSc;
        ID3D11BlendState* blend; FLOAT factor[4]; UINT mask;
        ID3D11DepthStencilState* dss; UINT stencilRef;
        ID3D11RenderTargetView* rtv[D3D11_SIMULTANEOUS_RENDER_TARGET_COUNT];
        ID3D11DepthStencilView* dsv;
        ID3D11UnorderedAccessView* omUav[D3D11_1_UAV_SLOT_COUNT];
        ID3D11UnorderedAccessView* csUav[D3D11_1_UAV_SLOT_COUNT];
        ID3D11Predicate* pred; BOOL predValue;
    };

    void RecSRVs(UINT st, UINT s, UINT n, ID3D11ShaderResourceView* const* v);
    void RecSamplers(UINT st, UINT s, UINT n, ID3D11SamplerState* const* v);
    void RecCBs(UINT st, UINT s, UINT n, ID3D11Buffer* const* v, const UINT* first, const UINT* num, bool v1);
    void RecShader(UINT st, void* shader, ID3D11ClassInstance* const* ci, UINT n);
    unsigned char* Put(UINT op, size_t bytes);
    void Hold(IUnknown* p);
    void Unexpected(const char* name);
    template<class T> void NoGet(const char* name, T** pp, UINT n)
    {
        Unexpected(name);
        for (UINT i = 0; pp && i < n; ++i) pp[i] = nullptr;
    }
    void Capture(ID3D11DeviceContext* imm, ID3D11DeviceContext1* imm1);
    void Apply(ID3D11DeviceContext* imm, ID3D11DeviceContext1* imm1);
    void ReleaseSnapshot();
    void Execute(ID3D11DeviceContext* imm, ID3D11DeviceContext1* imm1);

    volatile LONG              m_refs = 1;
    ID3D11Device*              m_device;
    UINT                       m_uavSlots;      // 64 at feature level 11.1, else 8
    UINT                       m_cbSlots;
    bool                       m_active = false;
    const char*                m_unexpected = nullptr;
    unsigned                   m_draws = 0;
    std::vector<ID3D11Buffer*> m_iaBuffers;     // ReadsIABuffer: the snapshot's and the recorded IA binds
    std::vector<unsigned char> m_ops;           // records of 8-byte fields, m_used bytes of it filled
    size_t                     m_used = 0;
    std::vector<IUnknown*>     m_held;          // one reference each, dropped after the replay
    Snapshot                   m_snap = {};
    ID3D11Resource*            m_mapRes = nullptr;   // mapped into the last record until Unmap
};

} // namespace wiz3d
