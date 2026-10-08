/* wiz3D - ID3D11DeviceContext proxy implementation (Option B Stage 2)
 *
 * Pure passthrough port of NvDirectMode/d3d11/Context11Proxy. The stage-3 BB
 * tracking and stage-4 magic-header capture were stripped for the MVP — the
 * job here is to prove COM identity + refcounting are right, not to do any
 * stereo work yet. OMSet/RSSetViewports/CopyResource/CopySubresourceRegion
 * are forwarded unchanged; per-eye behaviour will be re-added in Stage 4.
 */

#include "StdAfx.h"
#include "Context11Proxy.h"
#include "../Shared/StereoMatrix.h"
#include "Device11Proxy.h"
#include "Texture2D11Proxy.h"
#include "RTV11Proxy.h"
#include "DSV11Proxy.h"
#include "Buffer11Proxy.h"
#include "SRV11Proxy.h"
#include "UAV11Proxy.h"
#include "Texture1D11Proxy.h"
#include "Texture3D11Proxy.h"
#include "proxy_factory.h"     // TryUnwrap* helpers
#include "AdapterFunctions.h"  // DDILog
#include "..\S3DAPI\ShaderProfileData.h"  // g_ProfileData matrix declarations
#include <d3dcompiler.h>       // D3DDisassemble: the UAV slots a compute shader reads
#include <cctype>
#include <sstream>
#include <mutex>

#pragma comment(lib, "d3dcompiler.lib")

// Slot-count maxima now live on Context11Proxy (see the header) so the
// eye-tracking arrays and the unwrap temporaries stay the same size.
using wiz3d::Context11Proxy;
static constexpr UINT kMaxSRVs       = Context11Proxy::kMaxSRVs;
static constexpr UINT kMaxSamplers   = Context11Proxy::kMaxSamplers;
static constexpr UINT kMaxCBs        = Context11Proxy::kMaxCBs;
static constexpr UINT kMaxRTVs       = Context11Proxy::kMaxRTVs;
static constexpr UINT kMaxUAVs       = Context11Proxy::kMaxUAVs;
static constexpr UINT kMaxVBs        = Context11Proxy::kMaxVBs;
static constexpr UINT kMaxSOBuffers  = Context11Proxy::kMaxSOBuffers;
static constexpr UINT kMaxClassInst  = Context11Proxy::kMaxClassInst;

// Stage 3c.1: lightweight inline unwrap for ID3D11Buffer*. Used at every
// method boundary that hands a buffer to the real D3D11 runtime — passing
// our Buffer11Proxy directly would crash the runtime because it doesn't
// understand our vtable layout past ID3D11Buffer methods. Returns the raw
// buffer (or the original pointer if not ours, including nullptr).
static inline ID3D11Buffer* UnwrapBuf(ID3D11Buffer* p)
{
    if (!p) return nullptr;
    if (auto* bp = wiz3d::TryUnwrapBuffer(static_cast<ID3D11Resource*>(p)))
        return bp->GetReal();
    return p;
}

// The right eye's indirect arguments: the args buffer's right sibling, which the right eye's dispatches write.
static inline ID3D11Buffer* RightArgs(ID3D11Buffer* p)
{
    if (auto* bp = p ? wiz3d::TryUnwrapBuffer(static_cast<ID3D11Resource*>(p)) : nullptr)
        return bp->GetRealRight() ? bp->GetRealRight() : bp->GetReal();
    return p;
}

// Stage 3c.1: unwrap ID3D11Resource* for the eye-aware Do* helpers. Tries
// Texture2D11Proxy first (which has eye-stereo siblings) then Buffer11Proxy
// (no stereo doubling, single real). 3c.2 adds Texture1D/Texture3D — both
// passthrough (no stereo) but still need unwrap so the real runtime gets
// the real pointer.
static inline ID3D11Resource* UnwrapResourceForEye(ID3D11Resource* p, bool pickRight)
{
    if (!p) return nullptr;
    if (auto* tex = wiz3d::TryUnwrapTexture2D(p))
    {
        ID3D11Resource* right = tex->GetRealRight();
        return (pickRight && right) ? right : static_cast<ID3D11Resource*>(tex->GetReal());
    }
    if (auto* tex1 = wiz3d::TryUnwrapTexture1D(p))
        return static_cast<ID3D11Resource*>(tex1->GetReal());
    if (auto* tex3 = wiz3d::TryUnwrapTexture3D(p))
        return static_cast<ID3D11Resource*>(tex3->GetReal());
    if (auto* buf = wiz3d::TryUnwrapBuffer(p))
        return static_cast<ID3D11Resource*>(buf->GetReal());
    return p;
}

// Stage 3c.2: eye-aware unwrap helpers for SRV/UAV. Right-eye sibling is
// optional — falls back to left when null or when picking left eye.
static inline ID3D11ShaderResourceView* UnwrapSRVForEye(ID3D11ShaderResourceView* p, bool pickRight)
{
    if (!p) return nullptr;
    if (auto* sp = wiz3d::TryUnwrapSRV(p))
    {
        ID3D11ShaderResourceView* right = sp->GetRealRight();
        // A right copy no mirror has written holds nothing of the game's: the right eye samples the left.
        if (right && sp->Texture() && sp->Texture()->RightUnwritten()) right = nullptr;
        return (pickRight && right) ? right : sp->GetReal();
    }
    return p;
}

static inline ID3D11UnorderedAccessView* UnwrapUAVForEye(ID3D11UnorderedAccessView* p, bool pickRight)
{
    if (!p) return nullptr;
    if (auto* up = wiz3d::TryUnwrapUAV(p))
    {
        ID3D11UnorderedAccessView* right = up->GetRealRight();
        return (pickRight && right) ? right : up->GetReal();
    }
    return p;
}

#pragma comment(lib, "dxguid.lib")

namespace wiz3d
{

Context11Proxy::Context11Proxy(ID3D11DeviceContext* real, Device11Proxy* parent)
    : m_real(real)
    , m_real1(nullptr)
    , m_real2(nullptr)
    , m_real3(nullptr)
    , m_parent(parent)
    , m_refs(1)
    , m_currentBBBound(false)
    , m_activeEye(Eye::Left)
    , m_presentHookActive(false)
    , m_boundVS(nullptr)
    , m_boundPS(nullptr)
{
    for (UINT i = 0; i < kMaxVSCBSlots; ++i) m_boundVSCBs[i] = nullptr;
    for (UINT i = 0; i < kMaxPSCBSlots; ++i) m_boundPSCBs[i] = nullptr;
    ResetEyeTracking();
    if (m_real)
    {
        if (FAILED(m_real->QueryInterface(__uuidof(ID3D11DeviceContext1), reinterpret_cast<void**>(&m_real1)))) m_real1 = nullptr;
        if (FAILED(m_real->QueryInterface(__uuidof(ID3D11DeviceContext2), reinterpret_cast<void**>(&m_real2)))) m_real2 = nullptr;
        if (FAILED(m_real->QueryInterface(__uuidof(ID3D11DeviceContext3), reinterpret_cast<void**>(&m_real3)))) m_real3 = nullptr;
    }
}

Context11Proxy::~Context11Proxy()
{
    // Closures hold no AddRef'd state in 4b.1 (Draw/DrawIndexed capture only
    // POD args), so a plain clear is safe. Later stages that record state-
    // setting calls with captured COM pointers will release them in
    // ClearFrameCommands; the dtor will route there.
    ClearFrameCommands();
    ReleaseUavTwins(~0u);
    ResetEyeTracking();
    if (m_real3) { m_real3->Release(); m_real3 = nullptr; }
    if (m_real2) { m_real2->Release(); m_real2 = nullptr; }
    if (m_real1) { m_real1->Release(); m_real1 = nullptr; }
}

void Context11Proxy::ClearFrameCommands()
{
    m_frameCommands.clear();
}

// The scene camera's p00 a profile feature publishes (stereo::PublishCameraP00), 0 = unknown.
static float s_camP00 = 0.f;
static bool  s_uiProjectionParallax = false;   // the UiProjectionParallax feature is on (read by the static shift code)
static unsigned s_scaledByCamera = 0;   // scaled camera matrices shifted with s_camP00, this frame
static unsigned s_lightOnlyKept = 0;    // light matrices (read only by shadow passes) left unshifted, this frame
// Report only: the clean camera matrices of a frame by p00 (thousandths), so the frame report names the scene camera
// and shows how often the last one cached is another (weapon, UI).
static std::map<int, std::pair<unsigned, float>> s_cleanP00Votes;
static float    s_sceneP00 = 0.f;
static unsigned s_sceneP00Votes = 0, s_cleanP00Total = 0;

// The game's per-game features (ProfileFeature.h), built from its profile the first time a draw point needs them.
ProfileFeatures& Context11Proxy::Features()
{
    if (!m_featuresBuilt)
    {
        m_featuresBuilt = true;
        m_features = CreateProfileFeatures();
        s_uiProjectionParallax = UiProjectionParallax();
        FeatureLog("Context11Proxy %p: %u profile feature(s)\n", this, (unsigned)m_features.size());
    }
    return m_features;
}

void Context11Proxy::NoteRightConstants(Buffer11Proxy* b)
{
    if (b) b->NoteRightWrite();
}

bool Context11Proxy::WriteRightConstants(Buffer11Proxy* b, const unsigned char* data, size_t size, int variant)
{
    ID3D11Buffer* r = b ? b->RightCopy(variant) : nullptr;
    if (!r || !data || !size) return false;
    D3D11_BUFFER_DESC bd = {};
    r->GetDesc(&bd);
    if (bd.ByteWidth != size) return false;
    D3D11_MAPPED_SUBRESOURCE m = {};
    if (bd.Usage == D3D11_USAGE_DYNAMIC)
    {
        if (FAILED(m_real->Map(r, 0, D3D11_MAP_WRITE_DISCARD, 0, &m)) || !m.pData) return false;
        memcpy(m.pData, data, size);
        m_real->Unmap(r, 0);
    }
    else if (bd.Usage == D3D11_USAGE_DEFAULT) m_real->UpdateSubresource(r, 0, nullptr, data, 0, 0);
    else return false;
    if (variant) b->NoteMovedWrite(); else { b->NoteRightWrite(); b->SetRightFromShadow(false); }
    return true;
}

Buffer11Proxy* Context11Proxy::BoundCB(bool pixel, UINT slot) const
{
    return slot < kMaxCBs ? m_cbSlots[pixel ? ST_PS : ST_VS].proxy[slot] : nullptr;
}

ID3D11ShaderResourceView* Context11Proxy::LeftSRV(bool pixel, UINT slot) const
{
    return slot < kMaxSRVs ? m_srvSlots[pixel ? ST_PS : ST_VS].left[slot] : nullptr;
}

ID3D11ShaderResourceView* Context11Proxy::RightSRV(bool pixel, UINT slot) const
{
    return slot < kMaxSRVs ? m_srvSlots[pixel ? ST_PS : ST_VS].right[slot] : nullptr;
}

void Context11Proxy::BindLeftGeometryConstants()
{
    const StageIdx geo[] = { ST_VS, ST_HS, ST_DS, ST_GS };
    for (StageIdx st : geo) if (m_cbSlots[st].high) BindStageCBs(st, false);
}

float Context11Proxy::EyeShiftSign() const
{
    float s = 1.f;
    for (const auto& f : m_features) s *= f->EyeShiftSign();
    return s;
}

float Context11Proxy::CameraP00() const
{
    for (const auto& f : m_features) if (f->CameraP00() > 0.f) return f->CameraP00();
    return 0.f;
}

bool Context11Proxy::LightViewPass() const
{
    for (const auto& f : m_features) if (f->LightViewPass()) return true;
    return false;
}

bool Context11Proxy::VertexCameraMatrices() const
{
    for (const auto& f : m_features) if (f->VertexCameraMatrices()) return true;
    return false;
}

bool Context11Proxy::UiProjectionParallax() const
{
    for (const auto& f : m_features) if (f->UiProjectionParallax()) return true;
    return false;
}

bool Context11Proxy::BoundCameraMove(CameraMove& out)
{
    for (const auto& f : m_features) if (f->BoundCameraMove(*this, out)) return true;
    return false;
}

Buffer11Proxy* Context11Proxy::BoundNamedCB(const std::string& cbuffer) const
{
    const bool cs = m_editAtDispatch;
    const ShaderAnalysis11Result* infos[2] = { m_boundInfo[cs ? 2 : 0], cs ? nullptr : m_boundInfo[1] };
    const StageIdx stages[2] = { cs ? ST_CS : ST_VS, ST_PS };
    for (int s = 0; s < 2; ++s)
        if (infos[s])
            for (const auto& named : infos[s]->rightCBs)
                if (named.first == cbuffer && (UINT)named.second < kMaxCBs && m_cbSlots[stages[s]].proxy[named.second])
                    return m_cbSlots[stages[s]].proxy[named.second];
    return nullptr;
}

// The pass's output against the presented aspect, so a light's or probe's view is left alone: a draw's first target
// (or depth), a dispatch's right-eye texture UAVs (a dispatch writing only buffers counts as the scene's).
bool Context11Proxy::OutputAtPresentAspect(bool compute) const
{
    if (!m_parent || !m_parent->GetPresentWidth() || !m_parent->GetPresentHeight()) return false;
    const float present = (float)m_parent->GetPresentWidth() / m_parent->GetPresentHeight();
    auto atAspect = [&](ID3D11View* v, bool& isTexture) -> bool
    {
        isTexture = false;
        ID3D11Resource* r = nullptr;
        if (v) v->GetResource(&r);
        if (!r) return false;
        D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
        r->GetType(&dim);
        D3D11_TEXTURE2D_DESC td = {};
        if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D) static_cast<ID3D11Texture2D*>(r)->GetDesc(&td);
        r->Release();
        if (!td.Width || !td.Height) return false;
        isTexture = true;
        return fabsf((float)td.Width / td.Height - present) < 0.03f * present;
    };
    bool isTexture = false;
    if (!compute)
        return atAspect(m_numRTVs && m_rtvLeft[0] ? static_cast<ID3D11View*>(m_rtvLeft[0])
                                                  : static_cast<ID3D11View*>(m_dsvLeft), isTexture);
    bool sawTexture = false;
    for (UINT i = 0; i < m_csUAVSlots.high; ++i)
    {
        if (!m_csUAVSlots.right[i] || m_csUAVSlots.right[i] == m_csUAVSlots.left[i]) continue;
        if (atAspect(m_csUAVSlots.left[i], isTexture)) return true;
        sawTexture |= isTexture;
    }
    return !sawTexture;
}

// Each feature cbuffer of the bound shaders: the game's bytes plus every feature's edit, written once to the right copy
// until something changes them; a moved-origin pass reads a second copy, bound for this draw only.
void Context11Proxy::EditRightCBs(bool compute)
{
    if (Features().empty()) return;
    const ShaderAnalysis11Result* infos[2] = { m_boundInfo[compute ? 2 : 0], compute ? nullptr : m_boundInfo[1] };
    if (!(infos[0] && !infos[0]->rightCBs.empty()) && !(infos[1] && !infos[1]->rightCBs.empty())) return;
    const StageIdx stages[2] = { compute ? ST_CS : ST_VS, ST_PS };
    const float eyeShift = wiz3D_GetEffectiveEyeShift() * EyeShiftSign(), conv = stereo::Convergence();
    int aspect = -1;   // the output's aspect, read once and only when an edit is due
    m_editAtDispatch = compute;
    m_editMovedOrigin = false;
    for (const auto& f : m_features) m_editMovedOrigin |= f->MovedOriginPass(*this);
    const int v = m_editMovedOrigin ? 1 : 0;
    for (int s = 0; s < 2; ++s)
    {
        const ShaderAnalysis11Result* info = infos[s];
        if (!info) continue;
        for (const auto& named : info->rightCBs)
        {
            const UINT slot = (UINT)named.second;
            Buffer11Proxy* b = slot < kMaxCBs ? m_cbSlots[stages[s]].proxy[slot] : nullptr;
            if (!b || !b->GetRealRight() || b->Shadow().empty()) continue;
            ID3D11Buffer* copy = b->RightCopy(v);
            if (!copy) continue;
            if (b->NeedsRightEdit(v, eyeShift, conv))
            {
                if (aspect < 0) aspect = OutputAtPresentAspect(compute) ? 1 : 0;
                if (!aspect) return;
                m_editScratch.assign(b->Shadow().begin(), b->Shadow().end());
                bool edited = false;
                m_editRetry = false;
                m_editShader = info;
                for (auto& f : m_features)
                    edited |= f->EditRightConstants(*this, named.first, m_editScratch.data(), m_editScratch.size());
                m_editShader = nullptr;
                // The moved copy holds only what is written here, so it is written edited or not.
                if (edited || v) WriteRightConstants(b, m_editScratch.data(), m_editScratch.size(), v);
                if (!m_editRetry) b->MarkRightEdited(v, eyeShift, conv);
            }
            if (v) BindRightCB(stages[s], slot, copy);
        }
    }
}

ID3D11ShaderResourceView* Context11Proxy::ComputeSRVForRight(UINT slot) const
{
    if (slot >= kMaxSRVs) return nullptr;
    const SRVEyeSlots& v = m_srvSlots[ST_CS];
    return v.right[slot] ? v.right[slot] : v.left[slot];
}

void Context11Proxy::SubstituteComputeSRV(UINT slot, ID3D11ShaderResourceView* v)
{
    if (slot >= kMaxSRVs) return;
    m_real->CSSetShaderResources(slot, 1, &v);
    m_substitutedCSSRVs.push_back(slot);
}

// After the right eye's binds: the texture slots a feature named (ShaderAnalysis11Result::leftSRVSlots) read the game's
// (left) copy, for this draw or dispatch only (BindEye(false) puts the left set back anyway).
void Context11Proxy::BindLeftSRVs(bool compute)
{
    const ShaderAnalysis11Result* infos[2] = { m_boundInfo[compute ? 2 : 0], compute ? nullptr : m_boundInfo[1] };
    const StageIdx stages[2] = { compute ? ST_CS : ST_VS, ST_PS };
    for (int s = 0; s < 2; ++s)
    {
        const ShaderAnalysis11Result* info = infos[s];
        if (!info || info->leftSRVSlots.empty()) continue;
        const SRVEyeSlots& v = m_srvSlots[stages[s]];
        for (int slot : info->leftSRVSlots)
        {
            if (slot < 0 || (UINT)slot >= kMaxSRVs || !v.left[slot] || v.left[slot] == v.right[slot]) continue;
            ID3D11ShaderResourceView* l = v.left[slot];
            switch (stages[s])
            {
            case ST_VS: m_real->VSSetShaderResources(slot, 1, &l); break;
            case ST_PS: m_real->PSSetShaderResources(slot, 1, &l); break;
            case ST_CS: m_real->CSSetShaderResources(slot, 1, &l); break;
            default: break;
            }
            ++m_leftSRVsThisFrame;
        }
    }
}

void Context11Proxy::BindRightCB(StageIdx st, UINT slot, ID3D11Buffer* b)
{
    const CBEyeSlots& s = m_cbSlots[st];
    if (s.anyRange && m_real1 && s.num[slot])
    {
        const UINT first = s.first[slot], num = s.num[slot];
        switch (st)
        {
        case ST_VS: m_real1->VSSetConstantBuffers1(slot, 1, &b, &first, &num); break;
        case ST_PS: m_real1->PSSetConstantBuffers1(slot, 1, &b, &first, &num); break;
        case ST_CS: m_real1->CSSetConstantBuffers1(slot, 1, &b, &first, &num); break;
        default: break;
        }
        return;
    }
    switch (st)
    {
    case ST_VS: m_real->VSSetConstantBuffers(slot, 1, &b); break;
    case ST_PS: m_real->PSSetConstantBuffers(slot, 1, &b); break;
    case ST_CS: m_real->CSSetConstantBuffers(slot, 1, &b); break;
    default: break;
    }
}

bool Context11Proxy::BoundShadersTagged(const std::string& name) const
{
    const bool cs = m_editAtDispatch;
    const ShaderAnalysis11Result* infos[2] = { m_boundInfo[cs ? 2 : 0], cs ? nullptr : m_boundInfo[1] };
    for (const ShaderAnalysis11Result* info : infos)
        if (info)
            for (const std::string& t : info->tags) if (t == name) return true;
    return false;
}

// Present counter, so a buffer's last eye shift can be dated against the frame being drawn.
static unsigned s_frameId = 0;

// The matrix detector's state: the camera's x scale and the ring of forward view-projections (Shared/StereoMatrix.h).
static wiz3d::StereoMatrixState s_mat;

static DWORD BoundShaderCRC(Device11Proxy* parent, void* shader);
static float ProfileShaderMultiplier(Device11Proxy* parent, void* vs);

void Context11Proxy::BeginDraw(UINT vertexCount)
{
    ++m_drawsThisFrame;
    if (m_activeEye != Eye::Right)
    {
        TeachBoundMatrices(vertexCount <= 6);
        if (gInfo.CollectDebugInformation) TallyDrawCoverage();
    }
    for (auto& f : Features()) f->OnLeftDraw(*this);
}

// The bound VS's matrix registers, taught to the buffers at its slots: writes are attributed by the shader bound then,
// and Far Cry 4 fills its buffers long before binding them. Not from a fullscreen pass, whose matrices are not the camera's.
void Context11Proxy::TeachBoundMatrices(bool screenSpace)
{
    const ShaderAnalysis11Result* info = m_boundInfo[0];
    if (screenSpace || !info || !info->parsed) return;
    for (const auto& cb : info->projection.matrixData.cb)
    {
        if (cb.first >= kMaxVSCBSlots || !m_boundVSCBs[cb.first]) continue;
        Buffer11Proxy* b = TryUnwrapBuffer(m_boundVSCBs[cb.first]);
        if (!b) continue;
        for (const auto& pmd : cb.second)
        {
            if (pmd.incorrectProjection) continue;
            b->NoteMatrixUse(pmd.matrixRegister, LightViewPass());
            Buffer11Proxy::MatrixTarget mt = { pmd.matrixRegister, pmd.matrixIsTransposed, FALSE };
            b->LearnMatrix(mt);
        }
    }
}

// Why a draw's geometry may reach the right eye uncorrected, for the frame report (CollectDebugInformation).
void Context11Proxy::TallyDrawCoverage()
{
    if (!m_boundVS) return;
    const ShaderAnalysis11Result* info = m_boundInfo[0];
    if (!info || !info->parsed)                 { ++m_drawsVSUnparsedThisFrame; return; }
    if (info->projection.matrixData.cb.empty()) { ++m_drawsVSNoMatrixThisFrame; return; }
    // A shift within the last few Presents is current: a game can present more than once per frame.
    unsigned newest = 0;
    bool anyEver = false, anySlot = false;
    for (auto it = info->projection.matrixData.cb.begin();
         it != info->projection.matrixData.cb.end(); ++it)
    {
        const DWORD slot = it->first;
        if (slot >= kMaxVSCBSlots || !m_boundVSCBs[slot]) continue;
        anySlot = true;
        Buffer11Proxy* b = TryUnwrapBuffer(m_boundVSCBs[slot]);
        if (!b || !b->EverShifted()) continue;
        anyEver = true;
        if (b->ShiftedFrame() > newest) newest = b->ShiftedFrame();
    }
    // The shift lands in the right-eye sibling, so it only reaches the right eye
    // if that sibling is what gets bound for the duplicated draw.
    for (auto it = info->projection.matrixData.cb.begin();
         it != info->projection.matrixData.cb.end(); ++it)
    {
        const DWORD slot = it->first;
        if (slot >= kMaxCBs || !m_boundVSCBs[slot]) continue;
        ID3D11Buffer* r = m_cbSlots[ST_VS].right[slot];
        if (!r || r == m_cbSlots[ST_VS].left[slot])
        {
            ++m_drawsVSNoSiblingCBThisFrame;
            ++m_uncorrectedVSDraws[BoundShaderCRC(m_parent, m_boundVS)];
            break;
        }
    }

    if (!anySlot) return;
    if (!anyEver)
    {
        ++m_drawsVSNeverShiftedThisFrame;
        ++m_uncorrectedVSDraws[BoundShaderCRC(m_parent, m_boundVS)];
        return;
    }
    if (s_frameId > newest + 4)
    {
        ++m_drawsVSStaleShiftThisFrame;
        ++m_uncorrectedVSDraws[BoundShaderCRC(m_parent, m_boundVS)];
    }
}

void Context11Proxy::EndFrame()
{
    // The detector ages its cached view-projections by this, so a camera that has moved on no longer confirms a target.
    s_mat.frameId = ++s_frameId;
    static unsigned s_drawnFrames = 0;
    const bool report = gInfo.CollectDebugInformation && m_drawsThisFrame &&
                        (++s_drawnFrames <= 3 || s_drawnFrames % 60 == 0);
    s_sceneP00Votes = s_cleanP00Total = 0;
    for (const auto& v : s_cleanP00Votes)
    {
        s_cleanP00Total += v.second.first;
        if (v.second.first > s_sceneP00Votes) { s_sceneP00Votes = v.second.first; s_sceneP00 = v.second.second; }
    }
    s_cleanP00Votes.clear();
    for (auto& f : Features()) f->OnFrameEnd(*this, s_frameId, report);
    if (report) LogFrameStats(s_frameId);
    ReleaseUavTwins(600);

    m_flippedWritesThisFrame = 0;
    m_leftSRVsThisFrame = 0;
    m_sharedUavDispatchesThisFrame = m_sharedUavSlotsThisFrame = 0;
    m_sharedUavByCS.clear();
    m_twinCopiesThisFrame = m_twinFailedThisFrame = m_twinCounterUnknownThisFrame = m_twinWriteOnlyThisFrame = 0;
    m_twinBytesThisFrame = 0;
    s_scaledByCamera = s_lightOnlyKept = 0;
    s_mat.ringInsertsThisFrame = s_mat.evictedHotThisFrame = 0;
    s_mat.invStaleThisFrame = s_mat.invStaleShapeThisFrame = 0;
    m_drawsThisFrame = m_dispatchesThisFrame = m_cmdListsThisFrame = 0;
    m_cbTargetedThisFrame = m_cbUntargetedThisFrame = m_cbMatShiftedThisFrame = m_cbMatRejectedThisFrame = 0;
    m_invScansThisFrame = m_invHitsThisFrame = m_invPatchedThisFrame = m_invLearnedThisFrame = 0;
    m_updCBWritesThisFrame = m_updCBPatchedThisFrame = 0;
    m_invBestMissThisFrame = 1e30f;
    m_invDrawChecksThisFrame = m_invDrawPatchedThisFrame = m_fwdDrawPatchedThisFrame = 0;
    m_invDrawByPS.clear();
    m_overlayDrawsThisFrame = m_overlayRestoredThisFrame = m_plainRestoredThisFrame = m_learnedWithheldThisFrame = 0;
    m_dsBuffersPatchedThisFrame = 0;
    for (unsigned& w : m_untargetedWhy) w = 0;
    m_bindResetsThisFrame = 0;
    m_getCBsThisFrame = m_getSRVsThisFrame = m_getUAVsThisFrame = 0;
    m_drawsVSUnparsedThisFrame = m_drawsVSNoMatrixThisFrame = m_drawsVSNeverShiftedThisFrame = 0;
    m_drawsVSStaleShiftThisFrame = m_drawsVSNoSiblingCBThisFrame = 0;
    m_uncorrectedVSDraws.clear();
    m_drawsDuplicatedThisFrame = m_drawsMonoThisFrame = m_drawsUavSkippedThisFrame = m_drawsUavDupThisFrame = 0;
    m_drawsMonoDsvThisFrame = m_drawsNoRightRTVThisFrame = m_drawsDepthOnlyThisFrame = m_drawsNoRightTargetThisFrame = 0;
    m_dispatchesDuplicatedThisFrame = 0;
}

void Context11Proxy::LogFrameStats(unsigned frame)
{
    DDILog("  [frame %u] ctx=%p: draws=%u dispatches=%u cmdLists=%u recorded=%zu | CB writes: targeted=%u untargeted=%u"
           " | matrices: shifted=%u rejected=%u | duplicated=%u mono=%u uavSkip=%u uavDup=%u monoDSV=%u noRightRTV=%u"
           " depthOnly=%u noRightTarget=%u dispatchesDuplicated=%u\n",
           frame, this, m_drawsThisFrame, m_dispatchesThisFrame, m_cmdListsThisFrame, m_frameCommands.size(),
           m_cbTargetedThisFrame, m_cbUntargetedThisFrame, m_cbMatShiftedThisFrame, m_cbMatRejectedThisFrame,
           m_drawsDuplicatedThisFrame, m_drawsMonoThisFrame, m_drawsUavSkippedThisFrame, m_drawsUavDupThisFrame,
           m_drawsMonoDsvThisFrame, m_drawsNoRightRTVThisFrame, m_drawsDepthOnlyThisFrame,
           m_drawsNoRightTargetThisFrame, m_dispatchesDuplicatedThisFrame);

    // The vertex shaders of draws that reached the right eye uncorrected, most frequent first.
    std::vector<std::pair<unsigned, DWORD>> byUse;
    for (const auto& u : m_uncorrectedVSDraws) byUse.push_back(std::make_pair(u.second, u.first));
    std::sort(byUse.rbegin(), byUse.rend());
    std::string shaders;
    char one[32];
    for (size_t k = 0; k < byUse.size() && k < 12; ++k)
    {
        sprintf_s(one, " %08lX=%u", byUse[k].second, byUse[k].first);
        shaders += one;
    }
    DDILog("  [frame %u] uncorrected draws: vsUnparsed=%u vsNoMatrix=%u cbNeverShifted=%u cbStale=%u noSiblingCB=%u"
           " | by VS:%s\n", frame, m_drawsVSUnparsedThisFrame, m_drawsVSNoMatrixThisFrame,
           m_drawsVSNeverShiftedThisFrame, m_drawsVSStaleShiftThisFrame, m_drawsVSNoSiblingCBThisFrame,
           shaders.empty() ? " none" : shaders.c_str());

    std::string byCS;
    for (const auto& c : m_sharedUavByCS)
    {
        sprintf_s(one, " %08lX x%u", c.first, c.second);
        byCS += one;
    }
    DDILog("  [frame %u] right dispatches with shared UAVs twinned: %u (slots %u, copies %u, %.1f MB,"
           " write-only dropped %u, failed %u, counters unknown %u, twins held %u), by CS:%s\n", frame,
           m_sharedUavDispatchesThisFrame, m_sharedUavSlotsThisFrame, m_twinCopiesThisFrame,
           m_twinBytesThisFrame / 1048576.0, m_twinWriteOnlyThisFrame, m_twinFailedThisFrame,
           m_twinCounterUnknownThisFrame, (unsigned)m_uavTwins.size(), byCS.empty() ? " none" : byCS.c_str());
    DDILog("  [frame %u] left-eye textures read by the right: %u binds\n", frame, m_leftSRVsThisFrame);
    if (s_camP00 > 0.f || s_lightOnlyKept)
        DDILog("  [frame %u] camera: scaled camera matrices shifted with the camera's p00 %.4f: %u (cache %.4f)"
               " | light matrices left unshifted %u\n", frame, s_camP00, s_scaledByCamera, s_mat.projXScale,
               s_lightOnlyKept);
    // ring = cached view-projections to test against; bestMiss = the closest miss over the pass bound (<= 1 passes).
    if (gInfo.DetectInverseMatrix)
        DDILog("  [frame %u] invDetect: ring=%u fresh=%u inserts=%u evictedHot=%u scans=%u hits=%u patched=%u"
               " stale=%u (unshaped %u) learned=%u updCB=%u patched=%u bestMiss=%.2f\n", frame,
               s_mat.RingValid(), s_mat.RingFresh(), s_mat.ringInsertsThisFrame, s_mat.evictedHotThisFrame,
               m_invScansThisFrame, m_invHitsThisFrame, m_invPatchedThisFrame, s_mat.invStaleThisFrame,
               s_mat.invStaleShapeThisFrame, m_invLearnedThisFrame, m_updCBWritesThisFrame, m_updCBPatchedThisFrame,
               m_invBestMissThisFrame < 1e29f ? m_invBestMissThisFrame : -1.f);
    if (gInfo.DetectInverseMatrix)
    {
        std::string byPS;
        for (const auto& p : m_invDrawByPS)
        {
            sprintf_s(one, " %08lX x%u", p.first, p.second);
            byPS += one;
        }
        unsigned knownInverse = 0;
        for (const auto& k : m_psCameraMatrices) knownInverse += k.second.inverse ? 1 : 0;
        DDILog("  [frame %u] shaderCamera: buffers checked at the right draw/dispatch %u, inverses patched %u, previous"
               " cameras %u | known registers %u inverse + %u forward | by PS/CS:%s\n", frame, m_invDrawChecksThisFrame,
               m_invDrawPatchedThisFrame, m_fwdDrawPatchedThisFrame, knownInverse,
               (unsigned)m_psCameraMatrices.size() - knownInverse, byPS.empty() ? " none" : byPS.c_str());
    }
    DDILog("  [frame %u] camera scale: scene p00 %.4f (%u of %u clean camera matrices this frame), last cached %.4f\n",
           frame, s_sceneP00, s_sceneP00Votes, s_cleanP00Total, s_mat.projXScale);
    DDILog("  [frame %u] overlay: back-buffer draws without depth test %u, right copies restored %u | plain constants"
           " restored for matrix-less shaders %u | learned forward shifts withheld at write %u | domain-stage buffers"
           " patched %u\n", frame, m_overlayDrawsThisFrame, m_overlayRestoredThisFrame, m_plainRestoredThisFrame,
           m_learnedWithheldThisFrame, m_dsBuffersPatchedThisFrame);
    DDILog("  [frame %u] untargeted CB writes: noVS=%u vsUnparsed=%u vsNoMatrices=%u notAtVSSlot=%u atNonMatrixSlot=%u"
           " | binding resets %u\n", frame, m_untargetedWhy[0], m_untargetedWhy[1], m_untargetedWhy[2],
           m_untargetedWhy[3], m_untargetedWhy[4], m_bindResetsThisFrame);
    DDILog("  [frame %u] state getters answered with the game's bindings: constant buffers %u, shader resources %u,"
           " uavs %u\n", frame, m_getCBsThisFrame, m_getSRVsThisFrame, m_getUAVsThisFrame);
}

void Context11Proxy::ReplayFrameCommands(Eye eye)
{
    // Snapshot + flip the active eye for the replay pass. Each recorded
    // closure re-enters our proxy methods, so OMSet/etc. pick the
    // eye-appropriate real handle via m_activeEye automatically.
    Eye saved = m_activeEye;
    m_activeEye = eye;
    if (FrameTraceActive())
        FrameTrace("--- replay begin (eye=%c, %zu commands) ---\n",
                   eye == Eye::Right ? 'R' : 'L', m_frameCommands.size());
    for (auto& fn : m_frameCommands)
        fn();
    if (FrameTraceActive())
        FrameTrace("--- replay end (eye=%c) ---\n",
                   eye == Eye::Right ? 'R' : 'L');
    m_activeEye = saved;
}

void Context11Proxy::SetPresentHookActive(bool active)
{
    // Draw duplication issues both eyes live, so recording would be overhead
    // and replaying it would double every draw a second time.
    m_presentHookActive = active && !gInfo.DuplicateDraws;
}

// A tracked binding holds a reference, as the runtime does for a bound object (new one taken before the old dropped).
template<class T> static void HoldRef(T*& slot, T* p)
{
    if (p) p->AddRef();
    if (slot) slot->Release();
    slot = p;
}
template<class T> static void DropRef(T*& slot)
{
    if (slot) { slot->Release(); slot = nullptr; }
}

// Everything reads as unbound: after ClearState, a state-clearing ExecuteCommandList or a state swap.
void Context11Proxy::ResetEyeTracking()
{
    ++m_bindResetsThisFrame;
    for (int st = 0; st < ST_COUNT; ++st)
    {
        SRVEyeSlots& s = m_srvSlots[st];
        for (UINT i = 0; i < s.high; ++i) { DropRef(s.left[i]); DropRef(s.right[i]); DropRef(s.game[i]); }
        CBEyeSlots& c = m_cbSlots[st];
        for (UINT i = 0; i < c.high; ++i) { DropRef(c.left[i]); DropRef(c.right[i]); DropRef(c.proxy[i]); DropRef(c.game[i]); }
    }
    for (UINT i = 0; i < kMaxRTVs; ++i) { DropRef(m_rtvLeft[i]); DropRef(m_rtvRight[i]); DropRef(m_rtvGame[i]); }
    DropRef(m_dsvLeft);
    DropRef(m_dsvRight);
    DropRef(m_dsvGame);
    for (UINT i = 0; i < m_csUAVSlots.high; ++i)
    { DropRef(m_csUAVSlots.left[i]); DropRef(m_csUAVSlots.right[i]); DropRef(m_csUAVSlots.game[i]); }
    for (UINT i = 0; i < kMaxVSCBSlots; ++i) DropRef(m_boundVSCBs[i]);
    for (UINT i = 0; i < kMaxPSCBSlots; ++i) DropRef(m_boundPSCBs[i]);
    ClearOmUavRefs();
    memset(m_srvSlots, 0, sizeof(m_srvSlots));
    memset(m_cbSlots,  0, sizeof(m_cbSlots));
    m_csUAVSlots  = {};
    m_numRTVs     = 0;
    m_numRTVGame  = 0;
    m_omAnyStereo = false;
    m_omHasUAVs   = false;
    m_omMonoDSV   = false;
    m_omUavBound  = false;
    m_omBackBuffer = false;
    m_depthTestOn  = true;
}

void Context11Proxy::TrackSRVs(StageIdx stage, UINT StartSlot, UINT NumViews,
                               ID3D11ShaderResourceView* const* ppSRVs)
{
    if (!gInfo.DuplicateDraws || stage >= ST_COUNT) return;
    SRVEyeSlots& s = m_srvSlots[stage];
    for (UINT i = 0; i < NumViews; ++i)
    {
        UINT slot = StartSlot + i;
        if (slot >= kMaxSRVs) break;
        ID3D11ShaderResourceView* p = ppSRVs ? ppSRVs[i] : nullptr;
        HoldRef(s.left[slot],  UnwrapSRVForEye(p, false));
        HoldRef(s.right[slot], UnwrapSRVForEye(p, true));
        HoldRef(s.game[slot],  p);
        if (slot + 1 > s.high) s.high = slot + 1;
    }
    s.anyStereo = false;
    for (UINT i = 0; i < s.high; ++i)
        if (s.left[i] != s.right[i]) { s.anyStereo = true; break; }
}

void Context11Proxy::TrackCBs(StageIdx stage, UINT StartSlot, UINT NumBuffers,
                              ID3D11Buffer* const* ppCBs, const UINT* pFirst, const UINT* pNum)
{
    if (!gInfo.DuplicateDraws || stage >= ST_COUNT) return;
    CBEyeSlots& s = m_cbSlots[stage];
    for (UINT i = 0; i < NumBuffers; ++i)
    {
        UINT slot = StartSlot + i;
        if (slot >= kMaxCBs) break;
        ID3D11Buffer* p = ppCBs ? ppCBs[i] : nullptr;
        ID3D11Buffer* left  = p;
        ID3D11Buffer* right = p;
        Buffer11Proxy* proxy = nullptr;
        if (p)
        {
            if (auto* bp = TryUnwrapBuffer(static_cast<ID3D11Resource*>(p)))
            {
                proxy = bp;
                left  = bp->GetReal();
                right = bp->GetRealRight() ? bp->GetRealRight() : left;
            }
        }
        HoldRef(s.left[slot],  left);
        HoldRef(s.right[slot], right);
        HoldRef(s.proxy[slot], proxy);
        HoldRef(s.game[slot],  p);
        s.first[slot] = pFirst ? pFirst[i] : 0;
        s.num[slot]   = pNum ? pNum[i] : 0;
        s.size[slot]  = 0;
        if (left)
        {
            D3D11_BUFFER_DESC bd = {};
            left->GetDesc(&bd);
            s.size[slot] = bd.ByteWidth / 16;
        }
        if (slot + 1 > s.high) s.high = slot + 1;
    }
    s.anyStereo = false;
    s.anyRange = false;
    for (UINT i = 0; i < s.high; ++i)
    {
        if (s.left[i] != s.right[i]) s.anyStereo = true;
        if (s.num[i]) s.anyRange = true;
    }
}

void Context11Proxy::TrackOM(UINT NumViews, ID3D11RenderTargetView* const* ppRTVs,
                             ID3D11DepthStencilView* pDSV)
{
    if (!gInfo.DuplicateDraws) return;
    m_omHasUAVs = false;
    // A plain OMSetRenderTargets unbinds any output-merger UAVs.
    m_omUavBound = false;
    ClearOmUavRefs();
    m_numRTVs = NumViews <= kMaxRTVs ? NumViews : kMaxRTVs;
    bool anyRTVStereo = false;
    for (UINT i = 0; i < kMaxRTVs; ++i)
    {
        ID3D11RenderTargetView* p = (ppRTVs && i < m_numRTVs) ? ppRTVs[i] : nullptr;
        ID3D11RenderTargetView* left = p;
        // Mono targets must be written exactly once, so the right-eye pass
        // leaves them unbound rather than blending into them a second time.
        ID3D11RenderTargetView* right = nullptr;
        if (auto* rp = TryUnwrapRTV(p))
        {
            left  = rp->GetReal();
            right = rp->GetRealRight();
            if (right) anyRTVStereo = true;
        }
        HoldRef(m_rtvLeft[i], left);
        HoldRef(m_rtvRight[i], right);
    }
    // The first target is the swap chain's back buffer: an overlay pass once the depth test is off.
    m_omBackBuffer = false;
    if (RTV11Proxy* r0 = (ppRTVs && m_numRTVs) ? TryUnwrapRTV(ppRTVs[0]) : nullptr)
    {
        ID3D11Resource* res = nullptr;
        r0->GetReal()->GetResource(&res);
        if (res)
        {
            Texture2D11Proxy* tp = m_parent ? m_parent->LookupProxyByReal(res) : nullptr;
            m_omBackBuffer = tp && tp->IsBackBuffer();
            res->Release();
        }
    }
    // A mono depth-stencil stays bound for the right pass: losing depth testing is worse than testing against the
    // depth the left pass wrote (which then also takes the right pass's writes).
    ID3D11DepthStencilView* dsvLeft = pDSV;
    ID3D11DepthStencilView* dsvRight = pDSV;
    bool dsvStereo = false;
    if (auto* dp = TryUnwrapDSV(pDSV))
    {
        dsvLeft   = dp->GetReal();
        dsvRight  = dp->GetRealRight() ? dp->GetRealRight() : dp->GetReal();
        dsvStereo = (dp->GetRealRight() != nullptr);
    }
    HoldRef(m_dsvLeft, dsvLeft);
    HoldRef(m_dsvRight, dsvRight);
    m_omMonoDSV   = (pDSV != nullptr) && !dsvStereo;
    m_omAnyStereo = anyRTVStereo || dsvStereo;

    for (auto& f : Features()) f->OnTargetsSet(*this);
}

void Context11Proxy::RecordGameFacingOM(UINT NumViews,
                                        ID3D11RenderTargetView* const* ppRTVs,
                                        ID3D11DepthStencilView* pDSV)
{
    m_numRTVGame = NumViews <= kMaxRTVs ? NumViews : kMaxRTVs;
    for (UINT i = 0; i < kMaxRTVs; ++i)
        HoldRef(m_rtvGame[i], (ppRTVs && i < m_numRTVGame) ? ppRTVs[i] : nullptr);
    HoldRef(m_dsvGame, pDSV);
}

void STDMETHODCALLTYPE Context11Proxy::OMGetRenderTargets(
    UINT NumViews, ID3D11RenderTargetView** ppRenderTargetViews,
    ID3D11DepthStencilView** ppDepthStencilView)
{
    if (ppRenderTargetViews)
    {
        for (UINT i = 0; i < NumViews; ++i)
        {
            ID3D11RenderTargetView* v = (i < m_numRTVGame) ? m_rtvGame[i] : nullptr;
            if (v) v->AddRef();
            ppRenderTargetViews[i] = v;
        }
    }
    if (ppDepthStencilView)
    {
        if (m_dsvGame) m_dsvGame->AddRef();
        *ppDepthStencilView = m_dsvGame;
    }
}

void STDMETHODCALLTYPE Context11Proxy::OMGetRenderTargetsAndUnorderedAccessViews(
    UINT NumRTVs, ID3D11RenderTargetView** ppRenderTargetViews,
    ID3D11DepthStencilView** ppDepthStencilView,
    UINT UAVStartSlot, UINT NumUAVs, ID3D11UnorderedAccessView** ppUnorderedAccessViews)
{
    if (ppRenderTargetViews || ppDepthStencilView)
        OMGetRenderTargets(NumRTVs, ppRenderTargetViews, ppDepthStencilView);
    // UAVs aren't eye-doubled, so the real ones are the game-facing ones.
    if (ppUnorderedAccessViews)
        m_real->OMGetRenderTargetsAndUnorderedAccessViews(
            0, nullptr, nullptr, UAVStartSlot, NumUAVs, ppUnorderedAccessViews);
}

// The getters answer from the tracked game-facing bindings; with duplication off nothing is tracked and the real
// context answers as before.
void Context11Proxy::GetTrackedCBs(StageIdx stage, UINT StartSlot, UINT NumBuffers, ID3D11Buffer** pp,
                                   UINT* pFirst, UINT* pNum)
{
    if (!gInfo.DuplicateDraws || stage >= ST_COUNT)
    {
        const bool ranged = (pFirst || pNum) && m_real1;
        switch (stage)
        {
        case ST_VS: if (ranged) m_real1->VSGetConstantBuffers1(StartSlot, NumBuffers, pp, pFirst, pNum); else m_real->VSGetConstantBuffers(StartSlot, NumBuffers, pp); break;
        case ST_PS: if (ranged) m_real1->PSGetConstantBuffers1(StartSlot, NumBuffers, pp, pFirst, pNum); else m_real->PSGetConstantBuffers(StartSlot, NumBuffers, pp); break;
        case ST_GS: if (ranged) m_real1->GSGetConstantBuffers1(StartSlot, NumBuffers, pp, pFirst, pNum); else m_real->GSGetConstantBuffers(StartSlot, NumBuffers, pp); break;
        case ST_HS: if (ranged) m_real1->HSGetConstantBuffers1(StartSlot, NumBuffers, pp, pFirst, pNum); else m_real->HSGetConstantBuffers(StartSlot, NumBuffers, pp); break;
        case ST_DS: if (ranged) m_real1->DSGetConstantBuffers1(StartSlot, NumBuffers, pp, pFirst, pNum); else m_real->DSGetConstantBuffers(StartSlot, NumBuffers, pp); break;
        case ST_CS: if (ranged) m_real1->CSGetConstantBuffers1(StartSlot, NumBuffers, pp, pFirst, pNum); else m_real->CSGetConstantBuffers(StartSlot, NumBuffers, pp); break;
        default: break;
        }
        return;
    }
    ++m_getCBsThisFrame;
    const CBEyeSlots& s = m_cbSlots[stage];
    for (UINT i = 0; i < NumBuffers; ++i)
    {
        const UINT slot = StartSlot + i;
        ID3D11Buffer* b = slot < kMaxCBs ? s.game[slot] : nullptr;
        if (b) b->AddRef();
        if (pp) pp[i] = b;
        // A whole-buffer bind reports its full constant count, as the runtime does.
        if (pFirst) pFirst[i] = b ? s.first[slot] : 0;
        if (pNum)   pNum[i]   = b ? (s.num[slot] ? s.num[slot] : s.size[slot]) : 0;
    }
}

void Context11Proxy::GetTrackedSRVs(StageIdx stage, UINT StartSlot, UINT NumViews, ID3D11ShaderResourceView** pp)
{
    if (!gInfo.DuplicateDraws || stage >= ST_COUNT)
    {
        switch (stage)
        {
        case ST_VS: m_real->VSGetShaderResources(StartSlot, NumViews, pp); break;
        case ST_PS: m_real->PSGetShaderResources(StartSlot, NumViews, pp); break;
        case ST_GS: m_real->GSGetShaderResources(StartSlot, NumViews, pp); break;
        case ST_HS: m_real->HSGetShaderResources(StartSlot, NumViews, pp); break;
        case ST_DS: m_real->DSGetShaderResources(StartSlot, NumViews, pp); break;
        case ST_CS: m_real->CSGetShaderResources(StartSlot, NumViews, pp); break;
        default: break;
        }
        return;
    }
    ++m_getSRVsThisFrame;
    const SRVEyeSlots& s = m_srvSlots[stage];
    for (UINT i = 0; i < NumViews; ++i)
    {
        const UINT slot = StartSlot + i;
        ID3D11ShaderResourceView* v = slot < kMaxSRVs ? s.game[slot] : nullptr;
        if (v) v->AddRef();
        if (pp) pp[i] = v;
    }
}

void Context11Proxy::GetTrackedUAVs(UINT StartSlot, UINT NumUAVs, ID3D11UnorderedAccessView** pp)
{
    if (!gInfo.DuplicateDraws) { m_real->CSGetUnorderedAccessViews(StartSlot, NumUAVs, pp); return; }
    ++m_getUAVsThisFrame;
    for (UINT i = 0; i < NumUAVs; ++i)
    {
        const UINT slot = StartSlot + i;
        ID3D11UnorderedAccessView* v = slot < kMaxUAVs ? m_csUAVSlots.game[slot] : nullptr;
        if (v) v->AddRef();
        if (pp) pp[i] = v;
    }
}

void Context11Proxy::BindStageSRVs(StageIdx stage, bool right)
{
    SRVEyeSlots& s = m_srvSlots[stage];
    ID3D11ShaderResourceView* const* v = right ? s.right : s.left;
    switch (stage)
    {
    case ST_VS: m_real->VSSetShaderResources(0, s.high, v); break;
    case ST_PS: m_real->PSSetShaderResources(0, s.high, v); break;
    case ST_GS: m_real->GSSetShaderResources(0, s.high, v); break;
    case ST_HS: m_real->HSSetShaderResources(0, s.high, v); break;
    case ST_DS: m_real->DSSetShaderResources(0, s.high, v); break;
    case ST_CS: m_real->CSSetShaderResources(0, s.high, v); break;
    default: break;
    }
}

void Context11Proxy::TrackCSUAVs(UINT StartSlot, UINT NumUAVs,
                                 ID3D11UnorderedAccessView* const* pp, const UINT* initialCounts)
{
    UAVEyeSlots& s = m_csUAVSlots;
    for (UINT i = 0; i < NumUAVs; ++i)
    {
        const UINT slot = StartSlot + i;
        if (slot >= kMaxUAVs) break;
        ID3D11UnorderedAccessView* v = pp ? pp[i] : nullptr;
        HoldRef(s.left[slot],  UnwrapUAVForEye(v, false));
        HoldRef(s.right[slot], UnwrapUAVForEye(v, true));
        HoldRef(s.game[slot],  v);
        if (slot + 1 > s.high) s.high = slot + 1;
        m_csInitCount[slot] = initialCounts ? initialCounts[i] : ~0u;
        m_csInitFresh[slot] = initialCounts != nullptr;
    }
    s.anyStereo = false;
    for (UINT i = 0; i < s.high; ++i)
        if (s.left[i] != s.right[i]) { s.anyStereo = true; break; }
}

void Context11Proxy::BindCSUAVs(bool right)
{
    UAVEyeSlots& s = m_csUAVSlots;
    if (!s.high) return;
    if (!right)
    {
        m_real->CSSetUnorderedAccessViews(0, s.high, s.left, nullptr);
        return;
    }
    // A shared UAV (no right sibling) is its twin, copied before the left dispatch: the right reads what the left read
    // and its writes stay out of the left eye's buffers.
    ID3D11UnorderedAccessView* uavs[kMaxUAVs] = {};
    unsigned shared = 0;
    for (UINT i = 0; i < s.high; ++i)
    {
        const bool isShared = s.left[i] && s.right[i] == s.left[i];
        uavs[i] = isShared ? (m_csTwinsPrepared ? m_csTwin[i] : nullptr) : s.right[i];
        if (isShared) ++shared;
    }
    if (shared)
    {
        ++m_sharedUavDispatchesThisFrame;
        m_sharedUavSlotsThisFrame += shared;
        ++m_sharedUavByCS[m_boundInfo[2] ? m_boundInfo[2]->crc32 : 0];
    }
    m_real->CSSetUnorderedAccessViews(0, s.high, uavs, m_csTwinsPrepared ? m_csTwinCount : nullptr);
    m_csTwinsPrepared = false;
}

// The UAV slots a compute shader reads (loads, or atomics that return the old value), from its disassembly once per CRC;
// all ones when unknown. A slot it only writes needs no twin copy: the right's writes are dropped (null view).
static unsigned long long CsUavReadMask(const ShaderAnalysis11Result* info)
{
    if (!info) return ~0ull;
    static std::mutex s_lock;
    static std::map<DWORD, unsigned long long> s_masks;
    {
        std::lock_guard<std::mutex> g(s_lock);
        auto it = s_masks.find(info->crc32);
        if (it != s_masks.end()) return it->second;
    }
    unsigned long long mask = ~0ull;
    ID3DBlob* text = nullptr;
    if (info->bytecode && !info->bytecode->empty() &&
        SUCCEEDED(D3DDisassemble(info->bytecode->data(), info->bytecode->size(), 0, nullptr, &text)) && text)
    {
        mask = 0;
        std::istringstream in(std::string((const char*)text->GetBufferPointer(), text->GetBufferSize()));
        std::string line;
        while (std::getline(in, line))
        {
            const size_t a = line.find_first_not_of(' ');
            if (a == std::string::npos) continue;
            const std::string op = line.substr(a, line.find_first_of(" (", a) - a);
            if (op.compare(0, 12, "ld_uav_typed") && op.compare(0, 6, "ld_raw") && op.compare(0, 13, "ld_structured") &&
                op.compare(0, 10, "imm_atomic"))
                continue;
            for (size_t p = line.find('u', a + op.size()); p != std::string::npos; p = line.find('u', p + 1))
            {
                if (p > 0 && (isalnum((unsigned char)line[p - 1]) || line[p - 1] == '_')) continue;
                if (p + 1 >= line.size() || !isdigit((unsigned char)line[p + 1])) continue;
                const int slot = atoi(line.c_str() + p + 1);
                if (slot >= 0 && slot < 64) mask |= 1ull << slot;
            }
        }
        text->Release();
    }
    std::lock_guard<std::mutex> g(s_lock);
    s_masks[info->crc32] = mask;
    return mask;
}

// Before a left dispatch the right eye repeats: each shared UAV the shader reads is copied to its twin (write-only ones
// get none); the right's counters start at the game's bind-time counts while those are still fresh.
void Context11Proxy::PrepareSharedUavTwins()
{
    UAVEyeSlots& s = m_csUAVSlots;
    const bool repeats = gInfo.DuplicateDraws && gInfo.Input.StereoActive && m_activeEye == Eye::Left && s.anyStereo;
    const unsigned long long reads = repeats ? CsUavReadMask(m_boundInfo[2]) : 0;
    for (UINT i = 0; i < s.high; ++i)
    {
        m_csTwin[i] = nullptr;
        m_csTwinCount[i] = m_csInitFresh[i] ? m_csInitCount[i] : ~0u;
        if (!repeats || !s.left[i] || s.right[i] != s.left[i]) continue;
        if (i < 64 && !(reads & (1ull << i))) { ++m_twinWriteOnlyThisFrame; continue; }
        m_csTwin[i] = TwinFor(s.left[i]);
        if (!m_csTwin[i]) ++m_twinFailedThisFrame;
        D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
        s.left[i]->GetDesc(&ud);
        if (ud.ViewDimension == D3D11_UAV_DIMENSION_BUFFER &&
            (ud.Buffer.Flags & (D3D11_BUFFER_UAV_FLAG_APPEND | D3D11_BUFFER_UAV_FLAG_COUNTER)) && !m_csInitFresh[i])
            ++m_twinCounterUnknownThisFrame;
    }
    for (UINT i = 0; i < s.high; ++i) m_csInitFresh[i] = false;
    m_csTwinsPrepared = repeats;
}

ID3D11UnorderedAccessView* Context11Proxy::TwinFor(ID3D11UnorderedAccessView* v)
{
    ID3D11Device* dev = m_parent ? m_parent->GetReal() : nullptr;
    ID3D11Resource* src = nullptr;
    v->GetResource(&src);
    if (!src || !dev) { if (src) src->Release(); return nullptr; }
    D3D11_UNORDERED_ACCESS_VIEW_DESC ud = {};
    v->GetDesc(&ud);
    UavTwin& t = m_uavTwins[v];
    if (t.uav && (t.src != src || memcmp(&t.ud, &ud, sizeof(ud)) != 0))
    {
        t.uav->Release();
        t.res->Release();
        t = UavTwin();
    }
    if (!t.uav)
    {
        D3D11_RESOURCE_DIMENSION dim = D3D11_RESOURCE_DIMENSION_UNKNOWN;
        src->GetType(&dim);
        const UINT noShare = ~(UINT)(D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX |
                                     D3D11_RESOURCE_MISC_GDI_COMPATIBLE);
        HRESULT hr = E_FAIL;
        if (dim == D3D11_RESOURCE_DIMENSION_BUFFER)
        {
            D3D11_BUFFER_DESC d = {};
            static_cast<ID3D11Buffer*>(src)->GetDesc(&d);
            d.MiscFlags &= noShare;
            hr = dev->CreateBuffer(&d, nullptr, reinterpret_cast<ID3D11Buffer**>(&t.res));
            t.bytes = d.ByteWidth;
        }
        else if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE2D)
        {
            D3D11_TEXTURE2D_DESC d = {};
            static_cast<ID3D11Texture2D*>(src)->GetDesc(&d);
            d.MiscFlags &= noShare;
            hr = dev->CreateTexture2D(&d, nullptr, reinterpret_cast<ID3D11Texture2D**>(&t.res));
            t.bytes = (UINT64)d.Width * d.Height * d.ArraySize * 4;
        }
        else if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE3D)
        {
            D3D11_TEXTURE3D_DESC d = {};
            static_cast<ID3D11Texture3D*>(src)->GetDesc(&d);
            d.MiscFlags &= noShare;
            hr = dev->CreateTexture3D(&d, nullptr, reinterpret_cast<ID3D11Texture3D**>(&t.res));
            t.bytes = (UINT64)d.Width * d.Height * d.Depth * 4;
        }
        else if (dim == D3D11_RESOURCE_DIMENSION_TEXTURE1D)
        {
            D3D11_TEXTURE1D_DESC d = {};
            static_cast<ID3D11Texture1D*>(src)->GetDesc(&d);
            d.MiscFlags &= noShare;
            hr = dev->CreateTexture1D(&d, nullptr, reinterpret_cast<ID3D11Texture1D**>(&t.res));
            t.bytes = (UINT64)d.Width * d.ArraySize * 4;
        }
        if (SUCCEEDED(hr) && t.res && FAILED(dev->CreateUnorderedAccessView(t.res, &ud, &t.uav))) t.uav = nullptr;
        if (!t.uav)
        {
            if (t.res) t.res->Release();
            t = UavTwin();
            src->Release();
            m_uavTwins.erase(v);
            return nullptr;
        }
        t.src = src;
        t.ud = ud;
    }
    m_real->CopyResource(t.res, src);
    src->Release();
    t.lastFrame = s_frameId;
    ++m_twinCopiesThisFrame;
    m_twinBytesThisFrame += t.bytes;
    return t.uav;
}

// Twins whose UAV the game has not dispatched with for a while (a freed view's address may come back as another view).
void Context11Proxy::ReleaseUavTwins(unsigned olderThan)
{
    for (auto it = m_uavTwins.begin(); it != m_uavTwins.end();)
    {
        if (olderThan == ~0u || s_frameId - it->second.lastFrame > olderThan)
        {
            if (it->second.uav) it->second.uav->Release();
            if (it->second.res) it->second.res->Release();
            it = m_uavTwins.erase(it);
        }
        else ++it;
    }
}

// Compute output has no right-eye copy unless the dispatch runs again with the right-eye UAVs bound (GTA V drives
// its lighting and post-processing through compute).
bool Context11Proxy::BeginRightEyeDispatch()
{
    if (!gInfo.DuplicateDraws) return false;
    if (!gInfo.Input.StereoActive) return false;
    if (m_activeEye != Eye::Left) return false;
    if (!m_csUAVSlots.anyStereo) return false;
    m_activeEye = Eye::Right;
    BindCSUAVs(true);
    for (UINT st = 0; st < ST_COUNT; ++st)
    {
        if (m_srvSlots[st].anyStereo) BindStageSRVs((StageIdx)st, true);
        if (m_cbSlots[st].anyStereo)  BindStageCBs((StageIdx)st, true);
    }
    // Compute passes that feed the scene (shadow and lighting inputs) need the right camera too.
    if (gInfo.DetectInverseMatrix) PatchCameraMatrices(ST_CS);
    EditRightCBs(true);
    BindLeftSRVs(true);
    for (auto& f : Features()) f->OnRightDispatch(*this);
    return true;
}

void Context11Proxy::EndRightEyeDispatch()
{
    m_activeEye = Eye::Left;
    BindCSUAVs(false);
    for (UINT st = 0; st < ST_COUNT; ++st)
    {
        if (m_srvSlots[st].anyStereo) BindStageSRVs((StageIdx)st, false);
        if (m_cbSlots[st].anyStereo)  BindStageCBs((StageIdx)st, false);
    }
    // A feature's substitute for one right dispatch: the game's view goes back.
    for (UINT slot : m_substitutedCSSRVs)
    {
        ID3D11ShaderResourceView* l = m_srvSlots[ST_CS].left[slot];
        m_real->CSSetShaderResources(slot, 1, &l);
    }
    m_substitutedCSSRVs.clear();
}

void Context11Proxy::BindStageCBs(StageIdx stage, bool right)
{
    CBEyeSlots& s = m_cbSlots[stage];
    ID3D11Buffer* const* v = right ? s.right : s.left;
    if (s.anyRange && m_real1)
    {
        // A partial bind must be re-issued through the 11.1 entry; slots
        // bound whole spell out their full size (4096 constants max).
        UINT first[kMaxCBs], num[kMaxCBs];
        for (UINT i = 0; i < s.high; ++i)
        {
            first[i] = s.num[i] ? s.first[i] : 0;
            num[i]   = s.num[i] ? s.num[i] : (s.size[i] > 4096 ? 4096 : s.size[i]);
            if (!num[i]) num[i] = 16;
        }
        switch (stage)
        {
        case ST_VS: m_real1->VSSetConstantBuffers1(0, s.high, v, first, num); break;
        case ST_PS: m_real1->PSSetConstantBuffers1(0, s.high, v, first, num); break;
        case ST_GS: m_real1->GSSetConstantBuffers1(0, s.high, v, first, num); break;
        case ST_HS: m_real1->HSSetConstantBuffers1(0, s.high, v, first, num); break;
        case ST_DS: m_real1->DSSetConstantBuffers1(0, s.high, v, first, num); break;
        case ST_CS: m_real1->CSSetConstantBuffers1(0, s.high, v, first, num); break;
        default: break;
        }
        return;
    }
    switch (stage)
    {
    case ST_VS: m_real->VSSetConstantBuffers(0, s.high, v); break;
    case ST_PS: m_real->PSSetConstantBuffers(0, s.high, v); break;
    case ST_GS: m_real->GSSetConstantBuffers(0, s.high, v); break;
    case ST_HS: m_real->HSSetConstantBuffers(0, s.high, v); break;
    case ST_DS: m_real->DSSetConstantBuffers(0, s.high, v); break;
    case ST_CS: m_real->CSSetConstantBuffers(0, s.high, v); break;
    default: break;
    }
}

// D3D11.1 partial constant-buffer binds. Same unwrap and per-eye tracking as
// the 11.0 setters, with the constant range kept so BindEye can re-issue it.
#define STEREO_SET_CBS1(STAGE_PREFIX, STAGE_IDX)                                                         \
void STDMETHODCALLTYPE Context11Proxy::STAGE_PREFIX##SetConstantBuffers1(                                 \
    UINT StartSlot, UINT NumBuffers, ID3D11Buffer* const* ppConstantBuffers,                             \
    const UINT* pFirstConstant, const UINT* pNumConstants)                                               \
{                                                                                                        \
    TrackCBs(STAGE_IDX, StartSlot, NumBuffers, ppConstantBuffers, pFirstConstant, pNumConstants);        \
    ID3D11Buffer* rawCBs[kMaxCBs] = { 0 };                                                               \
    UINT cap = NumBuffers <= kMaxCBs ? NumBuffers : kMaxCBs;                                             \
    for (UINT i = 0; i < cap; ++i)                                                                       \
    {                                                                                                    \
        ID3D11Buffer* p = ppConstantBuffers ? ppConstantBuffers[i] : nullptr;                            \
        UINT slot = StartSlot + i;                                                                       \
        if (STAGE_IDX == ST_VS && slot < kMaxVSCBSlots) HoldRef(m_boundVSCBs[slot], p);                  \
        if (STAGE_IDX == ST_PS && slot < kMaxPSCBSlots) HoldRef(m_boundPSCBs[slot], p);                  \
        if (p)                                                                                           \
        {                                                                                                \
            if (auto* bp = wiz3d::TryUnwrapBuffer(static_cast<ID3D11Resource*>(p)))                      \
            {                                                                                            \
                if (STAGE_IDX != ST_PS && STAGE_IDX != ST_CS) bp->TagVSBound();                          \
                rawCBs[i] = bp->GetReal();                                                               \
            }                                                                                            \
            else rawCBs[i] = p;                                                                          \
        }                                                                                                \
    }                                                                                                    \
    if (m_real1)                                                                                         \
        m_real1->STAGE_PREFIX##SetConstantBuffers1(StartSlot, NumBuffers,                                \
            ppConstantBuffers ? rawCBs : nullptr, pFirstConstant, pNumConstants);                        \
}
STEREO_SET_CBS1(VS, ST_VS)
STEREO_SET_CBS1(HS, ST_HS)
STEREO_SET_CBS1(DS, ST_DS)
STEREO_SET_CBS1(GS, ST_GS)
STEREO_SET_CBS1(PS, ST_PS)
STEREO_SET_CBS1(CS, ST_CS)
#undef STEREO_SET_CBS1

// D3D11.1 resource/view entry points: each unwraps to the real object and applies to both eyes of a stereo resource;
// the 11.1 copy flags are hints, dropped so copies take the eye-aware 11.0 paths.
static ID3D11View* UnwrapViewForEye(ID3D11View* p, bool pickRight)
{
    if (!p) return nullptr;
    if (auto* r = wiz3d::TryUnwrapRTV(static_cast<ID3D11RenderTargetView*>(p)))
        return (pickRight && r->GetRealRight()) ? r->GetRealRight() : r->GetReal();
    if (auto* d = wiz3d::TryUnwrapDSV(static_cast<ID3D11DepthStencilView*>(p)))
        return (pickRight && d->GetRealRight()) ? d->GetRealRight() : d->GetReal();
    if (auto* sv = wiz3d::TryUnwrapSRV(static_cast<ID3D11ShaderResourceView*>(p)))
        return (pickRight && sv->GetRealRight()) ? sv->GetRealRight() : sv->GetReal();
    if (auto* u = wiz3d::TryUnwrapUAV(static_cast<ID3D11UnorderedAccessView*>(p)))
        return (pickRight && u->GetRealRight()) ? u->GetRealRight() : u->GetReal();
    return p;
}

void STDMETHODCALLTYPE Context11Proxy::SetResourceMinLOD(ID3D11Resource* pResource, FLOAT MinLOD)
{
    ID3D11Resource* l = UnwrapResourceForEye(pResource, false);
    ID3D11Resource* r = UnwrapResourceForEye(pResource, true);
    m_real->SetResourceMinLOD(l, MinLOD);
    if (r != l) m_real->SetResourceMinLOD(r, MinLOD);
}

void STDMETHODCALLTYPE Context11Proxy::CopySubresourceRegion1(
    ID3D11Resource* pDstResource, UINT DstSubresource, UINT DstX, UINT DstY, UINT DstZ,
    ID3D11Resource* pSrcResource, UINT SrcSubresource, const D3D11_BOX* pSrcBox, UINT CopyFlags)
{
    (void)CopyFlags;
    CopySubresourceRegion(pDstResource, DstSubresource, DstX, DstY, DstZ, pSrcResource, SrcSubresource, pSrcBox);
}

void STDMETHODCALLTYPE Context11Proxy::UpdateSubresource1(
    ID3D11Resource* pDstResource, UINT DstSubresource, const D3D11_BOX* pDstBox,
    const void* pSrcData, UINT SrcRowPitch, UINT SrcDepthPitch, UINT CopyFlags)
{
    (void)CopyFlags;
    UpdateSubresource(pDstResource, DstSubresource, pDstBox, pSrcData, SrcRowPitch, SrcDepthPitch);
}

void STDMETHODCALLTYPE Context11Proxy::DiscardResource(ID3D11Resource* pResource)
{
    if (!m_real1) return;
    ID3D11Resource* l = UnwrapResourceForEye(pResource, false);
    ID3D11Resource* r = UnwrapResourceForEye(pResource, true);
    m_real1->DiscardResource(l);
    if (r != l) m_real1->DiscardResource(r);
}

void STDMETHODCALLTYPE Context11Proxy::DiscardView(ID3D11View* pResourceView)
{
    if (!m_real1) return;
    ID3D11View* l = UnwrapViewForEye(pResourceView, false);
    ID3D11View* r = UnwrapViewForEye(pResourceView, true);
    m_real1->DiscardView(l);
    if (r != l) m_real1->DiscardView(r);
}

void STDMETHODCALLTYPE Context11Proxy::DiscardView1(ID3D11View* pResourceView, const D3D11_RECT* pRects, UINT NumRects)
{
    if (!m_real1) return;
    ID3D11View* l = UnwrapViewForEye(pResourceView, false);
    ID3D11View* r = UnwrapViewForEye(pResourceView, true);
    m_real1->DiscardView1(l, pRects, NumRects);
    if (r != l) m_real1->DiscardView1(r, pRects, NumRects);
}

void STDMETHODCALLTYPE Context11Proxy::ClearView(ID3D11View* pView, const FLOAT Color[4], const D3D11_RECT* pRect, UINT NumRects)
{
    if (!m_real1) return;
    ID3D11View* l = UnwrapViewForEye(pView, false);
    ID3D11View* r = UnwrapViewForEye(pView, true);
    m_real1->ClearView(l, Color, pRect, NumRects);
    if (r != l) m_real1->ClearView(r, Color, pRect, NumRects);
}

void Context11Proxy::BindEye(bool right)
{
    // The duplicated draw and this rebind both call m_real directly, so they
    // bypass the proxy methods that carry FrameTrace. Log them here instead.
    if (FrameTraceActive())
    {
        FrameTrace("    BindEye(%c) numRTV=%u rtv0=%p dsv=%p",
                   right ? 'R' : 'L', m_numRTVs,
                   m_numRTVs ? (right ? m_rtvRight[0] : m_rtvLeft[0]) : nullptr,
                   right ? m_dsvRight : m_dsvLeft);
        for (UINT st = 0; st < ST_COUNT; ++st)
            if (m_srvSlots[st].anyStereo || m_cbSlots[st].anyStereo)
                FrameTrace(" [st%u srv=%d cb=%d]", st,
                           (int)m_srvSlots[st].anyStereo, (int)m_cbSlots[st].anyStereo);
        FrameTrace("\n");
    }
    if (m_omAnyStereo)
    {
        if (m_omUavBound)
        {
            // Plain OMSetRenderTargets would unbind the UAVs; sibling-less
            // slots go null on the right so shared UAVs take only left writes.
            ID3D11UnorderedAccessView* uavs[kMaxUAVs] = {};
            for (UINT i = 0; i < m_omUavCount; ++i)
            {
                ID3D11UnorderedAccessView* l = UnwrapUAVForEye(m_omUavGame[i], false);
                ID3D11UnorderedAccessView* r = UnwrapUAVForEye(m_omUavGame[i], true);
                uavs[i] = right ? ((r != l) ? r : nullptr) : l;
            }
            // Initial counts once per game bind, or eye switches reset them.
            const UINT* counts = nullptr;
            if (right && m_omUavHasInit && !m_omUavRightInited)
            {
                counts = m_omUavInit;
                m_omUavRightInited = true;
            }
            m_real->OMSetRenderTargetsAndUnorderedAccessViews(
                m_numRTVs, right ? m_rtvRight : m_rtvLeft,
                right ? m_dsvRight : m_dsvLeft,
                m_omUavStart, m_omUavCount, uavs, counts);
        }
        else
            m_real->OMSetRenderTargets(m_numRTVs,
                                       right ? m_rtvRight : m_rtvLeft,
                                       right ? m_dsvRight : m_dsvLeft);
    }
    for (UINT st = 0; st < ST_COUNT; ++st)
    {
        if (m_srvSlots[st].anyStereo) BindStageSRVs((StageIdx)st, right);
        if (m_cbSlots[st].anyStereo)  BindStageCBs((StageIdx)st, right);
    }
}

void Context11Proxy::ClearOmUavRefs()
{
    for (UINT i = 0; i < m_omUavCount; ++i)
        if (m_omUavGame[i]) { m_omUavGame[i]->Release(); m_omUavGame[i] = nullptr; }
    m_omUavCount = 0;
}

bool Context11Proxy::BeginRightEyeDraw()
{
    if (!gInfo.DuplicateDraws) return false;
    // Stereo toggled off: skip the whole right-eye pass, not just the shift.
    if (!gInfo.Input.StereoActive) return false;
    if (m_activeEye != Eye::Left) return false;
    // Skipping UAV draws outright left GTA V's bloom bright-pass RTV black in
    // the right eye — its histogram UAVs are shared, so demand only stereo RTs.
    if (m_omHasUAVs)
    {
        if (!(m_omUavBound && m_omAnyStereo))
        {
            ++m_drawsUavSkippedThisFrame;
            return false;
        }
        ++m_drawsUavDupThisFrame;
    }
    else if (!m_omAnyStereo)
    {
        ++m_drawsMonoThisFrame;
        return false;
    }
    // Before the right eye binds anything: give it constants patched for the
    // shader that is about to run, not for whatever was bound at write time.
    // An overlay (back buffer, no depth test) keeps the game's bytes: its projection draws the HUD at screen depth.
    if (OverlayPass())
    {
        ++m_overlayDrawsThisFrame;
        UnshiftOverlayCBs();
    }
    else
    {
        if (gInfo.PatchConstantsAtDraw) PatchBoundCBsForDraw();
        // Under VertexCameraMatrices the vertex stage's declared camera matrices outside its position chain (last
        // frame's camera for per-object velocity), then the pixel stage's.
        if (gInfo.DetectInverseMatrix && VertexCameraMatrices()) PatchCameraMatrices(ST_VS);
        if (gInfo.DetectInverseMatrix) PatchCameraMatrices(ST_PS);
    }
    ++m_drawsDuplicatedThisFrame;
    if (m_omMonoDSV) ++m_drawsMonoDsvThisFrame;
    // Counted apart: a duplicated draw whose right-eye RTVs are all null writes nowhere, while a depth-only pass (a
    // shadow map) binds no RTV and still reaches the right eye through its DSV.
    if (m_numRTVs == 0)
    {
        ++m_drawsDepthOnlyThisFrame;
        if (!m_dsvRight) ++m_drawsNoRightTargetThisFrame;
    }
    else
    {
        bool anyRight = false;
        for (UINT i = 0; i < m_numRTVs && !anyRight; ++i)
            if (m_rtvRight[i]) anyRight = true;
        if (!anyRight)
        {
            ++m_drawsNoRightRTVThisFrame;
            ++m_drawsNoRightTargetThisFrame;
        }
    }
    m_activeEye = Eye::Right;
    BindEye(true);
    EditRightCBs(false);
    BindLeftSRVs(false);
    for (auto& f : Features()) f->OnRightDraw(*this);
    return true;
}

void Context11Proxy::EndRightEyeDraw()
{
    for (auto& f : Features()) f->OnRightDrawDone(*this);
    BindEye(false);
    m_activeEye = Eye::Left;
}

// Re-issue a draw for the right eye immediately after the game's own call.
#define DUPLICATE_DRAW(CALL)              \
    do {                                  \
        if (BeginRightEyeDraw())          \
        {                                 \
            m_real->CALL;                 \
            EndRightEyeDraw();            \
        }                                 \
        for (auto& f_ : Features())       \
            f_->OnDrawDone(*this);        \
    } while (0)

// Stage 4b.4 (more state setters): record-and-replay for *SetShaderResources
// across all 6 shader stages. Each stage's method body is identical except
// for the method name, so a macro keeps the boilerplate tractable. Stage 3c.2
// wraps SRVs so we unwrap before forwarding (the runtime can't see our
// vtable past ID3D11ShaderResourceView signatures), and the closure unwraps
// again per-eye on replay so right-eye siblings get bound at the right pass.
// Same gate (m_presentHookActive) as OMSet so games whose swap chain
// bypasses us stay safely in passthrough.
#define RECORD_SRV_SET(STAGE_PREFIX)                                                        \
void STDMETHODCALLTYPE Context11Proxy::STAGE_PREFIX##SetShaderResources(                    \
    UINT StartSlot, UINT NumViews, ID3D11ShaderResourceView* const* ppShaderResourceViews)  \
{                                                                                           \
    TrackSRVs(ST_##STAGE_PREFIX, StartSlot, NumViews, ppShaderResourceViews);               \
    ID3D11ShaderResourceView* rawSRVs[kMaxSRVs] = { 0 };                                    \
    UINT setCap = NumViews <= kMaxSRVs ? NumViews : kMaxSRVs;                               \
    bool pickRight = (m_activeEye == Eye::Right);                                           \
    for (UINT i = 0; i < setCap; ++i)                                                       \
        rawSRVs[i] = UnwrapSRVForEye(ppShaderResourceViews ? ppShaderResourceViews[i]       \
                                                           : nullptr, pickRight);           \
    m_real->STAGE_PREFIX##SetShaderResources(StartSlot, NumViews,                           \
        ppShaderResourceViews ? rawSRVs : nullptr);                                         \
    if (!m_presentHookActive) return;                                                       \
    std::vector<ComRefHolder> refs;                                                         \
    refs.reserve(NumViews);                                                                 \
    for (UINT i = 0; i < NumViews; ++i)                                                     \
        refs.emplace_back(ppShaderResourceViews ? ppShaderResourceViews[i] : nullptr);      \
    m_frameCommands.emplace_back(                                                           \
        [this, StartSlot, NumViews, refs]() {                                               \
            ID3D11ShaderResourceView* raw[kMaxSRVs] = { 0 };                                \
            UINT cap = NumViews <= kMaxSRVs ? NumViews : kMaxSRVs;                          \
            bool pr = (m_activeEye == Eye::Right);                                          \
            for (UINT i = 0; i < cap; ++i)                                                  \
                raw[i] = UnwrapSRVForEye(                                                   \
                    static_cast<ID3D11ShaderResourceView*>(refs[i].p), pr);                 \
            m_real->STAGE_PREFIX##SetShaderResources(StartSlot, NumViews, raw);             \
        });                                                                                 \
}
RECORD_SRV_SET(VS)
RECORD_SRV_SET(PS)
RECORD_SRV_SET(GS)
RECORD_SRV_SET(HS)
RECORD_SRV_SET(DS)
RECORD_SRV_SET(CS)
#undef RECORD_SRV_SET

// *SetSamplers — same shape as *SetShaderResources but with ID3D11SamplerState.
#define RECORD_SAMPLER_SET(STAGE_PREFIX)                                                    \
void STDMETHODCALLTYPE Context11Proxy::STAGE_PREFIX##SetSamplers(                           \
    UINT StartSlot, UINT NumSamplers, ID3D11SamplerState* const* ppSamplers)                \
{                                                                                           \
    m_real->STAGE_PREFIX##SetSamplers(StartSlot, NumSamplers, ppSamplers);                  \
    if (!m_presentHookActive) return;                                                       \
    std::vector<ComRefHolder> refs;                                                         \
    refs.reserve(NumSamplers);                                                              \
    for (UINT i = 0; i < NumSamplers; ++i)                                                  \
        refs.emplace_back(ppSamplers ? ppSamplers[i] : nullptr);                            \
    m_frameCommands.emplace_back(                                                           \
        [this, StartSlot, NumSamplers, refs]() {                                            \
            ID3D11SamplerState* raw[kMaxSamplers] = { 0 };                                  \
            UINT cap = NumSamplers <= kMaxSamplers ? NumSamplers : kMaxSamplers;            \
            for (UINT i = 0; i < cap; ++i)                                                  \
                raw[i] = static_cast<ID3D11SamplerState*>(refs[i].p);                       \
            m_real->STAGE_PREFIX##SetSamplers(StartSlot, NumSamplers, raw);                 \
        });                                                                                 \
}
RECORD_SAMPLER_SET(VS)
RECORD_SAMPLER_SET(PS)
RECORD_SAMPLER_SET(GS)
RECORD_SAMPLER_SET(HS)
RECORD_SAMPLER_SET(DS)
RECORD_SAMPLER_SET(CS)
#undef RECORD_SAMPLER_SET

// *SetConstantBuffers — same shape with ID3D11Buffer. Stage 4c will modify
// the closure body to apply per-eye CB writes (left-right view projection
// matrix), but for 4b.4 it's straight passthrough record-and-replay.
#define RECORD_CB_SET(STAGE_PREFIX, IS_VS_PIPELINE)                                         \
void STDMETHODCALLTYPE Context11Proxy::STAGE_PREFIX##SetConstantBuffers(                    \
    UINT StartSlot, UINT NumBuffers, ID3D11Buffer* const* ppConstantBuffers)                \
{                                                                                           \
    /* Stage 3c.1: unwrap wrapped buffers before forwarding. Also stage-tag */              \
    /* the proxy when VS-pipeline so 4c.1's Map filter can consult it. */                   \
    TrackCBs(ST_##STAGE_PREFIX, StartSlot, NumBuffers, ppConstantBuffers);                  \
    ID3D11Buffer* rawCBs[kMaxCBs] = { 0 };                                                  \
    UINT cap = NumBuffers <= kMaxCBs ? NumBuffers : kMaxCBs;                                \
    for (UINT i = 0; i < cap; ++i)                                                          \
    {                                                                                       \
        ID3D11Buffer* p = ppConstantBuffers ? ppConstantBuffers[i] : nullptr;               \
        if (p)                                                                              \
        {                                                                                   \
            if (auto* bp = wiz3d::TryUnwrapBuffer(static_cast<ID3D11Resource*>(p)))         \
            {                                                                               \
                if (IS_VS_PIPELINE) bp->TagVSBound();                                       \
                rawCBs[i] = bp->GetReal();                                                  \
            }                                                                               \
            else                                                                            \
            {                                                                               \
                rawCBs[i] = p;                                                              \
            }                                                                               \
        }                                                                                   \
    }                                                                                       \
    m_real->STAGE_PREFIX##SetConstantBuffers(StartSlot, NumBuffers,                         \
        ppConstantBuffers ? rawCBs : nullptr);                                              \
    if (!m_presentHookActive) return;                                                       \
    std::vector<ComRefHolder> refs;                                                         \
    refs.reserve(NumBuffers);                                                               \
    for (UINT i = 0; i < NumBuffers; ++i)                                                   \
        refs.emplace_back(ppConstantBuffers ? ppConstantBuffers[i] : nullptr);              \
    m_frameCommands.emplace_back(                                                           \
        [this, StartSlot, NumBuffers, refs]() {                                             \
            ID3D11Buffer* raw[kMaxCBs] = { 0 };                                             \
            UINT replayCap = NumBuffers <= kMaxCBs ? NumBuffers : kMaxCBs;                  \
            for (UINT i = 0; i < replayCap; ++i)                                            \
                raw[i] = UnwrapBuf(static_cast<ID3D11Buffer*>(refs[i].p));                  \
            m_real->STAGE_PREFIX##SetConstantBuffers(StartSlot, NumBuffers, raw);           \
        });                                                                                 \
}
RECORD_CB_SET(GS, 1)
RECORD_CB_SET(HS, 1)
RECORD_CB_SET(DS, 1)
RECORD_CB_SET(CS, 0)
#undef RECORD_CB_SET

// Stage 4e.2: VS variant of RECORD_CB_SET that ALSO updates m_boundVSCBs[]
// so Unmap can recover which slot each CB is at when consulting the bound
// VS's projection-matrix register table.
void STDMETHODCALLTYPE Context11Proxy::VSSetConstantBuffers(
    UINT StartSlot, UINT NumBuffers, ID3D11Buffer* const* ppConstantBuffers)
{
    TrackCBs(ST_VS, StartSlot, NumBuffers, ppConstantBuffers);
    ID3D11Buffer* rawCBs[kMaxCBs] = { 0 };
    UINT cap = NumBuffers <= kMaxCBs ? NumBuffers : kMaxCBs;
    for (UINT i = 0; i < cap; ++i)
    {
        ID3D11Buffer* p = ppConstantBuffers ? ppConstantBuffers[i] : nullptr;
        // 4e.2 slot snapshot — stored even when m_presentHookActive is off,
        // because Unmap path always uses it.
        UINT slot = StartSlot + i;
        if (slot < kMaxVSCBSlots) HoldRef(m_boundVSCBs[slot], p);

        if (p)
        {
            if (auto* bp = wiz3d::TryUnwrapBuffer(static_cast<ID3D11Resource*>(p)))
            {
                bp->TagVSBound();
                rawCBs[i] = bp->GetReal();
            }
            else
            {
                rawCBs[i] = p;
            }
        }
    }
    m_real->VSSetConstantBuffers(StartSlot, NumBuffers, ppConstantBuffers ? rawCBs : nullptr);
    if (!m_presentHookActive) return;
    std::vector<ComRefHolder> refs;
    refs.reserve(NumBuffers);
    for (UINT i = 0; i < NumBuffers; ++i)
        refs.emplace_back(ppConstantBuffers ? ppConstantBuffers[i] : nullptr);
    m_frameCommands.emplace_back(
        [this, StartSlot, NumBuffers, refs]() {
            ID3D11Buffer* raw[kMaxCBs] = { 0 };
            UINT replayCap = NumBuffers <= kMaxCBs ? NumBuffers : kMaxCBs;
            for (UINT i = 0; i < replayCap; ++i)
                raw[i] = UnwrapBuf(static_cast<ID3D11Buffer*>(refs[i].p));
            m_real->VSSetConstantBuffers(StartSlot, NumBuffers, raw);
        });
}

// PS variant of RECORD_CB_SET: same body minus the VS-pipeline tag, plus the
// m_boundPSCBs[] slot snapshot the profile matrix lookup needs in Unmap.
void STDMETHODCALLTYPE Context11Proxy::PSSetConstantBuffers(
    UINT StartSlot, UINT NumBuffers, ID3D11Buffer* const* ppConstantBuffers)
{
    TrackCBs(ST_PS, StartSlot, NumBuffers, ppConstantBuffers);
    ID3D11Buffer* rawCBs[kMaxCBs] = { 0 };
    UINT cap = NumBuffers <= kMaxCBs ? NumBuffers : kMaxCBs;
    for (UINT i = 0; i < cap; ++i)
    {
        ID3D11Buffer* p = ppConstantBuffers ? ppConstantBuffers[i] : nullptr;
        UINT slot = StartSlot + i;
        if (slot < kMaxPSCBSlots) HoldRef(m_boundPSCBs[slot], p);

        if (p)
        {
            if (auto* bp = wiz3d::TryUnwrapBuffer(static_cast<ID3D11Resource*>(p)))
                rawCBs[i] = bp->GetReal();
            else
                rawCBs[i] = p;
        }
    }
    m_real->PSSetConstantBuffers(StartSlot, NumBuffers, ppConstantBuffers ? rawCBs : nullptr);
    if (!m_presentHookActive) return;
    std::vector<ComRefHolder> refs;
    refs.reserve(NumBuffers);
    for (UINT i = 0; i < NumBuffers; ++i)
        refs.emplace_back(ppConstantBuffers ? ppConstantBuffers[i] : nullptr);
    m_frameCommands.emplace_back(
        [this, StartSlot, NumBuffers, refs]() {
            ID3D11Buffer* raw[kMaxCBs] = { 0 };
            UINT replayCap = NumBuffers <= kMaxCBs ? NumBuffers : kMaxCBs;
            for (UINT i = 0; i < replayCap; ++i)
                raw[i] = UnwrapBuf(static_cast<ID3D11Buffer*>(refs[i].p));
            m_real->PSSetConstantBuffers(StartSlot, NumBuffers, raw);
        });
}

// *SetShader — takes the stage-specific shader interface plus the
// class-instance array. Class instances are rarely non-null (used for
// dynamic shader linking) but the array is captured for fidelity.
#define RECORD_SHADER_SET(STAGE_PREFIX, SHADER_TYPE)                                        \
void STDMETHODCALLTYPE Context11Proxy::STAGE_PREFIX##SetShader(                             \
    SHADER_TYPE* pShader, ID3D11ClassInstance* const* ppClassInstances, UINT NumClassInstances) \
{                                                                                           \
    TrackStageShader(ST_##STAGE_PREFIX, pShader);                                           \
    m_real->STAGE_PREFIX##SetShader(pShader, ppClassInstances, NumClassInstances);          \
    if (!m_presentHookActive) return;                                                       \
    ComRefHolder shaderRef(pShader);                                                        \
    std::vector<ComRefHolder> ciRefs;                                                       \
    ciRefs.reserve(NumClassInstances);                                                      \
    for (UINT i = 0; i < NumClassInstances; ++i)                                            \
        ciRefs.emplace_back(ppClassInstances ? ppClassInstances[i] : nullptr);              \
    m_frameCommands.emplace_back(                                                           \
        [this, shaderRef, ciRefs, NumClassInstances]() {                                    \
            ID3D11ClassInstance* raw[kMaxClassInst] = { 0 };                                \
            UINT cap = NumClassInstances <= kMaxClassInst ? NumClassInstances : kMaxClassInst; \
            for (UINT i = 0; i < cap; ++i)                                                  \
                raw[i] = static_cast<ID3D11ClassInstance*>(ciRefs[i].p);                    \
            m_real->STAGE_PREFIX##SetShader(                                                \
                static_cast<SHADER_TYPE*>(shaderRef.p),                                     \
                ciRefs.empty() ? nullptr : raw,                                             \
                NumClassInstances);                                                         \
        });                                                                                 \
}
RECORD_SHADER_SET(GS, ID3D11GeometryShader)
RECORD_SHADER_SET(HS, ID3D11HullShader)
RECORD_SHADER_SET(DS, ID3D11DomainShader)

void Context11Proxy::TrackStageShader(StageIdx stage, void* shader)
{
    if (stage != ST_DS) return;
    m_boundDS     = static_cast<ID3D11DomainShader*>(shader);
    m_boundDSInfo = (shader && m_parent) ? m_parent->LookupShaderProjection(shader) : nullptr;
}

void STDMETHODCALLTYPE Context11Proxy::CSSetShader(
    ID3D11ComputeShader* pShader, ID3D11ClassInstance* const* ppClassInstances, UINT NumClassInstances)
{
    m_boundInfo[2] = m_parent ? m_parent->LookupShaderProjection(pShader) : nullptr;
    m_real->CSSetShader(pShader, ppClassInstances, NumClassInstances);
    if (!m_presentHookActive) return;
    ComRefHolder shaderRef(pShader);
    std::vector<ComRefHolder> ciRefs;
    ciRefs.reserve(NumClassInstances);
    for (UINT i = 0; i < NumClassInstances; ++i)
        ciRefs.emplace_back(ppClassInstances ? ppClassInstances[i] : nullptr);
    m_frameCommands.emplace_back(
        [this, shaderRef, ciRefs, NumClassInstances]() {
            ID3D11ClassInstance* raw[kMaxClassInst] = { 0 };
            UINT cap = NumClassInstances <= kMaxClassInst ? NumClassInstances : kMaxClassInst;
            for (UINT i = 0; i < cap; ++i)
                raw[i] = static_cast<ID3D11ClassInstance*>(ciRefs[i].p);
            m_real->CSSetShader(
                static_cast<ID3D11ComputeShader*>(shaderRef.p),
                ciRefs.empty() ? nullptr : raw, NumClassInstances);
        });
}

// Stage 4e.2: VS variant that ALSO snapshots m_boundVS for the Unmap path's
// projection-data lookup. Snapshot stored regardless of m_presentHookActive
// because Unmap-time consultation is independent of the replay-recording
// gate. NB: not AddRef'd — the game owns the shader; if it Releases under
// us, we'll get a stale-positive lookup that either misses (no entry in
// shaderProjections map) or hits an entry that's no longer the live shader,
// which downgrades to the original heuristic / no-op for that CB.
void STDMETHODCALLTYPE Context11Proxy::VSSetShader(
    ID3D11VertexShader* pShader, ID3D11ClassInstance* const* ppClassInstances,
    UINT NumClassInstances)
{
    m_boundVS = pShader;
    m_boundInfo[0] = m_parent ? m_parent->LookupShaderProjection(pShader) : nullptr;
    m_real->VSSetShader(pShader, ppClassInstances, NumClassInstances);
    if (!m_presentHookActive) return;
    ComRefHolder shaderRef(pShader);
    std::vector<ComRefHolder> ciRefs;
    ciRefs.reserve(NumClassInstances);
    for (UINT i = 0; i < NumClassInstances; ++i)
        ciRefs.emplace_back(ppClassInstances ? ppClassInstances[i] : nullptr);
    m_frameCommands.emplace_back(
        [this, shaderRef, ciRefs, NumClassInstances]() {
            ID3D11ClassInstance* raw[kMaxClassInst] = { 0 };
            UINT cap = NumClassInstances <= kMaxClassInst ? NumClassInstances : kMaxClassInst;
            for (UINT i = 0; i < cap; ++i)
                raw[i] = static_cast<ID3D11ClassInstance*>(ciRefs[i].p);
            m_real->VSSetShader(
                static_cast<ID3D11VertexShader*>(shaderRef.p),
                ciRefs.empty() ? nullptr : raw, NumClassInstances);
        });
}

// PS variant: snapshots m_boundPS so Unmap can match the bound pixel shader's CRC against BaseProfile.xml's
// <PixelShader> matrix declarations (not AddRef'd, as m_boundVS).
void STDMETHODCALLTYPE Context11Proxy::PSSetShader(
    ID3D11PixelShader* pShader, ID3D11ClassInstance* const* ppClassInstances,
    UINT NumClassInstances)
{
    m_boundPS = pShader;
    m_boundInfo[1] = m_parent ? m_parent->LookupShaderProjection(pShader) : nullptr;
    m_real->PSSetShader(pShader, ppClassInstances, NumClassInstances);
    if (!m_presentHookActive) return;
    ComRefHolder shaderRef(pShader);
    std::vector<ComRefHolder> ciRefs;
    ciRefs.reserve(NumClassInstances);
    for (UINT i = 0; i < NumClassInstances; ++i)
        ciRefs.emplace_back(ppClassInstances ? ppClassInstances[i] : nullptr);
    m_frameCommands.emplace_back(
        [this, shaderRef, ciRefs, NumClassInstances]() {
            ID3D11ClassInstance* raw[kMaxClassInst] = { 0 };
            UINT cap = NumClassInstances <= kMaxClassInst ? NumClassInstances : kMaxClassInst;
            for (UINT i = 0; i < cap; ++i)
                raw[i] = static_cast<ID3D11ClassInstance*>(ciRefs[i].p);
            m_real->PSSetShader(
                static_cast<ID3D11PixelShader*>(shaderRef.p),
                ciRefs.empty() ? nullptr : raw, NumClassInstances);
        });
}
#undef RECORD_SHADER_SET

// The profile's <VertexShader CRC Multiplier> scale for the bound shader's shift; Multiplier="0" is the classic
// crosshair fix (that shader's draws stay mono).
static float ProfileShaderMultiplier(Device11Proxy* parent, void* vs)
{
    if (g_ProfileData.VSCRCData.empty()) return 1.f;
    const DWORD crc = BoundShaderCRC(parent, vs);
    if (!crc) return 1.f;
    ShaderProfileDataMap::const_iterator it = g_ProfileData.VSCRCData.find(crc);
    return (it != g_ProfileData.VSCRCData.end()) ? it->second.m_Multiplier : 1.f;
}

// CRC of a bound shader, 0 if unknown: what a BaseProfile.xml <VertexShader>/<PixelShader> entry targets.
static DWORD BoundShaderCRC(Device11Proxy* parent, void* shader)
{
    if (!parent || !shader) return 0;
    const ShaderAnalysis11Result* info = parent->LookupShaderProjection(shader);
    return info ? info->crc32 : 0;
}

// Stage 4b.7: record-and-replay for draw/dispatch. Pure POD captures for the
// non-Indirect variants; Indirect/Dispatch-with-buffer use ComRefHolder to
// keep the arg buffer alive across replay. The closure body is the same shape
// as the original passthrough — no Do* helpers needed since draws don't
// reference our wrapped resources directly (the bound RTV/VB/IB/CB are picked
// up from the bound pipeline state, which the *Set* closures already replay
// with eye selection).

void STDMETHODCALLTYPE Context11Proxy::Draw(UINT VertexCount, UINT StartVertexLocation)
{
    BeginDraw(VertexCount);
    if (FrameTraceActive())
        FrameTrace("    Draw eye=%c vcount=%u start=%u vs=0x%08lX ps=0x%08lX\n",
                   m_activeEye == Eye::Right ? 'R' : 'L',
                   VertexCount, StartVertexLocation,
                   BoundShaderCRC(m_parent, m_boundVS),
                   BoundShaderCRC(m_parent, m_boundPS));
    m_real->Draw(VertexCount, StartVertexLocation);
    DUPLICATE_DRAW(Draw(VertexCount, StartVertexLocation));
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this, VertexCount, StartVertexLocation]()
        {
            if (FrameTraceActive())
                FrameTrace("    Draw eye=%c vcount=%u start=%u (replay)\n",
                           m_activeEye == Eye::Right ? 'R' : 'L',
                           VertexCount, StartVertexLocation);
            m_real->Draw(VertexCount, StartVertexLocation);
        });
}

void STDMETHODCALLTYPE Context11Proxy::DrawIndexed(
    UINT IndexCount, UINT StartIndexLocation, INT BaseVertexLocation)
{
    BeginDraw(IndexCount);
    if (FrameTraceActive())
        FrameTrace("    DrawIndexed eye=%c icount=%u start=%u base=%d vs=0x%08lX ps=0x%08lX\n",
                   m_activeEye == Eye::Right ? 'R' : 'L',
                   IndexCount, StartIndexLocation, BaseVertexLocation,
                   BoundShaderCRC(m_parent, m_boundVS),
                   BoundShaderCRC(m_parent, m_boundPS));
    m_real->DrawIndexed(IndexCount, StartIndexLocation, BaseVertexLocation);
    DUPLICATE_DRAW(DrawIndexed(IndexCount, StartIndexLocation, BaseVertexLocation));
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this, IndexCount, StartIndexLocation, BaseVertexLocation]()
        {
            if (FrameTraceActive())
                FrameTrace("    DrawIndexed eye=%c icount=%u start=%u base=%d (replay)\n",
                           m_activeEye == Eye::Right ? 'R' : 'L',
                           IndexCount, StartIndexLocation, BaseVertexLocation);
            m_real->DrawIndexed(IndexCount, StartIndexLocation, BaseVertexLocation);
        });
}

void STDMETHODCALLTYPE Context11Proxy::DrawInstanced(
    UINT VertexCountPerInstance, UINT InstanceCount,
    UINT StartVertexLocation, UINT StartInstanceLocation)
{
    BeginDraw(VertexCountPerInstance);
    m_real->DrawInstanced(VertexCountPerInstance, InstanceCount,
                          StartVertexLocation, StartInstanceLocation);
    DUPLICATE_DRAW(DrawInstanced(VertexCountPerInstance, InstanceCount,
                                 StartVertexLocation, StartInstanceLocation));
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this, VertexCountPerInstance, InstanceCount,
         StartVertexLocation, StartInstanceLocation]()
        {
            m_real->DrawInstanced(VertexCountPerInstance, InstanceCount,
                                   StartVertexLocation, StartInstanceLocation);
        });
}

void STDMETHODCALLTYPE Context11Proxy::DrawIndexedInstanced(
    UINT IndexCountPerInstance, UINT InstanceCount, UINT StartIndexLocation,
    INT BaseVertexLocation, UINT StartInstanceLocation)
{
    BeginDraw(IndexCountPerInstance);
    m_real->DrawIndexedInstanced(IndexCountPerInstance, InstanceCount,
                                  StartIndexLocation, BaseVertexLocation,
                                  StartInstanceLocation);
    DUPLICATE_DRAW(DrawIndexedInstanced(IndexCountPerInstance, InstanceCount,
                                        StartIndexLocation, BaseVertexLocation,
                                        StartInstanceLocation));
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this, IndexCountPerInstance, InstanceCount, StartIndexLocation,
         BaseVertexLocation, StartInstanceLocation]()
        {
            m_real->DrawIndexedInstanced(
                IndexCountPerInstance, InstanceCount, StartIndexLocation,
                BaseVertexLocation, StartInstanceLocation);
        });
}

void STDMETHODCALLTYPE Context11Proxy::DrawAuto()
{
    BeginDraw(UINT_MAX);   // count unknown; assume geometry
    m_real->DrawAuto();
    DUPLICATE_DRAW(DrawAuto());
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this]() { m_real->DrawAuto(); });
}

void STDMETHODCALLTYPE Context11Proxy::DrawInstancedIndirect(
    ID3D11Buffer* pBufferForArgs, UINT AlignedByteOffsetForArgs)
{
    BeginDraw(UINT_MAX);   // count unknown; assume geometry
    m_real->DrawInstancedIndirect(UnwrapBuf(pBufferForArgs), AlignedByteOffsetForArgs);
    DUPLICATE_DRAW(DrawInstancedIndirect(RightArgs(pBufferForArgs), AlignedByteOffsetForArgs));
    if (!m_presentHookActive) return;
    ComRefHolder bufRef(pBufferForArgs);
    m_frameCommands.emplace_back(
        [this, bufRef, AlignedByteOffsetForArgs]()
        {
            m_real->DrawInstancedIndirect(
                UnwrapBuf(static_cast<ID3D11Buffer*>(bufRef.p)), AlignedByteOffsetForArgs);
        });
}

void STDMETHODCALLTYPE Context11Proxy::DrawIndexedInstancedIndirect(
    ID3D11Buffer* pBufferForArgs, UINT AlignedByteOffsetForArgs)
{
    BeginDraw(UINT_MAX);   // count unknown; assume geometry
    m_real->DrawIndexedInstancedIndirect(UnwrapBuf(pBufferForArgs), AlignedByteOffsetForArgs);
    DUPLICATE_DRAW(DrawIndexedInstancedIndirect(RightArgs(pBufferForArgs), AlignedByteOffsetForArgs));
    if (!m_presentHookActive) return;
    ComRefHolder bufRef(pBufferForArgs);
    m_frameCommands.emplace_back(
        [this, bufRef, AlignedByteOffsetForArgs]()
        {
            m_real->DrawIndexedInstancedIndirect(
                UnwrapBuf(static_cast<ID3D11Buffer*>(bufRef.p)), AlignedByteOffsetForArgs);
        });
}

void STDMETHODCALLTYPE Context11Proxy::Dispatch(
    UINT ThreadGroupCountX, UINT ThreadGroupCountY, UINT ThreadGroupCountZ)
{
    ++m_dispatchesThisFrame;
    PrepareSharedUavTwins();
    m_real->Dispatch(ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ);
    if (BeginRightEyeDispatch())
    {
        ++m_dispatchesDuplicatedThisFrame;
        m_real->Dispatch(ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ);
        EndRightEyeDispatch();
    }
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this, ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ]()
        {
            m_real->Dispatch(ThreadGroupCountX, ThreadGroupCountY, ThreadGroupCountZ);
        });
}

void STDMETHODCALLTYPE Context11Proxy::DispatchIndirect(
    ID3D11Buffer* pBufferForArgs, UINT AlignedByteOffsetForArgs)
{
    ++m_dispatchesThisFrame;
    PrepareSharedUavTwins();
    m_real->DispatchIndirect(UnwrapBuf(pBufferForArgs), AlignedByteOffsetForArgs);
    if (BeginRightEyeDispatch())
    {
        ++m_dispatchesDuplicatedThisFrame;
        m_real->DispatchIndirect(RightArgs(pBufferForArgs), AlignedByteOffsetForArgs);
        EndRightEyeDispatch();
    }
    if (!m_presentHookActive) return;
    ComRefHolder bufRef(pBufferForArgs);
    m_frameCommands.emplace_back(
        [this, bufRef, AlignedByteOffsetForArgs]()
        {
            m_real->DispatchIndirect(
                UnwrapBuf(static_cast<ID3D11Buffer*>(bufRef.p)), AlignedByteOffsetForArgs);
        });
}

HRESULT STDMETHODCALLTYPE Context11Proxy::QueryInterface(REFIID riid, void** ppvObj)
{
    if (!ppvObj) return E_POINTER;
    if (riid == IID_IUnknown || riid == IID_ID3D11DeviceChild ||
        riid == IID_ID3D11DeviceContext)
    {
        *ppvObj = static_cast<ID3D11DeviceContext*>(this);
        AddRef();
        return S_OK;
    }
    // Claim Context1/2/3 with `this` so games accessing the immediate context
    // via Device1::GetImmediateContext1 (or QI for the higher versions) land
    // on our proxy rather than the unwrapped real ctx. Without this, every
    // state-setter call after that would bypass our record-for-replay logic.
    if (riid == __uuidof(ID3D11DeviceContext1) && m_real1)
    {
        *ppvObj = static_cast<ID3D11DeviceContext1*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == __uuidof(ID3D11DeviceContext2) && m_real2)
    {
        *ppvObj = static_cast<ID3D11DeviceContext2*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == __uuidof(ID3D11DeviceContext3) && m_real3)
    {
        *ppvObj = static_cast<ID3D11DeviceContext3*>(this);
        AddRef();
        return S_OK;
    }
    *ppvObj = nullptr;
    return E_NOINTERFACE;
}

void Context11Proxy::DoOMSetRenderTargets(
    UINT NumViews, ID3D11RenderTargetView* const* ppRenderTargetViews,
    ID3D11DepthStencilView* pDepthStencilView)
{
    // Stage 4a: pick the left- or right-eye real handle for each wrapped
    // RTV/DSV based on m_activeEye. When the proxy isn't stereo, both
    // GetReal() and GetRealRight() resolve to the same left-eye handle (the
    // latter is null, so we fall back to left). Stage 4b.8 will flip
    // m_activeEye between L/R passes during the per-frame replay.
    bool pickRight = (m_activeEye == Eye::Right);
    RecordGameFacingOM(NumViews, ppRenderTargetViews, pDepthStencilView);
    if (FrameTraceActive())
    {
        ID3D11RenderTargetView* raw0 = (NumViews > 0 && ppRenderTargetViews)
                                           ? ppRenderTargetViews[0] : nullptr;
        RTV11Proxy* rtv0 = raw0 ? TryUnwrapRTV(raw0) : nullptr;
        DSV11Proxy* dsv  = TryUnwrapDSV(pDepthStencilView);
        // ptr!=0 with proxy==0 means the game bound a real RTV we can't resolve
        // — a bypass, not an unbind. The two used to look identical here.
        FrameTrace("  OMSet eye=%c NumViews=%u rtv0={ptr=%p proxy=%p stereo=%d right=%p} dsv={proxy=%p stereo=%d right=%p}\n",
                   pickRight ? 'R' : 'L', NumViews, raw0,
                   rtv0, rtv0 ? rtv0->GetRealRight() != nullptr : 0,
                   rtv0 ? rtv0->GetRealRight() : nullptr,
                   dsv,  dsv  ? dsv->GetRealRight() != nullptr  : 0,
                   dsv  ? dsv->GetRealRight()  : nullptr);
    }
    ID3D11RenderTargetView* realRTVs[kMaxRTVs] = { 0 };
    ID3D11RenderTargetView* const* rtvsToUse = ppRenderTargetViews;
    if (NumViews > 0 && ppRenderTargetViews)
    {
        UINT cap = NumViews <= kMaxRTVs ? NumViews : kMaxRTVs;
        for (UINT i = 0; i < cap; ++i)
        {
            RTV11Proxy* p = TryUnwrapRTV(ppRenderTargetViews[i]);
            if (!p)
            {
                realRTVs[i] = ppRenderTargetViews[i];
                continue;
            }
            ID3D11RenderTargetView* right = p->GetRealRight();
            realRTVs[i] = (pickRight && right) ? right : p->GetReal();
        }
        rtvsToUse = realRTVs;
    }
    ID3D11DepthStencilView* realDSV = pDepthStencilView;
    if (DSV11Proxy* d = TryUnwrapDSV(pDepthStencilView))
    {
        ID3D11DepthStencilView* right = d->GetRealRight();
        realDSV = (pickRight && right) ? right : d->GetReal();
    }
    m_real->OMSetRenderTargets(NumViews, rtvsToUse, realDSV);
}

void STDMETHODCALLTYPE Context11Proxy::OMSetRenderTargets(
    UINT NumViews, ID3D11RenderTargetView* const* ppRenderTargetViews,
    ID3D11DepthStencilView* pDepthStencilView)
{
    TrackOM(NumViews, ppRenderTargetViews, pDepthStencilView);
    DoOMSetRenderTargets(NumViews, ppRenderTargetViews, pDepthStencilView);

    // Stage 4b.4: record-for-replay, but only when the Present hook is
    // active. Without a flush-each-frame trigger the vector would grow
    // unbounded, so games whose swap chain bypasses us stay safely in pure
    // passthrough mode. Capture the wrapped pointers by value (ComRefHolder
    // copy ctor AddRefs) so the lambda holds its own refs for the frame
    // even if the game releases. At replay time the closure re-calls
    // DoOMSetRenderTargets, which re-runs eye-aware unwrap with whatever
    // m_activeEye is set to at that point.
    if (!m_presentHookActive) return;
    std::vector<ComRefHolder> rtvRefs;
    rtvRefs.reserve(NumViews);
    for (UINT i = 0; i < NumViews; ++i)
        rtvRefs.emplace_back(ppRenderTargetViews ? ppRenderTargetViews[i] : nullptr);
    ComRefHolder dsvRef(pDepthStencilView);
    m_frameCommands.emplace_back(
        [this, NumViews, rtvRefs, dsvRef]()
        {
            // Rebuild raw-pointer array from the captured holders.
            ID3D11RenderTargetView* raw[kMaxRTVs] = { 0 };
            UINT cap = NumViews <= kMaxRTVs ? NumViews : kMaxRTVs;
            for (UINT i = 0; i < cap; ++i)
                raw[i] = static_cast<ID3D11RenderTargetView*>(rtvRefs[i].p);
            DoOMSetRenderTargets(NumViews, raw,
                static_cast<ID3D11DepthStencilView*>(dsvRef.p));
        });
}

void Context11Proxy::DoOMSetRenderTargetsAndUnorderedAccessViews(
    UINT NumRTVs, ID3D11RenderTargetView* const* ppRenderTargetViews,
    ID3D11DepthStencilView* pDepthStencilView,
    UINT UAVStartSlot, UINT NumUAVs,
    ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT* pUAVInitialCounts)
{
    bool pickRight = (m_activeEye == Eye::Right);
    // KEEP_RENDER_TARGETS leaves the existing binding in place, so don't
    // overwrite what we recorded from the last real bind.
    if (NumRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL)
        RecordGameFacingOM(NumRTVs, ppRenderTargetViews, pDepthStencilView);
    // Was untraced, so binds through this path looked like "no RTV bound".
    if (FrameTraceActive())
    {
        ID3D11RenderTargetView* raw0 = (NumRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL &&
                                        NumRTVs > 0 && ppRenderTargetViews)
                                       ? ppRenderTargetViews[0] : nullptr;
        RTV11Proxy* r0 = raw0 ? TryUnwrapRTV(raw0) : nullptr;
        FrameTrace("  OMSetUAV eye=%c NumRTVs=%u NumUAVs=%u rtv0={ptr=%p proxy=%p stereo=%d}\n",
                   pickRight ? 'R' : 'L', NumRTVs, NumUAVs, raw0,
                   r0, r0 ? (r0->GetRealRight() != nullptr) : 0);
    }
    ID3D11RenderTargetView* realRTVs[kMaxRTVs] = { 0 };
    ID3D11RenderTargetView* const* rtvsToUse = ppRenderTargetViews;
    if (NumRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL &&
        NumRTVs > 0 && ppRenderTargetViews)
    {
        UINT cap = NumRTVs <= kMaxRTVs ? NumRTVs : kMaxRTVs;
        for (UINT i = 0; i < cap; ++i)
        {
            RTV11Proxy* p = TryUnwrapRTV(ppRenderTargetViews[i]);
            if (!p)
            {
                realRTVs[i] = ppRenderTargetViews[i];
                continue;
            }
            ID3D11RenderTargetView* right = p->GetRealRight();
            realRTVs[i] = (pickRight && right) ? right : p->GetReal();
        }
        rtvsToUse = realRTVs;
    }
    ID3D11DepthStencilView* realDSV = pDepthStencilView;
    if (NumRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL)
    {
        if (DSV11Proxy* d = TryUnwrapDSV(pDepthStencilView))
        {
            ID3D11DepthStencilView* right = d->GetRealRight();
            realDSV = (pickRight && right) ? right : d->GetReal();
        }
    }
    // Stage 3c.2: unwrap UAVs (eye-aware) before forwarding.
    ID3D11UnorderedAccessView* realUAVs[kMaxUAVs] = { 0 };
    ID3D11UnorderedAccessView* const* uavsToUse = ppUnorderedAccessViews;
    if (NumUAVs != D3D11_KEEP_UNORDERED_ACCESS_VIEWS &&
        NumUAVs > 0 && ppUnorderedAccessViews)
    {
        UINT ucap = NumUAVs <= kMaxUAVs ? NumUAVs : kMaxUAVs;
        for (UINT i = 0; i < ucap; ++i)
            realUAVs[i] = UnwrapUAVForEye(ppUnorderedAccessViews[i], pickRight);
        uavsToUse = realUAVs;
    }
    m_real->OMSetRenderTargetsAndUnorderedAccessViews(
        NumRTVs, rtvsToUse, realDSV,
        UAVStartSlot, NumUAVs, uavsToUse, pUAVInitialCounts);
}

void STDMETHODCALLTYPE Context11Proxy::OMSetRenderTargetsAndUnorderedAccessViews(
    UINT NumRTVs, ID3D11RenderTargetView* const* ppRenderTargetViews,
    ID3D11DepthStencilView* pDepthStencilView,
    UINT UAVStartSlot, UINT NumUAVs,
    ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT* pUAVInitialCounts)
{
    // D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL leaves the OM bindings
    // alone, so the tracked pair stays valid and must not be overwritten.
    if (NumRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL)
        TrackOM(NumRTVs, ppRenderTargetViews, pDepthStencilView);
    if (gInfo.DuplicateDraws && NumUAVs > 0 &&
        NumUAVs != D3D11_KEEP_UNORDERED_ACCESS_VIEWS)
    {
        m_omHasUAVs = true;
        // Track the UAV binding (AddRef'd — the game may release views while
        // the binding persists) so BindEye can re-issue it per eye.
        ClearOmUavRefs();
        m_omUavBound = true;
        m_omUavStart = UAVStartSlot;
        m_omUavCount = NumUAVs <= kMaxUAVs ? NumUAVs : kMaxUAVs;
        m_omUavHasInit = (pUAVInitialCounts != nullptr);
        m_omUavRightInited = false;
        for (UINT i = 0; i < m_omUavCount; ++i)
        {
            ID3D11UnorderedAccessView* p =
                ppUnorderedAccessViews ? ppUnorderedAccessViews[i] : nullptr;
            if (p) p->AddRef();
            m_omUavGame[i] = p;
            m_omUavInit[i] = pUAVInitialCounts ? pUAVInitialCounts[i] : (UINT)-1;
        }
    }
    DoOMSetRenderTargetsAndUnorderedAccessViews(
        NumRTVs, ppRenderTargetViews, pDepthStencilView,
        UAVStartSlot, NumUAVs, ppUnorderedAccessViews, pUAVInitialCounts);

    if (!m_presentHookActive) return;
    // Stage 4b.4: record. Both RTV and UAV arrays need capture.
    std::vector<ComRefHolder> rtvRefs;
    if (NumRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL && ppRenderTargetViews)
    {
        rtvRefs.reserve(NumRTVs);
        for (UINT i = 0; i < NumRTVs; ++i)
            rtvRefs.emplace_back(ppRenderTargetViews[i]);
    }
    ComRefHolder dsvRef(pDepthStencilView);

    std::vector<ComRefHolder> uavRefs;
    if (ppUnorderedAccessViews)
    {
        uavRefs.reserve(NumUAVs);
        for (UINT i = 0; i < NumUAVs; ++i)
            uavRefs.emplace_back(ppUnorderedAccessViews[i]);
    }
    std::vector<UINT> initialCounts;
    if (pUAVInitialCounts)
        initialCounts.assign(pUAVInitialCounts, pUAVInitialCounts + NumUAVs);

    m_frameCommands.emplace_back(
        [this, NumRTVs, rtvRefs, dsvRef,
         UAVStartSlot, NumUAVs, uavRefs, initialCounts]()
        {
            ID3D11RenderTargetView* rawRTVs[kMaxRTVs] = { 0 };
            ID3D11RenderTargetView* const* rtvArg = nullptr;
            if (NumRTVs != D3D11_KEEP_RENDER_TARGETS_AND_DEPTH_STENCIL && !rtvRefs.empty())
            {
                UINT cap = NumRTVs <= kMaxRTVs ? NumRTVs : kMaxRTVs;
                for (UINT i = 0; i < cap; ++i)
                    rawRTVs[i] = static_cast<ID3D11RenderTargetView*>(rtvRefs[i].p);
                rtvArg = rawRTVs;
            }
            // UAVs reconstructed similarly. Stage 3c.2: now wrapped, so the
            // inner DoOMSet... helper will unwrap eye-aware. Hand it the raw
            // proxies retrieved from the captured refs.
            ID3D11UnorderedAccessView* rawUAVs[kMaxUAVs] = { 0 };
            ID3D11UnorderedAccessView* const* uavArg = nullptr;
            if (!uavRefs.empty())
            {
                UINT cap = NumUAVs <= kMaxUAVs ? NumUAVs : kMaxUAVs;
                for (UINT i = 0; i < cap; ++i)
                    rawUAVs[i] = static_cast<ID3D11UnorderedAccessView*>(uavRefs[i].p);
                uavArg = rawUAVs;
            }
            const UINT* countsArg = initialCounts.empty() ? nullptr : initialCounts.data();
            DoOMSetRenderTargetsAndUnorderedAccessViews(
                NumRTVs, rtvArg,
                static_cast<ID3D11DepthStencilView*>(dsvRef.p),
                UAVStartSlot, NumUAVs, uavArg, countsArg);
        });
}

void STDMETHODCALLTYPE Context11Proxy::RSSetViewports(UINT NumViewports, const D3D11_VIEWPORT* pViewports)
{
    m_real->RSSetViewports(NumViewports, pViewports);
    if (!m_presentHookActive) return;
    // Capture for replay. Without this the right-eye pass runs all draws
    // against whatever viewport is current at Present time (likely the SBS
    // compose viewport or the last UI viewport), producing the right-eye
    // geometry glitching / missing-elements / doubled-cursor symptoms that
    // followed the May 2026 kMaxUnwrapArray fix.
    std::vector<D3D11_VIEWPORT> vps;
    if (pViewports) vps.assign(pViewports, pViewports + NumViewports);
    m_frameCommands.emplace_back(
        [this, NumViewports, vps]()
        {
            m_real->RSSetViewports(
                NumViewports, vps.empty() ? nullptr : vps.data());
        });
}

// The right eye's copy of a resource: a buffer's right sibling or a stereo texture's, else the resource itself.
static inline ID3D11Resource* RightEyeResource(ID3D11Resource* p)
{
    if (auto* buf = p ? wiz3d::TryUnwrapBuffer(p) : nullptr)
        if (buf->GetRealRight()) return static_cast<ID3D11Resource*>(buf->GetRealRight());
    return UnwrapResourceForEye(p, true);
}

// Under duplication no replay pass redoes a copy for the right eye: the destination's right copy (a stereo texture's
// or a buffer's sibling) is written now; null when it has none.
static inline ID3D11Resource* RightMirrorTarget(ID3D11Resource* pDst, bool pickRight)
{
    if (!gInfo.DuplicateDraws || pickRight || !pDst) return nullptr;
    ID3D11Resource* r = RightEyeResource(pDst);
    return r != UnwrapResourceForEye(pDst, false) ? r : nullptr;
}

void STDMETHODCALLTYPE Context11Proxy::CopyStructureCount(
    ID3D11Buffer* pDstBuffer, UINT DstAlignedByteOffset, ID3D11UnorderedAccessView* pSrcView)
{
    bool pickRight = (m_activeEye == Eye::Right);
    m_real->CopyStructureCount(UnwrapBuf(pDstBuffer), DstAlignedByteOffset,
                               UnwrapUAVForEye(pSrcView, pickRight));
    // A destination with a right sibling takes the right UAV's count too (indirect arguments the right eye reads).
    if (ID3D11Resource* rd = RightMirrorTarget(pDstBuffer, pickRight))
        m_real->CopyStructureCount(static_cast<ID3D11Buffer*>(rd), DstAlignedByteOffset, UnwrapUAVForEye(pSrcView, true));
}

void STDMETHODCALLTYPE Context11Proxy::ClearUnorderedAccessViewUint(
    ID3D11UnorderedAccessView* pUnorderedAccessView, const UINT Values[4])
{
    bool pickRight = (m_activeEye == Eye::Right);
    m_real->ClearUnorderedAccessViewUint(UnwrapUAVForEye(pUnorderedAccessView, pickRight), Values);
    // The right view too, as for render targets: without it right-eye counters and histograms accumulate.
    ID3D11UnorderedAccessView* r = UnwrapUAVForEye(pUnorderedAccessView, true);
    if (gInfo.DuplicateDraws && !pickRight && r != UnwrapUAVForEye(pUnorderedAccessView, false))
        m_real->ClearUnorderedAccessViewUint(r, Values);
}

void STDMETHODCALLTYPE Context11Proxy::ClearUnorderedAccessViewFloat(
    ID3D11UnorderedAccessView* pUnorderedAccessView, const FLOAT Values[4])
{
    bool pickRight = (m_activeEye == Eye::Right);
    m_real->ClearUnorderedAccessViewFloat(UnwrapUAVForEye(pUnorderedAccessView, pickRight), Values);
    ID3D11UnorderedAccessView* r = UnwrapUAVForEye(pUnorderedAccessView, true);
    if (gInfo.DuplicateDraws && !pickRight && r != UnwrapUAVForEye(pUnorderedAccessView, false))
        m_real->ClearUnorderedAccessViewFloat(r, Values);
}

// Under duplication there is no replay pass to redo an operation for the right
// eye, so anything writing a stereo resource has to mirror into the sibling now.
static inline bool NeedsRightMirror(ID3D11Resource* pDst, bool pickRight)
{
    return gInfo.DuplicateDraws && !pickRight && pDst &&
           UnwrapResourceForEye(pDst, true) != UnwrapResourceForEye(pDst, false);
}

void STDMETHODCALLTYPE Context11Proxy::GenerateMips(ID3D11ShaderResourceView* pShaderResourceView)
{
    bool pickRight = (m_activeEye == Eye::Right);
    m_real->GenerateMips(UnwrapSRVForEye(pShaderResourceView, pickRight));
    if (gInfo.DuplicateDraws && !pickRight &&
        UnwrapSRVForEye(pShaderResourceView, true) !=
        UnwrapSRVForEye(pShaderResourceView, false))
        m_real->GenerateMips(UnwrapSRVForEye(pShaderResourceView, true));
}

void Context11Proxy::DoCopyResource(
    ID3D11Resource* pDstResource, ID3D11Resource* pSrcResource)
{
    bool pickRight = (m_activeEye == Eye::Right);
    m_real->CopyResource(UnwrapResourceForEye(pDstResource, pickRight),
                         UnwrapResourceForEye(pSrcResource, pickRight));
    if (ID3D11Resource* rd = RightMirrorTarget(pDstResource, pickRight))
    {
        m_real->CopyResource(rd, RightEyeResource(pSrcResource));
        if (auto* t = TryUnwrapTexture2D(pDstResource)) t->MarkRightWritten();
    }
    // A copied-into buffer no longer holds its last written bytes.
    if (Buffer11Proxy* db = TryUnwrapBuffer(pDstResource)) { db->SetRightFromShadow(false); db->ForgetShadow(); }
}

void STDMETHODCALLTYPE Context11Proxy::CopyResource(
    ID3D11Resource* pDstResource, ID3D11Resource* pSrcResource)
{
    DoCopyResource(pDstResource, pSrcResource);
    if (!m_presentHookActive) return;
    ComRefHolder dstRef(pDstResource);
    ComRefHolder srcRef(pSrcResource);
    m_frameCommands.emplace_back(
        [this, dstRef, srcRef]()
        {
            DoCopyResource(static_cast<ID3D11Resource*>(dstRef.p),
                           static_cast<ID3D11Resource*>(srcRef.p));
        });
}

void Context11Proxy::DoCopySubresourceRegion(
    ID3D11Resource* pDstResource, UINT DstSubresource, UINT DstX, UINT DstY,
    UINT DstZ, ID3D11Resource* pSrcResource, UINT SrcSubresource,
    const D3D11_BOX* pSrcBox)
{
    bool pickRight = (m_activeEye == Eye::Right);
    m_real->CopySubresourceRegion(
        UnwrapResourceForEye(pDstResource, pickRight), DstSubresource, DstX, DstY, DstZ,
        UnwrapResourceForEye(pSrcResource, pickRight), SrcSubresource, pSrcBox);
    if (ID3D11Resource* rd = RightMirrorTarget(pDstResource, pickRight))
    {
        m_real->CopySubresourceRegion(rd, DstSubresource, DstX, DstY, DstZ,
                                      RightEyeResource(pSrcResource), SrcSubresource, pSrcBox);
        if (auto* t = TryUnwrapTexture2D(pDstResource)) t->MarkRightWritten();
    }
    if (Buffer11Proxy* db = TryUnwrapBuffer(pDstResource)) { db->SetRightFromShadow(false); db->ForgetShadow(); }
}

void STDMETHODCALLTYPE Context11Proxy::CopySubresourceRegion(
    ID3D11Resource* pDstResource, UINT DstSubresource, UINT DstX, UINT DstY,
    UINT DstZ, ID3D11Resource* pSrcResource, UINT SrcSubresource,
    const D3D11_BOX* pSrcBox)
{
    DoCopySubresourceRegion(pDstResource, DstSubresource, DstX, DstY, DstZ,
                            pSrcResource, SrcSubresource, pSrcBox);
    if (!m_presentHookActive) return;
    ComRefHolder dstRef(pDstResource);
    ComRefHolder srcRef(pSrcResource);
    bool hasBox = (pSrcBox != nullptr);
    D3D11_BOX box = {};
    if (hasBox) box = *pSrcBox;
    m_frameCommands.emplace_back(
        [this, dstRef, DstSubresource, DstX, DstY, DstZ,
         srcRef, SrcSubresource, hasBox, box]()
        {
            DoCopySubresourceRegion(
                static_cast<ID3D11Resource*>(dstRef.p),
                DstSubresource, DstX, DstY, DstZ,
                static_cast<ID3D11Resource*>(srcRef.p),
                SrcSubresource, hasBox ? &box : nullptr);
        });
}

// Stage 4c: walk the captured CB bytes 64-byte (4x4 float) at a time. If the
// chunk matches the D3D row-major perspective projection pattern, apply the
// configured eye-shift to the m[2][0] element so the right eye renders with
// a horizontally-offset projection.
//
// D3D row-major perspective projection layout (HLSL default mul order):
//   m[0][0]=xScale   m[0][1]=0     m[0][2]=0       m[0][3]=0
//   m[1][0]=0        m[1][1]=yScale m[1][2]=0      m[1][3]=0
//   m[2][0]=0        m[2][1]=0     m[2][2]=zFactor m[2][3]=1
//   m[3][0]=0        m[3][1]=0     m[3][2]=zOffset m[3][3]=0
//
// As floats[0..15], we check floats[11]==1 (m[2][3]) and floats[15]==0
// (m[3][3]) — distinguishes perspective projections from identity, view,
// world, and orthographic. m[0][0] != 0 too (some xScale). When matched,
// add eyeShift to floats[8] (m[2][0]). This is the standard "horizontal
// off-axis projection" stereo trick used by iZ3D and 3DMigoto for view-
// shift without parallax errors.
// Stage 4e.2: targeted modifier. matrices[] enumerates known matrix start
// registers (matrixRegister) + their transposed flag. Each register is
// 16 bytes (4 floats). Non-transposed (HLSL row_major): matrix spans
// 4 consecutive registers row-by-row; m[2][0] is at the (register+2)*16
// byte offset, component 0. Transposed (HLSL default column_major): same
// 4 registers but stored column-by-column; m[2][0] is at register*16 + 8.
// We also need m[0][0] (xScale) to scale the shift by — at offset 0 of
// the first register in both layouts.
struct EyeShiftMatrix
{
    DWORD matrixRegister;   // register index inside CB
    BOOL  matrixIsTransposed;
    // Set for matrices declared <Inverse Value="1"/> in BaseProfile.xml: the CB
    // holds P^-1 (deferred passes reconstructing position), not P.
    BOOL  matrixIsInverse;
    // Recalled from the buffer instead of confirmed by the bound shader, so the
    // contents get a shape check before we trust them.
    BOOL  matrixFromLearned;
};

// Matrix helpers live in Shared/StereoMatrix.h; these keep the call sites.
static float EyeShiftB(float eyeShift) { return s_mat.EyeShiftB(eyeShift); }
using wiz3d::MatrixBasisNorms;
using wiz3d::BasesOrthogonal;
static float ProjXScaleFromMatrix(const float* f, bool transposed) { return s_mat.ProjXScaleFromMatrix(f, transposed); }

// Zero-parallax distance in world units, 0 with convergence off. The forward model is x += sep*(w - conv), so the
// forward and the inverse patch both read conv from here.
static float StereoConvergence()
{
    const CameraPreset* p = gInfo.Input.GetActivePreset();
    return (p && p->One_div_ZPS != 0.f) ? 1.f / p->One_div_ZPS : 0.f;
}

// DX9 shears view space by x' = x + m31*z + m41 (BaseStereoRenderer-inl.h:141): eyeShift is m31, which only re-centres,
// and m41 is the eye offset whose 1/z falloff is the depth cue; A = -B * One_div_ZPS relates the two.
static float EyeOffsetM41(float eyeShift)
{
    return -eyeShift * StereoConvergence();
}

// The core's stereo state, for profile features (ProfileFeature.h).
namespace stereo
{
float Convergence()               { return StereoConvergence(); }
float FallbackB(float eyeShift)   { return EyeShiftB(eyeShift); }
void  PublishCameraP00(float p00) { s_camP00 = p00; }
}

// The analysis finds a projection-shaped matrix, not necessarily the camera's: orthographic (HUD) and light or shadow
// projections are rejected as the legacy DDI path does (ConstantBufferWrapper.cpp), under the same gInfo flags.
static bool ShouldSkipProjectionMatrix(const float* f, bool transposed)
{
    // Orthographic: no perspective divide, i.e. the row/column that produces w
    // is (0,0,0,non-zero). A camera projection puts +/-1 in one of those slots.
    if (!gInfo.SkipCheckOrthoMatrix)
    {
        const float w0 = transposed ? f[12] : f[3];
        const float w1 = transposed ? f[13] : f[7];
        const float w2 = transposed ? f[14] : f[11];
        const float w3 = f[15];
        // Exact zeros by default; CheckShadowMatrix allows the ~1e-8 a
        // concatenated light matrix carries, which otherwise read as perspective.
        const bool ortho = gInfo.CheckShadowMatrix
            ? (fabsf(w3) > 1e-12f &&
               (fabsf(w0) + fabsf(w1) + fabsf(w2)) <= 1e-4f * fabsf(w3))
            : (w0 == 0.f && w1 == 0.f && w2 == 0.f && w3 != 0.f);
        if (ortho) return true;
    }

    if (gInfo.CheckShadowMatrix || gInfo.CheckExistenceInverseMatrix)
    {
        D3DXMATRIX m(f);
        D3DXMATRIX inv;
        if (!D3DXMatrixInverse(&inv, nullptr, &m))
        {
            // Singular — legacy treats "no inverse" as not-a-camera-projection.
            if (gInfo.CheckExistenceInverseMatrix) return true;
        }
        else if (gInfo.CheckShadowMatrix)
        {
            const float probe = transposed ? inv._43 : inv._34;
            if (fabsf(probe) < 0.01f) return true;   // perspective shadow map
        }
    }
    return false;
}

// Numeric inverse-matrix detection (DetectInverseMatrix): the method is in Shared/StereoMatrix.h.
static void CacheForwardVP(const float* f, bool transposed) { s_mat.CacheForwardVP(f, transposed); }

// The first inverse-shaped block only: a buffer can hold a previous frame's reprojection matrix beside the current one.
static bool DetectInverseInCB(const unsigned char* data, size_t byteCount,
                              wiz3d::StereoMatrixState::DetectedInverse& out, float& bestMiss)
{
    return s_mat.DetectInverses(data, byteCount, &out, 1, bestMiss) != 0;
}

// outShifted / outRejected / outInvPatched count for the frame report (may be null). fullColumn (draw duplication):
// shift the whole first column and add the eye offset; else the replay's m[2][0] shear, its inverse likewise.
static void ApplyTargetedEyeShiftToCB(unsigned char* data, size_t byteCount,
                                      float eyeShift,
                                      const std::vector<EyeShiftMatrix>& matrices,
                                      bool fullColumn,
                                      unsigned* outShifted = nullptr,
                                      unsigned* outRejected = nullptr,
                                      unsigned* outInvPatched = nullptr,
                                      const void* srcBuffer = nullptr)
{
    if (eyeShift == 0.f || matrices.empty()) return;
    constexpr size_t kRegBytes  = 16;
    constexpr size_t kMat4Bytes = 4 * kRegBytes;
    Buffer11Proxy* lightBuf = srcBuffer ? TryUnwrapBuffer(static_cast<ID3D11Resource*>(const_cast<void*>(srcBuffer))) : nullptr;
    for (const auto& m : matrices)
    {
        size_t base = size_t(m.matrixRegister) * kRegBytes;
        if (base + kMat4Bytes > byteCount) continue;
        float* f = reinterpret_cast<float*>(data + base);
        const bool transposed = (m.matrixIsTransposed != 0);
        // Only shadow passes read it as a position matrix: a light's view, the same for both eyes.
        if (lightBuf && !m.matrixIsInverse && lightBuf->LightOnlyMatrix(m.matrixRegister)) { ++s_lightOnlyKept; continue; }

        // Inverse (unprojection) matrices take the exact inverse of the whole forward shift (PatchInverse). No guard
        // below applies to them, so a learned one must re-match the ring; a profile-declared one is trusted.
        if (m.matrixIsInverse)
        {
            bool shapeReject = false;
            if (m.matrixFromLearned && !s_mat.VerifyInverseAgainstRing(f, transposed, &shapeReject))
            {
                ++s_mat.invStaleThisFrame;
                if (shapeReject) ++s_mat.invStaleShapeThisFrame;
                if (outRejected) ++*outRejected;
                continue;
            }
            s_mat.PatchInverse(f, transposed, eyeShift, fullColumn ? StereoConvergence() : 0.f);
            if (outShifted) ++*outShifted;
            if (outInvPatched) ++*outInvPatched;
            continue;
        }

        float xScale = f[0];
        if (xScale == 0.f) continue;
        // A recalled target is a guess about a buffer that may be pooled and now hold something else, so it needs the
        // unit w basis of a real view-projection (a near-zero |w| is a HUD's orthographic matrix).
        if (m.matrixFromLearned)
        {
            float lxn, lwn;
            MatrixBasisNorms(f, transposed, lxn, lwn);
            if (fabsf(lwn - 1.f) >= 0.05f)
            {
                if (outRejected) ++*outRejected;
                continue;
            }
            // And orthogonal x and w bases: UE3 reuses one buffer per slot, so a learned register also lands on
            // material blocks {color, unit direction} that pass the w test (BioShock Infinite's white flashes).
            if (!BasesOrthogonal(f, transposed, lxn, lwn))
            {
                if (outRejected) ++*outRejected;
                continue;
            }
        }
        if (ShouldSkipProjectionMatrix(f, transposed))
        {
            if (outRejected) ++*outRejected;
            continue;
        }
        // The camera scale and the detector's ring take only clean view-projections (unit w, orthogonal bases): no
        // scaled WVP, light projection or UE3 material block. Pre-shift values: the game derives its inverse from them.
        float basisXn, basisWn;
        MatrixBasisNorms(f, transposed, basisXn, basisWn);
        if (basisWn > 1e-6f && fabsf(basisWn - 1.f) < 0.05f &&
            BasesOrthogonal(f, transposed, basisXn, basisWn))
        {
            s_mat.projXScale = basisXn / basisWn;
            if (gInfo.DetectInverseMatrix) CacheForwardVP(f, transposed);
            // The frame's most frequent clean focal is the scene camera's: the reference for UI projections, scaled
            // worlds and last frame's camera alike.
            auto& vote = s_cleanP00Votes[(int)lroundf(s_mat.projXScale * 1000.f)];
            ++vote.first;
            vote.second = s_mat.projXScale;
        }
        // A scaled world (MP3's light volumes: unit mesh * radius) has no unit w row; its x scale is the camera's p00,
        // not the cached last unit-w matrix's.
        const bool unitW = basisWn > 1e-6f && fabsf(basisWn - 1.f) < 0.05f;
        const float own = ProjXScaleFromMatrix(f, transposed);
        // UiProjectionParallax: a clean matrix whose focal is beyond 2x from the scene camera's is a UI projection, not
        // another FOV (a weapon's stays within), and its draw takes the scene's parallax.
        const float scene = s_camP00 > 0.f ? s_camP00 : s_sceneP00;
        const bool uiFocal = s_uiProjectionParallax && unitW && scene > 0.f && (own < 0.5f * scene || own > 2.f * scene);
        if (!unitW && s_camP00 > 0.f) ++s_scaledByCamera;
        const float camXScale = uiFocal ? scene : (!unitW && s_camP00 > 0.f) ? s_camP00 : own;

        if (fullColumn)
        {
            const float b  = eyeShift * camXScale;
            const float m41 = EyeOffsetM41(eyeShift) * camXScale;
            if (transposed)
            {
                f[0] += b * f[12];  f[1] += b * f[13];
                f[2] += b * f[14];  f[3] += b * f[15];
                f[3] += m41;        // constant on x: the 1/z parallax term
            }
            else
            {
                f[0] += b * f[3];   f[4]  += b * f[7];
                f[8] += b * f[11];  f[12] += b * f[15];
                f[12] += m41;       // constant on x: the 1/z parallax term
            }
        }
        else if (transposed)
        {
            // m[2][0] = register 0, component 2
            f[2] += eyeShift * xScale;
        }
        else
        {
            // m[2][0] = register 2, component 0
            f[8] += eyeShift * xScale;
        }
        if (outShifted) ++*outShifted;
    }
}


// Patches for the shader about to read the constants, not whichever was bound at write time (incidental when a CB is
// filled long before its draw, as in GTA V). Unchanged contents and the same shader cost one compare.
void Context11Proxy::PatchBoundCBsForDraw()
{
    if (!m_boundVS || !m_parent) return;
    const float eyeShift = wiz3D_GetEffectiveEyeShift()
                         * ProfileShaderMultiplier(m_parent, m_boundVS) * EyeShiftSign();
    if (eyeShift == 0.f) return;
    PatchStageCBsForDraw(ST_VS, m_boundVS, m_parent->LookupShaderProjection(m_boundVS), eyeShift);
    // Tessellated geometry projects in the domain shader; its buffers come after the vertex stage's, so a buffer both
    // read ends up carrying both shaders' matrices.
    if (m_boundDS && m_boundDSInfo) PatchStageCBsForDraw(ST_DS, m_boundDS, m_boundDSInfo, eyeShift);
}

// One stage's constant buffers for the draw: its projection matrices shifted and learned inverses re-applied; the vertex
// stage also gives back the game's bytes where it reads no matrix, unless the domain stage reads one there.
void Context11Proxy::PatchStageCBsForDraw(StageIdx stage, void* shader, const ShaderAnalysis11Result* info, float eyeShift)
{
    if (!info || !info->parsed) return;
    const DWORD crc = BoundShaderCRC(m_parent, shader);

    const CBEyeSlots& s = m_cbSlots[stage];
    for (UINT slot = 0; slot < s.high && slot < kMaxVSCBSlots; ++slot)
    {
        Buffer11Proxy* b = s.proxy[slot];
        if (!b || !b->GetRealRight() || b->Shadow().empty()) continue;
        auto it = info->projection.matrixData.cb.find(slot);
        // No matrix read here: the shader wants the game's bytes, which a write under another shader may have shifted.
        if (it == info->projection.matrixData.cb.end())
        {
            bool dsReads = false;
            if (stage == ST_VS && m_boundDS && m_boundDSInfo && m_boundDSInfo->parsed)
                for (UINT k = 0; k < m_cbSlots[ST_DS].high && k < kMaxVSCBSlots; ++k)
                    dsReads |= (m_cbSlots[ST_DS].proxy[k] == b && m_boundDSInfo->projection.matrixData.cb.count(k) != 0);
            // Under VertexCameraMatrices a declared matrix outside the chain is PatchCameraMatrices' to shift or restore.
            bool rdefReads = false;
            if (VertexCameraMatrices())
                for (const auto& m : info->readMatrices) rdefReads |= (m.first == (int)slot);
            if (stage == ST_VS && !dsReads && !rdefReads && b->RightForwardShifted()) { RestoreRightCB(b); ++m_plainRestoredThisFrame; }
            continue;
        }
        if (!b->NeedsDrawPatch(crc, eyeShift)) continue;

        std::vector<EyeShiftMatrix> targets;
        // A ranged bind (11.1 first constant) places the shader's register that far into the buffer.
        const DWORD rangeBase = s.num[slot] ? s.first[slot] : 0;
        for (const auto& pmd : it->second)
        {
            if (pmd.incorrectProjection) continue;
            EyeShiftMatrix em = { pmd.matrixRegister + rangeBase, pmd.matrixIsTransposed, FALSE, FALSE };
            targets.push_back(em);
        }
        // The vertex stage's matrices in the same buffer stay shifted when the domain stage rewrites it.
        if (stage == ST_DS && m_boundInfo[0] && m_boundInfo[0]->parsed)
            for (UINT k = 0; k < m_cbSlots[ST_VS].high && k < kMaxVSCBSlots; ++k)
            {
                if (m_cbSlots[ST_VS].proxy[k] != b) continue;
                auto vit = m_boundInfo[0]->projection.matrixData.cb.find(k);
                if (vit == m_boundInfo[0]->projection.matrixData.cb.end()) continue;
                const DWORD vsBase = m_cbSlots[ST_VS].num[k] ? m_cbSlots[ST_VS].first[k] : 0;
                for (const auto& pmd : vit->second)
                {
                    if (pmd.incorrectProjection) continue;
                    bool dup = false;
                    for (const auto& t : targets) dup |= (t.matrixRegister == pmd.matrixRegister + vsBase);
                    if (dup) continue;
                    EyeShiftMatrix em = { pmd.matrixRegister + vsBase, pmd.matrixIsTransposed, FALSE, FALSE };
                    targets.push_back(em);
                }
            }
        // The whole sibling is rewritten, so any inverse it carries is re-applied: an inverse VP does not depend on the
        // consumer, and the ring re-verification still guards it.
        for (const auto& mem : b->LearnedMatrices())
        {
            if (!mem.inverse) continue;
            EyeShiftMatrix em = { mem.reg, mem.transposed, TRUE, TRUE };
            targets.push_back(em);
        }
        if (targets.empty()) { b->MarkDrawPatched(crc, eyeShift); continue; }

        m_drawPatchScratch.assign(b->Shadow().begin(), b->Shadow().end());
        unsigned shifted = 0, inverses = 0;
        ApplyTargetedEyeShiftToCB(m_drawPatchScratch.data(), m_drawPatchScratch.size(),
                                  eyeShift, targets, true, &shifted, nullptr, &inverses, b);
        NoteRightConstants(b);
        if (WriteWholeRightCB(b, m_drawPatchScratch))
        {
            b->MarkDrawPatched(crc, eyeShift);
            if (shifted) b->MarkShifted(s_frameId);
            b->NoteRightShift(shifted - inverses, inverses);
            b->InvalidateInverseDrawCheck();
            if (stage == ST_DS) ++m_dsBuffersPatchedThisFrame;
        }
    }
}

// The right sibling gets the game's bytes back; the shader about to read them wants no shift there.
void Context11Proxy::RestoreRightCB(Buffer11Proxy* b)
{
    if (!b || !b->GetRealRight() || b->Shadow().empty()) return;
    NoteRightConstants(b);
    if (!WriteWholeRightCB(b, b->Shadow())) return;
    b->SetRightFromShadow(true);
    b->MarkDrawPatched(0, 0.f);
    b->InvalidateInverseDrawCheck();
}

// An overlay draw (back buffer, no depth test) reads every vertex-stage buffer as the game wrote it.
void Context11Proxy::UnshiftOverlayCBs()
{
    const CBEyeSlots& s = m_cbSlots[ST_VS];
    for (UINT slot = 0; slot < s.high && slot < kMaxVSCBSlots; ++slot)
    {
        Buffer11Proxy* b = s.proxy[slot];
        if (!b || !b->GetRealRight() || b->Shadow().empty() || b->RightIsShadow()) continue;
        RestoreRightCB(b);
        ++m_overlayRestoredThisFrame;
    }
}

// WRITE_DISCARD works on a dynamic buffer only; a DEFAULT one (UpdateSubresource style) is written the way the game
// writes it, or it is never patched.
bool Context11Proxy::WriteWholeRightCB(Buffer11Proxy* b, const std::vector<unsigned char>& bytes)
{
    D3D11_MAPPED_SUBRESOURCE mapped = {};
    if (SUCCEEDED(m_real->Map(b->GetRealRight(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)) && mapped.pData)
    {
        memcpy(mapped.pData, bytes.data(), bytes.size());
        m_real->Unmap(b->GetRealRight(), 0);
        return true;
    }
    D3D11_BUFFER_DESC bd = {};
    b->GetRealRight()->GetDesc(&bd);
    if (bd.Usage != D3D11_USAGE_DEFAULT || bd.ByteWidth != bytes.size()) return false;
    m_real->UpdateSubresource(b->GetRealRight(), 0, nullptr, bytes.data(), 0, 0);
    return true;
}

// The camera matrices a pixel or compute shader reads (RDEF): an inverse view-projection (deferred and tiled lighting)
// and last frame's view-projection (motion blur). They are written before the frame's camera reaches the detector's
// ring, so they are identified at the right draw or dispatch; what a shader's register held once it holds every time.
// Last frame's camera (motion blur's P * V_prev * V_cur^-1) is no ring entry while the camera moves, yet it keeps the
// scene camera's x scale with a unit w basis and orthogonal bases; no light or HUD matrix declared next to it does.
static bool CameraFamily(const float* f, bool transposed)
{
    const float scene = s_camP00 > 0.f ? s_camP00 : s_sceneP00;
    if (scene <= 0.f) return false;
    float xn, wn;
    MatrixBasisNorms(f, transposed, xn, wn);
    if (wn <= 1e-6f || fabsf(wn - 1.f) >= 0.05f || !BasesOrthogonal(f, transposed, xn, wn)) return false;
    return fabsf(xn / wn - scene) <= 0.002f * scene;
}

void Context11Proxy::PatchCameraMatrices(StageIdx stage)
{
    const ShaderAnalysis11Result* ps = m_boundInfo[stage == ST_CS ? 2 : stage == ST_VS ? 0 : 1];
    if (!ps || ps->readMatrices.empty() || !m_parent) return;
    const float multiplier = stage == ST_CS ? 1.f : ProfileShaderMultiplier(m_parent, m_boundVS);
    const float eyeShift = wiz3D_GetEffectiveEyeShift() * multiplier * EyeShiftSign();
    if (eyeShift == 0.f) return;
    const float conv = StereoConvergence();
    const CBEyeSlots& s = m_cbSlots[stage];
    std::vector<EyeShiftMatrix> targets;
    for (UINT slot = 0; slot < s.high && slot < kMaxCBs; ++slot)
    {
        Buffer11Proxy* b = s.proxy[slot];
        if (!b || !b->GetRealRight() || b->Shadow().empty()) continue;
        // A vertex buffer holding the position chain's matrix is the draw patch's.
        if (stage == ST_VS && ps->projection.matrixData.cb.count(slot)) continue;
        bool reads = false;
        for (const auto& m : ps->readMatrices) reads |= (m.first == (int)slot);
        if (!reads || !b->InverseDrawCheckDue(eyeShift, conv)) continue;
        b->MarkInverseDrawChecked(eyeShift, conv);
        ++m_invDrawChecksThisFrame;

        const std::vector<unsigned char>& bytes = b->Shadow();
        targets.clear();
        // Every bound stage's declared matrices in this buffer go in together: one stage's rebuild must not drop the
        // other's (Metro's full-screen VS reads m_iP and its lighting PS m_iVP from one buffer).
        for (int st = 0; st < 3; ++st)
        {
        const StageIdx other = st == 0 ? ST_VS : st == 1 ? ST_PS : ST_CS;
        const ShaderAnalysis11Result* info = m_boundInfo[st];
        if (!info || (other == ST_CS) != (stage == ST_CS)) continue;
        if (other == ST_VS && !VertexCameraMatrices()) continue;
        for (UINT oslot = 0; oslot < m_cbSlots[other].high && oslot < kMaxCBs; ++oslot)
        {
        if (m_cbSlots[other].proxy[oslot] != b) continue;
        if (other == ST_VS && info->projection.matrixData.cb.count(oslot)) continue;
        for (const auto& m : info->readMatrices)
        {
            if (m.first != (int)oslot || (m.second & 15)) continue;
            // A ranged bind (11.1 first constant) places the shader's register that far into the buffer.
            const size_t off = m.second + (size_t)(m_cbSlots[other].num[oslot] ? m_cbSlots[other].first[oslot] : 0) * 16;
            if (off + 64 > bytes.size()) continue;
            const DWORD reg = (DWORD)(off / 16);
            bool dup = false;
            for (const auto& t : targets) dup |= (t.matrixRegister == reg);
            if (dup) continue;
            const float* f = reinterpret_cast<const float*>(bytes.data() + off);
            const unsigned long long key = ((unsigned long long)info->crc32 << 32) | (oslot << 16) | reg;
            auto known = m_psCameraMatrices.find(key);
            // A register classified once may hold something else later (pooled buffers, a load screen's bytes): an
            // inverse no longer shaped like one is forgotten and matched afresh.
            if (known != m_psCameraMatrices.end() && known->second.inverse)
            {
                float cand[16];
                wiz3d::CanonicalizeMat4(f, known->second.transposed, cand);
                if (!wiz3d::LooksLikeUnprojection(cand)) { m_psCameraMatrices.erase(known); known = m_psCameraMatrices.end(); }
            }
            if (known == m_psCameraMatrices.end())
                for (int o = 0; o < 2 && known == m_psCameraMatrices.end(); ++o)
                {
                    float cand[16];
                    wiz3d::CanonicalizeMat4(f, o == 1, cand);
                    const bool inverse = wiz3d::LooksLikeUnprojection(cand) && s_mat.MatchInverse(cand, nullptr);
                    if (inverse || s_mat.MatchForward(cand) || CameraFamily(f, o == 1))
                        known = m_psCameraMatrices.insert(std::make_pair(key, PsCameraMatrix{ inverse, o == 1 })).first;
                }
            if (known == m_psCameraMatrices.end()) continue;
            float cand[16];
            wiz3d::CanonicalizeMat4(f, known->second.transposed, cand);
            if (known->second.inverse && !wiz3d::LooksLikeUnprojection(cand)) continue;
            const EyeShiftMatrix em = { reg, known->second.transposed ? TRUE : FALSE, known->second.inverse ? TRUE : FALSE,
                                        FALSE };
            targets.push_back(em);
        }
        }
        }
        // A vertex buffer kept from the plain restore for its declared matrix, none of which is the camera's, is restored.
        if (targets.empty()) { if (stage == ST_VS && b->RightForwardShifted()) RestoreRightCB(b); continue; }
        // The whole sibling is rebuilt from the game's bytes: a geometry shift a write put there (a buffer shared with
        // the vertex stage) is re-applied from the buffer's remembered targets, under their guards.
        if (b->RightForwardShifted())
            for (const auto& mem : b->LearnedMatrices())
            {
                if (mem.inverse) continue;
                bool known = false;
                for (const auto& t : targets) known |= (t.matrixRegister == mem.reg);
                if (known) continue;
                const EyeShiftMatrix em = { mem.reg, mem.transposed, FALSE, TRUE };
                targets.push_back(em);
            }
        m_drawPatchScratch.assign(bytes.begin(), bytes.end());
        unsigned shifted = 0, inverses = 0;
        ApplyTargetedEyeShiftToCB(m_drawPatchScratch.data(), m_drawPatchScratch.size(), eyeShift, targets, true,
                                  &shifted, nullptr, &inverses, b);
        if (!shifted || !WriteWholeRightCB(b, m_drawPatchScratch)) continue;
        NoteRightConstants(b);
        b->NoteRightShift(shifted - inverses, inverses);
        m_invDrawPatchedThisFrame += inverses;
        m_fwdDrawPatchedThisFrame += shifted - inverses;
        if (gInfo.CollectDebugInformation) ++m_invDrawByPS[ps->crc32];
    }
}

// Hand-authored matrix declarations from BaseProfile.xml, covering what the
// bytecode analyzer cannot detect — chiefly inverse view-projection matrices.
static void AddProfileMatrixTargets(const ShaderProfileDataMap& map, DWORD crc,
                                    ID3D11Buffer* const* boundCBs, UINT numSlots,
                                    ID3D11Resource* pResource,
                                    std::vector<EyeShiftMatrix>& targets)
{
    if (!crc) return;
    ShaderProfileDataMap::const_iterator it = map.find(crc);
    if (it == map.end() || !it->second.m_pMatrices) return;
    const ShaderMatrices* sm = it->second.m_pMatrices;
    for (BYTE i = 0; i < sm->matrixSize; ++i)
    {
        const ShaderMatrixData& md = sm->matrix[i];
        if (md.incorrectProjection) continue;
        if (md.constantBuffer >= numSlots) continue;
        if (boundCBs[md.constantBuffer] != pResource) continue;
        EyeShiftMatrix em;
        em.matrixRegister     = md.matrixRegister;
        em.matrixIsTransposed = md.matrixIsTransposed;
        em.matrixIsInverse    = md.inverse;
        em.matrixFromLearned  = FALSE;
        targets.push_back(em);
    }
}

static void ApplyEyeShiftToCB(unsigned char* data, size_t byteCount, float eyeShift)
{
    if (eyeShift == 0.f) return;
    constexpr size_t kMat4Bytes = 16 * sizeof(float);
    if (byteCount < kMat4Bytes) return;
    for (size_t off = 0; off + kMat4Bytes <= byteCount; off += 4)
    {
        float* f = reinterpret_cast<float*>(data + off);
        // Perspective projection pattern: m[2][3]==1, m[3][3]==0,
        // m[0][0]!=0, m[1][1]!=0. The xScale / yScale checks reject
        // matrices that happen to have 1/0 in those slots for unrelated
        // reasons (lookups, bone weights, etc).
        if (f[11] != 1.f) continue;
        if (f[15] != 0.f) continue;
        if (f[0]  == 0.f) continue;
        if (f[5]  == 0.f) continue;
        // Hit — shift m[2][0]. Scaled by xScale so the shift magnitude
        // is proportional to the projected coord range (otherwise high-
        // FOV games get too much shift, narrow-FOV games too little).
        f[8] += eyeShift * f[0];
    }
}

HRESULT STDMETHODCALLTYPE Context11Proxy::Map(
    ID3D11Resource* pResource, UINT Subresource, D3D11_MAP MapType, UINT MapFlags,
    D3D11_MAPPED_SUBRESOURCE* pMappedResource)
{
    // Stage 3c.1: unwrap either texture or buffer proxies before forwarding.
    Buffer11Proxy*    buf = TryUnwrapBuffer(pResource);
    // Texture1D/3D too: our proxy passed through as a real resource had d3d11 dereference it (GTA V maps a volume LUT).
    ID3D11Resource*   realRes = UnwrapResourceForEye(pResource, false);
    HRESULT hr = m_real->Map(realRes, Subresource, MapType, MapFlags, pMappedResource);
    if (FAILED(hr) || !pMappedResource) return hr;
    // Duplication needs CB capture too, and never arms m_presentHookActive.
    if (!m_presentHookActive && !gInfo.DuplicateDraws) return hr;
    if (!gInfo.UseCOMWrapReplay) return hr;

    // Stage 4c: only record write maps on CONSTANT BUFFERS. Stage 4c.1
    // additionally requires the buffer to have been ever bound through a
    // vertex-pipeline stage — the IsVSBound tag is set on Buffer11Proxy by
    // *SetConstantBuffers when its stage is VS/GS/HS/DS.
    if (MapType != D3D11_MAP_WRITE_DISCARD &&
        MapType != D3D11_MAP_WRITE &&
        MapType != D3D11_MAP_WRITE_NO_OVERWRITE &&
        MapType != D3D11_MAP_READ_WRITE)
        return hr;
    // Under duplication every CB write must reach the sibling even when it needs no eye shift, or the right eye
    // reads stale constants.
    if (!buf || (!gInfo.DuplicateDraws && !buf->IsVSBound())) return hr;

    D3D11_BUFFER_DESC desc;
    buf->GetReal()->GetDesc(&desc);
    if ((desc.BindFlags & D3D11_BIND_CONSTANT_BUFFER) == 0) return hr;
    if (desc.ByteWidth == 0) return hr;

    ActiveMap am;
    am.resource    = pResource;
    am.subresource = Subresource;
    am.mapType     = MapType;
    am.mappedData  = pMappedResource->pData;
    am.byteWidth   = desc.ByteWidth;
    m_activeMaps.push_back(am);
    return hr;
}

// The registers the bound VS reads as projection matrices at the slots this buffer occupies. boundToVS: it is bound to
// that VS, so "none here" is final, as UE3 reuses one buffer per slot (BioShock Infinite's white flashes).
static void AnalyzerTargetsForCB(Device11Proxy* parent, void* boundVS,
                                 ID3D11Buffer* const* boundVSCBs,
                                 ID3D11Resource* pResource,
                                 std::vector<EyeShiftMatrix>& targets,
                                 bool& analyzerKnows, bool& boundToVS)
{
    analyzerKnows = false;
    boundToVS = false;
    if (!boundVS || !parent) return;
    const ShaderAnalysis11Result* info = parent->LookupShaderProjection(boundVS);
    if (!info || !info->parsed) return;
    analyzerKnows = true;
    // Only a shader with projection matrices somewhere is authoritative about a buffer it binds but reads none from: a
    // matrix-less (UI, post) shader knows nothing, and Far Cry 4 writes its shared view buffer under one.
    const bool vsHasMatrices = !info->projection.matrixData.cb.empty();
    for (UINT slot = 0; slot < Context11Proxy::kMaxCBs; ++slot)   // == kMaxVSCBSlots
    {
        if (static_cast<ID3D11Resource*>(boundVSCBs[slot]) != pResource) continue;
        if (vsHasMatrices) boundToVS = true;
        auto cbIt = info->projection.matrixData.cb.find(slot);
        if (cbIt == info->projection.matrixData.cb.end()) continue;
        for (const auto& pmd : cbIt->second)
        {
            if (pmd.incorrectProjection) continue;
            EyeShiftMatrix em = { pmd.matrixRegister, pmd.matrixIsTransposed, FALSE, FALSE };
            targets.push_back(em);
        }
    }
}

void STDMETHODCALLTYPE Context11Proxy::Unmap(ID3D11Resource* pResource, UINT Subresource)
{
    Buffer11Proxy*    buf = TryUnwrapBuffer(pResource);
    // Same Texture1D/3D gap as Map above.
    ID3D11Resource*   realRes = UnwrapResourceForEye(pResource, false);

    // Stage 4c: if Map captured this CB write, snapshot the bytes BEFORE
    // forwarding Unmap (which invalidates the mapped pointer), then push a
    // closure that re-maps + memcpy + applies the eye-shift heuristic +
    // unmaps at replay time. The closure only fires its modify-and-write
    // path on the right-eye pass (m_activeEye == Eye::Right) — the left
    // eye is the direct path the game just did, no replay needed.
    for (auto it = m_activeMaps.begin(); it != m_activeMaps.end(); ++it)
    {
        if (it->resource != pResource || it->subresource != Subresource) continue;
        if (it->mappedData && it->byteWidth)
        {
            std::vector<unsigned char> bytes(it->byteWidth);
            memcpy(bytes.data(), it->mappedData, it->byteWidth);
            UINT subres = it->subresource;
            D3D11_MAP mapType = it->mapType;
            ComRefHolder resRef(pResource);

            // Stage 4e.2: consult the analyzer for the currently bound VS.
            // If the bound shader has known projection matrices at any VS-CB
            // slot where this buffer is bound, build a targeted matrix list.
            // Empty list ⇒ fall back to the m[2][3]==1 / m[3][3]==0 heuristic.
            std::vector<EyeShiftMatrix> targets;
            // analyzerKnows: the bound VS was parsed, so empty targets mean "no projection matrix here", not "unknown".
            bool analyzerKnows = false, boundToVS = false;
            AnalyzerTargetsForCB(m_parent, m_boundVS, m_boundVSCBs, pResource,
                                 targets, analyzerKnows, boundToVS);

            // Profile-declared matrices on top of the analyzer's; a profile hit also counts as knowing this buffer.
            size_t beforeProfile = targets.size();
            AddProfileMatrixTargets(g_ProfileData.VSCRCData,
                                    BoundShaderCRC(m_parent, m_boundVS),
                                    m_boundVSCBs, kMaxVSCBSlots, pResource, targets);
            AddProfileMatrixTargets(g_ProfileData.PSCRCData,
                                    BoundShaderCRC(m_parent, m_boundPS),
                                    m_boundPSCBs, kMaxPSCBSlots, pResource, targets);
            if (targets.size() != beforeProfile) analyzerKnows = true;

            // Written for an overlay draw (back buffer, no depth test): no shift, and nothing learned from it.
            const bool overlayWrite = OverlayPass();
            if (overlayWrite) { targets.clear(); analyzerKnows = true; boundToVS = true; }

            // A buffer is often written before its consumer is bound, so the targets are remembered per buffer: without
            // them such writes reached the right eye unshifted, pinned at screen depth.
            if (Buffer11Proxy* lb = TryUnwrapBuffer(pResource))
            {
                lb->SetShadow(bytes.data(), bytes.size());
                if (!targets.empty())
                {
                    for (size_t ti = 0; ti < targets.size(); ++ti)
                    {
                        Buffer11Proxy::MatrixTarget mt = { targets[ti].matrixRegister,
                                                           targets[ti].matrixIsTransposed,
                                                           targets[ti].matrixIsInverse };
                        lb->LearnMatrix(mt);
                    }
                }
                // Learned registers fill that gap; they never override a parsed VS that has this buffer bound and says
                // "no matrix here".
                else if (!(analyzerKnows && boundToVS))
                {
                    // Under the draw patch each consumer gets its own matrices at its draw, so a pooled buffer's
                    // remembered forward targets stay off the plain constants written here; inverses still ride along.
                    const bool forwardWithheld = gInfo.PatchConstantsAtDraw;
                    const std::vector<Buffer11Proxy::MatrixTarget>& mem = lb->LearnedMatrices();
                    for (size_t ti = 0; ti < mem.size(); ++ti)
                    {
                        if (forwardWithheld && !mem[ti].inverse) { ++m_learnedWithheldThisFrame; continue; }
                        EyeShiftMatrix e = { mem[ti].reg, mem[ti].transposed, mem[ti].inverse, TRUE };
                        targets.push_back(e);
                        if (mem[ti].inverse) ++m_invLearnedThisFrame;
                    }
                    if (!targets.empty()) analyzerKnows = true;
                }
            }

            // Still unidentified: deferred lighting keeps its inverse view-projection in pixel-shader-only buffers no
            // shader analysis reaches, so the bytes are tested numerically.
            Buffer11Proxy* scanBp = TryUnwrapBuffer(pResource);
            wiz3d::StereoMatrixState::DetectedInverse hit;
            if (targets.empty() && !overlayWrite && gInfo.DetectInverseMatrix && bytes.size() >= 64 &&
                scanBp && scanBp->ShouldScanForInverse())
            {
                ++m_invScansThisFrame;
                if (DetectInverseInCB(bytes.data(), bytes.size(), hit, m_invBestMissThisFrame))
                {
                    ++m_invHitsThisFrame;
                    const BOOL itr = hit.transposed ? TRUE : FALSE;
                    EyeShiftMatrix em = { (DWORD)hit.reg, itr, TRUE, FALSE };
                    targets.push_back(em);
                    Buffer11Proxy::MatrixTarget mt = { (DWORD)hit.reg, itr, TRUE };
                    scanBp->LearnMatrix(mt);
                }
            }
            if (!targets.empty()) ++m_cbTargetedThisFrame;
            else                  ++m_cbUntargetedThisFrame;
            if (targets.empty() && gInfo.CollectDebugInformation)
            {
                const ShaderAnalysis11Result* vsInfo = m_boundVS ? m_parent->LookupShaderProjection(m_boundVS) : nullptr;
                const int why = !m_boundVS ? 0 : (!vsInfo || !vsInfo->parsed) ? 1
                              : vsInfo->projection.matrixData.cb.empty() ? 2 : boundToVS ? 4 : 3;
                ++m_untargetedWhy[why];
            }

            // Draw duplication keeps each eye's constants in its own buffer, so the patch lands in the sibling now; a
            // buffer no target names is copied through unshifted (no guessing).
            if (gInfo.DuplicateDraws)
            {
                Buffer11Proxy* bp = TryUnwrapBuffer(pResource);
                ID3D11Buffer* rightBuf = bp ? bp->GetRealRight() : nullptr;
                if (rightBuf)
                {
                    D3D11_MAPPED_SUBRESOURCE mapped = {};
                    if (SUCCEEDED(m_real->Map(rightBuf, subres, mapType, 0, &mapped))
                        && mapped.pData)
                    {
                        memcpy(mapped.pData, bytes.data(), bytes.size());
                        // Written while an x-flipped target is bound (a profile feature says so): the shift flips.
                        const float sign = EyeShiftSign();
                        float eyeShift = wiz3D_GetEffectiveEyeShift()
                                       * ProfileShaderMultiplier(m_parent, m_boundVS) * sign;
                        if (sign < 0.f && !targets.empty()) ++m_flippedWritesThisFrame;
                        const unsigned before = m_cbMatShiftedThisFrame, invBefore = m_invPatchedThisFrame;
                        if (!targets.empty())
                        {
                            ApplyTargetedEyeShiftToCB(
                                static_cast<unsigned char*>(mapped.pData),
                                bytes.size(), eyeShift, targets, true,
                                &m_cbMatShiftedThisFrame, &m_cbMatRejectedThisFrame,
                                &m_invPatchedThisFrame, pResource);
                            if (m_cbMatShiftedThisFrame != before) bp->MarkShifted(s_frameId);
                        }
                        const unsigned inv = m_invPatchedThisFrame - invBefore;
                        bp->NoteRightShift(m_cbMatShiftedThisFrame - before - inv, inv);
                        NoteRightConstants(bp);
                        m_real->Unmap(rightBuf, subres);
                    }
                }
                m_activeMaps.erase(it);
                break;
            }

            // Multiplier resolved now: the shader bound at write time is the
            // one the profile entry describes, not whatever replays later.
            const float profMult = ProfileShaderMultiplier(m_parent, m_boundVS);
            m_frameCommands.emplace_back(
                [this, resRef, subres, bytes, mapType, targets, profMult]()
                {
                    if (m_activeEye != Eye::Right) return;
                    auto* gameRes = static_cast<ID3D11Resource*>(resRef.p);
                    ID3D11Resource* real = UnwrapResourceForEye(gameRes, false);
                    D3D11_MAPPED_SUBRESOURCE mapped = {};
                    if (FAILED(m_real->Map(real, subres, mapType, 0, &mapped))
                        || !mapped.pData) return;
                    memcpy(mapped.pData, bytes.data(), bytes.size());
                    float eyeShift = wiz3D_GetEffectiveEyeShift() * profMult;
                    if (!targets.empty())
                    {
                        ApplyTargetedEyeShiftToCB(
                            static_cast<unsigned char*>(mapped.pData),
                            bytes.size(), eyeShift, targets, false,
                            &m_cbMatShiftedThisFrame, &m_cbMatRejectedThisFrame,
                            &m_invPatchedThisFrame, gameRes);
                    }
                    else
                    {
                        ApplyEyeShiftToCB(static_cast<unsigned char*>(mapped.pData),
                                          bytes.size(), eyeShift);
                    }
                    m_real->Unmap(real, subres);
                });
        }
        m_activeMaps.erase(it);
        break;
    }

    m_real->Unmap(realRes, Subresource);
}

void Context11Proxy::DoUpdateSubresource(
    ID3D11Resource* pDstResource, UINT DstSubresource, const D3D11_BOX* pDstBox,
    const void* pSrcData, UINT SrcRowPitch, UINT SrcDepthPitch)
{
    bool pickRight = (m_activeEye == Eye::Right);
    m_real->UpdateSubresource(UnwrapResourceForEye(pDstResource, pickRight),
                              DstSubresource, pDstBox,
                              pSrcData, SrcRowPitch, SrcDepthPitch);

    if (NeedsRightMirror(pDstResource, pickRight))
    {
        m_real->UpdateSubresource(UnwrapResourceForEye(pDstResource, true),
                                  DstSubresource, pDstBox,
                                  pSrcData, SrcRowPitch, SrcDepthPitch);
        if (auto* t = TryUnwrapTexture2D(pDstResource)) t->MarkRightWritten();
    }

    // Buffers have no right sibling in UnwrapResourceForEye, so a CB written here is mirrored explicitly: a whole-buffer
    // write gets the Map path's targets and inverse detection, a boxed partial write is copied as is.
    if (gInfo.DuplicateDraws)
    {
        Buffer11Proxy* bp = TryUnwrapBuffer(pDstResource);
        if (bp && bp->GetRealRight())
        {
            ++m_updCBWritesThisFrame;
            bool patched = false;
            const unsigned shiftedBefore = m_cbMatShiftedThisFrame, invBeforeUpd = m_invPatchedThisFrame;
            bp->SetRightFromShadow(false);
            if (!pDstBox && pSrcData)
            {
                D3D11_BUFFER_DESC bd = {};
                bp->GetRealRight()->GetDesc(&bd);
                std::vector<unsigned char> tmp(bd.ByteWidth);
                memcpy(tmp.data(), pSrcData, bd.ByteWidth);
                bp->SetShadow(tmp.data(), tmp.size());

                // Same attribution as the Map path: the bound VS first (UE3 commits CBs right before the draw that
                // consumes them), learned registers only where no parsed VS has this buffer bound.
                std::vector<EyeShiftMatrix> targets;
                bool analyzerKnows = false, boundToVS = false;
                AnalyzerTargetsForCB(m_parent, m_boundVS, m_boundVSCBs, pDstResource,
                                     targets, analyzerKnows, boundToVS);
                if (!targets.empty())
                {
                    for (size_t ti = 0; ti < targets.size(); ++ti)
                    {
                        Buffer11Proxy::MatrixTarget mt = { targets[ti].matrixRegister,
                                                           targets[ti].matrixIsTransposed,
                                                           targets[ti].matrixIsInverse };
                        bp->LearnMatrix(mt);
                    }
                }
                else if (!(analyzerKnows && boundToVS))
                {
                    // Same rule as the Map path: forward targets wait for the consumer's draw patch.
                    const bool forwardWithheld = gInfo.PatchConstantsAtDraw;
                    const std::vector<Buffer11Proxy::MatrixTarget>& mem = bp->LearnedMatrices();
                    for (size_t ti = 0; ti < mem.size(); ++ti)
                    {
                        if (forwardWithheld && !mem[ti].inverse) { ++m_learnedWithheldThisFrame; continue; }
                        EyeShiftMatrix e = { mem[ti].reg, mem[ti].transposed,
                                             mem[ti].inverse, TRUE };
                        targets.push_back(e);
                        if (mem[ti].inverse) ++m_invLearnedThisFrame;
                    }
                }
                if (targets.empty() && gInfo.DetectInverseMatrix && bd.ByteWidth >= 64 &&
                    bp->ShouldScanForInverse())
                {
                    ++m_invScansThisFrame;
                    wiz3d::StereoMatrixState::DetectedInverse hit;
                    if (DetectInverseInCB(tmp.data(), bd.ByteWidth, hit, m_invBestMissThisFrame))
                    {
                        ++m_invHitsThisFrame;
                        const BOOL itr = hit.transposed ? TRUE : FALSE;
                        EyeShiftMatrix em = { (DWORD)hit.reg, itr, TRUE, FALSE };
                        targets.push_back(em);
                        Buffer11Proxy::MatrixTarget mt = { (DWORD)hit.reg, itr, TRUE };
                        bp->LearnMatrix(mt);
                    }
                }
                if (!targets.empty())
                {
                    const unsigned before = m_cbMatShiftedThisFrame;
                    const float sign = EyeShiftSign();
                    float eyeShift = wiz3D_GetEffectiveEyeShift()
                                   * ProfileShaderMultiplier(m_parent, m_boundVS) * sign;
                    if (sign < 0.f) ++m_flippedWritesThisFrame;
                    ApplyTargetedEyeShiftToCB(tmp.data(), bd.ByteWidth, eyeShift,
                                              targets, true, &m_cbMatShiftedThisFrame,
                                              &m_cbMatRejectedThisFrame,
                                              &m_invPatchedThisFrame, pDstResource);
                    if (m_cbMatShiftedThisFrame != before) bp->MarkShifted(s_frameId);
                    m_real->UpdateSubresource(bp->GetRealRight(), DstSubresource,
                                              nullptr, tmp.data(), 0, 0);
                    ++m_updCBPatchedThisFrame;
                    patched = true;
                }
                const unsigned inv = m_invPatchedThisFrame - invBeforeUpd;
                bp->NoteRightShift(m_cbMatShiftedThisFrame - shiftedBefore - inv, inv);
            }
            if (!patched)
                m_real->UpdateSubresource(bp->GetRealRight(), DstSubresource, pDstBox,
                                          pSrcData, SrcRowPitch, SrcDepthPitch);
            if (pDstBox || !pSrcData) bp->ForgetShadow();
            NoteRightConstants(bp);
        }
    }
}

// True for the BC1..BC7 block-compressed format families. UpdateSubresource on
// BC textures sends 4x4-block rows, so byte counts are (height/4) * rowPitch
// not height * rowPitch.
static bool IsBCFormat(DXGI_FORMAT f)
{
    return (f >= DXGI_FORMAT_BC1_TYPELESS && f <= DXGI_FORMAT_BC5_SNORM)
        || (f >= DXGI_FORMAT_BC6H_TYPELESS && f <= DXGI_FORMAT_BC7_UNORM_SRGB);
}

// Compute how many source bytes UpdateSubresource will read so we can capture
// the right amount for replay. Returns 0 if uncomputable — caller should skip
// recording in that case. The previous Stage-4d attempt to record this call
// crashed Batman because it under-sized pSrcData; this version queries the
// destination resource's actual dimensions before copying.
static SIZE_T ComputeUpdateSubresourceCopyBytes(
    ID3D11Resource* res, UINT subresource, const D3D11_BOX* box,
    UINT srcRowPitch, UINT srcDepthPitch)
{
    if (!res) return 0;
    D3D11_RESOURCE_DIMENSION type;
    res->GetType(&type);
    switch (type)
    {
        case D3D11_RESOURCE_DIMENSION_TEXTURE2D:
        {
            D3D11_TEXTURE2D_DESC desc;
            static_cast<ID3D11Texture2D*>(res)->GetDesc(&desc);
            UINT mips = desc.MipLevels ? desc.MipLevels : 1;
            UINT mip = subresource % mips;
            UINT mipHeight = desc.Height >> mip;
            if (mipHeight == 0) mipHeight = 1;
            UINT height = box ? (box->bottom - box->top) : mipHeight;
            UINT rows = IsBCFormat(desc.Format) ? (height + 3) / 4 : height;
            return SIZE_T(rows) * srcRowPitch;
        }
        case D3D11_RESOURCE_DIMENSION_TEXTURE3D:
        {
            D3D11_TEXTURE3D_DESC desc;
            static_cast<ID3D11Texture3D*>(res)->GetDesc(&desc);
            UINT mips = desc.MipLevels ? desc.MipLevels : 1;
            UINT mip = subresource % mips;
            UINT mipDepth = desc.Depth >> mip;
            if (mipDepth == 0) mipDepth = 1;
            UINT depth = box ? (box->back - box->front) : mipDepth;
            return SIZE_T(depth) * srcDepthPitch;
        }
        case D3D11_RESOURCE_DIMENSION_TEXTURE1D:
            return srcRowPitch;
        case D3D11_RESOURCE_DIMENSION_BUFFER:
        {
            if (box) return box->right - box->left;
            D3D11_BUFFER_DESC desc;
            static_cast<ID3D11Buffer*>(res)->GetDesc(&desc);
            return desc.ByteWidth;
        }
        default:
            return 0;
    }
}

void STDMETHODCALLTYPE Context11Proxy::UpdateSubresource(
    ID3D11Resource* pDstResource, UINT DstSubresource, const D3D11_BOX* pDstBox,
    const void* pSrcData, UINT SrcRowPitch, UINT SrcDepthPitch)
{
    // Live call — applies to the active-eye sibling (left when called from
    // the game directly; right when called from replay).
    DoUpdateSubresource(pDstResource, DstSubresource, pDstBox,
                        pSrcData, SrcRowPitch, SrcDepthPitch);
    if (!m_presentHookActive) return;
    if (!pSrcData) return;

    // Only worth recording for replay when the destination is a stereo-doubled
    // Texture2D — that's the only resource kind where the right-eye sibling
    // diverges from the left. Texture1D/3D/Buffer have a single backing real,
    // so the live call already covered both eyes; recording would just bloat
    // memory and pay another driver-side upload cost for no visible benefit.
    Texture2D11Proxy* tex2d = TryUnwrapTexture2D(pDstResource);
    if (!tex2d || !tex2d->IsStereo()) return;

    // Use the wrapped resource's real (left) handle for sizing — both eye
    // siblings have identical descriptors by construction.
    SIZE_T bytes = ComputeUpdateSubresourceCopyBytes(
        static_cast<ID3D11Resource*>(tex2d->GetReal()),
        DstSubresource, pDstBox, SrcRowPitch, SrcDepthPitch);
    if (bytes == 0) return;

    std::vector<BYTE> data(bytes);
    memcpy(data.data(), pSrcData, bytes);

    ComRefHolder dstRef(pDstResource);
    bool hasBox = (pDstBox != nullptr);
    D3D11_BOX box = {};
    if (hasBox) box = *pDstBox;

    m_frameCommands.emplace_back(
        [this, dstRef, DstSubresource, hasBox, box, data, SrcRowPitch, SrcDepthPitch]()
        {
            DoUpdateSubresource(
                static_cast<ID3D11Resource*>(dstRef.p),
                DstSubresource,
                hasBox ? &box : nullptr,
                data.data(), SrcRowPitch, SrcDepthPitch);
        });
}

// ClearState resets every pipeline slot to defaults (shaders, SRVs, RTs,
// viewports, blend/depth/raster state, etc). Without recording, the right-eye
// replay sees subsequent state-setters applied on top of whatever state was
// in effect from the previous frame's replay tail — so eye-aware bindings
// can leak across the ClearState boundary. Closure has no captures; the real
// runtime handles the reset itself when replayed.
void STDMETHODCALLTYPE Context11Proxy::ClearState()
{
    m_real->ClearState();
    ResetEyeTracking();
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this]() { m_real->ClearState(); });
}

// The swapped-in state's bindings were never tracked; start from none rather than from the swapped-out state's.
void STDMETHODCALLTYPE Context11Proxy::SwapDeviceContextState(ID3DDeviceContextState* pState, ID3DDeviceContextState** ppPreviousState)
{
    if (!m_real1) return;
    m_real1->SwapDeviceContextState(pState, ppPreviousState);
    ResetEyeTracking();
}

// ExecuteCommandList submits a previously-built command list (typically from
// a deferred context). Without recording, deferred-context games (some
// DA2-era engines) submit entire batches of state + draws via this single
// call and the right-eye replay never sees them. Command lists are
// idempotent at the API level; replaying the same call is safe.
void STDMETHODCALLTYPE Context11Proxy::ExecuteCommandList(
    ID3D11CommandList* pCommandList, BOOL RestoreContextState)
{
    // A command list's bindings were baked on a deferred context we never wrapped (left-eye resources), so its draws
    // never reach the right eye: duplication issues it once, and the replay re-runs it into the left eye's resources.
    ++m_cmdListsThisFrame;
    m_real->ExecuteCommandList(pCommandList, RestoreContextState);
    // Not restoring leaves the context in its default state, as ClearState does.
    if (!RestoreContextState) ResetEyeTracking();
    if (!m_presentHookActive) return;
    ComRefHolder cmdRef(pCommandList);
    m_frameCommands.emplace_back(
        [this, cmdRef, RestoreContextState]()
        {
            m_real->ExecuteCommandList(
                static_cast<ID3D11CommandList*>(cmdRef.p), RestoreContextState);
        });
}

void Context11Proxy::DoResolveSubresource(
    ID3D11Resource* pDstResource, UINT DstSubresource,
    ID3D11Resource* pSrcResource, UINT SrcSubresource, DXGI_FORMAT Format)
{
    bool pickRight = (m_activeEye == Eye::Right);
    m_real->ResolveSubresource(
        UnwrapResourceForEye(pDstResource, pickRight), DstSubresource,
        UnwrapResourceForEye(pSrcResource, pickRight), SrcSubresource, Format);
    if (ID3D11Resource* rd = RightMirrorTarget(pDstResource, pickRight))
    {
        m_real->ResolveSubresource(rd, DstSubresource, RightEyeResource(pSrcResource), SrcSubresource, Format);
        if (auto* t = TryUnwrapTexture2D(pDstResource)) t->MarkRightWritten();
    }
}

void STDMETHODCALLTYPE Context11Proxy::ResolveSubresource(
    ID3D11Resource* pDstResource, UINT DstSubresource,
    ID3D11Resource* pSrcResource, UINT SrcSubresource, DXGI_FORMAT Format)
{
    DoResolveSubresource(pDstResource, DstSubresource,
                         pSrcResource, SrcSubresource, Format);
    if (!m_presentHookActive) return;
    ComRefHolder dstRef(pDstResource);
    ComRefHolder srcRef(pSrcResource);
    m_frameCommands.emplace_back(
        [this, dstRef, DstSubresource, srcRef, SrcSubresource, Format]()
        {
            DoResolveSubresource(
                static_cast<ID3D11Resource*>(dstRef.p), DstSubresource,
                static_cast<ID3D11Resource*>(srcRef.p), SrcSubresource, Format);
        });
}

void Context11Proxy::DoClearRenderTargetView(
    ID3D11RenderTargetView* pRenderTargetView, const FLOAT ColorRGBA[4])
{
    bool pickRight = (m_activeEye == Eye::Right);
    RTV11Proxy* rtv = TryUnwrapRTV(pRenderTargetView);
    ID3D11RenderTargetView* real = pRenderTargetView;
    if (rtv)
    {
        ID3D11RenderTargetView* right = rtv->GetRealRight();
        real = (pickRight && right) ? right : rtv->GetReal();
    }
    if (FrameTraceActive())
    {
        FrameTrace("  Clear eye=%c rtv={proxy=%p stereo=%d sibling=%c real=%p} color=[%.2f %.2f %.2f %.2f]\n",
                   pickRight ? 'R' : 'L',
                   rtv, rtv ? rtv->GetRealRight() != nullptr : 0,
                   (pickRight && rtv && rtv->GetRealRight()) ? 'R' : 'L',
                   real,
                   ColorRGBA ? ColorRGBA[0] : 0.f, ColorRGBA ? ColorRGBA[1] : 0.f,
                   ColorRGBA ? ColorRGBA[2] : 0.f, ColorRGBA ? ColorRGBA[3] : 0.f);
    }
    m_real->ClearRenderTargetView(real, ColorRGBA);
    // No replay pass to clear the sibling later, so do it now.
    if (gInfo.DuplicateDraws && rtv && rtv->GetRealRight() && !pickRight)
        m_real->ClearRenderTargetView(rtv->GetRealRight(), ColorRGBA);
}

void STDMETHODCALLTYPE Context11Proxy::ClearRenderTargetView(
    ID3D11RenderTargetView* pRenderTargetView, const FLOAT ColorRGBA[4])
{
    DoClearRenderTargetView(pRenderTargetView, ColorRGBA);
    if (!m_presentHookActive) return;
    ComRefHolder rtvRef(pRenderTargetView);
    FLOAT color[4] = { 0, 0, 0, 0 };
    if (ColorRGBA)
    {
        color[0] = ColorRGBA[0]; color[1] = ColorRGBA[1];
        color[2] = ColorRGBA[2]; color[3] = ColorRGBA[3];
    }
    m_frameCommands.emplace_back(
        [this, rtvRef, color]()
        {
            DoClearRenderTargetView(
                static_cast<ID3D11RenderTargetView*>(rtvRef.p), color);
        });
}

void Context11Proxy::DoClearDepthStencilView(
    ID3D11DepthStencilView* pDepthStencilView, UINT ClearFlags, FLOAT Depth, UINT8 Stencil)
{
    bool pickRight = (m_activeEye == Eye::Right);
    DSV11Proxy* dsv = TryUnwrapDSV(pDepthStencilView);
    ID3D11DepthStencilView* real = pDepthStencilView;
    if (dsv)
    {
        ID3D11DepthStencilView* right = dsv->GetRealRight();
        real = (pickRight && right) ? right : dsv->GetReal();
    }
    m_real->ClearDepthStencilView(real, ClearFlags, Depth, Stencil);
    // No replay pass to clear the sibling later, so do it now.
    if (gInfo.DuplicateDraws && dsv && dsv->GetRealRight() && !pickRight)
        m_real->ClearDepthStencilView(dsv->GetRealRight(), ClearFlags, Depth, Stencil);
}

void STDMETHODCALLTYPE Context11Proxy::ClearDepthStencilView(
    ID3D11DepthStencilView* pDepthStencilView, UINT ClearFlags, FLOAT Depth, UINT8 Stencil)
{
    DoClearDepthStencilView(pDepthStencilView, ClearFlags, Depth, Stencil);
    if (!m_presentHookActive) return;
    ComRefHolder dsvRef(pDepthStencilView);
    m_frameCommands.emplace_back(
        [this, dsvRef, ClearFlags, Depth, Stencil]()
        {
            DoClearDepthStencilView(
                static_cast<ID3D11DepthStencilView*>(dsvRef.p),
                ClearFlags, Depth, Stencil);
        });
}

// Stage 4b.4 Group C: remaining state setters. Same record-and-replay pattern
// as the macro-generated groups above, but each has a slightly different
// argument shape so they're written out individually. All gated on
// m_presentHookActive so recording is bounded.

void STDMETHODCALLTYPE Context11Proxy::IASetInputLayout(ID3D11InputLayout* pInputLayout)
{
    m_real->IASetInputLayout(pInputLayout);
    if (!m_presentHookActive) return;
    ComRefHolder layoutRef(pInputLayout);
    m_frameCommands.emplace_back(
        [this, layoutRef]()
        {
            m_real->IASetInputLayout(static_cast<ID3D11InputLayout*>(layoutRef.p));
        });
}

void STDMETHODCALLTYPE Context11Proxy::IASetVertexBuffers(
    UINT StartSlot, UINT NumBuffers, ID3D11Buffer* const* ppVertexBuffers,
    const UINT* pStrides, const UINT* pOffsets)
{
    // Stage 3c.1: unwrap each VB before forwarding to D3D11.
    ID3D11Buffer* rawVBs[kMaxVBs] = { 0 };
    UINT cap = NumBuffers <= kMaxVBs ? NumBuffers : kMaxVBs;
    for (UINT i = 0; i < cap; ++i)
        rawVBs[i] = ppVertexBuffers ? UnwrapBuf(ppVertexBuffers[i]) : nullptr;
    m_real->IASetVertexBuffers(StartSlot, NumBuffers,
        ppVertexBuffers ? rawVBs : nullptr, pStrides, pOffsets);
    if (!m_presentHookActive) return;
    std::vector<ComRefHolder> bufRefs;
    bufRefs.reserve(NumBuffers);
    for (UINT i = 0; i < NumBuffers; ++i)
        bufRefs.emplace_back(ppVertexBuffers ? ppVertexBuffers[i] : nullptr);
    std::vector<UINT> strides;
    if (pStrides) strides.assign(pStrides, pStrides + NumBuffers);
    std::vector<UINT> offsets;
    if (pOffsets) offsets.assign(pOffsets, pOffsets + NumBuffers);
    m_frameCommands.emplace_back(
        [this, StartSlot, NumBuffers, bufRefs, strides, offsets]()
        {
            ID3D11Buffer* raw[kMaxVBs] = { 0 };
            UINT replayCap = NumBuffers <= kMaxVBs ? NumBuffers : kMaxVBs;
            for (UINT i = 0; i < replayCap; ++i)
                raw[i] = UnwrapBuf(static_cast<ID3D11Buffer*>(bufRefs[i].p));
            m_real->IASetVertexBuffers(
                StartSlot, NumBuffers, raw,
                strides.empty() ? nullptr : strides.data(),
                offsets.empty() ? nullptr : offsets.data());
        });
}

void STDMETHODCALLTYPE Context11Proxy::IASetIndexBuffer(
    ID3D11Buffer* pIndexBuffer, DXGI_FORMAT Format, UINT Offset)
{
    m_real->IASetIndexBuffer(UnwrapBuf(pIndexBuffer), Format, Offset);
    if (!m_presentHookActive) return;
    ComRefHolder bufRef(pIndexBuffer);
    m_frameCommands.emplace_back(
        [this, bufRef, Format, Offset]()
        {
            m_real->IASetIndexBuffer(
                UnwrapBuf(static_cast<ID3D11Buffer*>(bufRef.p)), Format, Offset);
        });
}

void STDMETHODCALLTYPE Context11Proxy::IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY Topology)
{
    m_real->IASetPrimitiveTopology(Topology);
    if (!m_presentHookActive) return;
    m_frameCommands.emplace_back(
        [this, Topology]()
        {
            m_real->IASetPrimitiveTopology(Topology);
        });
}

void STDMETHODCALLTYPE Context11Proxy::RSSetState(ID3D11RasterizerState* pRasterizerState)
{
    m_real->RSSetState(pRasterizerState);
    if (!m_presentHookActive) return;
    ComRefHolder stateRef(pRasterizerState);
    m_frameCommands.emplace_back(
        [this, stateRef]()
        {
            m_real->RSSetState(static_cast<ID3D11RasterizerState*>(stateRef.p));
        });
}

void STDMETHODCALLTYPE Context11Proxy::RSSetScissorRects(UINT NumRects, const D3D11_RECT* pRects)
{
    m_real->RSSetScissorRects(NumRects, pRects);
    if (!m_presentHookActive) return;
    std::vector<D3D11_RECT> rects;
    if (pRects) rects.assign(pRects, pRects + NumRects);
    m_frameCommands.emplace_back(
        [this, NumRects, rects]()
        {
            m_real->RSSetScissorRects(
                NumRects, rects.empty() ? nullptr : rects.data());
        });
}

void STDMETHODCALLTYPE Context11Proxy::OMSetBlendState(
    ID3D11BlendState* pBlendState, const FLOAT BlendFactor[4], UINT SampleMask)
{
    m_real->OMSetBlendState(pBlendState, BlendFactor, SampleMask);
    if (!m_presentHookActive) return;
    ComRefHolder stateRef(pBlendState);
    FLOAT factor[4] = { 0, 0, 0, 0 };
    bool hasFactor = (BlendFactor != nullptr);
    if (hasFactor)
    {
        factor[0] = BlendFactor[0]; factor[1] = BlendFactor[1];
        factor[2] = BlendFactor[2]; factor[3] = BlendFactor[3];
    }
    m_frameCommands.emplace_back(
        [this, stateRef, factor, hasFactor, SampleMask]()
        {
            m_real->OMSetBlendState(
                static_cast<ID3D11BlendState*>(stateRef.p),
                hasFactor ? factor : nullptr, SampleMask);
        });
}

void STDMETHODCALLTYPE Context11Proxy::OMSetDepthStencilState(
    ID3D11DepthStencilState* pDepthStencilState, UINT StencilRef)
{
    m_real->OMSetDepthStencilState(pDepthStencilState, StencilRef);
    // The default state (null) tests depth.
    m_depthTestOn = true;
    if (pDepthStencilState)
    {
        D3D11_DEPTH_STENCIL_DESC dd = {};
        pDepthStencilState->GetDesc(&dd);
        m_depthTestOn = dd.DepthEnable != FALSE;
    }
    if (!m_presentHookActive) return;
    ComRefHolder stateRef(pDepthStencilState);
    m_frameCommands.emplace_back(
        [this, stateRef, StencilRef]()
        {
            m_real->OMSetDepthStencilState(
                static_cast<ID3D11DepthStencilState*>(stateRef.p), StencilRef);
        });
}

void STDMETHODCALLTYPE Context11Proxy::SOSetTargets(
    UINT NumBuffers, ID3D11Buffer* const* ppSOTargets, const UINT* pOffsets)
{
    // Stage 3c.1: unwrap each SO target before forwarding to D3D11.
    ID3D11Buffer* rawSOs[kMaxSOBuffers] = { 0 };
    UINT cap = NumBuffers <= kMaxSOBuffers ? NumBuffers : kMaxSOBuffers;
    for (UINT i = 0; i < cap; ++i)
        rawSOs[i] = ppSOTargets ? UnwrapBuf(ppSOTargets[i]) : nullptr;
    m_real->SOSetTargets(NumBuffers,
        ppSOTargets ? rawSOs : nullptr, pOffsets);
    if (!m_presentHookActive) return;
    std::vector<ComRefHolder> bufRefs;
    bufRefs.reserve(NumBuffers);
    for (UINT i = 0; i < NumBuffers; ++i)
        bufRefs.emplace_back(ppSOTargets ? ppSOTargets[i] : nullptr);
    std::vector<UINT> offsets;
    if (pOffsets) offsets.assign(pOffsets, pOffsets + NumBuffers);
    m_frameCommands.emplace_back(
        [this, NumBuffers, bufRefs, offsets]()
        {
            ID3D11Buffer* raw[kMaxSOBuffers] = { 0 };
            UINT replayCap = NumBuffers <= kMaxSOBuffers ? NumBuffers : kMaxSOBuffers;
            for (UINT i = 0; i < replayCap; ++i)
                raw[i] = UnwrapBuf(static_cast<ID3D11Buffer*>(bufRefs[i].p));
            m_real->SOSetTargets(
                NumBuffers, raw,
                offsets.empty() ? nullptr : offsets.data());
        });
}

void STDMETHODCALLTYPE Context11Proxy::SetPredication(
    ID3D11Predicate* pPredicate, BOOL PredicateValue)
{
    m_real->SetPredication(pPredicate, PredicateValue);
    if (!m_presentHookActive) return;
    ComRefHolder predRef(pPredicate);
    m_frameCommands.emplace_back(
        [this, predRef, PredicateValue]()
        {
            m_real->SetPredication(
                static_cast<ID3D11Predicate*>(predRef.p), PredicateValue);
        });
}

void STDMETHODCALLTYPE Context11Proxy::CSSetUnorderedAccessViews(
    UINT StartSlot, UINT NumUAVs,
    ID3D11UnorderedAccessView* const* ppUnorderedAccessViews,
    const UINT* pUAVInitialCounts)
{
    TrackCSUAVs(StartSlot, NumUAVs, ppUnorderedAccessViews, pUAVInitialCounts);
    // Stage 3c.2: unwrap UAVs eye-aware before forwarding.
    ID3D11UnorderedAccessView* rawSet[kMaxUAVs] = { 0 };
    UINT setCap = NumUAVs <= kMaxUAVs ? NumUAVs : kMaxUAVs;
    bool pickRight = (m_activeEye == Eye::Right);
    for (UINT i = 0; i < setCap; ++i)
        rawSet[i] = UnwrapUAVForEye(ppUnorderedAccessViews ? ppUnorderedAccessViews[i]
                                                           : nullptr, pickRight);
    m_real->CSSetUnorderedAccessViews(StartSlot, NumUAVs,
        ppUnorderedAccessViews ? rawSet : nullptr, pUAVInitialCounts);
    if (!m_presentHookActive) return;
    std::vector<ComRefHolder> uavRefs;
    uavRefs.reserve(NumUAVs);
    for (UINT i = 0; i < NumUAVs; ++i)
        uavRefs.emplace_back(ppUnorderedAccessViews ? ppUnorderedAccessViews[i] : nullptr);
    std::vector<UINT> initialCounts;
    if (pUAVInitialCounts)
        initialCounts.assign(pUAVInitialCounts, pUAVInitialCounts + NumUAVs);
    m_frameCommands.emplace_back(
        [this, StartSlot, NumUAVs, uavRefs, initialCounts]()
        {
            ID3D11UnorderedAccessView* raw[kMaxUAVs] = { 0 };
            UINT cap = NumUAVs <= kMaxUAVs ? NumUAVs : kMaxUAVs;
            bool pr = (m_activeEye == Eye::Right);
            for (UINT i = 0; i < cap; ++i)
                raw[i] = UnwrapUAVForEye(
                    static_cast<ID3D11UnorderedAccessView*>(uavRefs[i].p), pr);
            m_real->CSSetUnorderedAccessViews(
                StartSlot, NumUAVs, raw,
                initialCounts.empty() ? nullptr : initialCounts.data());
        });
}

void STDMETHODCALLTYPE Context11Proxy::GetDevice(ID3D11Device** ppDevice)
{
    // COM identity: GetDevice must return the wrapped device, not the real
    // one — otherwise a game that round-trips through GetDevice ends up
    // bypassing our wrapper for subsequent resource creation.
    if (!ppDevice) return;
    if (m_parent)
    {
        // Device11Proxy publicly inherits from ID3D11Device — real upcast,
        // static_cast keeps /W4 + warnings-as-errors happy.
        *ppDevice = static_cast<ID3D11Device*>(m_parent);
        m_parent->AddRef();
        return;
    }
    m_real->GetDevice(ppDevice);
}

} // namespace wiz3d
