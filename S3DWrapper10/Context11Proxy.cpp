/* wiz3D - ID3D11DeviceContext proxy: each stereo draw and dispatch runs again for the right eye. Split as DX9's renderer:
 * here lifetime, COM identity, shared state, profile features; _Main, _Modified, _ModifiedMain, _RHW, _Utils the rest. */

#include "StdAfx.h"
#include "Context11Proxy-inl.h"

#pragma comment(lib, "d3dcompiler.lib")

// Stage 3c.1: lightweight inline unwrap for ID3D11Buffer*. Used at every
// method boundary that hands a buffer to the real D3D11 runtime — passing
// our Buffer11Proxy directly would crash the runtime because it doesn't
// understand our vtable layout past ID3D11Buffer methods. Returns the raw
// buffer (or the original pointer if not ours, including nullptr).
ID3D11Buffer* UnwrapBuf(ID3D11Buffer* p)
{
    if (!p) return nullptr;
    if (auto* bp = wiz3d::TryUnwrapBuffer(static_cast<ID3D11Resource*>(p)))
        return bp->GetReal();
    return p;
}

// Stage 3c.1: unwrap ID3D11Resource* for the eye-aware Do* helpers. Tries
// Texture2D11Proxy first (which has eye-stereo siblings) then Buffer11Proxy
// (no stereo doubling, single real). 3c.2 adds Texture1D/Texture3D — both
// passthrough (no stereo) but still need unwrap so the real runtime gets
// the real pointer.
ID3D11Resource* UnwrapResourceForEye(ID3D11Resource* p, bool pickRight)
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
ID3D11ShaderResourceView* UnwrapSRVForEye(ID3D11ShaderResourceView* p, bool pickRight)
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

ID3D11UnorderedAccessView* UnwrapUAVForEye(ID3D11UnorderedAccessView* p, bool pickRight)
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
    if (m_real && m_parent && gInfo.DuplicateDraws && BatchRightEyeOn())
    {
        m_recorder = new RightRecorder11(m_parent->GetReal());
        m_mirror = m_recorder;
        m_mirror1 = m_recorder;
        DDILog("BatchRightEye: right-eye recorder %p\n", m_recorder);
    }
    if (m_real && m_parent && gInfo.CollectDebugInformation) m_gpuTimer = new GpuTimer11(m_parent->GetReal());
}

Context11Proxy::~Context11Proxy()
{
    // Closures hold no AddRef'd state in 4b.1 (Draw/DrawIndexed capture only
    // POD args), so a plain clear is safe. Later stages that record state-
    // setting calls with captured COM pointers will release them in
    // ClearFrameCommands; the dtor will route there.
    ClearFrameCommands();
    ReleaseUavTwins(~0u);
    for (auto& kv : m_clipSpaceGS) if (kv.second) kv.second->Release();
    for (auto& kv : m_clipSpaceRayGS) if (kv.second) kv.second->Release();
    for (auto& kv : m_ringStaging) if (kv.second) kv.second->Release();
    for (RingReadback& r : m_ringReadbacks) r.b->Release();
    for (auto& kv : m_clipSpaceDepth)
    {
        if (kv.second.tex) kv.second.tex->Release();
        if (kv.second.srv) kv.second.srv->Release();
        kv.first->Release();
    }
    if (m_clipSpaceCB) m_clipSpaceCB->Release();
    for (auto& kv : m_twins) if (kv.second.shader) kv.second.shader->Release();
    if (m_twinCB) m_twinCB->Release();
    ResetEyeTracking();
    if (m_recorder) { m_recorder->Release(); m_recorder = nullptr; }
    delete m_gpuTimer;
    m_gpuTimer = nullptr;
    if (m_real3) { m_real3->Release(); m_real3 = nullptr; }
    if (m_real2) { m_real2->Release(); m_real2 = nullptr; }
    if (m_real1) { m_real1->Release(); m_real1 = nullptr; }
}

void Context11Proxy::ClearFrameCommands()
{
    m_frameCommands.clear();
}

// The scene camera's p00 a profile feature publishes (stereo::PublishCameraP00), 0 = unknown.
float s_camP00 = 0.f;
bool  s_uiProjectionParallax = false;   // the UiProjectionParallax feature is on (read by the static shift code)
unsigned s_scaledByCamera = 0;   // scaled camera matrices shifted with s_camP00, this frame
unsigned s_lightOnlyKept = 0;    // light matrices (read only by shadow passes) left unshifted, this frame
// Report only: the clean camera matrices of a frame by p00 (thousandths), so the frame report names the scene camera
// and shows how often the last one cached is another (weapon, UI).
std::map<int, std::pair<unsigned, float>> s_cleanP00Votes;
float    s_sceneP00 = 0.f;
// The scene projection's z row (z_ndc = m22 + m23 / w_view) from the frame's last clean view-projection, so a
// vertex given in NDC (ClipSpaceSprites) gets back its view depth.
float    s_sceneZRow[2] = { 0.f, 0.f };
unsigned s_sceneZRowFrame = 0;
unsigned s_sceneP00Votes = 0, s_cleanP00Total = 0;

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
    if (!m_rightRecording)
    {
        for (StageIdx st : geo) if (m_cbSlots[st].high) BindStageCBs(st, false);
        return;
    }
    // BatchRightEye: the right siblings take the bytes the left draw read (the game's), in the recording; a stage holding
    // bytes we lack binds the left buffers, whose next write replays the recording first.
    auto ringRange = [](const CBEyeSlots& s, UINT slot, Buffer11Proxy* b, size_t& off, size_t& n)
    {
        const size_t size = b->RingSize();
        off = s.num[slot] ? (size_t)s.first[slot] * 16 : 0;
        n = 0;
        if (off >= size) return;
        n = s.num[slot] ? (size_t)s.num[slot] * 16 : 65536;
        if (off + n > size) n = size - off;
    };
    // The pixel or compute stage reading these bytes too keeps the right eye's (shifted) version of them.
    auto boundOutsideGeometry = [&](Buffer11Proxy* b, size_t off, size_t n)
    {
        for (StageIdx st : { ST_PS, ST_CS })
            for (UINT slot = 0; slot < m_cbSlots[st].high; ++slot)
            {
                if (m_cbSlots[st].proxy[slot] != b) continue;
                if (!b->IsRing()) return true;
                size_t o2 = 0, n2 = 0;
                ringRange(m_cbSlots[st], slot, b, o2, n2);
                if (n2 && o2 < off + n && off < o2 + n2) return true;
            }
        return false;
    };
    for (StageIdx st : geo)
    {
        CBEyeSlots& s = m_cbSlots[st];
        if (!s.high) continue;
        bool known = true;
        for (UINT slot = 0; known && slot < s.high; ++slot)
        {
            Buffer11Proxy* b = s.proxy[slot];
            if (!b) continue;
            size_t off = 0, n = 0;
            if (!b->GetRealRight()) known = false;
            else if (b->IsRing()) { ringRange(s, slot, b, off, n); known = n && !b->AnyForeign(off, n) && !boundOutsideGeometry(b, off, n); }
            else known = !b->Shadow().empty() && !boundOutsideGeometry(b, 0, 0);
        }
        if (!known)
        {
            BindStageCBs(st, false);
            for (UINT slot = 0; slot < s.high; ++slot)
            {
                Buffer11Proxy* b = s.proxy[slot];
                if (!b) continue;
                PendingLeft p = { b, 0, 0 };
                if (b->IsRing()) ringRange(s, slot, b, p.off, p.n);
                bool have = false;
                for (const PendingLeft& q : m_pendingLeftCBs) have |= (q.b == p.b && q.off == p.off && q.n == p.n);
                if (!have) m_pendingLeftCBs.push_back(p);
            }
            continue;
        }
        for (UINT slot = 0; slot < s.high; ++slot)
        {
            Buffer11Proxy* b = s.proxy[slot];
            if (!b) continue;
            if (!b->IsRing()) { if (!b->RightIsShadow()) RestoreRightCB(b); continue; }
            size_t off = 0, n = 0;
            ringRange(s, slot, b, off, n);
            if (RingKeepsShifts()) MergeRightRing(b, off, b->RingData() + off, n, 0.f);   // our last right write, as written
            WriteRingRange(b->GetRealRight(), off, b->RingData() + off, n);
            // Every slot holding this buffer syncs it again at the next right-eye draw.
            for (int any = 0; any < ST_COUNT; ++any)
                for (UINT k = 0; k < m_cbSlots[any].high; ++k)
                    if (m_cbSlots[any].proxy[k] == b) m_cbSlots[any].syncOff[1][k] = ~0u;
        }
        BindStageCBs(st, true);
    }
}

// A game write to a buffer a recorded draw reads in place; a ring's game writes reach only our shadow, its real buffer
// changes through WriteRingRange (PendingLeftRange), a DISCARD or a copy.
bool Context11Proxy::PendingLeftCB(Buffer11Proxy* b) const
{
    if (!b || b->IsRing()) return false;
    for (const PendingLeft& p : m_pendingLeftCBs) if (p.b == b) return true;
    return false;
}

ID3D11Buffer* Context11Proxy::GSConstantBufferNow(UINT slot) const
{
    const CBEyeSlots& s = m_cbSlots[ST_GS];
    if (slot >= s.high || slot >= kMaxCBs) return nullptr;
    return (m_activeEye == Eye::Right && s.anyStereo) ? s.right[slot] : s.left[slot];
}

ID3D11ShaderResourceView* Context11Proxy::GSResourceNow(UINT slot) const
{
    const SRVEyeSlots& s = m_srvSlots[ST_GS];
    if (slot >= kMaxSRVs) return nullptr;
    return (m_activeEye == Eye::Right && s.anyStereo) ? s.right[slot] : s.left[slot];
}

// Vertex and index buffers are read only through the input assembler, so one the recording never bound is unread by it.
bool Context11Proxy::IAOnlyUnread(ID3D11Buffer* real) const
{
    if (!real || !m_recorder || !m_rightPending) return false;
    D3D11_BUFFER_DESC d = {};
    real->GetDesc(&d);
    const UINT ia = D3D11_BIND_VERTEX_BUFFER | D3D11_BIND_INDEX_BUFFER;
    if (!(d.BindFlags & ia) || (d.BindFlags & ~ia)) return false;
    return !m_recorder->ReadsIABuffer(real);
}

bool Context11Proxy::PendingLeftRange(ID3D11Buffer* real, size_t off, size_t n) const
{
    for (const PendingLeft& p : m_pendingLeftCBs)
        if (p.b->GetReal() == real && (!p.n || (p.off < off + n && off < p.off + p.n))) return true;
    return false;
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

bool Context11Proxy::RingKeepsShifts() const
{
    for (const auto& f : m_features) if (f->RingKeepsShifts()) return true;
    return false;
}


bool Context11Proxy::KeepBareProjections() const
{
    for (const auto& f : m_features) if (f->KeepBareProjections()) return true;
    return false;
}

bool Context11Proxy::WholeBufferRanges() const
{
    for (const auto& f : m_features) if (f->WholeBufferRanges()) return true;
    return false;
}

bool Context11Proxy::ViewMatrices() const
{
    for (const auto& f : m_features) if (f->ViewMatrices()) return true;
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
unsigned s_frameId = 0;

// The matrix detector's state: the camera's x scale and the ring of forward view-projections (Shared/StereoMatrix.h).
wiz3d::StereoMatrixState s_mat;

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
