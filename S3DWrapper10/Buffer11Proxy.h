/* wiz3D - ID3D11Buffer proxy (Option B Stage 3c.1)
 *
 * Passthrough wrap of ID3D11Buffer. Carries a sticky "ever bound to a
 * vertex-pipeline stage" flag the Stage 4c projection-matrix heuristic
 * uses to filter PS-only / CS-only CBs out of the eye-shift modify path.
 *
 * Buffers are not stereo-doubled — there's no analogue to a right-eye RT
 * for a buffer (Stage 4c modifies CB bytes per eye via Map/Unmap instead).
 * So unlike Texture2D11Proxy this is just identity + metadata, single real
 * pointer.
 */

#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <vector>
#include <bitset>

namespace wiz3d
{

class Device11Proxy;

class Buffer11Proxy : public ID3D11Buffer
{
public:
    Buffer11Proxy(ID3D11Buffer* real, Device11Proxy* parent);
    virtual ~Buffer11Proxy();

    // IUnknown
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObj) override;
    ULONG   STDMETHODCALLTYPE AddRef() override                  { return InterlockedIncrement(&m_refs); }
    ULONG   STDMETHODCALLTYPE Release() override;

    // ID3D11DeviceChild
    void    STDMETHODCALLTYPE GetDevice(ID3D11Device** ppDevice) override;
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* pDataSize, void* pData) override                  { return m_real->GetPrivateData(guid, pDataSize, pData); }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT DataSize, const void* pData) override              { return m_real->SetPrivateData(guid, DataSize, pData); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* pData) override                { return m_real->SetPrivateDataInterface(guid, pData); }

    // ID3D11Resource
    void    STDMETHODCALLTYPE GetType(D3D11_RESOURCE_DIMENSION* pResourceDimension) override                       { m_real->GetType(pResourceDimension); }
    void    STDMETHODCALLTYPE SetEvictionPriority(UINT EvictionPriority) override                                  { m_real->SetEvictionPriority(EvictionPriority); }
    UINT    STDMETHODCALLTYPE GetEvictionPriority() override                                                       { return m_real->GetEvictionPriority(); }

    // ID3D11Buffer
    void    STDMETHODCALLTYPE GetDesc(D3D11_BUFFER_DESC* pDesc) override                                           { m_real->GetDesc(pDesc); }

    // wiz3D accessors
    ID3D11Buffer*  GetReal()       const { return m_real;   }
    Device11Proxy* GetParent()     const { return m_parent; }

    // Right-eye CB sibling, so draw duplication switches eyes by rebinding
    // rather than rewriting a shared buffer. Null unless gInfo.DuplicateDraws.
    ID3D11Buffer*  GetRealRight()  const { return m_realRight; }
    void           SetRealRight(ID3D11Buffer* p) { m_realRight = p; }
    // The right copy a pass reads (Context11Proxy::EditRightCBs): 0 = the sibling above, 1 = a second sibling for passes
    // that see the right camera with its origin moved, made on first use with the same desc.
    ID3D11Buffer*  RightCopy(int variant);

    // Stage 4c.1: sticky tag — set when *SetConstantBuffers fires on a
    // vertex-pipeline stage (VS / GS / HS / DS). Map/Unmap consults this
    // to decide whether to record + eye-shift the captured CB bytes.
    // Once set, never cleared — the buffer is eligible for stereo math
    // for its lifetime. m_evictionPriority field is the closest existing
    // metadata slot, but a dedicated bool keeps semantics clean.
    bool IsVSBound() const { return m_vsBound; }
    void TagVSBound()      { m_vsBound = true; }

    // When a matrix in this buffer was last eye-shifted, for the frame report's uncorrected-draw counts.
    bool EverShifted() const { return m_everShifted; }
    void MarkShifted(unsigned frame) { m_everShifted = true; m_shiftedFrame = frame; }
    unsigned ShiftedFrame() const { return m_shiftedFrame; }

    // Matrix registers learned for this buffer: a game often writes a CB before binding its consumer, and a register
    // holding a view-projection holds one whoever is bound. 24, as UE3's view buffer holds a dozen projective matrices.
    struct MatrixTarget { DWORD reg; BOOL transposed; BOOL inverse; };
    const std::vector<MatrixTarget>& LearnedMatrices() const { return m_learned; }
    void LearnMatrix(const MatrixTarget& t)
    {
        for (size_t i = 0; i < m_learned.size(); ++i)
            if (m_learned[i].reg == t.reg) return;
        if (m_learned.size() < 24) m_learned.push_back(t);
    }

    // Last bytes written, and a counter that changes with them: patching at draw time needs the current contents and
    // a way to tell whether the sibling is already correct for the shader about to read it.
    void SetShadow(const unsigned char* p, size_t n)
    {
        if (!p || !n || n > kMaxShadowBytes) return;
        m_shadow.assign(p, p + n);
        ++m_writeSeq;
    }
    const std::vector<unsigned char>& Shadow() const { return m_shadow; }
    // A copy or a boxed write filled the buffer behind the shadow's back: nothing may be rebuilt from it until the
    // next whole write.
    void ForgetShadow() { m_shadow.clear(); ++m_writeSeq; }
    // The shift is part of the key, as the eye offset is baked into the bytes: a buffer the game stops rewriting would
    // otherwise keep an old separation.
    bool NeedsDrawPatch(DWORD crc, float shift) const
    { return m_patchedSeq != m_writeSeq || m_patchedCRC != crc || m_patchedShift != shift; }
    void MarkDrawPatched(DWORD crc, float shift)
    { m_patchedSeq = m_writeSeq; m_patchedCRC = crc; m_patchedShift = shift; }
    static const size_t kMaxShadowBytes = 64 * 1024;

    // A D3D11.1 ring constant buffer (over 4096 constants, read through ranged binds): the game writes this shadow, and
    // Context11Proxy::SyncBoundConstants copies each bound range into the real buffers at the draw.
    void MakeRing(size_t byteWidth) { m_ring = true; m_ringShadow.assign(byteWidth, 0); m_ringForeign.assign((byteWidth + 255) / 256, 0); }
    bool IsRing() const { return m_ring; }
    unsigned char* RingData() { return m_ringShadow.data(); }
    size_t RingSize() const { return m_ringShadow.size(); }
    unsigned WriteSeq() const { return m_writeSeq; }
    void MarkRingWritten() { ++m_writeSeq; }
    // Pages (256 B) a GPU copy filled behind the shadow: the real buffers hold them, the shadow does not, until the
    // copy is read back (Context11Proxy::ServiceRingReadbacks). A block touching one is left as the game made it.
    void MarkForeign(size_t off, size_t n)  { for (size_t p = off / 256; p * 256 < off + n && p < m_ringForeign.size(); ++p) m_ringForeign[p] = 1; }
    void ClearForeign(size_t off, size_t n) { for (size_t p = off / 256; p * 256 < off + n && p < m_ringForeign.size(); ++p) m_ringForeign[p] = 0; }
    bool AnyForeign(size_t off, size_t n) const
    {
        for (size_t p = off / 256; p * 256 < off + n && p < m_ringForeign.size(); ++p) if (m_ringForeign[p]) return true;
        return false;
    }
    // What our in-place writes last left in the right sibling, the game's bytes they were built from, and their eye
    // shift (Context11Proxy::MergeRightRing); sized on first use.
    std::vector<unsigned char>& RightMirror() { return m_rightMirror; }
    std::vector<unsigned char>& GameAtRightWrite() { return m_gameAtRightWrite; }
    float& RightMirrorShift() { return m_rightMirrorShift; }

    // What the right sibling holds beyond the shadow bytes: forward (geometry) shifts and inverse patches applied to
    // them; plain shadow is neither. A copy into the buffer counts as unknown, i.e. shifted.
    void SetRightFromShadow(bool v) { m_rightForward = v ? 0 : 1; m_rightInverse = 0; }
    void NoteRightShift(unsigned forward, unsigned inverse) { m_rightForward = forward; m_rightInverse = inverse; }
    bool RightIsShadow() const { return !m_rightForward && !m_rightInverse; }
    bool RightForwardShifted() const { return m_rightForward != 0; }
    // A shader's camera matrices are rebuilt from the shadow once per write and stereo setting; a later rewrite of the
    // sibling by another path asks for the check again.
    bool InverseDrawCheckDue(float shift, float conv) const
    { return m_invCheckSeq != m_writeSeq || m_invCheckShift != shift || m_invCheckConv != conv; }
    void MarkInverseDrawChecked(float shift, float conv)
    { m_invCheckSeq = m_writeSeq; m_invCheckShift = shift; m_invCheckConv = conv; }
    void InvalidateInverseDrawCheck() { m_invCheckSeq = 0xFFFFFFFFu; }

    // Registers vertex shaders read as their position matrix, by pass. One read only in shadow passes is a light's view
    // (MP3's ShadowParam.gShadowMatrix), the same for both eyes.
    void NoteMatrixUse(DWORD reg, bool shadowPass) { if (reg < 256) { if (shadowPass) m_lightRegs.set(reg); else m_cameraRegs.set(reg); } }
    bool LightOnlyMatrix(DWORD reg) const { return reg < 256 && m_lightRegs.test(reg) && !m_cameraRegs.test(reg); }
    // Every write to the right sibling counts, so a feature's draw-time rewrite knows when its bytes are still in place
    // (pooled buffers change content between draws).
    void     NoteRightWrite() { ++m_rightSeq; }
    // The features' edit of a right copy (Context11Proxy::EditRightCBs) was made from these bytes and settings.
    bool NeedsRightEdit(int v, float eyeShift, float conv) const
    {
        const EditKey& k = m_edit[v & 1];
        return k.write != m_writeSeq || k.right != CopySeq(v) || k.shift != eyeShift || k.conv != conv;
    }
    void MarkRightEdited(int v, float eyeShift, float conv)
    {
        EditKey& k = m_edit[v & 1];
        k.write = m_writeSeq; k.right = CopySeq(v); k.shift = eyeShift; k.conv = conv;
    }
    void NoteMovedWrite() { ++m_movedSeq; }

    // Inverse-detection backoff: a buffer that keeps scanning clean is rescanned only sparsely (BioShock Infinite).
    bool ShouldScanForInverse()
    {
        const unsigned n = m_invScans++;
        return n < 16 || (n % 64) == 0;
    }

private:
    ID3D11Buffer*  m_real;      // owned (released in dtor)
    ID3D11Buffer*  m_realRight; // owned, null unless a stereo CB sibling
    Device11Proxy* m_parent;    // not owned
    LONG           m_refs;
    bool           m_vsBound;
    bool           m_everShifted = false;
    unsigned       m_shiftedFrame = 0;
    unsigned       m_invScans = 0;
    unsigned       m_writeSeq = 0;
    unsigned       m_patchedSeq = 0xFFFFFFFFu;
    DWORD          m_patchedCRC = 0;
    float          m_patchedShift = 0.f;
    unsigned       m_rightForward = 0, m_rightInverse = 0;
    unsigned       m_invCheckSeq = 0xFFFFFFFFu;
    float          m_invCheckShift = 0.f, m_invCheckConv = 0.f;
    std::vector<unsigned char> m_shadow;
    bool           m_ring = false;
    std::vector<unsigned char> m_ringShadow;
    std::vector<unsigned char> m_ringForeign;
    std::vector<unsigned char> m_rightMirror, m_gameAtRightWrite;
    float          m_rightMirrorShift = 0.f;
    std::vector<MatrixTarget> m_learned;
    std::bitset<256> m_lightRegs, m_cameraRegs;
    unsigned       m_rightSeq = 0;
    ID3D11Buffer*  m_realRightMoved = nullptr;   // owned, RightCopy(1)
    unsigned       m_movedSeq = 0;
    struct EditKey { unsigned write = ~0u, right = ~0u; float shift = 0.f, conv = 0.f; };
    EditKey        m_edit[2];
    unsigned       CopySeq(int v) const { return (v & 1) ? m_movedSeq : m_rightSeq; }
};

} // namespace wiz3d
