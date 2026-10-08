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
    // The shift is part of the key, as the eye offset is baked into the bytes: a buffer the game stops rewriting would
    // otherwise keep an old separation.
    bool NeedsDrawPatch(DWORD crc, float shift) const
    { return m_patchedSeq != m_writeSeq || m_patchedCRC != crc || m_patchedShift != shift; }
    void MarkDrawPatched(DWORD crc, float shift)
    { m_patchedSeq = m_writeSeq; m_patchedCRC = crc; m_patchedShift = shift; }
    static const size_t kMaxShadowBytes = 64 * 1024;

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
    std::vector<unsigned char> m_shadow;
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
