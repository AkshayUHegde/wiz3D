/* wiz3D - ID3D11UnorderedAccessView proxy (Option B Stage 3c.2)
 *
 * Passthrough wrap + private IID. Like SRV11Proxy carries an optional
 * right-eye sibling for eye-aware UAV binding once 4e wires it.
 */

#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>

namespace wiz3d
{

class Device11Proxy;
class Texture2D11Proxy;

class UAV11Proxy : public ID3D11UnorderedAccessView
{
public:
    UAV11Proxy(ID3D11UnorderedAccessView* realLeft, ID3D11UnorderedAccessView* realRight, Device11Proxy* parent);
    virtual ~UAV11Proxy();

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObj) override;
    ULONG   STDMETHODCALLTYPE AddRef() override                  { return InterlockedIncrement(&m_refs); }
    ULONG   STDMETHODCALLTYPE Release() override;

    void    STDMETHODCALLTYPE GetDevice(ID3D11Device** ppDevice) override;
    HRESULT STDMETHODCALLTYPE GetPrivateData(REFGUID guid, UINT* pDataSize, void* pData) override                  { return m_realLeft->GetPrivateData(guid, pDataSize, pData); }
    HRESULT STDMETHODCALLTYPE SetPrivateData(REFGUID guid, UINT DataSize, const void* pData) override              { return m_realLeft->SetPrivateData(guid, DataSize, pData); }
    HRESULT STDMETHODCALLTYPE SetPrivateDataInterface(REFGUID guid, const IUnknown* pData) override                { return m_realLeft->SetPrivateDataInterface(guid, pData); }

    void    STDMETHODCALLTYPE GetResource(ID3D11Resource** ppResource) override                                    { m_realLeft->GetResource(ppResource); }
    void    STDMETHODCALLTYPE GetDesc(D3D11_UNORDERED_ACCESS_VIEW_DESC* pDesc) override                            { m_realLeft->GetDesc(pDesc); }

    bool                         IsStereo()      const { return m_realRight != nullptr; }
    ID3D11UnorderedAccessView*   GetReal()       const { return m_realLeft;  }
    ID3D11UnorderedAccessView*   GetRealRight()  const { return m_realRight; }
    Device11Proxy*               GetParent()     const { return m_parent;    }
    // The viewed Texture2D proxy, when the view is of one, held by reference: a right-eye bind marks its right copy
    // written (Texture2D11Proxy::MarkRightWritten).
    void                         SetTexture(Texture2D11Proxy* t);
    Texture2D11Proxy*            Texture()       const { return m_texture;   }

private:
    ID3D11UnorderedAccessView* m_realLeft;
    ID3D11UnorderedAccessView* m_realRight;
    Device11Proxy*             m_parent;
    Texture2D11Proxy*          m_texture = nullptr;   // one reference held
    LONG                       m_refs;
};

} // namespace wiz3d
