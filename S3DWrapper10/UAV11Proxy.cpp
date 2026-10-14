/* wiz3D - ID3D11UnorderedAccessView proxy implementation */

#include "StdAfx.h"
#include "UAV11Proxy.h"
#include "Texture2D11Proxy.h"
#include "Device11Proxy.h"
#include "proxy_factory.h"

#pragma comment(lib, "dxguid.lib")

namespace wiz3d
{

UAV11Proxy::UAV11Proxy(ID3D11UnorderedAccessView* realLeft, ID3D11UnorderedAccessView* realRight, Device11Proxy* parent)
    : m_realLeft(realLeft)
    , m_realRight(realRight)
    , m_parent(parent)
    , m_refs(1)
{
}

UAV11Proxy::~UAV11Proxy()
{
    if (m_texture)   { m_texture->ReleaseInternal(); m_texture = nullptr; }
    if (m_realRight) { m_realRight->Release(); m_realRight = nullptr; }
    if (m_realLeft)  { m_realLeft->Release();  m_realLeft  = nullptr; }
}

void UAV11Proxy::SetTexture(Texture2D11Proxy* t)
{
    if (t) t->AddRefInternal();
    if (m_texture) m_texture->ReleaseInternal();
    m_texture = t;
}

ULONG STDMETHODCALLTYPE UAV11Proxy::Release()
{
    LONG r = InterlockedDecrement(&m_refs);
    if (r == 0) delete this;
    return (ULONG)r;
}

HRESULT STDMETHODCALLTYPE UAV11Proxy::QueryInterface(REFIID riid, void** ppvObj)
{
    if (!ppvObj) return E_POINTER;
    if (riid == IID_IUnknown        ||
        riid == IID_ID3D11DeviceChild ||
        riid == IID_ID3D11View      ||
        riid == IID_ID3D11UnorderedAccessView)
    {
        *ppvObj = static_cast<ID3D11UnorderedAccessView*>(this);
        AddRef();
        return S_OK;
    }
    if (riid == IID_wiz3D_UAV11Proxy)
    {
        *ppvObj = static_cast<IUnknown*>(static_cast<ID3D11UnorderedAccessView*>(this));
        AddRef();
        return S_OK;
    }
    return m_realLeft->QueryInterface(riid, ppvObj);
}

void STDMETHODCALLTYPE UAV11Proxy::GetDevice(ID3D11Device** ppDevice)
{
    if (!ppDevice) return;
    if (m_parent)
    {
        *ppDevice = static_cast<ID3D11Device*>(m_parent);
        m_parent->AddRef();
        return;
    }
    m_realLeft->GetDevice(ppDevice);
}

} // namespace wiz3d
