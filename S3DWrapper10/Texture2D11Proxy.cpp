/* wiz3D - ID3D11Texture2D proxy implementation (Option B Stage 3a) */

#include "StdAfx.h"
#include "Texture2D11Proxy.h"
#include "Device11Proxy.h"
#include "proxy_factory.h"     // IID_wiz3D_Texture2D11Proxy
#include "AdapterFunctions.h"  // DDILog
#include "ProfileFeature.h"    // PrivateBackBufferRefsOn

#pragma comment(lib, "dxguid.lib")

namespace wiz3d
{

Texture2D11Proxy::Texture2D11Proxy(ID3D11Texture2D* realLeft, ID3D11Texture2D* realRight, Device11Proxy* parent)
    : m_realLeft(realLeft)
    , m_realRight(realRight)
    , m_parent(parent)
    , m_refs(1)
{
}

Texture2D11Proxy::~Texture2D11Proxy()
{
    // Unregister BEFORE releasing the real, so the parent device's real→proxy
    // map can't return a stale proxy pointer for a real texture that's about
    // to be destroyed.
    if (m_parent && m_realLeft)
        m_parent->UnregisterRealToProxy(m_realLeft);
    if (m_parent && gInfo.CollectDebugInformation) m_parent->NoteTextureFreed();
    // RecycleRightCopies: the right copy waits for a texture created with the same desc.
    if (m_realRight)
    {
        if (m_parent && RecycleRightCopiesOn()) m_parent->RecycleRight(m_realRight);
        else m_realRight->Release();
        m_realRight = nullptr;
    }
    if (m_realLeft)  { m_realLeft->Release();  m_realLeft  = nullptr; }
}

ULONG STDMETHODCALLTYPE Texture2D11Proxy::Release()
{
    LONG r = InterlockedDecrement(&m_refs);
    if (r == 0) { delete this; return 0; }
    return (ULONG)(r - m_internalRefs);
}

void Texture2D11Proxy::AddRefInternal()
{
    if (m_backBuffer && PrivateBackBufferRefsOn()) InterlockedIncrement(&m_internalRefs);
    InterlockedIncrement(&m_refs);
}

void Texture2D11Proxy::ReleaseInternal()
{
    if (m_backBuffer && PrivateBackBufferRefsOn()) InterlockedDecrement(&m_internalRefs);
    Release();
}

void Texture2D11Proxy::MakeRefInternal()
{
    if (m_backBuffer && PrivateBackBufferRefsOn()) InterlockedIncrement(&m_internalRefs);
}

HRESULT STDMETHODCALLTYPE Texture2D11Proxy::QueryInterface(REFIID riid, void** ppvObj)
{
    if (!ppvObj) return E_POINTER;
    if (riid == IID_IUnknown        ||
        riid == IID_ID3D11DeviceChild ||
        riid == IID_ID3D11Resource  ||
        riid == IID_ID3D11Texture2D)
    {
        *ppvObj = static_cast<ID3D11Texture2D*>(this);
        AddRef();
        return S_OK;
    }
    // Stage 3b: private identity IID. Used by TryUnwrapTexture2D in
    // proxy_factory.cpp to detect a wiz3D proxy at COM boundaries before
    // forwarding to the real D3D11 runtime.
    if (riid == IID_wiz3D_Texture2D11Proxy)
    {
        *ppvObj = static_cast<IUnknown*>(static_cast<ID3D11Texture2D*>(this));
        AddRef();
        return S_OK;
    }
    // For unknown IIDs, forward to the real resource so things like
    // IDXGIResource (which games sometimes QI for cross-API sharing) still
    // work. We lose identity preservation for those — acceptable for
    // Stage 3 since the doubling logic doesn't depend on them.
    return m_realLeft->QueryInterface(riid, ppvObj);
}

void STDMETHODCALLTYPE Texture2D11Proxy::GetDevice(ID3D11Device** ppDevice)
{
    // Match Context11Proxy's pattern: return the wrapped device so the game's
    // round-trip Get* calls keep flowing through our wrapper.
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
