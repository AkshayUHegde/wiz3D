/* IZ3D_FILE: $Id$ 
*
* Project : iZ3D Stereo Driver
* Copyright (C) iZ3D Inc. 2002 - 2010
*
* $Author$
* $Revision$
* $Date$
* $LastChangedBy$
* $URL$
*/
// Direct3D8.cpp : Implementation of CDx8wrpApp and DLL registration.

#include "stdafx.h"
#include "d3d9_8.h"
#include "Direct3D8.h"
#include "Direct3DDevice8.h"
#include <algorithm>
#include <stdio.h>
#include <stdarg.h>

/////////////////////////////////////////////////////////////////////////////
// Diagnostic logging for AquaNox / DX8 game-not-detecting-adapter issue.
// Always-on (bypasses FINAL_RELEASE strip of DEBUG_TRACE). Appends to the
// wiz3D_d3d8_proxy.log next to the EXE so the wrapper's calls interleave
// with the proxy's calls in a single timeline.
static FILE* g_diagLog = NULL;
static void DiagLogV(const char* fmt, va_list ap)
{
    if (!g_diagLog)
    {
        char path[MAX_PATH];
        GetModuleFileNameA(NULL, path, MAX_PATH);
        char* p = strrchr(path, '\\');
        if (p) *(p + 1) = '\0';
        strcat(path, "wiz3D_Logs");
        CreateDirectoryA(path, NULL);
        strcat(path, "\\wiz3D_d3d8_proxy.log");
        g_diagLog = fopen(path, "a");
        if (g_diagLog)
            fprintf(g_diagLog, "[wrap] === instrumented S3DWrapperD3D8 attached ===\n");
    }
    if (!g_diagLog) return;
    fputs("[wrap] ", g_diagLog);
    vfprintf(g_diagLog, fmt, ap);
    fflush(g_diagLog);
}

static void DiagLog(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    DiagLogV(fmt, ap);
    va_end(ap);
}

// Exposed for use from S3DWrapper8.cpp (Direct3DCreate8).
extern "C" void DiagLogProxy(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    DiagLogV(fmt, ap);
    va_end(ap);
}

CDirect3D8::CDirect3D8()
: m_Adapter(-1)
{
    DiagLog("CDirect3D8::ctor this=%p\n", this);
}

STDMETHODIMP  CDirect3D8::RegisterSoftwareDevice(void* pInitializeFunction)
{
	DEBUG_TRACE3("RegisterSoftwareDevice(pInitializeFunction = %p)\n", pInitializeFunction);
	return m_pReal->RegisterSoftwareDevice(pInitializeFunction);
}

STDMETHODIMP_(UINT) CDirect3D8::GetAdapterCount()
{
	DEBUG_TRACE3("GetAdapterCount()\n");
	UINT cnt = m_pReal->GetAdapterCount();
	DiagLog("GetAdapterCount() = %u  (m_pReal=%p)\n", cnt, m_pReal);
	return cnt;
}

STDMETHODIMP  CDirect3D8::GetAdapterIdentifier(UINT Adapter,DWORD Flags,D3DADAPTER_IDENTIFIER8* pIdentifier)
{
	DEBUG_TRACE3("GetAdapterIdentifier(Adapter = %d, Flags = %d, pIdentifier = %p)\n", Adapter, Flags, pIdentifier);
	D3DADAPTER_IDENTIFIER9 ai;
	HRESULT hr;
	if(!pIdentifier) return E_POINTER;
	hr=m_pReal->GetAdapterIdentifier(Adapter,Flags & ~D3DENUM_WHQL_LEVEL,&ai);
	if(SUCCEEDED(hr))
	{
		strcpy(pIdentifier->Driver, ai.Driver);
		strcpy(pIdentifier->Description, ai.Description);
		pIdentifier->DriverVersion = ai.DriverVersion;
		pIdentifier->VendorId = ai.VendorId;
		pIdentifier->DeviceId = ai.DeviceId;
		pIdentifier->SubSysId = ai.SubSysId;
		pIdentifier->Revision = ai.Revision;
		pIdentifier->DeviceIdentifier = ai.DeviceIdentifier;
		pIdentifier->WHQLLevel = ai.WHQLLevel;
		DiagLog("GetAdapterIdentifier(%u, flags=0x%lX) hr=0x%lX  Driver='%s' Desc='%s' VID=0x%04X DID=0x%04X WHQL=%lu\n",
			Adapter, Flags, hr,
			pIdentifier->Driver, pIdentifier->Description,
			pIdentifier->VendorId, pIdentifier->DeviceId, pIdentifier->WHQLLevel);
	}
	else
	{
		DiagLog("GetAdapterIdentifier(%u, flags=0x%lX) FAILED hr=0x%lX\n", Adapter, Flags, hr);
	}
	return hr;
}

static D3DFORMAT adapterFmts[] = 
{
	D3DFMT_R5G6B5,		D3DFMT_X1R5G5B5,	
	D3DFMT_X8R8G8B8
};

STDMETHODIMP_(UINT) CDirect3D8::GetAdapterModeCount(UINT Adapter)
{
	DEBUG_TRACE3("GetAdapterModeCount(Adapter = %d)\n", Adapter);
	UINT res = 0;
	UINT perFmt[3] = {0,0,0};
	for (int i = 0; i < _countof(adapterFmts); i++)
	{
		UINT cnt = m_pReal->GetAdapterModeCount(Adapter, adapterFmts[i]);
		DEBUG_TRACE3("\t%s: %d\n", GetFormatString(adapterFmts[i]), cnt);
		perFmt[i] = cnt;
		res += cnt;
	}
	DiagLog("GetAdapterModeCount(%u) total=%u  (R5G6B5=%u X1R5G5B5=%u X8R8G8B8=%u)\n",
		Adapter, res, perFmt[0], perFmt[1], perFmt[2]);
	return res;
}

void CDirect3D8::BuildAdaperModesList( int adapter )
{
	if (adapter != m_Adapter)
	{
		m_Adapter = adapter;
		m_AdapterModes.clear();

		// The original algorithm seeded curdm from EnumAdapterModes(R5G6B5, 0).
		// On Win10/11 R5G6B5 and X1R5G5B5 aren't reported as adapter formats —
		// only X8R8G8B8 is — so the seed failed, curdm was uninitialized, and
		// the do/while exited with an empty list while GetAdapterModeCount
		// reported the correct total. Games (e.g. AquaNox) saw count > 0 but
		// every EnumAdapterModes() call returned NOTAVAILABLE, concluded
		// "no gfx board(s) found" and quit.
		//
		// Fix: enumerate all modes from all formats in order. The total still
		// matches GetAdapterModeCount (which sums per-format counts).
		for (int i = 0; i < _countof(adapterFmts); i++)
		{
			UINT n = m_pReal->GetAdapterModeCount(adapter, adapterFmts[i]);
			for (UINT j = 0; j < n; j++)
			{
				D3DDISPLAYMODE m;
				if (SUCCEEDED(m_pReal->EnumAdapterModes(adapter, adapterFmts[i], j, &m)))
					m_AdapterModes.push_back(m);
			}
		}
		DiagLog("BuildAdaperModesList(adapter=%d) collected %zu modes\n",
			adapter, m_AdapterModes.size());
	}
}

STDMETHODIMP  CDirect3D8::EnumAdapterModes(UINT Adapter,UINT Mode,D3DDISPLAYMODE* pMode)
{
	DEBUG_TRACE3("EnumAdapterModes(Adapter = %d, Mode = %d, pMode = %p)\n", Adapter, Mode, pMode);
	BuildAdaperModesList((int)Adapter);
	if (Mode < m_AdapterModes.size())
	{
		*pMode = m_AdapterModes[Mode];
		DEBUG_TRACE3("\t%s\n", GetDisplayModeString(pMode));
		if (Mode < 4 || Mode >= m_AdapterModes.size() - 2)
			DiagLog("EnumAdapterModes(%u, mode=%u) -> %ux%u@%uHz fmt=%d  (total=%zu)\n",
				Adapter, Mode, pMode->Width, pMode->Height, pMode->RefreshRate, pMode->Format,
				m_AdapterModes.size());
		return D3D_OK;
	}
	else
	{
		DiagLog("EnumAdapterModes(%u, mode=%u) -> NOTAVAILABLE  (total=%zu)\n",
			Adapter, Mode, m_AdapterModes.size());
		return D3DERR_NOTAVAILABLE;
	}
}

STDMETHODIMP  CDirect3D8::GetAdapterDisplayMode(UINT Adapter,D3DDISPLAYMODE* pMode)
{
	DEBUG_TRACE3("GetAdapterDisplayMode(Adapter = %d, pMode = %p)\n", Adapter, pMode);
	HRESULT hr = m_pReal->GetAdapterDisplayMode(Adapter,pMode);
	DEBUG_TRACE3("\t%s\n", GetDisplayModeString(pMode));
	if (SUCCEEDED(hr) && pMode)
		DiagLog("GetAdapterDisplayMode(%u) hr=0x%lX  %ux%u@%uHz fmt=%d\n",
			Adapter, hr, pMode->Width, pMode->Height, pMode->RefreshRate, pMode->Format);
	else
		DiagLog("GetAdapterDisplayMode(%u) hr=0x%lX\n", Adapter, hr);
	return hr;
}

STDMETHODIMP  CDirect3D8::CheckDeviceType(UINT Adapter,D3DDEVTYPE CheckType,D3DFORMAT DisplayFormat,D3DFORMAT BackBufferFormat,BOOL Windowed)
{
	DEBUG_TRACE3("CheckDeviceType(Adapter = %d, CheckType = %d, DisplayFormat = %s, BackBufferFormat = %s, Windowed = %d)\n", Adapter, CheckType,
		GetFormatString(DisplayFormat), GetFormatString(BackBufferFormat), Windowed);
	HRESULT hr = m_pReal->CheckDeviceType(Adapter,CheckType,DisplayFormat,BackBufferFormat,Windowed);
	DiagLog("CheckDeviceType(%u, devType=%d, disp=%d, back=%d, win=%d) hr=0x%lX\n",
		Adapter, CheckType, DisplayFormat, BackBufferFormat, Windowed, hr);
	return hr;
}

STDMETHODIMP  CDirect3D8::CheckDeviceFormat(UINT Adapter,D3DDEVTYPE DeviceType,D3DFORMAT AdapterFormat,DWORD Usage,D3DRESOURCETYPE RType,D3DFORMAT CheckFormat)
{
	DEBUG_TRACE3("CheckDeviceFormat(Adapter = %d, DeviceType = %d, AdapterFormat = %s, Usage = %d, RType = %d, CheckFormat = %s)\n", Adapter, DeviceType,
		GetFormatString(AdapterFormat), Usage, RType, GetFormatString(CheckFormat));
	HRESULT hr = m_pReal->CheckDeviceFormat(Adapter,DeviceType,AdapterFormat,Usage,RType,CheckFormat);
	DiagLog("CheckDeviceFormat(%u, devType=%d, adapt=%d, usage=0x%lX, rtype=%d, fmt=%d) hr=0x%lX\n",
		Adapter, DeviceType, AdapterFormat, Usage, RType, CheckFormat, hr);
	return hr;
}

STDMETHODIMP  CDirect3D8::CheckDeviceMultiSampleType(UINT Adapter,D3DDEVTYPE DeviceType,D3DFORMAT SurfaceFormat,BOOL Windowed,D3DMULTISAMPLE_TYPE MultiSampleType)
{
	DEBUG_TRACE3("CheckDeviceMultiSampleType(Adapter = %d, DeviceType = %d, SurfaceFormat = %s, Windowed = %d, MultiSampleType = %d)\n", Adapter, DeviceType, 
		GetFormatString(SurfaceFormat), Windowed, MultiSampleType);
	return m_pReal->CheckDeviceMultiSampleType(Adapter,DeviceType,SurfaceFormat,Windowed,MultiSampleType,NULL);
}

STDMETHODIMP  CDirect3D8::CheckDepthStencilMatch(UINT Adapter,D3DDEVTYPE DeviceType,D3DFORMAT AdapterFormat,D3DFORMAT RenderTargetFormat,D3DFORMAT DepthStencilFormat)
{
	DEBUG_TRACE3("CheckDepthStencilMatch(Adapter = %d, DeviceType = %d, AdapterFormat = %s, RenderTargetFormat = %s, DepthStencilFormat = %s)\n", Adapter, DeviceType, 
		GetFormatString(AdapterFormat), GetFormatString(RenderTargetFormat), GetFormatString(DepthStencilFormat));
	return m_pReal->CheckDepthStencilMatch(Adapter,DeviceType,AdapterFormat,RenderTargetFormat,DepthStencilFormat);
}

STDMETHODIMP  CDirect3D8::GetDeviceCaps(UINT Adapter,D3DDEVTYPE DeviceType,D3DCAPS8* pCaps)
{
	DEBUG_TRACE3("GetDeviceCaps(Adapter = %d, DeviceType = %d, pCaps = %p)\n", Adapter, DeviceType, pCaps);
	D3DCAPS9 caps9;
	HRESULT hr;
	hr=m_pReal->GetDeviceCaps(Adapter,DeviceType,&caps9);
	if(SUCCEEDED(hr))
		ConvertCaps(caps9, pCaps);
	DiagLog("GetDeviceCaps(%u, devType=%d) hr=0x%lX\n", Adapter, DeviceType, hr);
	return hr;
}

STDMETHODIMP_(HMONITOR) CDirect3D8::GetAdapterMonitor(UINT Adapter)
{
	DEBUG_TRACE3("GetAdapterMonitor(Adapter = %d)\n", Adapter);
	HMONITOR hm = m_pReal->GetAdapterMonitor(Adapter);
	DiagLog("GetAdapterMonitor(%u) = %p\n", Adapter, hm);
	return hm;
}

STDMETHODIMP  CDirect3D8::CreateDevice(UINT Adapter,D3DDEVTYPE DeviceType,HWND hFocusWindow,DWORD BehaviorFlags,D3DPRESENT_PARAMETERS8* pPresentationParameters,IDirect3DDevice8** ppReturnedDeviceInterface)
{
	DiagLog("CreateDevice(adapter=%u, devType=%d, hwnd=%p, beh=0x%lX) entered\n",
		Adapter, DeviceType, hFocusWindow, BehaviorFlags);
	HRESULT hr;
	if (ppReturnedDeviceInterface == NULL)
	{
		DiagLog("CreateDevice: ppReturnedDeviceInterface == NULL -> E_POINTER\n");
		return E_POINTER;
	}
	CComPtr<IDirect3DDevice9> pDev9;
	D3DPRESENT_PARAMETERS pp9;
	ConvertPresentParameters(pPresentationParameters,&pp9);

	DiagLog("CreateDevice pp8: BB=%ux%u fmt=%d count=%u MS=%d SwapEff=%d hWnd=%p Win=%d AutoDS=%d DSFmt=%d Flags=0x%lX RR=%u Interval=0x%lX\n",
		pPresentationParameters->BackBufferWidth, pPresentationParameters->BackBufferHeight,
		pPresentationParameters->BackBufferFormat, pPresentationParameters->BackBufferCount,
		pPresentationParameters->MultiSampleType, pPresentationParameters->SwapEffect,
		pPresentationParameters->hDeviceWindow, pPresentationParameters->Windowed,
		pPresentationParameters->EnableAutoDepthStencil, pPresentationParameters->AutoDepthStencilFormat,
		pPresentationParameters->Flags, pPresentationParameters->FullScreen_RefreshRateInHz,
		pPresentationParameters->FullScreen_PresentationInterval);
	DiagLog("CreateDevice pp9: BB=%ux%u fmt=%d count=%u MS=%d MSQ=%lu SwapEff=%d hWnd=%p Win=%d AutoDS=%d DSFmt=%d Flags=0x%lX RR=%u Interval=0x%lX\n",
		pp9.BackBufferWidth, pp9.BackBufferHeight, pp9.BackBufferFormat, pp9.BackBufferCount,
		pp9.MultiSampleType, pp9.MultiSampleQuality, pp9.SwapEffect, pp9.hDeviceWindow,
		pp9.Windowed, pp9.EnableAutoDepthStencil, pp9.AutoDepthStencilFormat, pp9.Flags,
		pp9.FullScreen_RefreshRateInHz, pp9.PresentationInterval);

	hr = m_pReal->CreateDevice(Adapter,DeviceType,hFocusWindow,BehaviorFlags,&pp9,&pDev9);
	DEBUG_TRACE1("CDirect3D8::CreateDevice(Adapter=%d, DeviceType=%d,  hFocusWindow=%d, BehaviorFlags=%d, ppReturnedDeviceInterface=%d)=%s\n",
		Adapter, DeviceType, hFocusWindow, BehaviorFlags,  ppReturnedDeviceInterface, GetResultString(hr));
	DiagLog("CreateDevice m_pReal->CreateDevice hr=0x%lX  pDev9=%p\n", hr, (void*)pDev9.p);

	if (FAILED(hr))
		DEBUG_FAILED(hr)
	else if (pDev9)
	{
		CreateWrapper(pDev9.p, ppReturnedDeviceInterface);
		CDirect3DDevice8* pDev8Wrap = (CDirect3DDevice8*)*ppReturnedDeviceInterface;
		pDev8Wrap->Init(this, Adapter, DeviceType, pp9.BackBufferFormat);
		pDev8Wrap->CreateMcClaud();
		DiagLog("CreateDevice wrapped device returned %p\n", *ppReturnedDeviceInterface);
	}
	return hr;
}
