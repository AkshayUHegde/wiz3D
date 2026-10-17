#pragma once

// ---------------------------------------------------------------------------
// Calibration wizard — vestigial as of wiz3D 0.4.0.
//
// Inherited from iZ3D's shutter-glass-era rig calibration flow (circa 2006-
// 2010). Walked the user through shutter sync direction, convergence
// comfort, separation comfort, and display geometry. Launched via the old
// iZ3D Control Center tray app or the Shift+Numpad* hotkey, with the UI
// rendered through UILib's custom CXWindow / CXLabel / CXButton components.
//
// Non-functional on current builds: depends on %ALLUSERSPROFILE%\
// resources.zip (button / checkbox / background PNGs) + OSDFont.fnt +
// language XML files. None of those ship through wiz3D's zip-release
// pipeline; they were deployed by the InnoSetup installer that was retired
// when wiz3D forked from iZ3D. The Control Center entry point was also
// retired at the fork.
//
// Modern targets (SR / SBS / HD3D / 3D Vision Direct) don't need this
// kind of per-user shutter-rig calibration -- the stereo geometry is
// fixed by the panel / API. The preset-switch hotkeys (Numpad 7/8/9)
// cover the one remaining use case (picking a comfortable separation
// percentage).
//
// The ToggleWizard hotkey (Shift+Numpad*, code 362) is still bound in
// the shared KeyboardHook config for backward compatibility, but it
// won't produce a visible wizard without resurrecting the asset
// pipeline. Not planned for 0.4.x or 0.5.x.
// ---------------------------------------------------------------------------

#include <string>
#include <map>
#include "..\uilib\Framework.h"
//#define DISABLE_WIZARD
#include <boost/serialization/access.hpp>
#include <boost/unordered_map.hpp>

interface IDirect3DDevice9;
class CBaseSwapChain;
class CXDesktopWindow;
class CXWindow;
class CXExtWindow;
class CXLabel;
class CXCanvas;
namespace uilib
{

	typedef std::map< std::basic_string< TCHAR >,  CAdapt<CComPtr<IDirect3DTexture9>>> TextureMap_t;

	class TexturesManager
	{
		IDirect3DDevice9* m_pDevice;
		TextureMap_t m_Data;
	public:
		TexturesManager() : m_pDevice(NULL) { }
		void SetDevice(IDirect3DDevice9* pDevice) { m_pDevice = pDevice; }
		void Clear() { m_Data.clear(); }
		LPDIRECT3DTEXTURE9 GetImage(const wchar_t* pResName);

		LPDIRECT3DTEXTURE9 LoadImage( const wchar_t* pResName );
	};

	class WizardSCData;

	class Wizard
	{
		// constants
		static const TCHAR* STR_THISDLLNAME;
		static const HMODULE s_hModule;

		static const LOGFONT s_lf;
		

		// fields
		TexturesManager m_TexManager;
		IDirect3DDevice9* m_pDevice;
		// static fields
		static bool s_bEnabled;

	public:
		static const LOGFONT s_Notlf;
		HRESULT Initialize(IDirect3DDevice9* pD3DDevice);
		IDirect3DDevice9* GetDevice() { return m_pDevice; }
		void Clear();
		Wizard(void);
		~Wizard(void);

		friend WizardSCData;
	};

	class WizardSCData
	{
		Wizard* m_pWizard;

		// hot keys codes
		static const int ID_Y_KEY = 0x59;
		static const int ID_N_KEY = 0x4e;
		static const int ID_P_KEY = 0x50;
		static const int ID_F_KEY = 0x46;
		static const int ID_R_KEY = 0x52;
		static const int ID_S_KEY = 0x53;
		static const int ID_B_KEY = 0x42;

		// fields
		bool m_bReourcesLoaded;
		DWORD m_Width;
		DWORD m_Height;
		CXDesktopWindow* m_pDesktop;

		LPDIRECT3DTEXTURE9 GetImage( const wchar_t* pResName );

		void SelectFirstControl();
		static bool IsWizardHotKeyPressed();
		void addPictureSignatures( CXExtWindow* pStep );
	public:
		void GetSize(DWORD& Width, DWORD& Height) { Width = m_Width; Height = m_Height; };
		IDirect3DDevice9* GetDevice() { return m_pWizard->GetDevice(); }

		// configuring wizard
		HRESULT ConfigureCommonStep(CXExtWindow* pWindow,
			const wchar_t* IDCaptionText,
			const wchar_t* IDFirstButtonText = 0,
			const wchar_t* IDSecondButtonText = 0,
			const wchar_t* IDThirdButtonText = 0,
			const wchar_t* IDFourthButtonText = 0);
		void CreateButton(const wchar_t* IDButtonText, const wchar_t* pResName, 
			LONG X, LONG Y, LONG Width, LONG Height, CXExtWindow* pWindow);
		//////////////////////////////////////////////////////////////////////////
		void ConfigureStereoSampleStep( CXExtWindow* pStep,
			const TCHAR* pLeftResName,
			const TCHAR* pRightResName );

		void ConfigureStereoStep( CXExtWindow* pStep,
			const TCHAR* pResName );

		void ProcessKeys();
		void Initialize( Wizard* pWizard, CBaseSwapChain* pSwapChain );
		void LoadResources( );
		void Show();
		void Hide();
		void Render(CBaseSwapChain* pSwapChain);

		bool Force3D();
		void Clear();
		WizardSCData(void);
		~WizardSCData(void);
	};
}
