/* 
* Project : iZ3D Stereo Driver
* Copyright (C) iZ3D Inc. 2002 - 2010
*/

#pragma once

struct GlobalInfo
{
	UINT		Trace;
	UINT		RouterType;
	TCHAR		DriverDirectory[MAX_PATH];
	TCHAR		ApplicationFileName[MAX_PATH];
	TCHAR		ApplicationName[MAX_PATH];
	TCHAR		LogFileDirectory[MAX_PATH];
	TCHAR		LogFileName[MAX_PATH];
	TCHAR		DumpDirectory[MAX_PATH];
	// 0.5.0: parity with DX9/DX10/11 — read from <GlobalSettings> in
	// wiz3D_Config.xml. Gates the DX8 shader dump in CreateVertexShader /
	// CreatePixelShader. Both default to false (ZeroMemory in ctor).
	bool		DumpVS;
	bool		DumpPS;

	GlobalInfo()
	{
		ZeroMemory(this, sizeof(GlobalInfo));
	}
};

extern GlobalInfo gInfo;
