/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*
* TryBuildModifiedVS DX11 port -- Phase 1 (shader modification + storage).
*
* The automatic bytecode-level stereo-fix pipeline from iZ3D's legacy DX10
* DDI path (S3DWrapper10/ShaderWrapper.cpp:229 -> ShaderAnalysis/modify.cpp),
* adapted for the new Context11Proxy world. Only runs when a game profile
* sets AlwaysModifyShaders=1 or an individual shader entry sets Modify=1,
* so default-case behaviour is unchanged.
*
* The six opt-in profiles in BaseProfile.xml today: HAWX.exe, AOGame.exe,
* BFBC2Game.exe, MoHMPGame.exe, NFS11.exe, LP2DX9.exe + LP2DX11.exe.
*
* Phase 2 (per-eye binding + the dedicated eye-info CB) is pending; this
* phase only creates the modified shader pair and stores it. The modified
* shader is NOT bound anywhere yet, so no rendering behaviour changes --
* this phase is pure instrumentation so we can log, per game, which shaders
* ShaderAnalysis's SM4/SM5 modifier can and cannot process. Task #18.
*/

#pragma once

#include "..\ShaderAnalysis\modify.h"
#include <d3d11.h>

namespace wiz3d
{

struct ShaderAnalysis11Result;

// One successfully-modified vertex shader + the register positions the
// modified bytecode will read eye-info from. The CB slot (CBIndex) was
// chosen by ShaderAnalysis::FindFreeRegisters as a slot the ORIGINAL
// shader doesn't read -- Phase 2 binds our eye-info constant buffer there
// just before issuing the right-eye draw of this shader.
struct ModifiedVS11
{
    ID3D11VertexShader* vs = nullptr;       // AddRef'd; released when Device11Proxy dies
    unsigned            cbIndex = 0;        // b<cbIndex> is where modified reads eye data
    unsigned            dp4Reg  = 0;        // register inside that CB for (sep, convSq, 0, ZNear)
    unsigned            zNearReg = 0;       // only valid when z-near-check was emitted
    float               storedZNear = 1.f;  // profile ZNear used at build time (diagnostic)
    float               storedMultiplier = 1.f;
    float               storedConvergenceShift = 0.f;
};

// Attempt bytecode-level modification of a VS. Returns true + populates
// `outMod` with the modified shader if ShaderAnalysis produced a valid
// result and D3D11 accepted the modified blob. Returns false otherwise
// (shader has no SV_Position output, parse failed, modify failed, or
// CreateVertexShader rejected the modified bytes). On false, outMod is
// left unchanged.
//
// `profileMultiplier` / `profileConvergenceShift` / `profileZNear` are
// stored on outMod for Phase 2's per-eye CB update to pick up without
// re-doing the profile lookup.
bool TryBuildModifiedVS11(ID3D11Device*                 pDevice,
                          const void*                   pBytecode,
                          SIZE_T                        byteLength,
                          const ShaderAnalysis11Result& analysis,
                          float                         profileMultiplier,
                          float                         profileConvergenceShift,
                          float                         profileZNear,
                          bool                          addZNearCheck,
                          ModifiedVS11&                 outMod);

}  // namespace wiz3d
