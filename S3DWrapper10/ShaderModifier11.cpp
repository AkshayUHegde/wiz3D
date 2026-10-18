/*
* TryBuildModifiedVS DX11 Phase 1 -- see ShaderModifier11.h for the design.
*/
#include "StdAfx.h"
#include "ShaderModifier11.h"
#include "ShaderAnalyzer11.h"
#include "..\ShaderAnalysis\disasm.h"
#include "..\ShaderAnalysis\modify.h"

#include <d3d11.h>

namespace wiz3d
{

// D3D_NAME_POSITION is the SV_Position system-value tag that marks the clip-
// space output register. iZ3D's ShaderAnalysis::ModifyShader needs this
// register index to know where to inject the eye-offset ALU.
static constexpr unsigned kD3DNamePosition = 1;

bool TryBuildModifiedVS11(ID3D11Device*                 pDevice,
                          const void*                   pBytecode,
                          SIZE_T                        byteLength,
                          const ShaderAnalysis11Result& analysis,
                          float                         profileMultiplier,
                          float                         profileConvergenceShift,
                          float                         profileZNear,
                          bool                          addZNearCheck,
                          ModifiedVS11&                 outMod)
{
    if (!pDevice || !pBytecode || byteLength < 32) return false;

    // Must have a parsed signature with SV_Position for the modifier to
    // know where to inject.
    unsigned posReg = ~0u;
    for (const auto& o : analysis.outputs)
    {
        if (o.sysValue == kD3DNamePosition) { posReg = o.reg; break; }
    }
    if (posReg == ~0u) return false;

    // Parse the DXBC blob into ShaderAnalysis's instruction list, run the
    // modifier, serialize the modified list back to a DXBC blob. The modifier
    // picks a free CB register internally and reports it via MData.
    shader_analyzer::TShaderList originalList;
    shader_analyzer::ParseShader(reinterpret_cast<const unsigned*>(pBytecode), originalList);
    if (originalList.empty()) return false;

    shader_analyzer::TShaderList modifiedList;
    ModifiedShaderData mData;
    mData.StereoZNear      = profileZNear;
    mData.Multiplier       = profileMultiplier;
    mData.ConvergenceShift = profileConvergenceShift;
    if (!ModifyShader(originalList, posReg, addZNearCheck, modifiedList, mData))
        return false;
    if (!mData.ModifiedShaderAvailable) return false;

    // Serialize back. The modified list may have grown; reserve roughly
    // (originalLen + 32) UINT32s to match the legacy DDI path's buffer
    // sizing in ShaderWrapper.cpp:231.
    const unsigned dxbcVersion = reinterpret_cast<const unsigned*>(pBytecode)[0];
    const unsigned originalLenInUints = reinterpret_cast<const unsigned*>(pBytecode)[1];
    std::vector<unsigned> modifiedBlob;
    if (!shader_analyzer::ShaderList2Blob(modifiedList, dxbcVersion, modifiedBlob,
                                          originalLenInUints + 32))
        return false;
    if (modifiedBlob.empty()) return false;

    // Create the modified shader on the real device. If D3D11 rejects it
    // (e.g. our serialization produced a malformed blob) we give up on this
    // shader and the game falls back to the matrix-patch path.
    ID3D11VertexShader* modShader = nullptr;
    HRESULT hr = pDevice->CreateVertexShader(modifiedBlob.data(),
                                             modifiedBlob.size() * sizeof(unsigned),
                                             nullptr, &modShader);
    if (FAILED(hr) || !modShader) return false;

    outMod.vs                     = modShader;   // ownership transfers to caller/map
    outMod.cbIndex                = mData.CBIndex;
    outMod.dp4Reg                 = mData.dp4VectorRegister;
    outMod.zNearReg               = mData.ZNearRegister;
    outMod.storedZNear            = profileZNear;
    outMod.storedMultiplier       = profileMultiplier;
    outMod.storedConvergenceShift = profileConvergenceShift;
    return true;
}

}  // namespace wiz3d
