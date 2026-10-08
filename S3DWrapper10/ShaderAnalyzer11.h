/* wiz3D - DXBC-driven shader analyzer adapter (Option B Stage 4e)
 *
 * Bridges the COM-wrap path (raw DXBC blobs from Create{Vertex,Pixel,...}Shader)
 * to the existing iZ3D shader_analyzer in ShaderAnalysis/, which was written
 * against the legacy DDI surface (D3D10DDIARG_STAGE_IO_SIGNATURES + a SHEX/SHDR
 * chunk pointer). Walks DXBC chunks, extracts the SHDR/SHEX chunk and the
 * OSGN/OSG5 output signature, builds a synthetic STAGE_IO_SIGNATURES from
 * the OSGN element table, then calls shader_analyzer::GetProjectionMatrices.
 */

#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "..\ShaderAnalysis\AnalyzeData.h"

struct D3D10DDIARG_SIGNATURE_ENTRY;

namespace wiz3d
{

// Result of analyzing a DXBC blob. ProjectionShaderData mirrors the legacy
// analyzer's output — see ShaderAnalysis/AnalyzeData.h.
struct ShaderAnalysis11Result
{
    bool                  parsed;     // SHDR/SHEX + OSGN chunks both present and parsed
    DWORD                 crc32;      // bytecode CRC32 — diagnostic, helps cross-match against legacy DDI logs
    ProjectionShaderData  projection; // matrix register info per CB
    // Register holding SV_Position. ModifyShader needs it to know which output to
    // shift; valid only when parsed is true.
    DWORD                 posRegister;
    // Located by the names the game's <Profile> gives (g_ProfileData), from RDEF; -1 = absent or not read.
    // StereoTexture's t-slot, and the cb slot + byte offset of its EyeOffset view inverse (row 0 = the camera's right axis).
    int                   stereoParmsSlot = -1;
    int                   viewInvCB       = -1;
    UINT                  viewInvOffset   = 0;
    // MirroredTarget's t-slot: a planar reflection, sampled at u = 0.5 - 0.5*x/w (MP3 mirrors and water).
    int                   reflectTexSlot  = -1;
    // AdditionalMatrixName matrices the shader reads: cb slot and first register each.
    std::vector<std::pair<int, UINT>> namedMatrices;
    // HudDepth: the w the game gives a HUD element under 3D Vision (MP3's ApparentDepth).
    int                   apparentDepthCB     = -1;
    UINT                  apparentDepthOffset = 0;
    // HudRect, same cbuffer: ndc = corner * rect.xy + rect.zw (MP3's TransPos).
    int                   transPosCB          = -1;
    UINT                  transPosOffset      = 0;
    // CameraTanX: .x = tan of the half FOV, 1/p00 of the camera (MP3's gDeferredProjParams).
    int                   projParamsCB        = -1;
    UINT                  projParamsOffset    = 0;
    // Cbuffers a profile feature edits for the right eye (ProfileFeature::EditRightConstants), by name, and their slots.
    std::vector<std::pair<std::string, int>> rightCBs;
    void NoteRightCB(const std::string& name, int slot)
    {
        for (const auto& n : rightCBs) if (n.first == name) return;
        rightCBs.emplace_back(name, slot);
    }
    // Names a feature recognised in this shader (a cbuffer, texture or variable it reads), for per-pass decisions.
    std::vector<std::string> tags;
    // Shader resources a feature named, and their t-slots (any SRV kind: texture or buffer).
    std::vector<std::pair<std::string, int>> namedSRVs;
    // Texture slots where the right eye reads the game's (left) copy: light-space maps a profile names (LeftEyeTextures).
    std::vector<int> leftSRVSlots;
    // The bytecode: during shader creation for every stage, kept afterwards for compute shaders only (null past the
    // keep budget).
    std::shared_ptr<const std::vector<BYTE>> bytecode;
};

// Analyze a DXBC bytecode blob. Returns true if a SHDR/SHEX + OSGN pair was
// found and at least walked; the caller still needs to inspect
// projection.matrixData.cb.empty() to know whether any projection matrices
// were detected. Safe to call with nullptr/0 — returns parsed=false.
bool AnalyzeShader11(const void* pBytecode, SIZE_T byteLength,
                     ShaderAnalysis11Result& outResult);

} // namespace wiz3d
