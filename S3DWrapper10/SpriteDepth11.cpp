/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The sprite-depth plan (SpriteDepth11.h) for ShaderPatch11.

#include "StdAfx.h"
#include "SpriteDepth11.h"
#include <d3d10umddi.h>

namespace wiz3d
{

ShaderPatchPlan SpriteDepthPlan(unsigned worldInput)
{
    using namespace DxbcEmit;
    ShaderPatchPlan plan;
    plan.redirectOutput = ShaderPatchPlan::kPositionOutput;
    plan.extraTemps = 1;
    plan.cbRegisters = 2;
    plan.beforeRet = [worldInput](std::vector<uint32_t>& o, const PatchRegs& r)
    {
        if ((r.outMask & 0x9) != 0x9) return false;   // x and w written
        const uint32_t s = r.temp, c = r.cbSlot;
        Ins(o, D3D10_SB_OPCODE_MOV, { Dst(kTemp, s, 0x7), Src(kInput, worldInput, 0, 1, 2, 0) });   // rS.xyz = world
        Ins(o, D3D10_SB_OPCODE_MOV, { Dst(kTemp, s, 0x8), Imm(1.f) });
        Ins(o, D3D10_SB_OPCODE_DP4, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 1, 2, 3), Cb(c, 0, 0, 1, 2, 3) });   // depth
        Ins(o, D3D10_SB_OPCODE_MAX, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Imm(0.001f) });
        Ins(o, D3D10_SB_OPCODE_DIV, { Dst(kTemp, s, 0x1), Cb(c, 1, 1, 1, 1, 1), Src(kTemp, s, 0, 0, 0, 0) });   // m41 / w
        Ins(o, D3D10_SB_OPCODE_ADD, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Cb(c, 1, 0, 0, 0, 0) });   // + b
        Ins(o, D3D10_SB_OPCODE_ADD, { Dst(kTemp, r.outTemp, 0x1), Src(kTemp, r.outTemp, 0, 0, 0, 0), Src(kTemp, s, 0, 0, 0, 0) });
        return true;
    };
    return plan;
}

} // namespace wiz3d
