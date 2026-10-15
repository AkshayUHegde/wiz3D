/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Right-eye twins of DXBC shaders (SM 4/5, any stage): the game's code plus a plan's instructions, a cbuffer of ours at a
// free slot and extra temps, the container rebuilt and rehashed; Context11Proxy::BindTwin binds one per right-eye call.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <functional>
#include <vector>

namespace wiz3d
{

// Registers a plan's code may use, as the patcher assigned them.
struct PatchRegs
{
    uint32_t cbSlot = 0;      // the wrapper's cbuffer
    uint32_t temp = 0;        // the first of the plan's extraTemps
    uint32_t outTemp = 0;     // the redirected output's values, written to the output after the plan's code
    uint32_t outMask = 0;     // the components the shader writes to it
    int      outReg = -1;     // the redirected output register (-1 = none)
};

struct ShaderPatchPlan
{
    // An output whose writes go to outTemp first (kPositionOutput = the SV_Position output); -1 = none.
    static const int kPositionOutput = -2;
    int      redirectOutput = -1;
    uint32_t extraTemps = 0;     // temps the plan's code uses past the shader's own
    uint32_t cbRegisters = 0;    // the wrapper's cbuffer, in registers (0 = none)
    // Emits the plan's tokens before every ret (DxbcEmit helps); false = this shader cannot take the plan.
    std::function<bool(std::vector<uint32_t>& out, const PatchRegs& regs)> beforeRet;
    // Every instruction past the declarations (nth = earlier ones of its opcode), as a copy the plan may edit in place
    // (same length), its own tokens emitted before it; false = refused. A plan sets this, beforeRet or both.
    std::function<bool(std::vector<uint32_t>& ins, unsigned nth, std::vector<uint32_t>& out, const PatchRegs& regs)> eachInstruction;
};

// The token offsets of an instruction's operands, past its extended opcode tokens; false = an operand it cannot size.
bool DxbcOperandOffsets(const uint32_t* ins, uint32_t len, std::vector<uint32_t>& offsets);

// The twin of a DXBC blob under a plan, its cbuffer at `cbSlot` (refused if the shader declares that slot). false = a
// shape the patcher cannot follow (relative output indexing, labels, interfaces, a geometry shader's emits).
bool PatchShader(const void* bc, size_t len, const ShaderPatchPlan& plan, unsigned cbSlot, std::vector<unsigned char>& out);

// The DXBC container hash: MD5 with Microsoft's final-block layout over everything after the 20-byte header.
void DxbcHash11(const unsigned char* container, size_t len, uint32_t outHash[4]);

// Token helpers for a plan's code: an instruction is Ins(out, opcode, { operand tokens... }), operands built with these.
namespace DxbcEmit
{
    enum Type { kTemp = 0, kInput = 1, kOutput = 2 };
    // A masked destination register (mask bits x=1 y=2 z=4 w=8).
    std::vector<uint32_t> Dst(Type type, uint32_t reg, uint32_t mask);
    // A swizzled source register (components 0..3), negated when neg.
    std::vector<uint32_t> Src(Type type, uint32_t reg, uint32_t x, uint32_t y, uint32_t z, uint32_t w, bool neg = false);
    // A swizzled two-dimensional cbuffer source cb[slot][reg].
    std::vector<uint32_t> Cb(uint32_t slot, uint32_t reg, uint32_t x, uint32_t y, uint32_t z, uint32_t w, bool neg = false);
    // A scalar float literal.
    std::vector<uint32_t> Imm(float v);
    // One instruction: opcode (a D3D10_SB_OPCODE_TYPE) and its operands, the length filled in.
    void Ins(std::vector<uint32_t>& out, uint32_t opcode, std::initializer_list<std::vector<uint32_t>> operands);
}

} // namespace wiz3d
