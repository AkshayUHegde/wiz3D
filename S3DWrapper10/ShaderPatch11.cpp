/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Right-eye shader twins (ShaderPatch11.h): a token rewrite of the shader's SHEX/SHDR, then the container rebuilt and
// rehashed. Layout and hash follow S3DWrapper12's DxbcPatch12, for shader model 4/5 (two-dimensional cbuffer operands).

#include "StdAfx.h"
#include "ShaderPatch11.h"
#include <d3d10umddi.h>
#include <math.h>
#include <string.h>

namespace wiz3d
{
namespace
{

struct Md5 { uint32_t a, b, c, d; uint8_t buf[64]; size_t bufLen; };

uint32_t Rotl(uint32_t x, int c) { return (x << c) | (x >> (32 - c)); }

const uint32_t* Md5K()
{
    static uint32_t k[64];
    static bool init = false;
    if (!init)
    {
        for (int i = 0; i < 64; ++i) k[i] = (uint32_t)(fabs(sin((double)(i + 1))) * 4294967296.0);
        init = true;
    }
    return k;
}

void Md5Block(Md5& m, const uint8_t* p)
{
    static const int r[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21 };
    const uint32_t* K = Md5K();
    uint32_t M[16];
    for (int i = 0; i < 16; ++i)
        M[i] = (uint32_t)p[i * 4] | ((uint32_t)p[i * 4 + 1] << 8) | ((uint32_t)p[i * 4 + 2] << 16) | ((uint32_t)p[i * 4 + 3] << 24);
    uint32_t a = m.a, b = m.b, c = m.c, d = m.d;
    for (int i = 0; i < 64; ++i)
    {
        uint32_t f; int g;
        if (i < 16)      { f = (b & c) | (~b & d); g = i; }
        else if (i < 32) { f = (d & b) | (~d & c); g = (5 * i + 1) & 15; }
        else if (i < 48) { f = b ^ c ^ d;          g = (3 * i + 5) & 15; }
        else             { f = c ^ (b | ~d);       g = (7 * i) & 15; }
        f += a + K[i] + M[g];
        a = d; d = c; c = b;
        b += Rotl(f, r[i]);
    }
    m.a += a; m.b += b; m.c += c; m.d += d;
}

void Md5Update(Md5& m, const void* data, size_t n)
{
    const uint8_t* p = (const uint8_t*)data;
    while (n)
    {
        const size_t take = (64 - m.bufLen) < n ? (64 - m.bufLen) : n;
        memcpy(m.buf + m.bufLen, p, take);
        m.bufLen += take; p += take; n -= take;
        if (m.bufLen == 64) { Md5Block(m, m.buf); m.bufLen = 0; }
    }
}

// Instruction length in DWORDs (customdata carries its own).
uint32_t OpLen(const uint32_t* t, size_t i, size_t end)
{
    if ((t[i] & 0x7FF) == D3D10_SB_OPCODE_CUSTOMDATA) return (i + 1 < end) ? t[i + 1] : 0;
    return DECODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(t[i]);
}

bool IsDecl(uint32_t op)
{
    return (op >= D3D10_SB_OPCODE_DCL_RESOURCE && op <= D3D10_SB_OPCODE_DCL_GLOBAL_FLAGS) ||
           (op >= D3D11_SB_OPCODE_DCL_STREAM && op <= D3D11_SB_OPCODE_DCL_RESOURCE_STRUCTURED) ||
           op == D3D11_SB_OPCODE_DCL_GS_INSTANCE_COUNT || op == D3D10_SB_OPCODE_CUSTOMDATA;
}

// DWORDs one operand occupies, or 0 when it runs off the end.
size_t OperandLen(const uint32_t* t, size_t j, size_t end)
{
    if (j >= end) return 0;
    const uint32_t tok = t[j];
    size_t n = 1;
    for (uint32_t ext = tok >> 31; ext; )
    {
        if (j + n >= end) return 0;
        ext = t[j + n] >> 31; ++n;
    }
    const uint32_t type = DECODE_D3D10_SB_OPERAND_TYPE(tok);
    const uint32_t ncomp = tok & 3;
    if (type == D3D10_SB_OPERAND_TYPE_IMMEDIATE32 || type == D3D10_SB_OPERAND_TYPE_IMMEDIATE64)
    {
        const size_t count = ncomp == 1 ? 1 : (ncomp == 2 ? 4 : 0);
        n += count * (type == D3D10_SB_OPERAND_TYPE_IMMEDIATE64 ? 2 : 1);
        return j + n <= end ? n : 0;
    }
    const uint32_t dims = DECODE_D3D10_SB_OPERAND_INDEX_DIMENSION(tok);
    for (uint32_t d = 0; d < dims; ++d)
    {
        size_t sub = 0;
        switch ((tok >> (22 + 3 * d)) & 7)
        {
            case 0: n += 1; break;
            case 1: n += 2; break;
            case 2: sub = OperandLen(t, j + n, end); if (!sub) return 0; n += sub; break;
            case 3: sub = OperandLen(t, j + n + 1, end); if (!sub) return 0; n += 1 + sub; break;
            case 4: sub = OperandLen(t, j + n + 2, end); if (!sub) return 0; n += 2 + sub; break;
            default: return 0;
        }
        if (j + n > end) return 0;
    }
    return n;
}

uint32_t Op(uint32_t op, uint32_t len)
{
    return ENCODE_D3D10_SB_OPCODE_TYPE(op) | ENCODE_D3D10_SB_TOKENIZED_INSTRUCTION_LENGTH(len);
}

uint32_t Swizzled(uint32_t type, uint32_t dims, uint32_t x, uint32_t y, uint32_t z, uint32_t w, bool neg)
{
    uint32_t tok = ENCODE_D3D10_SB_OPERAND_NUM_COMPONENTS(D3D10_SB_OPERAND_4_COMPONENT) |
                   ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE_MODE) |
                   ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SWIZZLE(x, y, z, w) |
                   ENCODE_D3D10_SB_OPERAND_TYPE(type) |
                   ENCODE_D3D10_SB_OPERAND_INDEX_DIMENSION(dims == 2 ? D3D10_SB_OPERAND_INDEX_2D : D3D10_SB_OPERAND_INDEX_1D) |
                   ENCODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(0, D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
    if (dims == 2) tok |= ENCODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(1, D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
    if (neg) tok |= ENCODE_D3D10_SB_OPERAND_EXTENDED(1);
    return tok;
}

uint32_t NegToken()
{
    return ENCODE_D3D10_SB_EXTENDED_OPERAND_TYPE(D3D10_SB_EXTENDED_OPERAND_MODIFIER) |
           ENCODE_D3D10_SB_EXTENDED_OPERAND_MODIFIER(D3D10_SB_OPERAND_MODIFIER_NEG);
}

D3D10_SB_OPERAND_TYPE OperandType(DxbcEmit::Type t)
{
    return t == DxbcEmit::kInput ? D3D10_SB_OPERAND_TYPE_INPUT : t == DxbcEmit::kOutput ? D3D10_SB_OPERAND_TYPE_OUTPUT
                                                                                         : D3D10_SB_OPERAND_TYPE_TEMP;
}

} // namespace

namespace DxbcEmit
{

std::vector<uint32_t> Dst(Type type, uint32_t reg, uint32_t mask)
{
    const uint32_t tok = ENCODE_D3D10_SB_OPERAND_NUM_COMPONENTS(D3D10_SB_OPERAND_4_COMPONENT) |
                         ENCODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE) |
                         ((mask & 0xF) << 4) | ENCODE_D3D10_SB_OPERAND_TYPE(OperandType(type)) |
                         ENCODE_D3D10_SB_OPERAND_INDEX_DIMENSION(D3D10_SB_OPERAND_INDEX_1D) |
                         ENCODE_D3D10_SB_OPERAND_INDEX_REPRESENTATION(0, D3D10_SB_OPERAND_INDEX_IMMEDIATE32);
    return { tok, reg };
}

std::vector<uint32_t> Src(Type type, uint32_t reg, uint32_t x, uint32_t y, uint32_t z, uint32_t w, bool neg)
{
    if (neg) return { Swizzled(OperandType(type), 1, x, y, z, w, true), NegToken(), reg };
    return { Swizzled(OperandType(type), 1, x, y, z, w, false), reg };
}

std::vector<uint32_t> Cb(uint32_t slot, uint32_t reg, uint32_t x, uint32_t y, uint32_t z, uint32_t w, bool neg)
{
    if (neg) return { Swizzled(D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER, 2, x, y, z, w, true), NegToken(), slot, reg };
    return { Swizzled(D3D10_SB_OPERAND_TYPE_CONSTANT_BUFFER, 2, x, y, z, w, false), slot, reg };
}

std::vector<uint32_t> Imm(float v)
{
    uint32_t u;
    memcpy(&u, &v, 4);
    return { ENCODE_D3D10_SB_OPERAND_NUM_COMPONENTS(D3D10_SB_OPERAND_1_COMPONENT) |
             ENCODE_D3D10_SB_OPERAND_TYPE(D3D10_SB_OPERAND_TYPE_IMMEDIATE32) |
             ENCODE_D3D10_SB_OPERAND_INDEX_DIMENSION(D3D10_SB_OPERAND_INDEX_0D), u };
}

void Ins(std::vector<uint32_t>& out, uint32_t opcode, std::initializer_list<std::vector<uint32_t>> operands)
{
    uint32_t len = 1;
    for (const auto& o : operands) len += (uint32_t)o.size();
    out.push_back(Op(opcode, len));
    for (const auto& o : operands) out.insert(out.end(), o.begin(), o.end());
}

} // namespace DxbcEmit

void DxbcHash11(const unsigned char* container, size_t len, uint32_t outHash[4])
{
    outHash[0] = outHash[1] = outHash[2] = outHash[3] = 0;
    if (!container || len < 20) return;
    Md5 m;
    m.a = 0x67452301u; m.b = 0xefcdab89u; m.c = 0x98badcfeu; m.d = 0x10325476u; m.bufLen = 0;
    const uint8_t* data = container + 20;
    const uint32_t length = (uint32_t)(len - 20);
    const uint32_t numBits = length * 8;
    const uint32_t numBitsPart2 = (numBits >> 2) | 1;
    const uint32_t leftover = length % 64;
    Md5Update(m, data, length - leftover);
    data += length - leftover;
    uint32_t block[16] = {};
    if (leftover >= 56)
    {
        Md5Update(m, data, leftover);
        block[0] = 0x80;
        Md5Update(m, block, 64 - leftover);
        block[0] = numBits;
        block[15] = numBitsPart2;
        Md5Update(m, block, 64);
    }
    else
    {
        Md5Update(m, &numBits, 4);
        if (leftover) Md5Update(m, data, leftover);
        const uint32_t padding = 64 - leftover - 4;
        block[0] = 0x80;
        memcpy(((uint8_t*)block) + padding - 4, &numBitsPart2, 4);
        Md5Update(m, block, padding);
    }
    outHash[0] = m.a; outHash[1] = m.b; outHash[2] = m.c; outHash[3] = m.d;
}

bool PatchShader(const void* bc, size_t len, const ShaderPatchPlan& plan, unsigned cbSlot, std::vector<unsigned char>& out)
{
    out.clear();
    const unsigned char* p = (const unsigned char*)bc;
    if (!p || len < 36 || memcmp(p, "DXBC", 4) != 0 || !plan.beforeRet) return false;
    const uint32_t np = *(const uint32_t*)(p + 28);
    if (np > 64 || 32 + (size_t)np * 4 > len) return false;
    const uint32_t* offs = (const uint32_t*)(p + 32);
    int codePart = -1;
    for (uint32_t i = 0; i < np; ++i)
    {
        if ((size_t)offs[i] + 8 > len) return false;
        const uint32_t sz = *(const uint32_t*)(p + offs[i] + 4);
        if ((size_t)offs[i] + 8 + sz > len) return false;
        if (!memcmp(p + offs[i], "SHEX", 4) || !memcmp(p + offs[i], "SHDR", 4)) codePart = (int)i;
    }
    if (codePart < 0) return false;
    const uint32_t* t = (const uint32_t*)(p + offs[codePart] + 8);
    const size_t count = *(const uint32_t*)(p + offs[codePart] + 4) / 4;
    if (count < 3 || t[1] != count) return false;
    const bool geometry = DECODE_D3D10_SB_TOKENIZED_PROGRAM_TYPE(t[0]) == D3D10_SB_GEOMETRY_SHADER;

    // Pass 1: the declarations. The new cbuffer follows the last declared one (or opens the declarations).
    int outReg = plan.redirectOutput >= 0 ? plan.redirectOutput : -1, temps = 0;
    size_t dclTempsAt = (size_t)-1, lastCbEnd = (size_t)-1, firstNonDcl = (size_t)-1;
    for (size_t i = 2; i < count; )
    {
        const uint32_t L = OpLen(t, i, count);
        if (!L || i + L > count) return false;
        const uint32_t op = t[i] & 0x7FF;
        if (op == D3D10_SB_OPCODE_LABEL || op == D3D11_SB_OPCODE_INTERFACE_CALL) return false;
        if (op == D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER && L >= 4)
        {
            if (plan.cbRegisters && t[i + 2] == cbSlot) return false;   // the slot is the shader's own
            lastCbEnd = i + L;
        }
        else if (op == D3D10_SB_OPCODE_DCL_OUTPUT_SIV && L >= 4 && plan.redirectOutput == ShaderPatchPlan::kPositionOutput &&
                 DECODE_D3D10_SB_OPERAND_TYPE(t[i + 1]) == D3D10_SB_OPERAND_TYPE_OUTPUT && t[i + L - 1] == D3D10_SB_NAME_POSITION)
            outReg = (int)t[i + 2];
        else if (op == D3D10_SB_OPCODE_DCL_TEMPS && L >= 2) { temps = (int)t[i + 1]; dclTempsAt = i; }
        else if (firstNonDcl == (size_t)-1 && !IsDecl(op)) firstNonDcl = i;
        i += L;
    }
    if (firstNonDcl == (size_t)-1) return false;
    if (plan.redirectOutput != -1 && (outReg < 0 || geometry)) return false;   // a GS's vertices leave at emit
    PatchRegs regs;
    regs.cbSlot = cbSlot;
    regs.outReg = outReg;
    regs.outTemp = (uint32_t)temps;
    regs.temp = (uint32_t)temps + (outReg >= 0 ? 1 : 0);
    const uint32_t newTemps = regs.temp + plan.extraTemps;

    // Pass 2: every write of the redirected output goes to outTemp.
    std::vector<uint32_t> src(t, t + count);
    for (size_t i = firstNonDcl; outReg >= 0 && i < count; )
    {
        const uint32_t L = OpLen(t, i, count);
        if (!IsDecl(t[i] & 0x7FF))
        {
            size_t j = i + 1;
            for (uint32_t ext = t[i] >> 31; ext; ++j) ext = t[j] >> 31;   // extended opcode tokens
            while (j < i + L)
            {
                const size_t n = OperandLen(t, j, i + L);
                if (!n) return false;
                const uint32_t o = t[j];
                if (DECODE_D3D10_SB_OPERAND_TYPE(o) == D3D10_SB_OPERAND_TYPE_OUTPUT)
                {
                    if ((o >> 31) || DECODE_D3D10_SB_OPERAND_INDEX_DIMENSION(o) != D3D10_SB_OPERAND_INDEX_1D ||
                        ((o >> 22) & 7) != D3D10_SB_OPERAND_INDEX_IMMEDIATE32)
                        return false;
                    if (t[j + 1] == (uint32_t)outReg)
                    {
                        regs.outMask |= DECODE_D3D10_SB_OPERAND_4_COMPONENT_SELECTION_MODE(o) == D3D10_SB_OPERAND_4_COMPONENT_MASK_MODE
                                            ? (o >> 4) & 0xF : 0xF;
                        src[j] = (o & ~(0xFFu << 12)) | ENCODE_D3D10_SB_OPERAND_TYPE(D3D10_SB_OPERAND_TYPE_TEMP);
                        src[j + 1] = regs.outTemp;
                    }
                }
                j += n;
            }
        }
        i += L;
    }
    if (outReg >= 0 && !regs.outMask) return false;

    // Pass 3: emit, with the cbuffer and the temps declared and the plan's code (then the output) before every ret.
    std::vector<uint32_t> o;
    o.reserve(count + 128);
    o.push_back(t[0]); o.push_back(0);
    auto emitCbDecl = [&]()
    {
        o.push_back(Op(D3D10_SB_OPCODE_DCL_CONSTANT_BUFFER, 4) |
                    ENCODE_D3D10_SB_D3D10_SB_CONSTANT_BUFFER_ACCESS_PATTERN(D3D10_SB_CONSTANT_BUFFER_IMMEDIATE_INDEXED));
        o.push_back(DxbcEmit::Cb(0, 0, 0, 1, 2, 3)[0]); o.push_back(cbSlot); o.push_back(plan.cbRegisters);
    };
    auto emitTemps = [&]() { o.push_back(Op(D3D10_SB_OPCODE_DCL_TEMPS, 2)); o.push_back(newTemps); };
    bool cbDone = !plan.cbRegisters, tempsDone = newTemps == (uint32_t)temps && dclTempsAt == (size_t)-1;
    for (size_t i = 2; i < count; )
    {
        const uint32_t L = OpLen(t, i, count);
        const uint32_t op = t[i] & 0x7FF;
        if (i == firstNonDcl || (!cbDone && i == lastCbEnd))
        {
            if (!cbDone) { emitCbDecl(); cbDone = true; }
            if (!tempsDone && dclTempsAt == (size_t)-1) { emitTemps(); tempsDone = true; }
        }
        if (i == dclTempsAt) { emitTemps(); tempsDone = true; i += L; continue; }
        if (op == D3D10_SB_OPCODE_RET || op == D3D10_SB_OPCODE_RETC)
        {
            if (!plan.beforeRet(o, regs)) return false;
            if (outReg >= 0)
                DxbcEmit::Ins(o, D3D10_SB_OPCODE_MOV, { DxbcEmit::Dst(DxbcEmit::kOutput, (uint32_t)outReg, regs.outMask),
                                                        DxbcEmit::Src(DxbcEmit::kTemp, regs.outTemp, 0, 1, 2, 3) });
        }
        for (uint32_t k = 0; k < L; ++k) o.push_back(src[i + k]);
        i += L;
    }
    if (!cbDone || !tempsDone) return false;
    o[1] = (uint32_t)o.size();

    // The container: same parts in the same order, the code swapped, offsets, size and hash recomputed.
    const size_t headerBytes = 32 + (size_t)np * 4;
    out.reserve(len + o.size() * 4);
    out.resize(headerBytes);
    memcpy(&out[0], p, 32);
    for (uint32_t i = 0; i < np; ++i)
    {
        *(uint32_t*)&out[32 + i * 4] = (uint32_t)out.size();
        const unsigned char* part = p + offs[i];
        if ((int)i == codePart)
        {
            const uint32_t bytes = (uint32_t)(o.size() * 4);
            out.insert(out.end(), part, part + 4);
            out.insert(out.end(), (const unsigned char*)&bytes, (const unsigned char*)&bytes + 4);
            out.insert(out.end(), (const unsigned char*)&o[0], (const unsigned char*)&o[0] + bytes);
        }
        else
        {
            const uint32_t sz = *(const uint32_t*)(part + 4);
            out.insert(out.end(), part, part + 8 + sz);
            while (out.size() & 3) out.push_back(0);
        }
    }
    *(uint32_t*)&out[24] = (uint32_t)out.size();
    uint32_t h[4];
    DxbcHash11(&out[0], out.size(), h);
    memcpy(&out[4], h, 16);
    return true;
}

} // namespace wiz3d
