/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The fog lookup plan (FogLookup11.h) for ShaderPatch11: lookups found in the shader's disassembly, patched by tokens.

#include "StdAfx.h"
#include "FogLookup11.h"
#include <d3dcompiler.h>
#include <d3d10umddi.h>
#include <set>
#include <sstream>
#include <string>
#include <string.h>

namespace wiz3d
{
namespace
{

int Component(char c) { return c == 'x' ? 0 : c == 'y' ? 1 : c == 'z' ? 2 : c == 'w' ? 3 : -1; }

std::string Mnemonic(const std::string& line) { return line.substr(0, line.find_first_of(" (")); }

// An instruction line's operands past its modifiers ("[precise(w)]"), split at ", " (a literal vector splits too, but
// only after the registers).
std::vector<std::string> Operands(const std::string& line)
{
    std::vector<std::string> ops;
    size_t sp = line.find(' ');
    while (sp != std::string::npos && sp + 1 < line.size() && line[sp + 1] == '[') sp = line.find(' ', sp + 1);
    if (sp == std::string::npos) return ops;
    for (size_t p = sp + 1; p < line.size(); )
    {
        const size_t q = line.find(", ", p);
        ops.push_back(line.substr(p, q == std::string::npos ? std::string::npos : q - p));
        if (q == std::string::npos) break;
        p = q + 2;
    }
    return ops;
}

bool IsSampleL(const std::string& m)
{
    return m == "sample_l" || m.compare(0, 18, "sample_l_indexable") == 0 || m.compare(0, 17, "sample_l_aoffimmi") == 0;
}

bool IsFlow(const std::string& m)
{
    static const char* const k[] = { "if_", "else", "endif", "loop", "endloop", "switch", "case", "default", "endswitch",
                                     "break", "continue", "ret", "call", "label" };
    for (const char* f : k) if (m.compare(0, strlen(f), f) == 0) return true;
    return false;
}

// "rN.c" (one component) or "rN.mask".
bool TempOperand(const std::string& op, int& reg, std::string& comps)
{
    char m[8] = {};
    if (sscanf_s(op.c_str(), "r%d.%7s", &reg, m, (unsigned)sizeof m) != 2) return false;
    comps = m;
    return true;
}

// "cbS[R].c...": slot, register, the component at position `pos` of the swizzle (0 for a scalar).
bool CbOperand(const std::string& op, int pos, int& slot, int& reg, int& comp)
{
    char m[8] = {};
    if (sscanf_s(op.c_str(), "cb%d[%d].%7s", &slot, &reg, m, (unsigned)sizeof m) != 3) return false;
    const size_t n = strlen(m);
    comp = Component(m[n == 1 ? 0 : pos < (int)n ? pos : 0]);
    return comp >= 0;
}

// The nearest earlier line of the block writing rReg's component c, -1 when control flow comes first.
int Writer(const std::vector<std::string>& code, int at, int reg, char c)
{
    for (int i = at; i-- > 0; )
    {
        if (IsFlow(Mnemonic(code[i]))) return -1;
        const std::vector<std::string> ops = Operands(code[i]);
        int r = -1;
        std::string mask;
        if (ops.empty() || ops[0].empty()) continue;
        if (ops[0][0] == '[') return -1;   // a destination it cannot read
        if (TempOperand(ops[0], r, mask) && r == reg && mask.find(c) != std::string::npos) return i;
    }
    return -1;
}

// The slice chain ending at line `at`: mad(depth, zP.x, zP.y), log, * zP.z, * invGrid.z.
bool SliceChain(const std::vector<std::string>& code, int at, FogLookup& k)
{
    std::vector<std::string> ops = Operands(code[at]);
    int a = -1, slot = -1, comp = -1;
    std::string ac;
    if (Mnemonic(code[at]) != "mul" || ops.size() != 3 || !TempOperand(ops[1], a, ac) || ac.size() != 1 ||
        !CbOperand(ops[2], 0, slot, k.invGrid, comp) || comp != 2)
        return false;
    k.view = slot;
    const char c = ac[0];
    int i = Writer(code, at, a, c);
    if (i < 0 || Mnemonic(code[i]) != "mul" || (ops = Operands(code[i])).size() != 3 ||
        !CbOperand(ops[2], 0, slot, k.zParams, comp) || slot != k.view || comp != 2)
        return false;
    if ((i = Writer(code, i, a, c)) < 0 || Mnemonic(code[i]) != "log") return false;
    int r1 = 0, c1 = 0, r2 = 0, c2 = 0, s2 = 0;
    if ((i = Writer(code, i, a, c)) < 0 || Mnemonic(code[i]) != "mad" || (ops = Operands(code[i])).size() != 4 ||
        !CbOperand(ops[2], 0, slot, r1, c1) || !CbOperand(ops[3], 0, s2, r2, c2))
        return false;
    return slot == k.view && s2 == k.view && r1 == k.zParams && r2 == k.zParams && c1 == 0 && c2 == 1;
}

} // namespace

bool FindFogLookups(const void* bc, size_t len, std::vector<FogLookup>& out)
{
    out.clear();
    ID3DBlob* text = nullptr;
    if (!bc || !len || FAILED(D3DDisassemble(bc, len, 0, nullptr, &text)) || !text) return false;
    const std::string s((const char*)text->GetBufferPointer(), text->GetBufferSize());
    text->Release();
    return FindFogLookupsInText(s, out);
}

bool FindFogLookupsInText(const std::string& s, std::vector<FogLookup>& out)
{
    out.clear();
    if (s.find("\nps_") == std::string::npos) return false;
    std::set<int> volumes;
    std::vector<std::string> code;
    std::istringstream in(s);
    std::string line;
    bool body = false;
    while (std::getline(in, line))
    {
        const size_t b = line.find_first_not_of(' ');
        if (b == std::string::npos || line[b] == '/') continue;
        line = line.substr(b);
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        int t = -1;
        if (line.compare(0, 22, "dcl_resource_texture3d") == 0 && sscanf_s(line.c_str() + line.rfind(' ') + 1, "t%d", &t) == 1)
            volumes.insert(t);
        if (line.compare(0, 4, "dcl_") == 0 || line.compare(0, 3, "ps_") == 0 || (!body && line[0] == '{')) continue;
        body = true;
        code.push_back(line);
    }
    int nth = 0;
    for (int i = 0; i < (int)code.size(); ++i)
    {
        if (!IsSampleL(Mnemonic(code[i]))) continue;
        FogLookup k = {};
        k.nth = nth++;
        const std::vector<std::string> ops = Operands(code[i]);
        std::string sw;
        if (ops.size() < 3 || !TempOperand(ops[1], k.reg, sw) || sw.size() != 4 ||
            sscanf_s(ops[2].c_str(), "t%d", &k.tslot) != 1 || !volumes.count(k.tslot))
            continue;
        k.cu = Component(sw[0]);
        k.cw = Component(sw[2]);
        const int wz = Writer(code, i, k.reg, sw[2]), wx = Writer(code, i, k.reg, sw[0]);
        if (wz < 0 || wx < 0 || !SliceChain(code, wz, k)) continue;
        // x: (pixels) * svposToUv, or NDC x * 0.5 + 0.5.
        const std::vector<std::string> xo = Operands(code[wx]);
        const std::string xm = Mnemonic(code[wx]);
        int xslot = -1;
        const bool pixels = xm == "mul" && xo.size() == 3 && CbOperand(xo[2], k.cu, xslot, k.svposToUv, k.svposComp) &&
                            xslot == k.view;
        const bool ndc = !pixels && xm == "mad" && k.cu == 0 && code[wx].find("l(0.500000, -0.500000") != std::string::npos &&
                         code[wx].find("l(0.500000, 0.500000") != std::string::npos;
        if (!pixels && !ndc) continue;
        if (!pixels) k.svposToUv = k.svposComp = -1;
        out.push_back(k);
    }
    return !out.empty();
}

ShaderPatchPlan FogLookupPlan(const std::vector<FogLookup>& lookups)
{
    using namespace DxbcEmit;
    ShaderPatchPlan plan;
    plan.extraTemps = 2;
    plan.cbRegisters = 1;
    plan.eachInstruction = [lookups](std::vector<uint32_t>& ins, unsigned nth, std::vector<uint32_t>& o, const PatchRegs& r)
    {
        if ((ins[0] & 0x7FF) != D3D10_SB_OPCODE_SAMPLE_L) return true;
        const FogLookup* k = nullptr;
        for (const auto& l : lookups) if (l.nth == (int)nth) k = &l;
        if (!k) return true;
        std::vector<uint32_t> at;
        if (!DxbcOperandOffsets(ins.data(), (uint32_t)ins.size(), at) || at.size() < 3) return false;
        const uint32_t c = ins[at[1]], t = ins[at[2]];
        if ((c >> 31) || DECODE_D3D10_SB_OPERAND_TYPE(c) != D3D10_SB_OPERAND_TYPE_TEMP ||
            DECODE_D3D10_SB_OPERAND_INDEX_DIMENSION(c) != D3D10_SB_OPERAND_INDEX_1D || ins[at[1] + 1] != (uint32_t)k->reg ||
            (t >> 31) || DECODE_D3D10_SB_OPERAND_TYPE(t) != D3D10_SB_OPERAND_TYPE_RESOURCE || ins[at[2] + 1] != (uint32_t)k->tslot)
            return false;
        const uint32_t v = (uint32_t)k->view, zp = (uint32_t)k->zParams, cp = r.temp, s = r.temp + 1, w = (uint32_t)k->cw;
        Ins(o, D3D10_SB_OPCODE_MOV, { Dst(kTemp, cp, 0xF), Src(kTemp, (uint32_t)k->reg, 0, 1, 2, 3) });
        Ins(o, D3D10_SB_OPCODE_MUL, { Dst(kTemp, s, 0x1), Cb(v, zp, 2, 2, 2, 2), Cb(v, (uint32_t)k->invGrid, 2, 2, 2, 2) });
        Ins(o, D3D10_SB_OPCODE_DIV, { Dst(kTemp, s, 0x1), Src(kTemp, (uint32_t)k->reg, w, w, w, w), Src(kTemp, s, 0, 0, 0, 0) });
        Ins(o, D3D10_SB_OPCODE_EXP, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0) });
        Ins(o, D3D10_SB_OPCODE_ADD, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Cb(v, zp, 1, 1, 1, 1, true) });
        Ins(o, D3D10_SB_OPCODE_DIV, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Cb(v, zp, 0, 0, 0, 0) });   // w
        Ins(o, D3D10_SB_OPCODE_MAX, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Imm(0.001f) });
        Ins(o, D3D10_SB_OPCODE_DIV, { Dst(kTemp, s, 0x1), Cb(r.cbSlot, 0, 1, 1, 1, 1), Src(kTemp, s, 0, 0, 0, 0) });   // C / w
        Ins(o, D3D10_SB_OPCODE_ADD, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Imm(-1.f) });
        if (k->svposToUv >= 0)
        {
            const uint32_t sc = (uint32_t)k->svposComp;
            Ins(o, D3D10_SB_OPCODE_MUL, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Cb(r.cbSlot, 0, 2, 2, 2, 2) });
            Ins(o, D3D10_SB_OPCODE_MUL, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Cb(v, (uint32_t)k->svposToUv, sc, sc, sc, sc) });
        }
        else
            Ins(o, D3D10_SB_OPCODE_MUL, { Dst(kTemp, s, 0x1), Src(kTemp, s, 0, 0, 0, 0), Cb(r.cbSlot, 0, 0, 0, 0, 0) });
        const uint32_t u = (uint32_t)k->cu;
        Ins(o, D3D10_SB_OPCODE_ADD, { Dst(kTemp, cp, 1u << u), Src(kTemp, cp, u, u, u, u), Src(kTemp, s, 0, 0, 0, 0) });
        ins[at[1] + 1] = cp;
        return true;
    };
    return plan;
}

} // namespace wiz3d
