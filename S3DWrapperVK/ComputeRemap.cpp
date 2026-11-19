/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// VKComputeRemap: shaders run again for the right eye rebuild positions with the left camera; stereo set 8 (sep 0 in
// the left run) moves them onto the right one by rule: ndc rows, froxel rays, 3D reprojection, view vector, masks.

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>
#include <spirv/unified1/GLSL.std.450.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <map>
#include <algorithm>
#include <array>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "FsPatch.h"

namespace wiz3d { namespace vk {

namespace {

struct Ins { size_t at; uint16_t n; spv::Op op; uint32_t block; };

struct Cs
{
    const uint32_t* w = nullptr;
    std::vector<Ins> ins;
    std::unordered_map<uint32_t, size_t> def;
    std::unordered_map<uint32_t, uint32_t> varClass, fconst, iconst;
    std::unordered_map<uint32_t, uint8_t> taint;       // pixel axes a value reads: 1 x, 2 y, 4 z; 0x80 an id vector
    std::unordered_set<uint32_t> idVars;               // WorkgroupId / LocalInvocationId / GlobalInvocationId
    uint32_t glsl = 0, tFloat = 0;

    const uint32_t* At(uint32_t id, spv::Op* op) const
    {
        auto it = def.find(id);
        if (it == def.end()) return nullptr;
        *op = ins[it->second].op;
        return w + ins[it->second].at;
    }
    uint8_t T(uint32_t id) const
    {
        auto it = taint.find(id);
        return it == taint.end() ? 0 : it->second == 0x80 ? 7 : it->second;
    }
    // Through the fork's wrappers: bitcasts, Select(exec, new, old), the denormal flush Select(tiny, x & sign, x).
    uint32_t Strip(uint32_t id) const
    {
        for (int k = 0; k < 32; ++k)
        {
            spv::Op op; const uint32_t* i = At(id, &op);
            if (!i) break;
            if (op == spv::Op::OpBitcast || op == spv::Op::OpCopyObject) { id = i[3]; continue; }
            if (op == spv::Op::OpSelect)
            {
                spv::Op top; const uint32_t* t = At(i[4], &top);
                id = (t && top == spv::Op::OpBitwiseAnd && t[3] == i[5]) ? i[5] : i[4];
                continue;
            }
            if (op == spv::Op::OpBitwiseAnd && iconst.count(i[4])) { id = i[3]; continue; }
            break;
        }
        return id;
    }
    const uint32_t* Glsl(uint32_t id, uint32_t inst) const
    {
        spv::Op op; const uint32_t* i = At(id, &op);
        return i && op == spv::Op::OpExtInst && i[3] == glsl && i[4] == inst ? i : nullptr;
    }
    const uint32_t* Op(uint32_t id, spv::Op want) const
    {
        spv::Op op; const uint32_t* i = At(id, &op);
        return i && op == want ? i : nullptr;
    }
    bool IsFloat(uint32_t id) const
    {
        spv::Op op; const uint32_t* i = At(id, &op);
        bool r = false, t = false;
        if (i) spv::HasResultAndType(op, &r, &t);
        return i && t && i[1] == tFloat;
    }
    // A matrix element: loaded through a buffer address or from a uniform/storage/push-constant block.
    bool FromBuffer(uint32_t id) const
    {
        const uint32_t* l = Op(Strip(id), spv::Op::OpLoad);
        if (!l) return false;
        spv::Op pop; const uint32_t* p = At(l[3], &pop);
        if (p && pop == spv::Op::OpConvertUToPtr) return true;
        uint32_t base = l[3];
        if (p && (pop == spv::Op::OpAccessChain || pop == spv::Op::OpInBoundsAccessChain)) base = p[3];
        auto c = varClass.find(base);
        return c != varClass.end() && (c->second == (uint32_t)spv::StorageClass::Uniform || c->second == (uint32_t)spv::StorageClass::StorageBuffer ||
            c->second == (uint32_t)spv::StorageClass::PushConstant || c->second == (uint32_t)spv::StorageClass::PhysicalStorageBuffer);
    }
    bool IsOne(uint32_t id) const { auto c = fconst.find(id); return c != fconst.end() && c->second == 0x3F800000u; }
    // A load through a buffer address as (base address, constant offset); base 0 when it is not one.
    std::pair<uint32_t, uint32_t> LoadBase(uint32_t id) const
    {
        const uint32_t* l = Op(Strip(id), spv::Op::OpLoad);
        const uint32_t* p = l ? Op(l[3], spv::Op::OpConvertUToPtr) : nullptr;
        if (!p) return { 0, 0 };
        if (const uint32_t* add = Op(p[3], spv::Op::OpIAdd))
        {
            auto c = iconst.find(add[4]);
            if (c != iconst.end()) return { Strip(add[3]), c->second };
            c = iconst.find(add[3]);
            if (c != iconst.end()) return { Strip(add[4]), c->second };
        }
        return { Strip(p[3]), 0 };
    }
};

// The id operands of an instruction with a result, literals skipped where the grammar puts them.
template <class F> void ForIds(const uint32_t* i, uint16_t n, spv::Op op, F f)
{
    switch (op)
    {
    case spv::Op::OpExtInst: for (uint16_t k = 5; k < n; ++k) f(i[k]); break;
    case spv::Op::OpCompositeExtract: f(i[3]); break;
    case spv::Op::OpCompositeInsert: case spv::Op::OpVectorShuffle: f(i[3]); f(i[4]); break;
    case spv::Op::OpPhi: for (uint16_t k = 3; k < n; k += 2) f(i[k]); break;
    case spv::Op::OpLoad: case spv::Op::OpAccessChain: case spv::Op::OpInBoundsAccessChain: case spv::Op::OpVariable: break;
    default: for (uint16_t k = 3; k < n; ++k) f(i[k]); break;
    }
}

struct Emit
{
    std::vector<uint32_t> v;
    void Op(spv::Op op, std::initializer_list<uint32_t> a)
    {
        v.push_back(((uint32_t)(a.size() + 1) << 16) | (uint32_t)op);
        v.insert(v.end(), a.begin(), a.end());
    }
};

inline uint32_t Bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

} // namespace

// fusedMad: a translator rounding a MAD's product before its add emits FMul then FAdd (NoContraction) where the rules
// below expect Fma; a product only its add reads goes back to that Fma.
bool FuseUnfusedMad(const uint32_t* code, size_t words, std::vector<uint32_t>& out, unsigned* fusedOut)
{
    *fusedOut = 0;
    if (words < 5 || code[0] != spv::MagicNumber) return false;
    struct In { size_t at; uint16_t n; spv::Op op; };
    std::vector<In> ins;
    for (size_t i = 5; i < words; )
    {
        const uint16_t n = (uint16_t)(code[i] >> 16);
        if (!n || i + n > words) return false;
        ins.push_back({ i, n, (spv::Op)(code[i] & 0xFFFF) });
        i += n;
    }
    std::unordered_set<uint32_t> noContraction;
    std::unordered_map<uint32_t, size_t> def;
    std::unordered_map<uint32_t, uint32_t> uses;   // operand words inside functions, results left out
    uint32_t glsl = 0;
    bool inFunction = false;
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        const uint16_t n = ins[k].n;
        if (ins[k].op == spv::Op::OpDecorate && n == 3 && i[2] == (uint32_t)spv::Decoration::NoContraction) noContraction.insert(i[1]);
        if (ins[k].op == spv::Op::OpExtInstImport && n >= 3 && !strcmp((const char*)(i + 2), "GLSL.std.450")) glsl = i[1];
        if (ins[k].op == spv::Op::OpFunction) inFunction = true;
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(ins[k].op, &hasRes, &hasType);
        const uint16_t resAt = hasRes ? (hasType ? 2 : 1) : 0;
        if (hasRes) def[i[resAt]] = k;
        if (inFunction)
            for (uint16_t a = 1; a < n; ++a) if (a != resAt) ++uses[i[a]];
    }
    if (!glsl) return false;
    std::unordered_map<size_t, std::array<uint32_t, 5>> fma;   // FAdd index -> type, result, a, b, c
    std::unordered_set<size_t> dropped;
    std::unordered_set<uint32_t> undecorate;
    for (size_t k = 0; k < ins.size(); ++k)
    {
        if (ins[k].op != spv::Op::OpFAdd || ins[k].n != 5) continue;
        const uint32_t* i = code + ins[k].at;
        if (!noContraction.count(i[2])) continue;
        for (int side = 0; side < 2; ++side)
        {
            const uint32_t p = i[3 + side];
            auto d = def.find(p);
            if (d == def.end() || ins[d->second].op != spv::Op::OpFMul || ins[d->second].n != 5 || !noContraction.count(p) || uses[p] != 1) continue;
            const uint32_t* f = code + ins[d->second].at;
            if (f[1] != i[1]) continue;
            fma[k] = { i[1], i[2], f[3], f[4], i[4 - side] };
            dropped.insert(d->second);
            undecorate.insert(p);
            undecorate.insert(i[2]);
            break;
        }
    }
    if (fma.empty()) return false;
    out.clear();
    out.reserve(words + fma.size() * 3);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        if (dropped.count(k)) continue;
        if (ins[k].op == spv::Op::OpDecorate && ins[k].n == 3 && i[2] == (uint32_t)spv::Decoration::NoContraction && undecorate.count(i[1])) continue;
        auto it = fma.find(k);
        if (it != fma.end())
        {
            const auto& v = it->second;
            const uint32_t e[8] = { (8u << 16) | (uint32_t)spv::Op::OpExtInst, v[0], v[1], glsl, GLSLstd450Fma, v[2], v[3], v[4] };
            out.insert(out.end(), e, e + 8);
            continue;
        }
        out.insert(out.end(), i, i + ins[k].n);
    }
    *fusedOut = (unsigned)fma.size();
    return true;
}

// `wide` (VKComputeRemap=2): the ndc rule also takes FAdd(FMul(a, b), c) chains and colour-output fragment shaders,
// as in a sun shadow mask rebuilding 1/(d34*ndcx + d35*ndcy + d36*z + d37).
bool PatchComputeRemap(const uint32_t* code, size_t words, uint32_t stereoSet, const ComputeRemapRules& rules,
                       std::vector<uint32_t>& out, ComputeRemapCounts& counts)
{
    const bool wide = rules.wide, uvRays = rules.uvRays, storeCamera = rules.storeCamera, counters = rules.counters;
    const bool tiledLights = rules.tiledLights, rayRows = rules.rayRows, softRead = rules.softRead, viewRows = rules.viewRows;
    const bool flareX = rules.flareX, uvAffine = rules.uvAffine, froxelLookups = rules.froxelLookups, fragRays = rules.fragRays;
    const bool cornerRays = rules.cornerRays;
    counts = ComputeRemapCounts();
    if (words < 5 || code[0] != spv::MagicNumber) return false;
    Cs m;
    m.w = code;
    uint32_t block = 0;
    for (size_t i = 5; i < words; )
    {
        const uint16_t n = (uint16_t)(code[i] >> 16);
        if (!n || i + n > words) return false;
        const spv::Op op = (spv::Op)(code[i] & 0xFFFF);
        if (op == spv::Op::OpLabel) ++block;
        m.ins.push_back({ i, n, op, block });
        i += n;
    }
    bool compute = false, fragment = false, storageWrite = false, hasImageQuery = false, queriesSize = false;
    uint32_t tInt = 0, tPtrUF = 0, wgVar = 0, fragCoordVar = 0;
    size_t firstType = SIZE_MAX, firstFunc = SIZE_MAX;
    std::unordered_map<uint32_t, uint32_t> builtinOf;
    std::unordered_map<uint64_t, uint32_t> ptrOf;   // (storage class << 32 | pointee) -> pointer type, declared once
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(in.op, &hasRes, &hasType);
        if (hasRes) m.def[hasType ? i[2] : i[1]] = k;
        switch (in.op)
        {
        case spv::Op::OpEntryPoint:
            if (i[1] == (uint32_t)spv::ExecutionModel::GLCompute) compute = true;
            if (i[1] == (uint32_t)spv::ExecutionModel::Fragment) compute = fragment = true;
            break;
        case spv::Op::OpExtInstImport: if (!strcmp((const char*)(i + 2), "GLSL.std.450")) m.glsl = i[1]; break;
        case spv::Op::OpCapability: if (i[1] == (uint32_t)spv::Capability::ImageQuery) hasImageQuery = true; break;
        case spv::Op::OpDecorate:
            // A fragment shader's pixel is FragCoord.
            if (in.n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn &&
                (i[3] == (uint32_t)spv::BuiltIn::WorkgroupId || i[3] == (uint32_t)spv::BuiltIn::LocalInvocationId ||
                 i[3] == (uint32_t)spv::BuiltIn::GlobalInvocationId || i[3] == (uint32_t)spv::BuiltIn::FragCoord))
                m.idVars.insert(i[1]);
            if (in.n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::WorkgroupId) wgVar = i[1];
            if (in.n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::FragCoord) fragCoordVar = i[1];
            break;
        case spv::Op::OpImageWrite: storageWrite = true; break;
        case spv::Op::OpTypeFloat: if (i[2] == 32 && !m.tFloat) m.tFloat = i[1]; break;
        case spv::Op::OpTypeInt: if (i[2] == 32 && (!tInt || i[3] == 1)) tInt = i[1]; break;
        case spv::Op::OpTypePointer:
            if (i[2] == (uint32_t)spv::StorageClass::Uniform && i[3] == m.tFloat && m.tFloat && !tPtrUF) tPtrUF = i[1];
            if (!ptrOf.count(((uint64_t)i[2] << 32) | i[3])) ptrOf[((uint64_t)i[2] << 32) | i[3]] = i[1];
            break;
        case spv::Op::OpConstant:
            if (i[1] == m.tFloat && m.tFloat) m.fconst[i[2]] = i[3];
            else if (in.n == 4 || in.n == 5) m.iconst[i[2]] = i[3];   // 64-bit: the low word (buffer-address offsets)
            break;
        case spv::Op::OpVariable: m.varClass[i[2]] = i[3]; break;
        case spv::Op::OpFunction: if (firstFunc == SIZE_MAX) firstFunc = k; break;
        default: break;
        }
        const bool typeSection = (in.op >= spv::Op::OpTypeVoid && in.op <= spv::Op::OpTypeForwardPointer) ||
                                 (in.op >= spv::Op::OpConstantTrue && in.op <= spv::Op::OpSpecConstantOp) ||
                                 (in.op == spv::Op::OpVariable && firstFunc == SIZE_MAX);
        if (typeSection && firstType == SIZE_MAX) firstType = k;
    }
    // Latte's mul_nonIEEE (Cemu): a float function of two float pointers whose body multiplies their loads.
    std::unordered_set<uint32_t> mulFns;
    for (size_t k = 0; k < m.ins.size() && m.tFloat; ++k)
    {
        if (m.ins[k].op != spv::Op::OpFunction || code[m.ins[k].at + 1] != m.tFloat) continue;
        std::vector<uint32_t> par;
        size_t j = k + 1;
        for (; j < m.ins.size() && m.ins[j].op == spv::Op::OpFunctionParameter; ++j) par.push_back(code[m.ins[j].at + 2]);
        for (; par.size() == 2 && j < m.ins.size() && m.ins[j].op != spv::Op::OpFunctionEnd; ++j)
        {
            const uint32_t* f = code + m.ins[j].at;
            const uint32_t* a = m.ins[j].op == spv::Op::OpFMul ? m.Op(f[3], spv::Op::OpLoad) : nullptr;
            const uint32_t* b = a ? m.Op(f[4], spv::Op::OpLoad) : nullptr;
            if (b && ((a[3] == par[0] && b[3] == par[1]) || (a[3] == par[1] && b[3] == par[0]))) { mulFns.insert(code[m.ins[k].at + 2]); break; }
        }
    }
    // uvAffine: shaders rebuilding from varyings alone, no FragCoord.
    if (!compute || !m.tFloat || !m.glsl || (m.idVars.empty() && mulFns.empty() && !rayRows && !softRead && !flareX && !(uvAffine && fragment) &&
        !(froxelLookups && fragment)) || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    // Fragment shaders only when they store into images (box decals writing G-buffer layers after rebuilding the
    // pixel's position from depth); a colour-output composite keeps its camera.
    if (fragment && !storageWrite && !wide && !froxelLookups && !fragRays) return false;

    // FragCoord.w, a component load or an extract; its pointer type and index, or vector type, to read it again.
    uint32_t fwPtrType = 0, fwIndex = 0, fwVecType = 0;
    auto fragW = [&](uint32_t id) -> bool
    {
        if (!fragCoordVar) return false;
        if (const uint32_t* l = m.Op(id, spv::Op::OpLoad))
        {
            const uint32_t* p = m.Op(l[3], spv::Op::OpAccessChain);
            auto c = p && (p[0] >> 16) == 5 ? m.iconst.find(p[4]) : m.iconst.end();
            if (!p || p[3] != fragCoordVar || c == m.iconst.end() || c->second != 3) return false;
            fwPtrType = p[1]; fwIndex = p[4];
            return true;
        }
        const uint32_t* x = m.Op(id, spv::Op::OpCompositeExtract);
        const uint32_t* l = x && (x[0] >> 16) == 5 && x[4] == 3 ? m.Op(x[3], spv::Op::OpLoad) : nullptr;
        if (!l || l[3] != fragCoordVar) return false;
        fwVecType = l[1];
        return true;
    };
    // A noperspective varying as a translator emulates it, v * FragCoord.w: the input load v, else 0.
    auto npInner = [&](uint32_t id) -> uint32_t
    {
        const uint32_t* f = m.Op(id, spv::Op::OpFMul);
        for (int o = 0; f && o < 2; ++o)
            if (fragW(f[4 - o]) && m.Op(f[3 + o], spv::Op::OpLoad) && !m.T(f[3 + o])) return f[3 + o];
        return 0;
    };
    // perspInputs: also (1/FragCoord.w) * (v * FragCoord.w), the guest's perspective multiply, which is v where w is 1.
    auto npLoad = [&](uint32_t id) -> uint32_t
    {
        if (const uint32_t v = npInner(id)) return v;
        const uint32_t* f = rules.perspInputs ? m.Op(id, spv::Op::OpFMul) : nullptr;
        for (int o = 0; f && o < 2; ++o)
        {
            const uint32_t* d = m.Op(f[4 - o], spv::Op::OpFDiv);
            if (d && m.IsOne(d[3]) && fragW(d[4]))
                if (const uint32_t v = npInner(f[3 + o])) return v;
        }
        return 0;
    };

    // Pixel taint, in program order (a loop's back edge is missed, which only loses matches).
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(in.op, &hasRes, &hasType);
        if (!hasRes || !hasType) continue;
        uint8_t t = 0;
        switch (in.op)
        {
        case spv::Op::OpLoad:
        {
            if (m.idVars.count(i[3])) { t = 0x80; break; }
            const uint32_t* p = m.Op(i[3], spv::Op::OpAccessChain);
            if (p && (p[0] >> 16) >= 5 && m.idVars.count(p[3]))
            {
                auto c = m.iconst.find(p[4]);
                t = c != m.iconst.end() && c->second < 3 ? (uint8_t)(1u << c->second) : 7;
            }
            // Shared memory addressed by the invocation holds that pixel's value: its depth is read at the pixel.
            else if (tiledLights && p && (p[0] >> 16) >= 6)
            {
                auto c = m.varClass.find(p[3]);
                if (c != m.varClass.end() && c->second == (uint32_t)spv::StorageClass::Workgroup) t = m.T(p[5]);
            }
            break;
        }
        case spv::Op::OpCompositeExtract:
        {
            auto b = m.taint.find(i[3]);
            if (b != m.taint.end()) t = b->second == 0x80 ? (in.n > 4 && i[4] < 3 ? (uint8_t)(1u << i[4]) : 7) : b->second;
            break;
        }
        case spv::Op::OpSelect:
        {
            spv::Op top; const uint32_t* tt = m.At(i[4], &top);
            t = m.T((tt && top == spv::Op::OpBitwiseAnd && tt[3] == i[5]) ? i[5] : i[4]);
            break;
        }
        case spv::Op::OpFMul:
            // A noperspective varying (v * FragCoord.w) is a varying, as v alone, not the pixel.
            if (!npLoad(i[2])) ForIds(i, in.n, in.op, [&](uint32_t id) { t |= m.T(id); });
            break;
        default: ForIds(i, in.n, in.op, [&](uint32_t id) { t |= m.T(id); }); break;
        }
        if (t) m.taint[i[2]] = t;
    }

    std::unordered_map<size_t, std::vector<std::pair<uint16_t, uint32_t>>> rewrite;   // instruction -> (word, new id)
    std::unordered_map<size_t, Emit> before;                                          // emitted ahead of an instruction
    std::unordered_map<size_t, Emit> replace;                                         // emitted in place of one
    uint32_t next = code[3];
    const uint32_t cI0 = next++, cI1 = next++, cI2 = next++, cOne = next++, cFar = next++, vStereo = next++;
    const uint32_t cHalf = next++, cTiny = next++, cI4 = next++;
    // A draw reads fsSep (member 4): the shift its depth was drawn with, which pinned full-screen right-eye draws keep
    // while their sep is 0 (EnsureEyeSet). Compute sets carry the same value in both.
    const uint32_t cSep = fragment ? cI4 : cI0;
    auto stereo = [&](Emit& e, uint32_t& sep, uint32_t& conv, uint32_t& wMin)
    {
        const uint32_t p0 = next++, p1 = next++, p2 = next++;
        sep = next++; conv = next++; wMin = next++;
        e.Op(spv::Op::OpAccessChain, { 0, p0, vStereo, cSep });   // pointer type patched below
        e.Op(spv::Op::OpLoad, { m.tFloat, sep, p0 });
        e.Op(spv::Op::OpAccessChain, { 0, p1, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { m.tFloat, conv, p1 });
        e.Op(spv::Op::OpAccessChain, { 0, p2, vStereo, cI2 });
        e.Op(spv::Op::OpLoad, { m.tFloat, wMin, p2 });
    };
    // Half the width of the image a pixel coordinate fetches: pixels per ndc unit, whatever the render resolution.
    uint32_t fetchImgType = 0, fetchImgVar = 0, fetchVecType = 0, fetchUintType = 0;
    for (size_t k = firstFunc; tiledLights && k < m.ins.size() && !fetchImgVar; ++k)
    {
        if (m.ins[k].op != spv::Op::OpImageFetch) continue;
        const uint32_t* s = code + m.ins[k].at;
        const uint32_t* l = m.Op(s[3], spv::Op::OpLoad);
        // The pixel vector itself, or its bitcast to signed (a hand-assembled module).
        const uint32_t* bc = m.Op(s[4], spv::Op::OpBitcast);
        const uint32_t* cc = m.Op(bc ? bc[3] : s[4], spv::Op::OpCompositeConstruct);
        spv::Op xop; const uint32_t* x = cc && (cc[0] >> 16) == 5 ? m.At(cc[3], &xop) : nullptr;
        if (l && x) { fetchImgType = l[1]; fetchImgVar = l[3]; fetchVecType = cc[1]; fetchUintType = x[1]; }
    }
    auto halfWidth = [&](Emit& e) -> uint32_t
    {
        const uint32_t img = next++, sz = next++, wi = next++, wf = next++, hw = next++;
        queriesSize = true;
        e.Op(spv::Op::OpLoad, { fetchImgType, img, fetchImgVar });
        e.Op(spv::Op::OpImageQuerySizeLod, { fetchVecType, sz, img, cI0 });
        e.Op(spv::Op::OpCompositeExtract, { fetchUintType, wi, sz, 0 });
        e.Op(spv::Op::OpConvertUToF, { m.tFloat, wf, wi });
        e.Op(spv::Op::OpFMul, { m.tFloat, hw, wf, cHalf });
        return hw;
    };
    // A full-screen texture's x read at a point of clip w `w`, moved onto the right's screen: u + 0.5*sep*(1 - conv/w).
    // Texture units, so no game scale enters; w is floored above 0 because the left run's 0 * inf would be NaN.
    std::unordered_set<size_t> shifted;
    auto shiftU = [&](size_t ccAt, uint32_t w)
    {
        const uint32_t* cc = code + m.ins[ccAt].at;
        Emit& e = before[ccAt];
        uint32_t sep, conv, wMin;
        stereo(e, sep, conv, wMin);
        const uint32_t lo = next++, wc = next++, q = next++, df = next++, h = next++, hh = next++, u2 = next++;
        e.Op(spv::Op::OpExtInst, { m.tFloat, lo, m.glsl, GLSLstd450FMax, wMin, cTiny });
        e.Op(spv::Op::OpExtInst, { m.tFloat, wc, m.glsl, GLSLstd450FClamp, w, lo, cFar });
        e.Op(spv::Op::OpFDiv, { m.tFloat, q, conv, wc });
        e.Op(spv::Op::OpFSub, { m.tFloat, df, cOne, q });
        e.Op(spv::Op::OpFMul, { m.tFloat, h, sep, df });
        e.Op(spv::Op::OpFMul, { m.tFloat, hh, h, cHalf });
        e.Op(spv::Op::OpFAdd, { m.tFloat, u2, cc[3], hh });
        rewrite[ccAt].push_back({ 3, u2 });
        shifted.insert(ccAt);
    };
    // The texture read's coordinate when it is a float vector built in place and the image matches `dim`.
    auto sampleCoord = [&](size_t k, spv::Dim dim) -> size_t
    {
        const spv::Op op = m.ins[k].op;
        if (op != spv::Op::OpImageSampleExplicitLod && op != spv::Op::OpImageSampleImplicitLod && op != spv::Op::OpImageGather) return SIZE_MAX;
        const uint32_t* s = code + m.ins[k].at;
        spv::Op sop; const uint32_t* si = m.At(s[3], &sop);
        if (!si) return SIZE_MAX;
        const uint32_t* st = m.Op(si[1], spv::Op::OpTypeSampledImage);
        const uint32_t* it = st ? m.Op(st[2], spv::Op::OpTypeImage) : nullptr;
        const uint32_t* cc = m.Op(s[4], spv::Op::OpCompositeConstruct);
        if (!it || it[3] != (uint32_t)dim || !cc || (cc[0] >> 16) < 5 || !m.IsFloat(cc[3])) return SIZE_MAX;
        return m.def[s[4]];
    };

    // A value's expression through the fork's wrappers, hashed: equal for re-materialized copies (same constants,
    // same loads, same pixel builtins), distinct for anything else.
    std::unordered_map<uint32_t, uint64_t> sigMemo;
    std::function<uint64_t(uint32_t, int)> sig = [&](uint32_t v, int depth) -> uint64_t
    {
        v = m.Strip(v);
        auto it = sigMemo.find(v);
        if (it != sigMemo.end()) return it->second;
        auto mix = [](uint64_t h, uint64_t x) { return (h ^ x) * 0x100000001B3ull + 0x9E3779B97F4A7C15ull; };
        uint64_t h = 0xCBF29CE484222325ull;
        spv::Op op; const uint32_t* i = m.At(v, &op);
        if (!i || depth > 16) h = mix(h, 0xFFFF0000ull | v);
        else if (op == spv::Op::OpConstant) h = mix(mix(h, 1), i[3]);
        else if (op == spv::Op::OpLoad)
        {
            const uint32_t* p = m.Op(i[3], spv::Op::OpAccessChain);
            const auto lb = m.LoadBase(v);
            if (m.idVars.count(i[3])) h = mix(mix(h, 2), i[3]);
            else if (p && m.idVars.count(p[3])) h = mix(mix(mix(h, 3), p[3]), sig(p[4], depth + 1));
            else if (lb.first) h = mix(mix(mix(h, 4), sig(lb.first, depth + 1)), lb.second);
            else h = mix(mix(h, 5), i[3]);
        }
        else
        {
            bool hasRes = false, hasType = false;
            spv::HasResultAndType(op, &hasRes, &hasType);
            h = mix(h, 0x100u + (uint32_t)op);
            if (op == spv::Op::OpExtInst) h = mix(h, i[4]);
            if (op == spv::Op::OpCompositeExtract) for (uint16_t k = 4; k < (uint16_t)(i[0] >> 16); ++k) h = mix(h, i[k]);
            if (op == spv::Op::OpPhi || op == spv::Op::OpFunctionCall || !hasType) h = mix(h, 0xFFFF0000ull | v);
            else ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t id) { h = mix(h, sig(id, depth + 1)); });
        }
        sigMemo[v] = h;
        return h;
    };
    auto Sig = [&](uint32_t v) { return sig(v, 0); };
    uint32_t entryBlock = 0;
    for (size_t k = firstFunc; k < m.ins.size(); ++k) if (m.ins[k].op == spv::Op::OpLabel) { entryBlock = m.ins[k].block; break; }

    // One fma level: Fma(a, b, c), or with `wide` FAdd(FMul(a, b), c) either way round whose product holds one pixel
    // axis. `row` is the instruction that reads a and b.
    auto fmaOf = [&](uint32_t id, uint32_t ab[2], uint32_t& c, uint32_t& row) -> bool
    {
        if (const uint32_t* f = m.Glsl(id, GLSLstd450Fma)) { ab[0] = f[5]; ab[1] = f[6]; c = f[7]; row = id; return true; }
        const uint32_t* s = wide ? m.Op(id, spv::Op::OpFAdd) : nullptr;
        for (int o = 0; s && o < 2; ++o)
        {
            const uint32_t mul = m.Strip(s[3 + o]);
            const uint32_t* f = m.Op(mul, spv::Op::OpFMul);
            const uint8_t t = f ? (uint8_t)(m.T(m.Strip(f[3])) | m.T(m.Strip(f[4]))) : 0;
            if (t != 1 && t != 2) continue;
            ab[0] = f[3]; ab[1] = f[4]; c = s[4 - o]; row = mul;
            return true;
        }
        return false;
    };

    // ndc: the inverse-matrix reconstruction of a screen pixel.
    std::unordered_set<uint32_t> ndcDone;
    // 1/R; (position row, its ndc.x coefficient)
    struct NdcMatch { uint32_t rcp; std::vector<std::pair<uint32_t, uint32_t>> pos; };
    std::vector<NdcMatch> ndcMatches;
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        if (m.ins[k].op != spv::Op::OpFDiv) continue;
        const uint32_t* d = code + m.ins[k].at;
        if (d[1] != m.tFloat || !m.IsOne(d[3])) continue;
        uint32_t node = m.Strip(d[4]), N = 0, P = 0, root = 0, rootRow = 0;
        for (int depth = 0; depth < 4 && !P; ++depth)
        {
            uint32_t ab[2], c, row;
            if (!fmaOf(node, ab, c, row)) { P = node; break; }
            if (!root) { root = node; rootRow = row; }
            const uint32_t a = m.Strip(ab[0]), b = m.Strip(ab[1]);
            const uint8_t ta = m.T(a), tb = m.T(b);
            if ((ta == 1 && tb == 0) || (tb == 1 && ta == 0))
            {
                const uint32_t n = ta == 1 ? a : b;
                if (N || !m.FromBuffer(ta == 1 ? ab[1] : ab[0])) { N = 0; break; }
                N = n;
            }
            else if (!((ta == 2 && tb == 0) || (tb == 2 && ta == 0))) { P = node; break; }
            node = m.Strip(c);
        }
        if (!N || !P || P == root || !m.IsFloat(N) || !m.IsFloat(P)) continue;
        // wide: the depth term must be read at the pixel (both axes), not a d22*x + d23 divisor.
        if (wide && (m.T(P) & 3) != 3) continue;
        if (!ndcDone.insert(N).second) continue;
        // A translator may re-materialize the same ndc expression on several paths: rows reading a copy are moved too.
        const uint64_t sigN = Sig(N);
        std::unordered_set<uint32_t> alias;
        for (size_t c = firstFunc; c < m.ins.size(); ++c)
        {
            const spv::Op op = m.ins[c].op;
            if (op != spv::Op::OpFAdd && op != spv::Op::OpFSub && op != spv::Op::OpExtInst) continue;
            const uint32_t v = code[m.ins[c].at + 2];
            if (v != N && m.IsFloat(v) && m.T(v) == 1 && Sig(v) == sigN) alias.insert(v);
        }
        // Every matrix row that reads this ndc.x or a copy of it: (instruction, operand word, the ndc value it reads).
        struct Row { size_t c; uint16_t slot; uint32_t n; };
        std::vector<Row> rows;
        auto isNdc = [&](uint32_t s) { return s == N || alias.count(s) != 0; };
        size_t rootAt = m.def[rootRow];
        for (size_t c = firstFunc; c < m.ins.size(); ++c)
        {
            const uint32_t* f = code + m.ins[c].at;
            if (m.ins[c].op == spv::Op::OpExtInst && f[3] == m.glsl && f[4] == GLSLstd450Fma)
            {
                if (isNdc(m.Strip(f[5])) && m.FromBuffer(f[6])) rows.push_back({ c, 5, m.Strip(f[5]) });
                else if (isNdc(m.Strip(f[6])) && m.FromBuffer(f[5])) rows.push_back({ c, 6, m.Strip(f[6]) });
            }
            else if (m.ins[c].op == spv::Op::OpFMul && f[1] == m.tFloat)
            {
                if (isNdc(m.Strip(f[3])) && m.FromBuffer(f[4])) rows.push_back({ c, 3, m.Strip(f[3]) });
                else if (isNdc(m.Strip(f[4])) && m.FromBuffer(f[3])) rows.push_back({ c, 4, m.Strip(f[4]) });
            }
        }
        if (rows.empty()) continue;
        const size_t first = rows.front().c;
        if (m.def[P] >= first) continue;                      // the depth term must exist where the rows start
        const uint32_t blk = m.ins[first].block;
        bool hasRoot = false;
        for (const auto& r : rows) if (r.c == rootAt) hasRoot = true;
        if (!hasRoot || m.ins[rootAt].block != blk) continue;
        Emit& e = before[first];
        uint32_t sep, conv, wMin;
        stereo(e, sep, conv, wMin);
        // |1/w| when wide: an inverse scaled by -1 still divides right, but its 1/w would flip the correction.
        uint32_t Pw = P;
        if (wide) { Pw = next++; e.Op(spv::Op::OpExtInst, { m.tFloat, Pw, m.glsl, GLSLstd450FAbs, P }); }
        const uint32_t wp = next++, mx = next++, cp = next++, df = next++, corr = next++, N2 = next++;
        e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, Pw });
        e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
        e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, Pw });
        e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
        e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
        // A tiled-lights x is an integer pixel, not ndc: half a screen per ndc unit.
        uint32_t corrN = corr;
        if (tiledLights && fetchImgVar && (m.Op(N, spv::Op::OpConvertSToF) || m.Op(N, spv::Op::OpConvertUToF)))
        {
            const uint32_t hw = halfWidth(e);
            corrN = next++;
            e.Op(spv::Op::OpFMul, { m.tFloat, corrN, corr, hw });
        }
        e.Op(spv::Op::OpFSub, { m.tFloat, N2, N, corrN });
        ++counts.ndc;
        // corr reaches rows in its own block, or everywhere when that block is the function's first.
        const bool everywhere = blk == entryBlock;
        std::unordered_map<uint32_t, uint32_t> moved{ { N, N2 } };   // ndc value -> its right-camera twin
        NdcMatch nm{ d[2], {} };
        for (const auto& r : rows)
        {
            if (m.ins[r.c].block != blk && !everywhere) continue;
            auto mv = moved.find(r.n);
            if (mv == moved.end())
            {
                if (m.def[r.n] >= r.c) continue;
                const uint32_t n2 = next++;
                before[r.c].Op(spv::Op::OpFSub, { m.tFloat, n2, r.n, corrN });
                mv = moved.emplace(r.n, n2).first;
            }
            rewrite[r.c].push_back({ r.slot, mv->second });
            ++counts.rows;
            const uint32_t* f = code + m.ins[r.c].at;
            const uint16_t other = r.slot == 5 ? 6 : r.slot == 6 ? 5 : r.slot == 3 ? 4 : 3;
            if (r.c != rootAt && r.n == N) nm.pos.push_back({ f[2], f[other] });
        }
        ndcMatches.push_back(nm);
    }

    // uv ray (`uvRays`, level 3): a screen-uv varying times the depth read at it, in a shadow-sampling shader (a sun
    // mask): the product moves by u - 0.5*sep*(1 - conv/z); the depth read keeps its own u.
    bool shadowRead = false;
    for (const Ins& in : m.ins)
        if (in.op == spv::Op::OpImageSampleDrefExplicitLod || in.op == spv::Op::OpImageSampleDrefImplicitLod || in.op == spv::Op::OpImageDrefGather)
        { shadowRead = true; break; }
    uint32_t cTinyDx = 0;   // 1e-20: a derivative's floor (the uv-ray level 4 and the register form below)
    // Level 4 also takes a shader with no shadow read when the product reaches a texture coordinate with no division on
    // the way (a world x/z map lookup); a reprojection (TAA, velocity) divides by w and stays out.
    if (uvRays && fragment && (shadowRead || uvAffine || fragRays))
    {
        // uvAffine (level 4): a value of constants and buffer loads only, through arithmetic and the fork's sign flips.
        std::function<bool(uint32_t, int)> isUni = [&](uint32_t id, int depth) -> bool
        {
            if (m.fconst.count(id) || m.iconst.count(id) || m.FromBuffer(id)) return true;
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 12) return false;
            if (op == spv::Op::OpExtInst && i[3] != m.glsl) return false;
            // Integer steps too: a scale from push-constant dwords, Select(w == 0, 1920, w) as a float.
            if (op != spv::Op::OpFAdd && op != spv::Op::OpFSub && op != spv::Op::OpFMul && op != spv::Op::OpFDiv && op != spv::Op::OpFNegate &&
                op != spv::Op::OpExtInst && op != spv::Op::OpBitcast && op != spv::Op::OpBitwiseXor && op != spv::Op::OpBitwiseAnd &&
                op != spv::Op::OpBitwiseOr && op != spv::Op::OpCopyObject && op != spv::Op::OpConvertUToF && op != spv::Op::OpConvertSToF &&
                op != spv::Op::OpSelect && op != spv::Op::OpIEqual && op != spv::Op::OpINotEqual && op != spv::Op::OpShiftRightLogical) return false;
            bool ok = true;
            ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { ok = ok && isUni(a, depth + 1); });
            return ok;
        };
        // A coordinate's varying under uniform offsets (level 4: depth gathered at fma(0.25, c74, u)); below it, Strip.
        std::function<uint32_t(uint32_t, int)> rootU = [&](uint32_t id, int depth) -> uint32_t
        {
            id = m.Strip(id);
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!uvAffine || !i || depth > 8) return id;
            if ((op == spv::Op::OpFAdd || op == spv::Op::OpFSub) && isUni(i[4], 0)) return rootU(i[3], depth + 1);
            if (op == spv::Op::OpFAdd && isUni(i[3], 0)) return rootU(i[4], depth + 1);
            if (const uint32_t* f = m.Glsl(id, GLSLstd450Fma)) if (isUni(f[5], 0) && isUni(f[6], 0)) return rootU(f[7], depth + 1);
            return id;
        };
        std::unordered_map<uint32_t, uint32_t> sampleX;   // sample result -> its coordinate's x
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const spv::Op op = m.ins[k].op;
            if (op != spv::Op::OpImageSampleExplicitLod && op != spv::Op::OpImageSampleImplicitLod && !(uvAffine && op == spv::Op::OpImageGather)) continue;
            const uint32_t* s = code + m.ins[k].at;
            const uint32_t* cc = m.Op(s[4], spv::Op::OpCompositeConstruct);
            if (cc && (cc[0] >> 16) >= 5) sampleX[s[2]] = rootU(cc[3], 0);
        }
        // The one sample a value is computed from through arithmetic, else 0.
        std::function<uint32_t(uint32_t, int)> sampleOf = [&](uint32_t id, int depth) -> uint32_t
        {
            id = m.Strip(id);
            if (sampleX.count(id)) return id;
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 8) return 0;
            if (op == spv::Op::OpCompositeExtract) return sampleOf(i[3], depth + 1);
            if (op != spv::Op::OpFAdd && op != spv::Op::OpFSub && op != spv::Op::OpFMul && op != spv::Op::OpFDiv &&
                op != spv::Op::OpFNegate && op != spv::Op::OpExtInst && !(uvAffine && op == spv::Op::OpBitwiseXor)) return 0;
            uint32_t found = 0;
            ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a)
            {
                const uint32_t s2 = sampleOf(a, depth + 1);
                if (s2) found = (found && found != s2) ? ~0u : (found == ~0u ? ~0u : s2);
            });
            return found == ~0u ? 0 : found;
        };
        for (size_t k = firstFunc; shadowRead && k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFMul || rewrite.count(k)) continue;
            const uint32_t* f = code + m.ins[k].at;
            if (f[1] != m.tFloat) continue;
            for (int o = 0; o < 2; ++o)
            {
                const uint32_t x = m.Strip(f[3 + o]), z = f[4 - o];
                if (m.T(x) != 0 || !m.IsFloat(x)) continue;   // a varying, not the pixel's own coordinate
                const uint32_t s = sampleOf(z, 0);
                if (!s || sampleX[s] != x) continue;
                Emit& e = before[k];
                uint32_t sep, conv, wMin;
                stereo(e, sep, conv, wMin);
                const uint32_t az = next++, lo = next++, P = next++, wp = next++, mx = next++, cp = next++, df = next++;
                const uint32_t corr = next++, h = next++, x2 = next++;
                e.Op(spv::Op::OpExtInst, { m.tFloat, az, m.glsl, GLSLstd450FAbs, z });
                e.Op(spv::Op::OpExtInst, { m.tFloat, lo, m.glsl, GLSLstd450FMax, az, cTiny });
                e.Op(spv::Op::OpFDiv, { m.tFloat, P, cOne, lo });
                e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, P });
                e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
                e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, P });
                e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
                e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
                e.Op(spv::Op::OpFMul, { m.tFloat, h, corr, cHalf });
                e.Op(spv::Op::OpFSub, { m.tFloat, x2, f[3 + o], h });
                rewrite[k].push_back({ (uint16_t)(3 + o), x2 });
                ++counts.rays;
                break;
            }
        }

        // Level 4: the varying reaches the product affinely, A = k*u + c (as (u - 0.5 + jitter) * -2*c68 times depth);
        // A moves by k*h, k emitted from A's own chain ahead of the product (no derivatives past the sky test).
        struct Slope { bool ok; uint32_t id; };   // id 0: no u in it
        std::function<Slope(uint32_t, uint32_t, Emit&, int)> slopeOf = [&](uint32_t id, uint32_t u, Emit& e, int depth) -> Slope
        {
            id = m.Strip(id);
            if (id == u) return { true, cOne };
            if (isUni(id, 0)) return { true, 0 };
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 10) return { false, 0 };
            auto bin = [&](spv::Op o, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(o, { m.tFloat, r, a, b }); return r; };
            auto neg = [&](uint32_t a) { const uint32_t r = next++; e.Op(spv::Op::OpFNegate, { m.tFloat, r, a }); return r; };
            switch (op)
            {
            case spv::Op::OpFAdd: case spv::Op::OpFSub:
            {
                const Slope a = slopeOf(i[3], u, e, depth + 1), b = slopeOf(i[4], u, e, depth + 1);
                if (!a.ok || !b.ok) return { false, 0 };
                if (!b.id) return a;
                if (!a.id) return { true, op == spv::Op::OpFSub ? neg(b.id) : b.id };
                return { true, bin(op, a.id, b.id) };
            }
            case spv::Op::OpFMul:
            {
                const bool ua = isUni(i[3], 0), ub = isUni(i[4], 0);
                if (ua == ub) return { false, 0 };   // two varying factors: not affine
                const Slope s = slopeOf(ua ? i[4] : i[3], u, e, depth + 1);
                if (!s.ok || !s.id) return s;
                return { true, bin(spv::Op::OpFMul, s.id, ua ? i[3] : i[4]) };
            }
            case spv::Op::OpFNegate:
            {
                const Slope s = slopeOf(i[3], u, e, depth + 1);
                return s.ok && s.id ? Slope{ true, neg(s.id) } : s;
            }
            case spv::Op::OpBitwiseXor:   // the fork's negation: a float's bits Xor the sign bit
            {
                auto c = m.iconst.find(i[4]);
                if (c == m.iconst.end() || c->second != 0x80000000u) return { false, 0 };
                const Slope s = slopeOf(i[3], u, e, depth + 1);
                return s.ok && s.id ? Slope{ true, neg(s.id) } : s;
            }
            case spv::Op::OpExtInst:
            {
                if (i[3] != m.glsl || i[4] != GLSLstd450Fma) return { false, 0 };
                const bool ua = isUni(i[5], 0), ub = isUni(i[6], 0);
                if (!ua && !ub) return { false, 0 };
                Slope p = { true, 0 };
                if (!(ua && ub))
                {
                    const Slope s = slopeOf(ua ? i[6] : i[5], u, e, depth + 1);
                    if (!s.ok) return s;
                    if (s.id) p.id = bin(spv::Op::OpFMul, s.id, ua ? i[5] : i[6]);
                }
                const Slope c = slopeOf(i[7], u, e, depth + 1);
                if (!c.ok) return c;
                if (!c.id) return p;
                if (!p.id) return c;
                return { true, bin(spv::Op::OpFAdd, p.id, c.id) };
            }
            default: return { false, 0 };
            }
        };
        // A product affine in another screen varying V (depth * -V.x, the VS's clip copy): dA/du = dA/dV * dV/du,
        // dV/du = dFdx(V)/dFdx(u) taken at entry, where every lane of the quad is still live.
        size_t entryAt4 = SIZE_MAX;
        if (uvAffine)
        {
            uint32_t mainId4 = 0;
            bool inMain = false;
            for (const Ins& in : m.ins) if (in.op == spv::Op::OpEntryPoint) { mainId4 = code[in.at + 2]; break; }
            for (size_t k = firstFunc; k < m.ins.size() && entryAt4 == SIZE_MAX; ++k)
            {
                const spv::Op op = m.ins[k].op;
                if (!inMain) { inMain = op == spv::Op::OpFunction && code[m.ins[k].at + 2] == mainId4; continue; }
                if (op != spv::Op::OpLabel && op != spv::Op::OpVariable && op != spv::Op::OpFunctionParameter && op != spv::Op::OpLine && op != spv::Op::OpNoLine)
                    entryAt4 = k;
            }
        }
        // A load of an input (a component through a constant index, or a scalar): its variable and component (~0u);
        // np: the load as a noperspective varying, times FragCoord.w.
        auto inputOf = [&](uint32_t load, uint32_t& var, uint32_t& comp, bool& np) -> bool
        {
            const uint32_t v = npLoad(load);
            np = v != 0;
            if (np) load = v;
            const uint32_t* l = m.Op(load, spv::Op::OpLoad);
            if (!l) return false;
            spv::Op pop; const uint32_t* p = m.At(l[3], &pop);
            if (p && (pop == spv::Op::OpAccessChain || pop == spv::Op::OpInBoundsAccessChain) && (p[0] >> 16) == 5)
            {
                auto c = m.iconst.find(p[4]);
                if (c == m.iconst.end()) return false;
                var = p[3]; comp = c->second;
            }
            else { var = l[3]; comp = ~0u; }
            auto vc = m.varClass.find(var);
            return vc != m.varClass.end() && vc->second == (uint32_t)spv::StorageClass::Input;
        };
        std::map<std::pair<uint64_t, uint64_t>, uint32_t> kOfVar;   // (V, u), np in bit 31 of each var -> dV/du
        auto entrySlope = [&](uint32_t vLoad, uint32_t uLoad) -> uint32_t
        {
            uint32_t vv = 0, vc = 0, uv = 0, uc = 0;
            bool vn = false, un = false;
            if (entryAt4 == SIZE_MAX || !inputOf(vLoad, vv, vc, vn) || !inputOf(uLoad, uv, uc, un)) return 0;
            uint32_t& k = kOfVar[{ ((uint64_t)(vv | (vn ? 0x80000000u : 0)) << 32) | vc, ((uint64_t)(uv | (un ? 0x80000000u : 0)) << 32) | uc }];
            if (k) return k;
            Emit& e = before[entryAt4];
            auto ddx = [&](uint32_t var, uint32_t comp, bool np) -> uint32_t
            {
                spv::Op op; const uint32_t* vi = m.At(var, &op);
                const uint32_t* pt = vi ? m.Op(vi[1], spv::Op::OpTypePointer) : nullptr;
                if (!pt) return 0;
                if (comp == ~0u && pt[3] != m.tFloat) return 0;
                const uint32_t l = next++, dd = next++;
                e.Op(spv::Op::OpLoad, { pt[3], l, var });
                if (np)
                {
                    // d(v * FragCoord.w)/dx, FragCoord.w read again at entry.
                    uint32_t v = l;
                    if (comp != ~0u) { v = next++; e.Op(spv::Op::OpCompositeExtract, { m.tFloat, v, l, comp }); }
                    uint32_t w = next++;
                    if (fwPtrType) { const uint32_t p = next++; e.Op(spv::Op::OpAccessChain, { fwPtrType, p, fragCoordVar, fwIndex }); e.Op(spv::Op::OpLoad, { m.tFloat, w, p }); }
                    else { const uint32_t f = next++; e.Op(spv::Op::OpLoad, { fwVecType, f, fragCoordVar }); e.Op(spv::Op::OpCompositeExtract, { m.tFloat, w, f, 3 }); }
                    const uint32_t pw = next++;
                    e.Op(spv::Op::OpFMul, { m.tFloat, pw, v, w });
                    e.Op(spv::Op::OpDPdx, { m.tFloat, dd, pw });
                    return dd;
                }
                e.Op(spv::Op::OpDPdx, { pt[3], dd, l });
                if (comp == ~0u) return dd;
                const uint32_t x = next++;
                e.Op(spv::Op::OpCompositeExtract, { m.tFloat, x, dd, comp });
                return x;
            };
            const uint32_t dv = ddx(vv, vc, vn), du = ddx(uv, uc, un);
            if (!dv || !du) return 0;
            // dv*du / max(du*du, 1e-20): 0 rather than inf where u does not change along x.
            if (!cTinyDx) cTinyDx = next++;
            const uint32_t p = next++, q = next++, r = next++;
            e.Op(spv::Op::OpFMul, { m.tFloat, p, dv, du });
            e.Op(spv::Op::OpFMul, { m.tFloat, q, du, du });
            e.Op(spv::Op::OpExtInst, { m.tFloat, r, m.glsl, GLSLstd450FMax, q, cTinyDx });
            k = next++;
            e.Op(spv::Op::OpFDiv, { m.tFloat, k, p, r });
            return k;
        };
        // The one non-uniform load A is built from, through the arithmetic slopeOf follows; 0 for none or several.
        std::function<bool(uint32_t, int, uint32_t&)> soleLoad = [&](uint32_t id, int depth, uint32_t& leaf) -> bool
        {
            id = m.Strip(id);
            if (isUni(id, 0)) return true;
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 10) return false;
            if (op == spv::Op::OpLoad || npLoad(id)) { if (leaf && leaf != id) return false; leaf = id; return true; }
            const bool fma = op == spv::Op::OpExtInst && i[3] == m.glsl && i[4] == GLSLstd450Fma;
            if (op != spv::Op::OpFAdd && op != spv::Op::OpFSub && op != spv::Op::OpFMul && op != spv::Op::OpFNegate && op != spv::Op::OpBitwiseXor && !fma) return false;
            bool ok = true;
            ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { ok = ok && soleLoad(a, depth + 1, leaf); });
            return ok;
        };
        // No shadow read: a product counts only if it reaches a texture coordinate by arithmetic alone, no division
        // (a reprojection's / w).
        std::unordered_map<uint32_t, std::vector<size_t>> users;
        if (uvAffine && !shadowRead)
            for (size_t k = firstFunc; k < m.ins.size(); ++k)
            {
                bool hasRes = false, hasType = false;
                spv::HasResultAndType(m.ins[k].op, &hasRes, &hasType);
                if (!hasRes || !hasType) continue;
                ForIds(code + m.ins[k].at, m.ins[k].n, m.ins[k].op, [&](uint32_t a) { users[a].push_back(k); });
            }
        auto reachesLookup = [&](uint32_t start) -> bool
        {
            std::vector<uint32_t> todo{ start };
            std::unordered_set<uint32_t> seen{ start };
            while (!todo.empty() && seen.size() < 512)
            {
                const uint32_t id = todo.back(); todo.pop_back();
                auto u = users.find(id);
                if (u == users.end()) continue;
                for (size_t k : u->second)
                {
                    const spv::Op op = m.ins[k].op;
                    const uint32_t* i = code + m.ins[k].at;
                    if (op == spv::Op::OpImageSampleImplicitLod || op == spv::Op::OpImageSampleExplicitLod || op == spv::Op::OpImageGather ||
                        op == spv::Op::OpImageFetch)
                    { if (i[4] == id) return true; continue; }
                    const bool glslArith = op == spv::Op::OpExtInst && i[3] == m.glsl &&
                        (i[4] == GLSLstd450Fma || i[4] == GLSLstd450FClamp || i[4] == GLSLstd450FMin || i[4] == GLSLstd450FMax ||
                         i[4] == GLSLstd450Floor || i[4] == GLSLstd450Fract);
                    const bool arith = op == spv::Op::OpFAdd || op == spv::Op::OpFSub || op == spv::Op::OpFMul || op == spv::Op::OpFNegate ||
                        op == spv::Op::OpCompositeConstruct || op == spv::Op::OpCompositeExtract || op == spv::Op::OpVectorShuffle ||
                        op == spv::Op::OpBitcast || op == spv::Op::OpBitwiseXor || op == spv::Op::OpCopyObject || glslArith;
                    if (arith && seen.insert(i[2]).second) todo.push_back(i[2]);
                }
            }
            return false;
        };
        for (size_t k = firstFunc; uvAffine && k < m.ins.size(); ++k)
        {
            const uint32_t* f = code + m.ins[k].at;
            const bool fmul = m.ins[k].op == spv::Op::OpFMul;
            const bool fma = m.ins[k].op == spv::Op::OpExtInst && f[3] == m.glsl && f[4] == GLSLstd450Fma;
            if ((!fmul && !fma) || rewrite.count(k) || f[1] != m.tFloat) continue;
            const uint16_t a0 = fmul ? 3 : 5;
            for (int o = 0; o < 2; ++o)
            {
                const uint32_t A = f[a0 + o], z = f[a0 + 1 - o];
                if (m.T(A) != 0) continue;
                const uint32_t s = sampleOf(z, 0);
                // The plain product: the level-3 loop above.
                if (!s || (shadowRead && m.Strip(A) == sampleX[s])) continue;
                if (!shadowRead && !reachesLookup(f[2])) continue;
                Emit e;
                Slope sl = slopeOf(A, sampleX[s], e, 0);
                if (!sl.ok)
                {
                    uint32_t leaf = 0;
                    Emit ev;
                    if (soleLoad(A, 0, leaf) && leaf && leaf != sampleX[s])
                    {
                        const Slope sv = slopeOf(A, leaf, ev, 0);
                        const uint32_t kv = sv.ok && sv.id ? entrySlope(leaf, sampleX[s]) : 0;
                        if (kv) { const uint32_t t = next++; ev.Op(spv::Op::OpFMul, { m.tFloat, t, sv.id, kv }); e = ev; sl = { true, t }; }
                    }
                }
                if (!sl.ok || !sl.id) continue;
                uint32_t sep, conv, wMin;
                stereo(e, sep, conv, wMin);
                const uint32_t az = next++, lo = next++, P = next++, wp = next++, mx = next++, cp = next++, df = next++;
                const uint32_t corr = next++, h = next++, kh = next++, a2 = next++;
                e.Op(spv::Op::OpExtInst, { m.tFloat, az, m.glsl, GLSLstd450FAbs, z });
                e.Op(spv::Op::OpExtInst, { m.tFloat, lo, m.glsl, GLSLstd450FMax, az, cTiny });
                e.Op(spv::Op::OpFDiv, { m.tFloat, P, cOne, lo });
                e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, P });
                e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
                e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, P });
                e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
                e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
                e.Op(spv::Op::OpFMul, { m.tFloat, h, corr, cHalf });
                e.Op(spv::Op::OpFMul, { m.tFloat, kh, sl.id, h });
                e.Op(spv::Op::OpFSub, { m.tFloat, a2, A, kh });
                Emit& b = before[k];
                b.v.insert(b.v.end(), e.v.begin(), e.v.end());
                rewrite[k].push_back({ (uint16_t)(a0 + o), a2 });
                ++counts.rays;
                break;
            }
        }

        // fragRays: the pixel's own uv, fma(FragCoord.x*k, 2, -1), times its linearised depth over the left's P00;
        // u counts when a sample takes it as x. Below level 4: u = FragCoord.x * uniforms, ray = k*u + c (buffer k, c).
        if (fragRays)
        {
            // Level 3, no shadow read (a light pass): only a position that reaches no texture coordinate at all; a
            // screen-space effect projects it back through the left camera and must keep its round trip.
            std::unordered_map<uint32_t, std::vector<size_t>> usersR;
            if (!uvAffine && !shadowRead)
                for (size_t k = firstFunc; k < m.ins.size(); ++k)
                {
                    bool hasRes = false, hasType = false;
                    spv::HasResultAndType(m.ins[k].op, &hasRes, &hasType);
                    if (hasRes && hasType) ForIds(code + m.ins[k].at, m.ins[k].n, m.ins[k].op, [&](uint32_t a) { usersR[a].push_back(k); });
                }
            auto reachesCoord = [&](uint32_t start) -> bool
            {
                std::vector<uint32_t> todo{ start };
                std::unordered_set<uint32_t> seen{ start };
                while (!todo.empty())
                {
                    if (seen.size() > 4096) return true;   // too far to follow: stay out
                    const uint32_t id = todo.back(); todo.pop_back();
                    auto u = usersR.find(id);
                    if (u == usersR.end()) continue;
                    for (size_t k : u->second)
                    {
                        const spv::Op op = m.ins[k].op;
                        const uint32_t* i = code + m.ins[k].at;
                        const bool sample = op == spv::Op::OpImageSampleImplicitLod || op == spv::Op::OpImageSampleExplicitLod ||
                            op == spv::Op::OpImageSampleDrefImplicitLod || op == spv::Op::OpImageSampleDrefExplicitLod ||
                            op == spv::Op::OpImageGather || op == spv::Op::OpImageDrefGather || op == spv::Op::OpImageFetch ||
                            op == spv::Op::OpImageRead;
                        if (sample) { if (i[4] == id) return true; continue; }
                        if (seen.insert(i[2]).second) todo.push_back(i[2]);
                    }
                }
                return false;
            };
            std::function<bool(uint32_t, int)> pixScaled = [&](uint32_t id, int depth) -> bool
            {
                const uint32_t* f = m.Op(id, spv::Op::OpFMul);
                if (!f || depth > 4) return false;
                for (int o = 0; o < 2; ++o)
                {
                    const uint32_t a = m.Strip(f[3 + o]);
                    if (!isUni(f[4 - o], 0)) continue;
                    if ((m.Op(a, spv::Op::OpLoad) && m.T(a) == 1) || pixScaled(a, depth + 1)) return true;
                }
                return false;
            };
            std::unordered_set<uint32_t> pixU;
            for (const auto& sx : sampleX)
                if (pixScaled(sx.second, 0)) pixU.insert(sx.second);
            // A fetch coordinate's pixel axis (0 x, 1 y, -1 none) through a NaN guard Select(isnan, 0, v), conversions,
            // clamp and trunc.
            std::function<int(uint32_t, int)> axisOf = [&](uint32_t id, int depth) -> int
            {
                spv::Op op; const uint32_t* i = m.At(id, &op);
                if (!i || depth > 8) return -1;
                auto isConst = [&](uint32_t c) { return m.iconst.count(c) || m.fconst.count(c); };
                switch (op)
                {
                case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpConvertFToS: case spv::Op::OpConvertFToU:
                    return axisOf(i[3], depth + 1);
                case spv::Op::OpSelect:
                    return isConst(i[4]) ? axisOf(i[5], depth + 1) : isConst(i[5]) ? axisOf(i[4], depth + 1) : -1;
                case spv::Op::OpExtInst:
                    if (i[3] == m.glsl && (i[4] == GLSLstd450Trunc || i[4] == GLSLstd450Floor || i[4] == GLSLstd450FClamp)) return axisOf(i[5], depth + 1);
                    return -1;
                case spv::Op::OpLoad: { const uint8_t t = m.T(id); return t == 1 ? 0 : t == 2 ? 1 : -1; }
                default: return -1;
                }
            };
            std::unordered_set<uint32_t> pixFetch;
            for (size_t k = firstFunc; !pixU.empty() && k < m.ins.size(); ++k)
            {
                if (m.ins[k].op != spv::Op::OpImageFetch) continue;
                const uint32_t* s = code + m.ins[k].at;
                const uint32_t* cc = m.Op(s[4], spv::Op::OpCompositeConstruct);
                if (cc && (cc[0] >> 16) >= 5 && axisOf(cc[3], 0) == 0 && axisOf(cc[4], 0) == 1) pixFetch.insert(s[2]);
            }
            // z: arithmetic on one read at the pixel and constants, the read under a division (linearised depth).
            std::function<bool(uint32_t, int, uint32_t&, bool&)> pixDepth = [&](uint32_t id, int depth, uint32_t& leaf, bool& div) -> bool
            {
                id = m.Strip(id);
                if (pixFetch.count(id) || (sampleX.count(id) && pixU.count(sampleX[id]))) { if (leaf && leaf != id) return false; leaf = id; return true; }
                if (isUni(id, 0)) return true;
                spv::Op op; const uint32_t* i = m.At(id, &op);
                if (!i || depth > 10) return false;
                if (op == spv::Op::OpCompositeExtract) return pixDepth(i[3], depth + 1, leaf, div);
                if (op == spv::Op::OpFDiv)
                {
                    if (!pixDepth(i[3], depth + 1, leaf, div)) return false;
                    const uint32_t before4 = leaf;
                    if (!pixDepth(i[4], depth + 1, leaf, div)) return false;
                    if (leaf && !before4) div = true;
                    return true;
                }
                const bool glslArith = op == spv::Op::OpExtInst && i[3] == m.glsl &&
                    (i[4] == GLSLstd450Fma || i[4] == GLSLstd450FAbs || i[4] == GLSLstd450FMax || i[4] == GLSLstd450FMin);
                if (op != spv::Op::OpFAdd && op != spv::Op::OpFSub && op != spv::Op::OpFMul && op != spv::Op::OpFNegate &&
                    op != spv::Op::OpBitwiseXor && !glslArith) return false;
                bool ok = true;
                ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { ok = ok && pixDepth(a, depth + 1, leaf, div); });
                return ok;
            };
            for (size_t k = firstFunc; !pixU.empty() && k < m.ins.size(); ++k)
            {
                const uint32_t* f = code + m.ins[k].at;
                const bool fmul = m.ins[k].op == spv::Op::OpFMul;
                const bool fma = m.ins[k].op == spv::Op::OpExtInst && f[3] == m.glsl && f[4] == GLSLstd450Fma;
                if ((!fmul && !fma) || rewrite.count(k) || f[1] != m.tFloat) continue;
                const uint16_t a0 = fmul ? 3 : 5;
                for (int o = 0; o < 2; ++o)
                {
                    const uint32_t A = f[a0 + o], z = f[a0 + 1 - o];
                    if (m.T(A) != 1) continue;   // the pixel's x alone: its y row stays
                    uint32_t leaf = 0;
                    bool div = false;
                    if (!pixDepth(z, 0, leaf, div) || !leaf || !div) continue;
                    if (!uvAffine && !shadowRead && reachesCoord(f[2])) continue;
                    Emit e;
                    Slope sl = { false, 0 };
                    for (uint32_t u : pixU)
                    {
                        Emit t;
                        const Slope s = slopeOf(A, u, t, 0);
                        if (s.ok && s.id) { sl = s; e = t; break; }
                    }
                    if (!sl.ok || !sl.id) continue;
                    uint32_t sep, conv, wMin;
                    stereo(e, sep, conv, wMin);
                    const uint32_t az = next++, lo = next++, P = next++, wp = next++, mx = next++, cp = next++, df = next++;
                    const uint32_t corr = next++, h = next++, kh = next++, a2 = next++;
                    e.Op(spv::Op::OpExtInst, { m.tFloat, az, m.glsl, GLSLstd450FAbs, z });
                    e.Op(spv::Op::OpExtInst, { m.tFloat, lo, m.glsl, GLSLstd450FMax, az, cTiny });
                    e.Op(spv::Op::OpFDiv, { m.tFloat, P, cOne, lo });
                    e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, P });
                    e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
                    e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, P });
                    e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
                    e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
                    e.Op(spv::Op::OpFMul, { m.tFloat, h, corr, cHalf });
                    e.Op(spv::Op::OpFMul, { m.tFloat, kh, sl.id, h });
                    e.Op(spv::Op::OpFSub, { m.tFloat, a2, A, kh });
                    Emit& b = before[k];
                    b.v.insert(b.v.end(), e.v.begin(), e.v.end());
                    rewrite[k].push_back({ (uint16_t)(a0 + o), a2 });
                    ++counts.rays;
                    break;
                }
            }
        }
    }

    // The same product in register form: mul_nonIEEE(ray, z) through Function-storage registers, the ray any view-ray
    // varying (a sun mask's V.z * -z, depth read at V.xy), moved by dRay/dNDC = dFdx(ray) / (2 dFdx(u)).
    if (uvRays && fragment && shadowRead && !mulFns.empty())
    {
        std::unordered_set<uint32_t> builtins;
        for (const Ins& in : m.ins)
            if (in.op == spv::Op::OpDecorate && in.n >= 4 && code[in.at + 2] == (uint32_t)spv::Decoration::BuiltIn) builtins.insert(code[in.at + 1]);
        // A Function-storage register a pointer addresses: component, ~0u the whole register, ~1u a dynamic index.
        auto slotOf = [&](uint32_t ptr, uint32_t& var, uint32_t& comp) -> bool
        {
            var = ptr; comp = ~0u;
            spv::Op pop; const uint32_t* p = m.At(ptr, &pop);
            if (p && (pop == spv::Op::OpAccessChain || pop == spv::Op::OpInBoundsAccessChain))
            {
                var = p[3];
                auto ic = (p[0] >> 16) == 5 ? m.iconst.find(p[4]) : m.iconst.end();
                comp = ic != m.iconst.end() ? ic->second : ~1u;
            }
            auto vc = m.varClass.find(var);
            return vc != m.varClass.end() && vc->second == (uint32_t)spv::StorageClass::Function;
        };
        struct St { size_t k; uint32_t comp, value; };
        std::unordered_map<uint32_t, std::vector<St>> regStores;   // in program order
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpStore) continue;
            const uint32_t* s = code + m.ins[k].at;
            uint32_t var, comp;
            if (slotOf(s[1], var, comp)) regStores[var].push_back({ k, comp, s[2] });
        }
        // The last store before `at` that wrote component c of `var` (comp ~0u: it stored the whole register).
        auto reaching = [&](uint32_t var, uint32_t c, size_t at, uint32_t& value, uint32_t& comp) -> bool
        {
            auto it = regStores.find(var);
            if (it == regStores.end()) return false;
            for (auto s = it->second.rbegin(); s != it->second.rend(); ++s)
            {
                if (s->k >= at) continue;
                if (s->comp == ~1u) return false;
                if (s->comp == c || s->comp == ~0u) { value = s->value; comp = s->comp; return true; }
                if (c == ~0u) return false;
            }
            return false;
        };
        auto isSample = [](spv::Op op) { return op == spv::Op::OpImageSampleImplicitLod || op == spv::Op::OpImageSampleExplicitLod || op == spv::Op::OpImageGather; };
        auto vecCount = [&](uint32_t id) -> uint32_t
        {
            spv::Op op; const uint32_t* i = m.At(id, &op);
            bool r = false, t = false;
            if (i) spv::HasResultAndType(op, &r, &t);
            const uint32_t* vt = i && t ? m.Op(i[1], spv::Op::OpTypeVector) : nullptr;
            return vt ? vt[3] : 0;
        };
        struct Src { uint32_t var = 0, comp = 0; };
        // Component c (~0u: the value) of `id` through the registers: the instruction computing it (`leaf`, a sample
        // standing for any of its components), or the stage input it was read from (`in`).
        std::function<bool(uint32_t, uint32_t, int, uint32_t&, Src&)> resolve = [&](uint32_t id, uint32_t c, int depth, uint32_t& leaf, Src& in) -> bool
        {
            id = m.Strip(id);
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 32) return false;
            const uint16_t n = (uint16_t)(i[0] >> 16);
            if (op == spv::Op::OpLoad)
            {
                uint32_t var = i[3], comp = c;
                spv::Op pop; const uint32_t* p = m.At(var, &pop);
                if (p && (pop == spv::Op::OpAccessChain || pop == spv::Op::OpInBoundsAccessChain)) var = p[3];
                auto vc = m.varClass.find(var);
                if (vc == m.varClass.end()) return false;
                const uint32_t cls = vc->second;
                if (cls == (uint32_t)spv::StorageClass::Uniform || cls == (uint32_t)spv::StorageClass::StorageBuffer ||
                    cls == (uint32_t)spv::StorageClass::PushConstant) { leaf = id; return c == ~0u; }
                if (var != i[3])
                {
                    auto ic = (p[0] >> 16) == 5 ? m.iconst.find(p[4]) : m.iconst.end();
                    if (ic == m.iconst.end() || c != ~0u) return false;
                    comp = ic->second;
                }
                if (cls == (uint32_t)spv::StorageClass::Input)
                {
                    if (builtins.count(var) || comp > 3) return false;
                    in = { var, comp };
                    return true;
                }
                uint32_t value, sc;
                if (cls != (uint32_t)spv::StorageClass::Function || !reaching(var, comp, m.def[id], value, sc)) return false;
                return resolve(value, sc == ~0u ? comp : ~0u, depth + 1, leaf, in);
            }
            if (op == spv::Op::OpCompositeExtract && n == 5 && c == ~0u) return resolve(i[3], i[4], depth + 1, leaf, in);
            if (op == spv::Op::OpVectorShuffle && c != ~0u)
            {
                const uint32_t n1 = vecCount(i[3]);
                if (!n1 || 5u + c >= n || i[5 + c] == ~0u) return false;
                return i[5 + c] < n1 ? resolve(i[3], i[5 + c], depth + 1, leaf, in) : resolve(i[4], i[5 + c] - n1, depth + 1, leaf, in);
            }
            if (op == spv::Op::OpCompositeConstruct && c != ~0u)
                return n - 3u == vecCount(id) && 3u + c < n && resolve(i[3 + c], ~0u, depth + 1, leaf, in);
            // A gather's uv plus a constant offset.
            if ((op == spv::Op::OpFAdd || op == spv::Op::OpFSub) && c != ~0u)
            {
                if (m.Op(m.Strip(i[4]), spv::Op::OpConstantComposite)) return resolve(i[3], c, depth + 1, leaf, in);
                if (op == spv::Op::OpFAdd && m.Op(m.Strip(i[3]), spv::Op::OpConstantComposite)) return resolve(i[4], c, depth + 1, leaf, in);
                return false;
            }
            leaf = id;
            return c == ~0u || isSample(op);
        };
        // The depth a product scales by: samples that all read at one input's x (`uv`), uniforms and constants, through
        // registers and mul calls.
        std::function<bool(uint32_t, int, Src&, int&)> depthOf = [&](uint32_t id, int depth, Src& uv, int& samples) -> bool
        {
            uint32_t leaf = 0; Src in;
            if (depth > 40 || !resolve(id, ~0u, 0, leaf, in) || in.var) return false;
            spv::Op op; const uint32_t* i = m.At(leaf, &op);
            if (!i) return false;
            if (isSample(op))
            {
                uint32_t l2 = 0; Src u;
                if (!resolve(i[4], 0, 0, l2, u) || !u.var) return false;
                if (samples++ && (u.var != uv.var || u.comp != uv.comp)) return false;
                uv = u;
                return true;
            }
            if (op == spv::Op::OpConstant || op == spv::Op::OpLoad) return true;   // resolve keeps only uniform loads
            if (op == spv::Op::OpFunctionCall)
            {
                if (!mulFns.count(i[3]) || (i[0] >> 16) != 6) return false;
                for (int a = 0; a < 2; ++a)
                {
                    uint32_t v, sc;
                    if (!reaching(i[4 + a], ~0u, m.def[leaf], v, sc) || !depthOf(v, depth + 1, uv, samples)) return false;
                }
                return true;
            }
            if (op != spv::Op::OpFNegate && op != spv::Op::OpFAdd && op != spv::Op::OpFSub && op != spv::Op::OpFMul &&
                op != spv::Op::OpFDiv && op != spv::Op::OpExtInst) return false;
            bool ok = true;
            ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { ok = ok && depthOf(a, depth + 1, uv, samples); });
            return ok;
        };
        // Forward from a product, through values and registers, to a shadow compare.
        auto reachesDref = [&](uint32_t start, size_t from) -> bool
        {
            std::unordered_set<uint32_t> ids{ start };
            std::unordered_set<uint64_t> slots;   // var << 32 | component
            auto tainted = [&](uint32_t var, uint32_t comp) -> bool
            {
                for (uint64_t s : slots)
                    if ((uint32_t)(s >> 32) == var && (comp == ~0u || (uint32_t)s == comp || (uint32_t)s == ~0u)) return true;
                return false;
            };
            for (size_t k = from + 1; k < m.ins.size() && m.ins[k].op != spv::Op::OpFunctionEnd; ++k)
            {
                const Ins& in = m.ins[k];
                const uint32_t* i = code + in.at;
                uint32_t var, comp;
                if (in.op == spv::Op::OpStore)
                {
                    if (!slotOf(i[1], var, comp)) continue;
                    if (ids.count(i[2])) slots.insert(((uint64_t)var << 32) | comp);
                    else if (comp != ~0u) slots.erase(((uint64_t)var << 32) | comp);
                    else for (auto s = slots.begin(); s != slots.end(); ) s = (uint32_t)(*s >> 32) == var ? slots.erase(s) : std::next(s);
                    continue;
                }
                bool hasRes = false, hasType = false;
                spv::HasResultAndType(in.op, &hasRes, &hasType);
                if (!hasRes || !hasType) continue;
                bool t = false;
                if (in.op == spv::Op::OpLoad) t = slotOf(i[3], var, comp) && tainted(var, comp);
                else if (in.op == spv::Op::OpFunctionCall) { for (uint16_t a = 4; a < in.n && !t; ++a) t = slotOf(i[a], var, comp) && tainted(var, ~0u); }
                else ForIds(i, in.n, in.op, [&](uint32_t a) { if (ids.count(a)) t = true; });
                if (!t) continue;
                if (in.op == spv::Op::OpImageSampleDrefImplicitLod || in.op == spv::Op::OpImageSampleDrefExplicitLod || in.op == spv::Op::OpImageDrefGather)
                    return true;
                ids.insert(i[2]);
            }
            return false;
        };
        // The entry point's first instruction past the variables: d/dx in uniform control flow.
        uint32_t mainId = 0;
        for (const Ins& in : m.ins) if (in.op == spv::Op::OpEntryPoint) { mainId = code[in.at + 2]; break; }
        size_t mainAt = SIZE_MAX, mainEnd = SIZE_MAX, entryAt = SIZE_MAX;
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const spv::Op op = m.ins[k].op;
            if (mainAt == SIZE_MAX) { if (op == spv::Op::OpFunction && code[m.ins[k].at + 2] == mainId) mainAt = k; continue; }
            if (op == spv::Op::OpFunctionEnd) { mainEnd = k; break; }
            if (entryAt == SIZE_MAX && op != spv::Op::OpLabel && op != spv::Op::OpVariable && op != spv::Op::OpFunctionParameter &&
                op != spv::Op::OpLine && op != spv::Op::OpNoLine) entryAt = k;
        }
        // An input's type when it is float or a float vector.
        auto inputType = [&](uint32_t var) -> uint32_t
        {
            spv::Op op; const uint32_t* v = m.At(var, &op);
            const uint32_t* pt = v ? m.Op(v[1], spv::Op::OpTypePointer) : nullptr;
            if (!pt) return 0;
            const uint32_t* vt = m.Op(pt[3], spv::Op::OpTypeVector);
            return pt[3] == m.tFloat || (vt && vt[2] == m.tFloat) ? pt[3] : 0;
        };
        std::unordered_map<uint32_t, uint32_t> dxOf;   // input -> its d/dx
        std::unordered_map<uint64_t, uint32_t> kOf;    // (ray, uv) -> dRay/dNDC
        auto dx = [&](const Src& s, Emit& e) -> uint32_t
        {
            const uint32_t ty = inputType(s.var);
            uint32_t& d = dxOf[s.var];
            if (!d) { const uint32_t l = next++; d = next++; e.Op(spv::Op::OpLoad, { ty, l, s.var }); e.Op(spv::Op::OpDPdx, { ty, d, l }); }
            if (ty == m.tFloat) return d;
            const uint32_t x = next++;
            e.Op(spv::Op::OpCompositeExtract, { m.tFloat, x, d, s.comp });
            return x;
        };
        for (size_t k = mainAt + 1; entryAt != SIZE_MAX && mainEnd != SIZE_MAX && k < mainEnd; ++k)
        {
            if (m.ins[k].op != spv::Op::OpFunctionCall || k <= entryAt) continue;
            const uint32_t* f = code + m.ins[k].at;
            uint32_t val[2], sc;
            if (f[1] != m.tFloat || (f[0] >> 16) != 6 || !mulFns.count(f[3])) continue;
            if (!reaching(f[4], ~0u, k, val[0], sc) || !reaching(f[5], ~0u, k, val[1], sc)) continue;
            for (int o = 0; o < 2; ++o)
            {
                uint32_t leaf = 0; Src ray, uv; int samples = 0;
                if (!resolve(val[o], ~0u, 0, leaf, ray) || !ray.var || !inputType(ray.var)) continue;
                if (!depthOf(val[1 - o], 0, uv, samples) || !samples || !inputType(uv.var)) continue;
                if (!reachesDref(f[2], k)) break;
                Emit& ent = before[entryAt];
                uint32_t& kk = kOf[((uint64_t)ray.var << 34) | ((uint64_t)ray.comp << 32) | ((uint64_t)uv.var << 2) | uv.comp];
                if (!kk)
                {
                    const uint32_t dr = dx(ray, ent), du = dx(uv, ent);
                    if (!cTinyDx) cTinyDx = next++;
                    const uint32_t den = next++, ad = next++, md = next++, sg = next++, q = next++;
                    kk = next++;
                    ent.Op(spv::Op::OpFAdd, { m.tFloat, den, du, du });
                    ent.Op(spv::Op::OpExtInst, { m.tFloat, ad, m.glsl, GLSLstd450FAbs, den });
                    ent.Op(spv::Op::OpExtInst, { m.tFloat, md, m.glsl, GLSLstd450FMax, ad, cTinyDx });
                    ent.Op(spv::Op::OpExtInst, { m.tFloat, sg, m.glsl, GLSLstd450FSign, den });
                    ent.Op(spv::Op::OpFDiv, { m.tFloat, q, dr, md });
                    ent.Op(spv::Op::OpFMul, { m.tFloat, kk, q, sg });
                }
                // Ahead of the call: its ray parameter becomes ray - sep*(max(1, wMin/|z|) - conv/|z|)*dRay/dNDC.
                Emit& e = before[k];
                uint32_t sep, conv, wMin;
                stereo(e, sep, conv, wMin);
                const uint32_t ra = next++, za = next++, az = next++, lo = next++, P = next++, wp = next++, mx = next++, cp = next++;
                const uint32_t df = next++, corr = next++, h = next++, x2 = next++;
                e.Op(spv::Op::OpLoad, { m.tFloat, ra, f[4 + o] });
                e.Op(spv::Op::OpLoad, { m.tFloat, za, f[5 - o] });
                e.Op(spv::Op::OpExtInst, { m.tFloat, az, m.glsl, GLSLstd450FAbs, za });
                e.Op(spv::Op::OpExtInst, { m.tFloat, lo, m.glsl, GLSLstd450FMax, az, cTiny });
                e.Op(spv::Op::OpFDiv, { m.tFloat, P, cOne, lo });
                e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, P });
                e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
                e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, P });
                e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
                e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
                e.Op(spv::Op::OpFMul, { m.tFloat, h, corr, kk });
                e.Op(spv::Op::OpFSub, { m.tFloat, x2, ra, h });
                e.Op(spv::Op::OpStore, { f[4 + o], x2 });
                ++counts.rays;
                break;
            }
        }
    }

    // varying ndc (level 3): a full-screen shadow mask whose x is the quad's clip copy moves x to the left eye's in
    // every row reading it; 1/w from the camera row `storeCamera` shaders keep in binding 1 [6], [7].
    uint32_t vCam = 0, tPtrCF = 0, cI6 = 0, cI7 = 0, cI16 = 0;
    bool newPtrCF = false, camStores = false;
    auto camSlot = [&](Emit& e, int slot) -> uint32_t   // slot 6 or 7; the ids exist only after the first call
    {
        if (!vCam)
        {
            vCam = next++; cI6 = next++; cI7 = next++; cI16 = next++;
            auto pf = ptrOf.find(((uint64_t)spv::StorageClass::StorageBuffer << 32) | m.tFloat);
            if (pf != ptrOf.end()) tPtrCF = pf->second; else { tPtrCF = next++; newPtrCF = true; }
        }
        const uint32_t p = next++;
        e.Op(spv::Op::OpAccessChain, { tPtrCF, p, vCam, cI0, slot == 6 ? cI6 : cI7 });
        return p;
    };
    if (uvRays && fragment && (shadowRead || (storeCamera && counters)))
    {
        std::unordered_set<uint32_t> builtinVar;
        for (const Ins& in : m.ins)
            if (in.op == spv::Op::OpDecorate && in.n >= 4 && code[in.at + 2] == (uint32_t)spv::Decoration::BuiltIn) builtinVar.insert(code[in.at + 1]);
        // The input components (bit c) of one non-builtin input a value is built from; 0 when it reads anything else.
        std::function<bool(uint32_t, int, uint32_t&, uint32_t&)> inputsOf = [&](uint32_t id, int depth, uint32_t& var, uint32_t& mask) -> bool
        {
            id = m.Strip(id);
            if (m.fconst.count(id) || m.iconst.count(id)) return true;
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 16) return false;
            uint32_t base = 0, comp = ~0u;
            if (op == spv::Op::OpLoad)
            {
                spv::Op pop; const uint32_t* p = m.At(i[3], &pop);
                base = i[3];
                if (p && (pop == spv::Op::OpAccessChain || pop == spv::Op::OpInBoundsAccessChain))
                {
                    base = p[3];
                    auto c = m.iconst.find(p[(p[0] >> 16) - 1]);
                    if (c != m.iconst.end()) comp = c->second;
                }
                auto vc = m.varClass.find(base);
                if (vc == m.varClass.end() || vc->second != (uint32_t)spv::StorageClass::Input) return false;
                if (builtinVar.count(base)) return true;
                if (comp > 3 || (var && var != base)) return false;
                var = base; mask |= 1u << comp;
                return true;
            }
            if (op != spv::Op::OpFAdd && op != spv::Op::OpFSub && op != spv::Op::OpFMul && op != spv::Op::OpFDiv &&
                op != spv::Op::OpFNegate && op != spv::Op::OpExtInst && op != spv::Op::OpCompositeExtract) return false;
            bool ok = true;
            ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { ok = ok && inputsOf(a, depth + 1, var, mask); });
            return ok;
        };
        // 0 for x/w (or x: a lighting FS may skip w == 1), 1 for y; -1 otherwise.
        auto axisOf = [&](uint32_t v) -> int
        {
            uint32_t var = 0, mask = 0;
            if (m.T(v) != 0 || !m.IsFloat(v) || !inputsOf(v, 0, var, mask) || !var) return -1;
            return (mask == 0x9u || mask == 0x1u) ? 0 : (mask == 0xAu || mask == 0x2u) ? 1 : -1;
        };
        // A value re-emitted at instruction `at` when it is defined there or later: loads, their chains and offsets.
        std::function<uint32_t(uint32_t, size_t, Emit&, int)> clone = [&](uint32_t id, size_t at, Emit& e, int depth) -> uint32_t
        {
            auto d = m.def.find(id);
            if (d == m.def.end() || d->second < at || m.ins[d->second].op == spv::Op::OpVariable) return id;
            const Ins& in = m.ins[d->second];
            if (depth > 8 || (in.op != spv::Op::OpBitcast && in.op != spv::Op::OpCopyObject && in.op != spv::Op::OpLoad &&
                in.op != spv::Op::OpAccessChain && in.op != spv::Op::OpInBoundsAccessChain && in.op != spv::Op::OpIAdd)) return 0;
            std::vector<uint32_t> w(code + in.at, code + in.at + in.n);
            const uint16_t to = in.op == spv::Op::OpLoad ? 4 : in.n;   // a load's memory operands are literals
            for (uint16_t k = 3; k < to; ++k)
            {
                const uint32_t c = clone(w[k], at, e, depth + 1);
                if (!c) return 0;
                w[k] = c;
            }
            w[2] = next++;
            e.v.insert(e.v.end(), w.begin(), w.end());
            return w[2];
        };
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFDiv) continue;
            const uint32_t* d = code + m.ins[k].at;
            if (d[1] != m.tFloat || !m.IsOne(d[3])) continue;
            // R's terms through its FAdd tree, as the float operands the adds take (a stripped load is the uint).
            std::vector<uint32_t> terms, stack{ d[4] };
            std::unordered_set<uint32_t> termIds;
            while (!stack.empty() && terms.size() <= 6)
            {
                const uint32_t v = stack.back(); stack.pop_back();
                if (const uint32_t* a = m.Op(m.Strip(v), spv::Op::OpFAdd)) { stack.push_back(a[3]); stack.push_back(a[4]); }
                else { terms.push_back(v); termIds.insert(m.Strip(v)); }
            }
            uint32_t X = 0, z = 0, zCoef = 0, cTerm = 0;
            int depthTerms = 0;
            bool ok = terms.size() <= 4;
            for (uint32_t t : terms)
            {
                if (!ok) break;
                if (m.FromBuffer(t)) { ok = !cTerm && m.IsFloat(t); cTerm = t; continue; }
                const uint32_t* f = m.Op(m.Strip(t), spv::Op::OpFMul);
                if (!f) { ok = false; break; }
                const bool b3 = m.FromBuffer(f[3]), b4 = m.FromBuffer(f[4]);
                if (b3 == b4) { ok = false; break; }
                const uint32_t v = m.Strip(b3 ? f[4] : f[3]);
                const int ax = axisOf(v);
                if (ax == 0) { ok = !X; X = v; }
                else if (ax == 1) {}
                else if ((m.T(v) & 3) == 3) { ++depthTerms; z = b3 ? f[4] : f[3]; zCoef = b3 ? f[3] : f[4]; }
                else ok = false;
            }
            if (!ok || !X || depthTerms != 1 || !cTerm) continue;
            if (storeCamera && counters && !camStores)
            {
                // The camera's w row: [6] its depth coefficient, [7] its constant (the camera-window slots' meaning).
                Emit& st = before[k];
                st.Op(spv::Op::OpStore, { camSlot(st, 6), zCoef });
                st.Op(spv::Op::OpStore, { camSlot(st, 7), cTerm });
                camStores = true;
            }
            if (!shadowRead) continue;
            // A matrix element in any shape: buffer loads (also vec4 loads split per component, cascade matrices
            // indexed by the cascade picked) and constants, nothing per pixel.
            std::unordered_map<uint32_t, bool> uniMemo;
            std::function<bool(uint32_t, int)> uniform = [&](uint32_t id, int depth) -> bool
            {
                id = m.Strip(id);
                if (m.fconst.count(id) || m.iconst.count(id) || m.FromBuffer(id)) return true;
                auto mm = uniMemo.find(id);
                if (mm != uniMemo.end()) return mm->second;
                spv::Op op; const uint32_t* i = m.At(id, &op);
                // shadPS4 builds each cascade vec4 as VectorShuffle(OpConstantComposite, loaded, 4 5 6 7).
                if (i && (op == spv::Op::OpConstantComposite || op == spv::Op::OpConstantNull || op == spv::Op::OpUndef)) return true;
                bool r = false;
                if (i && depth < 12 && m.T(id) == 0 &&
                    (op == spv::Op::OpCompositeExtract || op == spv::Op::OpCompositeConstruct || op == spv::Op::OpVectorShuffle ||
                     op == spv::Op::OpFAdd || op == spv::Op::OpFSub || op == spv::Op::OpFMul || op == spv::Op::OpFNegate))
                {
                    r = true;
                    ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { r = r && uniform(a, depth + 1); });
                }
                uniMemo[id] = r;
                return r;
            };
            // The rows reading x, R's own term aside.
            struct Row { size_t c; uint16_t slot; };
            std::vector<Row> rows;
            for (size_t c = firstFunc; c < m.ins.size(); ++c)
            {
                if (m.ins[c].op != spv::Op::OpFMul || rewrite.count(c)) continue;
                const uint32_t* f = code + m.ins[c].at;
                if (f[1] != m.tFloat || termIds.count(f[2])) continue;
                if (m.Strip(f[3]) == X && uniform(f[4], 0)) rows.push_back({ c, 3 });
                else if (m.Strip(f[4]) == X && uniform(f[3], 0)) rows.push_back({ c, 4 });
            }
            if (rows.empty()) continue;
            const size_t first = rows.front().c;
            const uint32_t blk = m.ins[first].block;
            if (m.def[X] >= first) continue;
            if (m.def[z] >= first) { z = m.Strip(z); if (m.def[z] >= first) continue; }
            Emit e;
            const uint32_t cz = clone(zCoef, first, e, 0), cd = clone(cTerm, first, e, 0);
            if (!cz || !cd) continue;
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            const uint32_t zc = next++, R2 = next++, Pw = next++, wp = next++, mx = next++, cp = next++, df = next++, corr = next++, X2 = next++;
            e.Op(spv::Op::OpFMul, { m.tFloat, zc, cz, z });
            e.Op(spv::Op::OpFAdd, { m.tFloat, R2, zc, cd });
            uint32_t Rw = R2;
            if (counters)
            {
                // The stored camera row when one is there (n32 != 0), else this shader's own.
                const uint32_t s6 = next++, s7 = next++, sz = next++, Rs = next++, a6 = next++, has = next++, mixd = next++;
                e.Op(spv::Op::OpLoad, { m.tFloat, s6, camSlot(e, 6) });
                e.Op(spv::Op::OpLoad, { m.tFloat, s7, camSlot(e, 7) });
                e.Op(spv::Op::OpFMul, { m.tFloat, sz, s6, z });
                e.Op(spv::Op::OpFAdd, { m.tFloat, Rs, sz, s7 });
                e.Op(spv::Op::OpExtInst, { m.tFloat, a6, m.glsl, GLSLstd450FAbs, s6 });
                e.Op(spv::Op::OpExtInst, { m.tFloat, has, m.glsl, GLSLstd450Step, cTiny, a6 });
                e.Op(spv::Op::OpExtInst, { m.tFloat, mixd, m.glsl, GLSLstd450FMix, R2, Rs, has });
                Rw = mixd;
            }
            e.Op(spv::Op::OpExtInst, { m.tFloat, Pw, m.glsl, GLSLstd450FAbs, Rw });
            e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, Pw });
            e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
            e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, Pw });
            e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
            e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
            e.Op(spv::Op::OpFSub, { m.tFloat, X2, X, corr });
            Emit& at = before[first];
            at.v.insert(at.v.end(), e.v.begin(), e.v.end());
            ++counts.ndc;
            const bool everywhere = blk == entryBlock;
            for (const Row& r : rows)
            {
                if (m.ins[r.c].block != blk && !everywhere) continue;
                rewrite[r.c].push_back({ r.slot, X2 });
                ++counts.rows;
            }
        }
    }

    // eye: the view vector camPos - worldPos starts at the left camera; the right camera sits sep*conv * (rotation *
    // inverse-projection column 0) away: the view rows' ndc.x coefficients, rotated.
    for (const NdcMatch& nm : ndcMatches)
    {
        if (nm.pos.size() != 3) continue;
        std::unordered_map<uint32_t, int> posOf;   // view-space position component -> its row
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFMul) continue;
            const uint32_t* f = code + m.ins[k].at;
            const uint32_t a = m.Strip(f[3]), b = m.Strip(f[4]);
            const uint32_t other = a == nm.rcp ? b : b == nm.rcp ? a : 0;
            for (int r = 0; r < 3 && other; ++r) if (other == nm.pos[r].first) posOf[f[2]] = r;
        }
        if (posOf.size() < 3) continue;
        // rotation * view + translation, one fma per component, from the translation load up.
        struct Chain { uint32_t coef[3] = {}; bool has[3] = {}; uint32_t trans = 0; };
        std::unordered_map<uint32_t, Chain> chain;
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpExtInst) continue;
            const uint32_t* f = code + m.ins[k].at;
            if (f[3] != m.glsl || f[4] != GLSLstd450Fma) continue;
            for (int o = 0; o < 2; ++o)
            {
                auto p = posOf.find(m.Strip(f[5 + o]));
                if (p == posOf.end() || !m.FromBuffer(f[6 - o])) continue;
                Chain c;
                auto prev = chain.find(m.Strip(f[7]));
                if (prev != chain.end()) c = prev->second;
                else if (m.FromBuffer(f[7])) c.trans = f[7];
                else break;
                if (c.has[p->second]) break;
                c.has[p->second] = true;
                c.coef[p->second] = f[6 - o];
                chain[f[2]] = c;
                break;
            }
        }
        // Light vectors (lightPos - world) have the same shape; the camera is the one loaded beside the translation.
        // A pass whose loads carry no buffer address keeps the vector only when it is the pass's single one.
        struct Cand { size_t k; const Chain* cw; bool camMinusWorld; bool sameBase; bool addressed; };
        std::vector<Cand> cands;
        auto full = [&](uint32_t v) -> const Chain* { auto c = chain.find(v); return c != chain.end() && c->second.has[0] && c->second.has[1] && c->second.has[2] ? &c->second : nullptr; };
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFSub || rewrite.count(k) || replace.count(k)) continue;
            const uint32_t* f = code + m.ins[k].at;
            if (f[1] != m.tFloat) continue;
            const Chain* cw = full(m.Strip(f[4]));
            bool camMinusWorld = cw && m.FromBuffer(f[3]);
            if (!camMinusWorld) { cw = full(m.Strip(f[3])); if (!cw || !m.FromBuffer(f[4])) continue; }
            const auto cam = m.LoadBase(camMinusWorld ? f[3] : f[4]), tr = m.LoadBase(cw->trans);
            cands.push_back({ k, cw, camMinusWorld, cam.first && cam.first == tr.first, cam.first && tr.first });
        }
        // Exactly one such vector or none: a shader may load a light beside the camera, and offsets in its own packed
        // block cannot say which is which.
        std::vector<const Cand*> take;
        for (const Cand& cd : cands)
            if (cd.addressed ? cd.sameBase : cands.size() == 3) take.push_back(&cd);
        if (take.size() != 3) continue;
        for (const Cand* cp : take)
        {
            const Cand& cd = *cp;
            const size_t k = cd.k;
            const uint32_t* f = code + m.ins[k].at;
            const Chain* cw = cd.cw;
            const bool camMinusWorld = cd.camMinusWorld;
            Emit& e = replace[k];
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            const uint32_t t0 = next++, t1 = next++, t2 = next++, s1 = next++, s2 = next++, sc = next++, dl = next++, raw = next++;
            e.Op(spv::Op::OpFMul, { m.tFloat, t0, cw->coef[0], nm.pos[0].second });
            e.Op(spv::Op::OpFMul, { m.tFloat, t1, cw->coef[1], nm.pos[1].second });
            e.Op(spv::Op::OpFMul, { m.tFloat, t2, cw->coef[2], nm.pos[2].second });
            e.Op(spv::Op::OpFAdd, { m.tFloat, s1, t0, t1 });
            e.Op(spv::Op::OpFAdd, { m.tFloat, s2, s1, t2 });
            e.Op(spv::Op::OpFMul, { m.tFloat, sc, sep, conv });
            e.Op(spv::Op::OpFMul, { m.tFloat, dl, sc, s2 });
            e.Op(spv::Op::OpFSub, { m.tFloat, raw, f[3], f[4] });
            e.Op(camMinusWorld ? spv::Op::OpFAdd : spv::Op::OpFSub, { m.tFloat, f[2], raw, dl });
            ++counts.eye;
        }
    }

    // ray: a froxel's view-space x, paired with a y of the same depth.
    struct RayArg { uint32_t n, d, b; };
    auto rayArgs = [&](const uint32_t* f, RayArg& r) -> bool
    {
        for (int o = 0; o < 2; ++o)
        {
            const uint32_t nRaw = f[3 + o], dRaw = f[4 - o];
            const uint32_t n = m.Strip(nRaw), dd = m.Strip(dRaw);
            const uint32_t* fm = m.Glsl(n, GLSLstd450Fma);
            if (!fm || m.T(dd) == 1 || m.T(dd) == 2 || dd == n) continue;
            // Scale and offset are camera constants; a constant -1 offset is a unit grid, not a frustum.
            const uint8_t t0 = m.T(m.Strip(fm[5])), t1 = m.T(m.Strip(fm[6]));
            if (!((t0 == 0) ^ (t1 == 0)) || m.T(m.Strip(fm[7])) != 0) continue;
            if (!m.FromBuffer(t0 == 0 ? fm[5] : fm[6]) || !m.FromBuffer(fm[7])) continue;
            r = { n, dd, fm[7] };
            return true;
        }
        return false;
    };
    std::unordered_set<uint32_t> yDepths;
    std::vector<std::pair<size_t, RayArg>> xRays;
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        if (m.ins[k].op != spv::Op::OpFMul) continue;
        const uint32_t* f = code + m.ins[k].at;
        RayArg r;
        if (f[1] != m.tFloat || !rayArgs(f, r)) continue;
        const uint8_t t = m.T(r.n);
        if (t == 2) yDepths.insert(r.d);
        else if (t == 1) xRays.push_back({ k, r });
    }
    std::unordered_set<uint32_t> rayIds;
    std::unordered_map<uint32_t, uint32_t> movedRays;   // a corrected ray -> its offset term b
    for (const auto& xr : xRays)
    {
        if (!yDepths.count(xr.second.d) || rewrite.count(xr.first)) continue;
        const uint32_t* f = code + m.ins[xr.first].at;
        const uint32_t dRaw = m.Strip(f[3]) == xr.second.n ? f[4] : f[3];
        Emit& e = replace[xr.first];
        uint32_t sep, conv, wMin;
        stereo(e, sep, conv, wMin);
        const uint32_t raw = next++, ab = next++, ad = next++, md = next++, df = next++, kk = next++, corr = next++;
        e.Op(spv::Op::OpFMul, { m.tFloat, raw, f[3], f[4] });
        e.Op(spv::Op::OpExtInst, { m.tFloat, ab, m.glsl, GLSLstd450FAbs, xr.second.b });
        e.Op(spv::Op::OpExtInst, { m.tFloat, ad, m.glsl, GLSLstd450FAbs, dRaw });
        e.Op(spv::Op::OpExtInst, { m.tFloat, md, m.glsl, GLSLstd450FMax, ad, wMin });
        e.Op(spv::Op::OpFSub, { m.tFloat, df, md, conv });
        e.Op(spv::Op::OpFMul, { m.tFloat, kk, sep, ab });
        e.Op(spv::Op::OpFMul, { m.tFloat, corr, kk, df });
        e.Op(spv::Op::OpFSub, { m.tFloat, f[2], raw, corr });
        rayIds.insert(f[2]);
        movedRays[xr.second.n] = xr.second.b;
        ++counts.rays;
    }
    // The same ray in a fused product: a froxel's far end, ray*zFar - xNear, the segment its light samples walk, moved
    // by sep*|b|*(zFar - conv) so the right eye's samples run along its own ray.
    for (size_t k = firstFunc; k < m.ins.size() && !movedRays.empty(); ++k)
    {
        if (m.ins[k].op != spv::Op::OpExtInst || rewrite.count(k) || replace.count(k)) continue;
        const uint32_t* fm = code + m.ins[k].at;
        if (fm[1] != m.tFloat || fm[3] != m.glsl || fm[4] != GLSLstd450Fma) continue;
        for (int o = 0; o < 2; ++o)
        {
            auto ray = movedRays.find(m.Strip(fm[5 + o]));
            if (ray == movedRays.end()) continue;
            Emit& e = replace[k];
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            const uint32_t raw = next++, ab = next++, ad = next++, md = next++, df = next++, kk = next++, corr = next++;
            e.Op(spv::Op::OpExtInst, { m.tFloat, raw, m.glsl, GLSLstd450Fma, fm[5], fm[6], fm[7] });
            e.Op(spv::Op::OpExtInst, { m.tFloat, ab, m.glsl, GLSLstd450FAbs, ray->second });
            e.Op(spv::Op::OpExtInst, { m.tFloat, ad, m.glsl, GLSLstd450FAbs, fm[6 - o] });
            e.Op(spv::Op::OpExtInst, { m.tFloat, md, m.glsl, GLSLstd450FMax, ad, wMin });
            e.Op(spv::Op::OpFSub, { m.tFloat, df, md, conv });
            e.Op(spv::Op::OpFMul, { m.tFloat, kk, sep, ab });
            e.Op(spv::Op::OpFMul, { m.tFloat, corr, kk, df });
            e.Op(spv::Op::OpFSub, { m.tFloat, fm[2], raw, corr });
            rayIds.insert(fm[2]);
            ++counts.rays;
            break;
        }
    }

    // reproj: values built from a corrected ray, then 3D samples whose x is a projection of one.
    if (!rayIds.empty())
    {
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const Ins& in = m.ins[k];
            const uint32_t* i = code + in.at;
            bool hasRes = false, hasType = false;
            spv::HasResultAndType(in.op, &hasRes, &hasType);
            if (!hasRes || !hasType || rayIds.count(i[2])) continue;
            bool dep = false;
            ForIds(i, in.n, in.op, [&](uint32_t id) { if (rayIds.count(id)) dep = true; });
            if (dep) rayIds.insert(i[2]);
        }
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const size_t ccAt = sampleCoord(k, spv::Dim::Dim3D);
            if (ccAt == SIZE_MAX || shifted.count(ccAt) || rewrite.count(ccAt)) continue;
            const uint32_t* cc = code + m.ins[ccAt].at;
            uint32_t u = m.Strip(cc[3]);
            for (int g = 0; g < 4; ++g)
            {
                spv::Op uop; const uint32_t* ui = m.At(u, &uop);
                if (!ui || uop != spv::Op::OpExtInst || ui[3] != m.glsl ||
                    (ui[4] != GLSLstd450FClamp && ui[4] != GLSLstd450NClamp && ui[4] != GLSLstd450FMin && ui[4] != GLSLstd450FMax)) break;
                u = m.Strip(ui[5]);
            }
            const uint32_t* uf = m.Glsl(u, GLSLstd450Fma);
            if (!uf) continue;
            for (int o = 0; o < 2; ++o)
            {
                // k/R or 1/R times the clip x row.
                uint32_t r = m.Strip(uf[5 + o]);
                if (const uint32_t* mul = m.Op(r, spv::Op::OpFMul))
                {
                    const uint32_t a = m.Strip(mul[3]), b = m.Strip(mul[4]);
                    const uint32_t* da = m.Op(a, spv::Op::OpFDiv); const uint32_t* db = m.Op(b, spv::Op::OpFDiv);
                    r = da && m.IsOne(da[3]) ? a : db && m.IsOne(db[3]) ? b : 0;
                }
                const uint32_t* dv = r ? m.Op(r, spv::Op::OpFDiv) : nullptr;
                if (!dv || !m.IsOne(dv[3]) || !rayIds.count(m.Strip(dv[4])) || !m.IsFloat(dv[4])) continue;
                // The texture u moves, not the clip x row (whose scale is the game's).
                shiftU(ccAt, dv[4]);
                ++counts.reproj;
                break;
            }
        }
    }
    // Tiled lights: lights are culled by the tile's column and centre planes, the left camera's; the right run tests
    // the left-eye columns its tile's depth range maps to (sep 0: the game's own).
    uint32_t cTile = 0, cTileInv = 0, cZero = 0;
    // The tile: column x16 = WorkgroupId.x * 16, table byte x16 + K, dword base byte >> 2, centre float(x0 + x16 + 8),
    // and the raw depths zMin/zMax (element 0/1 of the tile's range) in 1/(c1*z + c0).
    uint32_t tZMin = 0, tZMax = 0, tC0 = 0, tC1 = 0, tX16 = 0, tTable = 0, tBase = 0, tCx = 0;
    auto isConst = [&](uint32_t id, uint32_t v) { auto c = m.iconst.find(id); return c != m.iconst.end() && c->second == v; };
    // A vector component's index, extracted or loaded through a constant index (2: neither), and that vector.
    auto element = [&](uint32_t id) -> uint32_t
    {
        if (const uint32_t* x = m.Op(id, spv::Op::OpCompositeExtract)) return (x[0] >> 16) == 5 ? x[4] : 2;
        const uint32_t* l = m.Op(id, spv::Op::OpLoad);
        const uint32_t* p = l ? m.Op(l[3], spv::Op::OpAccessChain) : nullptr;
        auto c = p && (p[0] >> 16) == 5 ? m.iconst.find(p[4]) : m.iconst.end();
        return c != m.iconst.end() ? c->second : 2;
    };
    auto vectorOf = [&](uint32_t id) -> uint32_t
    {
        if (const uint32_t* x = m.Op(id, spv::Op::OpCompositeExtract)) { const uint32_t* l = m.Op(x[3], spv::Op::OpLoad); return l ? l[3] : 0; }
        const uint32_t* l = m.Op(id, spv::Op::OpLoad);
        const uint32_t* p = l ? m.Op(l[3], spv::Op::OpAccessChain) : nullptr;
        return p ? p[3] : 0;
    };
    for (size_t k = firstFunc; tiledLights && counts.ndc && fetchImgVar && wgVar && k < m.ins.size() && !tBase; ++k)
    {
        if (m.ins[k].op != spv::Op::OpShiftRightLogical) continue;
        const uint32_t* s = code + m.ins[k].at;
        const uint32_t* t = isConst(s[4], 2) ? m.Op(s[3], spv::Op::OpIAdd) : nullptr;
        const uint32_t* x = t && m.iconst.count(t[4]) ? m.Op(t[3], spv::Op::OpIMul) : nullptr;
        if (x && isConst(x[4], 16) && vectorOf(x[3]) == wgVar && element(x[3]) == 0) { tBase = s[2]; tTable = t[2]; tX16 = x[2]; }
    }
    for (size_t k = tBase ? m.def[tX16] : m.ins.size(); k < m.ins.size() && !tCx; ++k)
    {
        if (m.ins[k].op != spv::Op::OpConvertUToF) continue;
        const uint32_t* f = code + m.ins[k].at;
        const uint32_t* a = m.Op(f[3], spv::Op::OpIAdd);
        const uint32_t* b = a && isConst(a[4], 8) ? m.Op(a[3], spv::Op::OpIAdd) : nullptr;
        if (b && (b[3] == tX16 || b[4] == tX16)) tCx = f[2];
    }
    // 1/(c1*z + c0) and 1/-(c1*z + c0): c0, c1 buffer floats, z the tile's raw depth (no buffer value).
    for (size_t k = firstFunc; tCx && k < m.ins.size() && !(tZMin && tZMax); ++k)
    {
        if (m.ins[k].op != spv::Op::OpFDiv || !m.IsOne(code[m.ins[k].at + 3])) continue;
        uint32_t den = code[m.ins[k].at + 4];
        if (const uint32_t* ng = m.Op(den, spv::Op::OpFNegate)) den = ng[3];
        const uint32_t* a = m.Op(den, spv::Op::OpFAdd);
        for (int o = 0; a && o < 2; ++o)
        {
            const uint32_t* f = m.Op(a[3 + o], spv::Op::OpFMul);
            const uint32_t* c0 = m.Op(a[4 - o], spv::Op::OpBitcast);
            for (int q = 0; f && c0 && m.FromBuffer(c0[3]) && q < 2; ++q)
            {
                const uint32_t* c1 = m.Op(f[3 + q], spv::Op::OpBitcast);
                const uint32_t* zf = m.Op(f[4 - q], spv::Op::OpBitcast);
                if (!c1 || !zf || !m.FromBuffer(c1[3]) || m.FromBuffer(zf[3])) continue;
                const uint32_t el = element(zf[3]);
                if (el > 1 || (tC0 && (tC0 != c0[3] || tC1 != c1[3]))) continue;
                tC0 = c0[3]; tC1 = c1[3];
                (el ? tZMax : tZMin) = zf[3];
            }
        }
    }
    // A value read at `at` that the module defines later in the same block (a hand-ordered module): its loads and
    // index arithmetic emitted again at `at`, when nothing stores to that memory in between. 0: not available.
    std::function<uint32_t(Emit&, size_t, uint32_t, int)> avail = [&](Emit& e, size_t at, uint32_t id, int depth) -> uint32_t
    {
        auto d = m.def.find(id);
        if (d == m.def.end() || d->second < at) return id;
        const size_t k = d->second;
        if (depth > 6 || m.ins[k].block != m.ins[at].block) return 0;
        const uint32_t* i = code + m.ins[k].at;
        const spv::Op op = m.ins[k].op;
        std::vector<uint32_t> w(i + 1, i + m.ins[k].n);   // type, result, operands
        size_t ids = w.size();
        auto root = [&](uint32_t p) { for (const uint32_t* a; (a = m.Op(p, spv::Op::OpAccessChain)) != nullptr; ) p = a[3]; return p; };
        switch (op)
        {
        case spv::Op::OpLoad:
            ids = 3;
            for (size_t j = at; j < k; ++j)
            {
                const spv::Op o = m.ins[j].op;
                const uint32_t* s = code + m.ins[j].at;
                if (o == spv::Op::OpControlBarrier || o == spv::Op::OpFunctionCall) return 0;
                if ((o == spv::Op::OpStore || o == spv::Op::OpAtomicStore) && root(s[1]) == root(i[3])) return 0;
                if (o >= spv::Op::OpAtomicExchange && o <= spv::Op::OpAtomicXor && root(s[3]) == root(i[3])) return 0;
            }
            break;
        case spv::Op::OpCompositeExtract: ids = 3; break;
        case spv::Op::OpAccessChain: case spv::Op::OpIAdd: case spv::Op::OpIMul: case spv::Op::OpShiftRightLogical: break;
        case spv::Op::OpBitcast: break;
        default: return 0;
        }
        for (size_t j = 2; j < ids; ++j) if (!(w[j] = avail(e, at, w[j], depth + 1))) return 0;
        w[1] = next++;
        e.v.push_back(((uint32_t)(w.size() + 1) << 16) | (uint32_t)op);
        e.v.insert(e.v.end(), w.begin(), w.end());
        return w[1];
    };
    Emit tileInputs;
    uint32_t zMinAt = 0, zMaxAt = 0, c0At = 0, c1At = 0;
    if (tCx && tZMin && tZMax)
    {
        const size_t at = m.def[tTable];
        zMinAt = avail(tileInputs, at, tZMin, 0); zMaxAt = avail(tileInputs, at, tZMax, 0);
        c0At = avail(tileInputs, at, tC0, 0); c1At = avail(tileInputs, at, tC1, 0);
    }
    const bool tiledIds = zMinAt && zMaxAt && c0At && c1At;
    if (tiledIds)
    {
        const size_t at = m.def[tTable];
        const uint32_t* tb = code + m.ins[at].at;              // table byte = 16 * column + K
        const uint32_t* mul = code + m.ins[m.def[tX16]].at;    // 16 * column
        const uint32_t* shr = code + m.ins[m.def[tBase]].at;   // dword = byte >> 2
        const uint32_t tU = tb[1], cK = tb[4], c16 = mul[4], c2 = shr[4];
        cTile = next++; cTileInv = next++; cZero = next++;
        Emit& e = before[at];
        e.v.insert(e.v.end(), tileInputs.v.begin(), tileInputs.v.end());
        uint32_t sep, conv, wMin;
        stereo(e, sep, conv, wMin);
        const uint32_t hw = halfWidth(e), c0f = next++, c1f = next++;
        e.Op(spv::Op::OpBitcast, { m.tFloat, c0f, c0At });
        e.Op(spv::Op::OpBitcast, { m.tFloat, c1f, c1At });
        // The pixels a depth moves by in the right eye; 1/w = c1*z + c0, the game's own view depth.
        // NClamp: an empty tile's min is NaN.
        auto parallax = [&](uint32_t zRaw) -> uint32_t
        {
            const uint32_t z = next++, m1 = next++, r0 = next++, r1 = next++, r = next++, wp = next++, mx = next++, cp = next++, df = next++, cr = next++, px = next++;
            e.Op(spv::Op::OpBitcast, { m.tFloat, z, zRaw });
            e.Op(spv::Op::OpFMul, { m.tFloat, m1, c1f, z });
            e.Op(spv::Op::OpFAdd, { m.tFloat, r0, m1, c0f });
            e.Op(spv::Op::OpExtInst, { m.tFloat, r1, m.glsl, GLSLstd450FAbs, r0 });
            e.Op(spv::Op::OpExtInst, { m.tFloat, r, m.glsl, GLSLstd450NClamp, r1, cTiny, cFar });
            e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, r });
            e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
            e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, r });
            e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
            e.Op(spv::Op::OpFMul, { m.tFloat, cr, sep, df });
            e.Op(spv::Op::OpFMul, { m.tFloat, px, cr, hw });
            return px;
        };
        const uint32_t pa = parallax(zMinAt), pb = parallax(zMaxAt);
        const uint32_t sum = next++, mid = next++, dif = next++, ad = next++, half = next++, x0 = next++, pMax = next++, pMin = next++;
        const uint32_t lo = next++, x1 = next++, hi = next++, loT = next++, hiT = next++, kLo = next++, kHi = next++, wT = next++, nT = next++, nM = next++;
        const uint32_t kLoC = next++, kHiC = next++, kHiM = next++, uLo = next++, uHi = next++, lo16 = next++, hi16 = next++, loB = next++, hiB = next++, loS = next++, hiS = next++;
        e.Op(spv::Op::OpFAdd, { m.tFloat, sum, pa, pb });
        e.Op(spv::Op::OpFMul, { m.tFloat, mid, sum, cHalf });
        e.Op(spv::Op::OpFSub, { m.tFloat, dif, pa, pb });
        e.Op(spv::Op::OpExtInst, { m.tFloat, ad, m.glsl, GLSLstd450FAbs, dif });
        e.Op(spv::Op::OpFMul, { m.tFloat, half, ad, cHalf });
        e.Op(spv::Op::OpConvertUToF, { m.tFloat, x0, tX16 });
        e.Op(spv::Op::OpFAdd, { m.tFloat, pMax, mid, half });
        e.Op(spv::Op::OpFSub, { m.tFloat, pMin, mid, half });
        e.Op(spv::Op::OpFSub, { m.tFloat, lo, x0, pMax });
        e.Op(spv::Op::OpFAdd, { m.tFloat, x1, x0, cTile });
        e.Op(spv::Op::OpFSub, { m.tFloat, hi, x1, pMin });
        e.Op(spv::Op::OpFMul, { m.tFloat, loT, lo, cTileInv });
        e.Op(spv::Op::OpFMul, { m.tFloat, hiT, hi, cTileInv });
        e.Op(spv::Op::OpExtInst, { m.tFloat, kLo, m.glsl, GLSLstd450Floor, loT });
        e.Op(spv::Op::OpExtInst, { m.tFloat, kHi, m.glsl, GLSLstd450Ceil, hiT });
        e.Op(spv::Op::OpFAdd, { m.tFloat, wT, hw, hw });
        e.Op(spv::Op::OpFMul, { m.tFloat, nT, wT, cTileInv });
        e.Op(spv::Op::OpFSub, { m.tFloat, nM, nT, cOne });
        e.Op(spv::Op::OpExtInst, { m.tFloat, kLoC, m.glsl, GLSLstd450NClamp, kLo, cZero, nM });
        e.Op(spv::Op::OpExtInst, { m.tFloat, kHiC, m.glsl, GLSLstd450NClamp, kHi, cOne, nT });
        e.Op(spv::Op::OpFSub, { m.tFloat, kHiM, kHiC, cOne });
        e.Op(spv::Op::OpConvertFToU, { tU, uLo, kLoC });
        e.Op(spv::Op::OpConvertFToU, { tU, uHi, kHiM });
        e.Op(spv::Op::OpIMul, { tU, lo16, uLo, c16 });
        e.Op(spv::Op::OpIMul, { tU, hi16, uHi, c16 });
        e.Op(spv::Op::OpIAdd, { tU, loB, lo16, cK });
        e.Op(spv::Op::OpIAdd, { tU, hiB, hi16, cK });
        e.Op(spv::Op::OpShiftRightLogical, { tU, loS, loB, c2 });
        e.Op(spv::Op::OpShiftRightLogical, { tU, hiS, hiB, c2 });
        // Its left plane's four dwords from the lowest column, its right plane's (dwords 4..7) from the highest.
        for (size_t k = at; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpIAdd || code[m.ins[k].at + 3] != tBase) continue;
            auto c = m.iconst.find(code[m.ins[k].at + 4]);
            rewrite[k].push_back({ 3, c != m.iconst.end() && c->second >= 4 ? hiS : loS });
            ++counts.rows;
        }
        // a*cx -> the least of a*x over the left-eye span, as the planes' own margin takes it over the tile
        // (all ten rows are value - margin < limit).
        const uint32_t xMid = next++;
        before[m.def[tCx] + 1].Op(spv::Op::OpFSub, { m.tFloat, xMid, tCx, mid });
        for (size_t k = m.def[tCx] + 1; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFMul) continue;
            const uint32_t* f = code + m.ins[k].at;
            if (f[3] != tCx && f[4] != tCx) continue;
            const uint32_t a = f[3] == tCx ? f[4] : f[3], t = next++, ab = next++, h = next++;
            Emit& r = replace[k];
            r.Op(spv::Op::OpFMul, { m.tFloat, t, a, xMid });
            r.Op(spv::Op::OpExtInst, { m.tFloat, ab, m.glsl, GLSLstd450FAbs, a });
            r.Op(spv::Op::OpFMul, { m.tFloat, h, ab, half });
            r.Op(spv::Op::OpFSub, { m.tFloat, f[2], t, h });
            ++counts.rows;
        }
    }
    // Ray rows: world = (ray varying, or col0*u + col1*v + col3) + col2*depth from the left camera's inverse: the
    // right's adds -col0*D, D the parallax at this depth in the matrix's x units (dwords 16..31 texture u, 0..15 x).
    struct NewConst { uint32_t type, id, value; };
    std::vector<NewConst> rayConsts;
    if (rayRows)
    {
        struct Row { size_t k; uint32_t coef, depth, base, idx0, off, ptrType, loadType, addType; };
        // A float loaded from buffer[dword + off].
        auto dwordOf = [&](uint32_t id, Row& o, uint32_t& dword) -> bool
        {
            const uint32_t* l = m.Op(m.Strip(id), spv::Op::OpLoad);
            const uint32_t* a = l ? m.Op(l[3], spv::Op::OpAccessChain) : nullptr;
            const uint32_t* add = a && (a[0] >> 16) == 6 ? m.Op(a[5], spv::Op::OpIAdd) : nullptr;
            auto c = add ? m.iconst.find(add[3]) : m.iconst.end();
            if (c == m.iconst.end()) return false;
            dword = c->second;
            o.base = a[3]; o.idx0 = a[4]; o.off = add[4]; o.ptrType = a[1]; o.loadType = l[1]; o.addType = add[1];
            return true;
        };
        std::function<bool(uint32_t, int)> imageFed = [&](uint32_t id, int depth) -> bool
        {
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 8) return false;
            switch (op)
            {
            case spv::Op::OpImageSampleImplicitLod: case spv::Op::OpImageSampleExplicitLod: case spv::Op::OpImageFetch: return true;
            case spv::Op::OpCompositeExtract: case spv::Op::OpBitcast: return imageFed(i[3], depth + 1);
            case spv::Op::OpFMul: return imageFed(i[3], depth + 1) || imageFed(i[4], depth + 1);
            default: return false;
            }
        };
        // Four consecutive dwords of one column times one depth value.
        // (offset value, depth) -> dword -> row
        std::map<std::pair<uint32_t, uint32_t>, std::map<uint32_t, Row>> groups;
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFMul) continue;
            const uint32_t* f = code + m.ins[k].at;
            for (int o = 0; o < 2; ++o)
            {
                Row r = {};
                uint32_t dword = 0;
                if (!dwordOf(f[3 + o], r, dword) || !imageFed(f[4 - o], 0)) continue;
                r.k = k; r.coef = f[3 + o]; r.depth = f[4 - o];
                groups[{ r.off, m.Strip(r.depth) }].emplace(dword, r);
                break;
            }
        }
        for (const auto& g : groups)
        {
            for (uint32_t b : { 8u, 24u })
            {
                bool all = true;
                for (uint32_t c = 0; c < 4; ++c) all = all && g.second.count(b + c);
                if (!all || rewrite.size() + replace.size() > 0) continue;
                const Row& rw = g.second.at(b + 3);
                // 1/w: the w row's own sum, rebuilt ahead of the first row.
                const uint32_t rowW = code[m.ins[rw.k].at + 2];
                uint32_t partner = 0;
                for (size_t k = rw.k + 1; k < m.ins.size() && !partner; ++k)
                {
                    if (m.ins[k].op != spv::Op::OpFAdd) continue;
                    const uint32_t* a = code + m.ins[k].at;
                    if (a[3] == rowW) partner = a[4];
                    else if (a[4] == rowW) partner = a[3];
                }
                size_t first = rw.k;
                for (uint32_t c = 0; c < 4; ++c) first = (std::min)(first, g.second.at(b + c).k);
                auto pd = m.def.find(partner);
                auto zd = m.def.find(rw.depth);
                auto cd = m.def.find(rw.coef);
                if (!partner || pd == m.def.end() || pd->second >= first || zd == m.def.end() || zd->second >= first || cd == m.def.end() || cd->second >= first) continue;
                // Clip x (dwords 0..15): the box's own w and its VS's shift come into D; FragCoord.w = 1/w.
                uint32_t fcw = 0, boxW = 0;
                if (b == 8)
                    for (size_t k = firstFunc; k < first && !fcw; ++k)
                    {
                        if (m.ins[k].op != spv::Op::OpFDiv) continue;
                        const uint32_t* dv = code + m.ins[k].at;
                        const uint32_t* l = m.IsOne(dv[3]) ? m.Op(dv[4], spv::Op::OpLoad) : nullptr;
                        const uint32_t* a = l ? m.Op(l[3], spv::Op::OpAccessChain) : nullptr;
                        auto c = a && (a[0] >> 16) == 5 ? m.iconst.find(a[4]) : m.iconst.end();
                        if (c != m.iconst.end() && c->second == 3 && m.idVars.count(a[3])) { fcw = dv[4]; boxW = dv[2]; }
                    }
                if (b == 8 && !fcw) continue;
                Emit& e = before[first];
                uint32_t sep, conv, wMin;
                stereo(e, sep, conv, wMin);
                const uint32_t z2 = next++, W = next++, absW = next++;
                e.Op(spv::Op::OpFMul, { m.tFloat, z2, rw.coef, rw.depth });
                e.Op(spv::Op::OpFAdd, { m.tFloat, W, z2, partner });
                e.Op(spv::Op::OpExtInst, { m.tFloat, absW, m.glsl, GLSLstd450FAbs, W });
                uint32_t R = absW;
                if (b == 8) { R = next++; e.Op(spv::Op::OpFMul, { m.tFloat, R, absW, fcw }); }
                const uint32_t wp = next++, mx = next++, cp = next++, df = next++, corr = next++;
                e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, R });
                e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
                e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, R });
                e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
                e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
                uint32_t D = next++;
                if (b == 24) e.Op(spv::Op::OpFMul, { m.tFloat, D, corr, cHalf });
                else
                {
                    // The box's varying is already the left ray through its point, x moved by sep*(1 - conv/w).
                    const uint32_t ps = next++, sv = next++, cb = next++, db = next++, s2 = next++, dl = next++;
                    e.Op(spv::Op::OpAccessChain, { 0, ps, vStereo, cI0 });
                    e.Op(spv::Op::OpLoad, { m.tFloat, sv, ps });
                    e.Op(spv::Op::OpFMul, { m.tFloat, cb, conv, fcw });
                    e.Op(spv::Op::OpFSub, { m.tFloat, db, cOne, cb });
                    e.Op(spv::Op::OpFMul, { m.tFloat, s2, sv, db });
                    e.Op(spv::Op::OpFSub, { m.tFloat, dl, corr, s2 });
                    e.Op(spv::Op::OpFMul, { m.tFloat, D, dl, boxW });
                }
                for (uint32_t c = 0; c < 4; ++c)
                {
                    const Row& r = g.second.at(b + c);
                    const uint32_t* f = code + m.ins[r.k].at;
                    const uint32_t cc = next++, ia = next++, p = next++, u = next++, c0 = next++, t = next++, h = next++;
                    rayConsts.push_back({ r.addType, cc, b - 8 + c });
                    Emit& q = replace[r.k];
                    q.Op(spv::Op::OpIAdd, { r.addType, ia, cc, r.off });
                    q.Op(spv::Op::OpAccessChain, { r.ptrType, p, r.base, r.idx0, ia });
                    q.Op(spv::Op::OpLoad, { r.loadType, u, p });
                    q.Op(spv::Op::OpBitcast, { m.tFloat, c0, u });
                    q.Op(spv::Op::OpFMul, { m.tFloat, t, f[3], f[4] });
                    q.Op(spv::Op::OpFMul, { m.tFloat, h, c0, D });
                    q.Op(spv::Op::OpFSub, { m.tFloat, f[2], t, h });
                    ++counts.rows;
                }
                ++counts.ndc;
            }
        }
    }
    // Soft read: 1/(a*depth + b) of a depth read at a varying; in the right eye the read moves to the fragment's own
    // pixel (the left's expression stays for sep 0).
    uint32_t vFrag = 0, tPtrInVec4 = 0;
    bool newPtrInVec4 = false;
    if (softRead)
    {
        uint32_t tBool = 0, tVec4 = 0, tIvec2 = 0, fragVar = 0;
        for (const Ins& in : m.ins)
        {
            const uint32_t* i = code + in.at;
            if (in.op == spv::Op::OpTypeBool) tBool = i[1];
            if (in.op == spv::Op::OpTypeVector && i[2] == m.tFloat && i[3] == 4) tVec4 = i[1];
            if (in.op == spv::Op::OpTypeVector && tInt && i[2] == tInt && i[3] == 2) tIvec2 = i[1];
            if (in.op == spv::Op::OpDecorate && in.n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::FragCoord) fragVar = i[1];
        }
        size_t ccAt = SIZE_MAX;
        uint32_t imgVar = 0, imgType = 0;
        for (size_t k = firstFunc; k < m.ins.size() && ccAt == SIZE_MAX; ++k)
        {
            if (m.ins[k].op != spv::Op::OpFDiv) continue;
            const uint32_t* dv = code + m.ins[k].at;
            const uint32_t* add = m.IsOne(dv[3]) ? m.Op(m.Strip(dv[4]), spv::Op::OpFAdd) : nullptr;
            for (int o = 0; add && o < 2 && ccAt == SIZE_MAX; ++o)
            {
                const uint32_t* mul = m.Op(m.Strip(add[3 + o]), spv::Op::OpFMul);
                if (!mul || !m.FromBuffer(add[4 - o])) continue;
                for (int q = 0; q < 2; ++q)
                {
                    const uint32_t* ex = m.FromBuffer(mul[3 + q]) ? m.Op(m.Strip(mul[4 - q]), spv::Op::OpCompositeExtract) : nullptr;
                    spv::Op sop; const uint32_t* s = ex ? m.At(ex[3], &sop) : nullptr;
                    if (!s || (sop != spv::Op::OpImageSampleImplicitLod && sop != spv::Op::OpImageSampleExplicitLod)) continue;
                    const uint32_t* si = m.Op(s[3], spv::Op::OpSampledImage);
                    const uint32_t* img = si ? m.Op(si[3], spv::Op::OpLoad) : nullptr;
                    const uint32_t* cc = m.Op(s[4], spv::Op::OpCompositeConstruct);
                    if (!img || !cc || (cc[0] >> 16) != 5) continue;
                    ccAt = m.def[s[4]]; imgVar = img[3]; imgType = img[1];
                    break;
                }
            }
        }
        if (ccAt != SIZE_MAX && tBool && tVec4 && tIvec2 && code[1] >= 0x00010400)
        {
            if (!fragVar)
            {
                vFrag = fragVar = next++;
                auto pv = ptrOf.find(((uint64_t)spv::StorageClass::Input << 32) | tVec4);
                if (pv != ptrOf.end()) tPtrInVec4 = pv->second; else { tPtrInVec4 = next++; newPtrInVec4 = true; }
            }
            const uint32_t* cc = code + m.ins[ccAt].at;
            Emit& e = before[ccAt];
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            const uint32_t cZ = next++, fc = next++, fx = next++, img = next++, sz = next++, wi = next++, wf = next++, u = next++, nz = next++, sel = next++;
            rayConsts.push_back({ m.tFloat, cZ, Bits(0.f) });
            e.Op(spv::Op::OpLoad, { tVec4, fc, fragVar });
            e.Op(spv::Op::OpCompositeExtract, { m.tFloat, fx, fc, 0 });
            e.Op(spv::Op::OpLoad, { imgType, img, imgVar });
            e.Op(spv::Op::OpImageQuerySizeLod, { tIvec2, sz, img, cI0 });
            e.Op(spv::Op::OpCompositeExtract, { tInt, wi, sz, 0 });
            e.Op(spv::Op::OpConvertSToF, { m.tFloat, wf, wi });
            e.Op(spv::Op::OpFDiv, { m.tFloat, u, fx, wf });
            e.Op(spv::Op::OpFOrdNotEqual, { tBool, nz, sep, cZ });
            e.Op(spv::Op::OpSelect, { m.tFloat, sel, nz, u, cc[3] });
            rewrite[ccAt].push_back({ 3, sel });
            ++counts.ndc; ++counts.rows;
        }
    }
    // Flare: a table point (x or k*x) first met with the uv's x alone (x - uv, or k*x + k*(-uv) stepping toward it);
    // no depth there, so the right eye's point takes the far field's parallax, x + fsSep/2.
    if (flareX && fragment)
    {
        // The varying axes a value reads: 1 = an input's component 0, 2 = its component 1, through arithmetic.
        std::unordered_map<uint32_t, uint8_t> axesMemo;
        std::function<uint8_t(uint32_t, int)> axes = [&](uint32_t id, int depth) -> uint8_t
        {
            auto it = axesMemo.find(id);
            if (it != axesMemo.end()) return it->second;
            spv::Op op; const uint32_t* i = m.At(id, &op);
            uint8_t r = 0;
            if (i && depth < 16 && op == spv::Op::OpLoad && i[1] == m.tFloat)
            {
                const uint32_t* a = m.Op(i[3], spv::Op::OpAccessChain);
                auto vc = a ? m.varClass.find(a[3]) : m.varClass.end();
                auto c = (vc != m.varClass.end() && vc->second == (uint32_t)spv::StorageClass::Input && (a[0] >> 16) >= 5) ?
                         m.iconst.find(a[(a[0] >> 16) - 1]) : m.iconst.end();
                if (c != m.iconst.end() && c->second < 2) r = (uint8_t)(1u << c->second);
            }
            else if (i && depth < 16 && (op == spv::Op::OpFAdd || op == spv::Op::OpFSub || op == spv::Op::OpFMul || op == spv::Op::OpFNegate ||
                                         op == spv::Op::OpExtInst || op == spv::Op::OpBitcast || op == spv::Op::OpCopyObject))
                ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { r |= axes(a, depth + 1); });
            axesMemo[id] = r;
            return r;
        };
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFSub && m.ins[k].op != spv::Op::OpFAdd) continue;
            const uint32_t* f = code + m.ins[k].at;
            if (f[1] != m.tFloat) continue;
            uint16_t slot = 0;
            uint32_t scale = 0;
            for (int o = 0; o < 2 && !slot; ++o)
            {
                if (axes(f[4 - o], 0) != 1) continue;
                if (m.FromBuffer(f[3 + o])) { slot = (uint16_t)(3 + o); break; }
                const uint32_t* mul = m.Op(f[3 + o], spv::Op::OpFMul);
                for (int q = 0; mul && q < 2 && !slot; ++q)
                    if (m.fconst.count(mul[3 + q]) && m.FromBuffer(mul[4 - q])) { slot = (uint16_t)(3 + o); scale = mul[3 + q]; }
            }
            if (!slot) continue;
            Emit& e = before[k];
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            uint32_t h = next++;
            const uint32_t x2 = next++;
            e.Op(spv::Op::OpFMul, { m.tFloat, h, sep, cHalf });
            if (scale) { const uint32_t hs = next++; e.Op(spv::Op::OpFMul, { m.tFloat, hs, h, scale }); h = hs; }
            e.Op(spv::Op::OpFAdd, { m.tFloat, x2, f[slot], h });
            rewrite[k].push_back({ slot, x2 });
            ++counts.ndc; ++counts.rows;
            break;
        }
    }
    // View rows: world = (col0*FragCoord.x + col1*FragCoord.y + col3)*w, the left camera's: x - D is the left's pixel,
    // D = fsSep*(max(w,wMin) - conv)/w ndc times the viewport scale in push member 2 (half the width).
    uint32_t tPtrPF = 0;
    bool newPtrPF = false;
    if (viewRows && fragment && !counts.ndc)
    {
        uint32_t pushVar = 0;
        for (const Ins& in : m.ins)
        {
            if (in.op != spv::Op::OpVariable || code[in.at + 3] != (uint32_t)spv::StorageClass::PushConstant) continue;
            const uint32_t* pt = m.Op(code[in.at + 1], spv::Op::OpTypePointer);
            const uint32_t* st = pt ? m.Op(pt[3], spv::Op::OpTypeStruct) : nullptr;
            if (st && (st[0] >> 16) >= 5 && st[4] == m.tFloat) pushVar = code[in.at + 2];
        }
        uint32_t fx = 0, fy = 0, fcw = 0, w = 0;
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const uint32_t* i = code + m.ins[k].at;
            if (m.ins[k].op == spv::Op::OpLoad)
            {
                const uint32_t* a = m.Op(i[3], spv::Op::OpAccessChain);
                auto c = a && (a[0] >> 16) == 5 && m.idVars.count(a[3]) ? m.iconst.find(a[4]) : m.iconst.end();
                if (c == m.iconst.end()) continue;
                if (c->second == 0 && !fx) fx = i[2];
                if (c->second == 1 && !fy) fy = i[2];
                if (c->second == 3 && !fcw) fcw = i[2];
            }
            else if (m.ins[k].op == spv::Op::OpFDiv && fcw && !w && m.IsOne(i[3]) && i[4] == fcw) w = i[2];
        }
        std::vector<std::pair<size_t, uint16_t>> rows;   // FMul, word of x
        for (size_t k = firstFunc; fx && fy && w && k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFMul) continue;
            const uint32_t* f = code + m.ins[k].at;
            for (int o = 0; o < 2; ++o)
            {
                if (f[3 + o] != fx) continue;
                bool row = false;
                for (size_t q = k + 1; q < m.ins.size() && !row; ++q)
                {
                    if (m.ins[q].op != spv::Op::OpFAdd) continue;
                    const uint32_t* a = code + m.ins[q].at;
                    if (a[3] != f[2] && a[4] != f[2]) continue;
                    const uint32_t other = a[3] == f[2] ? a[4] : a[3];
                    const uint32_t* y = m.Op(other, spv::Op::OpFAdd);
                    for (int t = 0; y && t < 2 && !row; ++t)
                    {
                        const uint32_t* my = m.Op(y[3 + t], spv::Op::OpFMul);
                        row = my && (my[3] == fy || my[4] == fy);
                    }
                    // View x, (d40*x + d42)*w: no y term. A screen read k*x + offset never meets w.
                    for (size_t u = q + 1; !row && m.FromBuffer(other) && u < m.ins.size(); ++u)
                    {
                        if (m.ins[u].op != spv::Op::OpFMul) continue;
                        const uint32_t* mw = code + m.ins[u].at;
                        row = (mw[3] == a[2] && mw[4] == w) || (mw[4] == a[2] && mw[3] == w);
                    }
                }
                if (row) rows.push_back({ k, (uint16_t)(3 + o) });
                break;
            }
        }
        if (!rows.empty() && pushVar && m.def[w] < rows[0].first)
        {
            auto pf = ptrOf.find(((uint64_t)spv::StorageClass::PushConstant << 32) | m.tFloat);
            if (pf != ptrOf.end()) tPtrPF = pf->second; else { tPtrPF = next++; newPtrPF = true; }
            Emit& e = before[rows[0].first];
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            const uint32_t pxs = next++, hp = next++, wc = next++, df = next++, q = next++, sd = next++, D = next++, x2 = next++;
            e.Op(spv::Op::OpAccessChain, { tPtrPF, pxs, pushVar, cI2 });
            e.Op(spv::Op::OpLoad, { m.tFloat, hp, pxs });
            e.Op(spv::Op::OpExtInst, { m.tFloat, wc, m.glsl, GLSLstd450FMax, w, wMin });
            e.Op(spv::Op::OpFSub, { m.tFloat, df, wc, conv });
            e.Op(spv::Op::OpFDiv, { m.tFloat, q, df, w });
            e.Op(spv::Op::OpFMul, { m.tFloat, sd, sep, q });
            e.Op(spv::Op::OpFMul, { m.tFloat, D, sd, hp });
            e.Op(spv::Op::OpFSub, { m.tFloat, x2, fx, D });
            for (const auto& r : rows) { rewrite[r.first].push_back({ r.second, x2 }); ++counts.rows; }
            ++counts.ndc;
        }
    }
    // Froxel lookups: fog volumes built in the left frustum, read at (uv*k, slice(log2(clamp(D, near, far)))), are read
    // at uv - h, h = fsSep/2*(max(1, wMin/D) - conv/D); a uv through a phi moves on the incoming carrying the varying.
    unsigned froxels = 0;
    // Constants and buffer loads through arithmetic.
    std::function<bool(uint32_t, int)> uniOnly = [&](uint32_t id, int depth) -> bool
    {
        if (m.fconst.count(id) || m.iconst.count(id) || m.FromBuffer(id)) return true;
        spv::Op op; const uint32_t* i = m.At(id, &op);
        if (!i || depth > 12 || (op == spv::Op::OpExtInst && i[3] != m.glsl)) return false;
        if (op != spv::Op::OpFAdd && op != spv::Op::OpFSub && op != spv::Op::OpFMul && op != spv::Op::OpFDiv && op != spv::Op::OpFNegate &&
            op != spv::Op::OpExtInst && op != spv::Op::OpBitcast && op != spv::Op::OpBitwiseXor && op != spv::Op::OpBitwiseAnd &&
            op != spv::Op::OpCopyObject) return false;
        bool ok = true;
        ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { ok = ok && uniOnly(a, depth + 1); });
        return ok;
    };
    if (froxelLookups && fragment)
    {
        uint32_t tBoolF = 0;
        for (const Ins& in : m.ins) if (in.op == spv::Op::OpTypeBool) { tBoolF = code[in.at + 1]; break; }
        // A Log2's argument when it is a depth clamped by uniform bounds (FMin/FMax/FClamp over one value), else 0.
        auto clampOf = [&](uint32_t id) -> uint32_t
        {
            const uint32_t top = m.Strip(id);
            uint32_t v = top;
            int layers = 0;
            for (; layers < 3; ++layers)
            {
                spv::Op op; const uint32_t* i = m.At(v, &op);
                if (!i || op != spv::Op::OpExtInst || i[3] != m.glsl) break;
                const uint32_t f = i[4], n = i[0] >> 16;
                if (f != GLSLstd450FMin && f != GLSLstd450FMax && f != GLSLstd450FClamp && f != GLSLstd450NClamp &&
                    f != GLSLstd450NMin && f != GLSLstd450NMax) break;
                uint32_t live = 0;
                for (uint32_t a = 5; a < n; ++a)
                    if (!uniOnly(m.Strip(i[a]), 0)) { if (live) return 0; live = m.Strip(i[a]); }
                if (!live) return 0;
                v = live;
            }
            return layers ? top : 0;
        };
        // The one clamped depth under a slice coordinate, through arithmetic (no phis: it must dominate the read).
        auto sliceDepth = [&](uint32_t z) -> uint32_t
        {
            std::vector<uint32_t> todo{ z };
            std::unordered_set<uint32_t> seen;
            uint32_t found = 0;
            while (!todo.empty() && seen.size() < 512)
            {
                const uint32_t v = todo.back(); todo.pop_back();
                if (!seen.insert(v).second) continue;
                spv::Op op; const uint32_t* i = m.At(v, &op);
                if (!i || op == spv::Op::OpPhi || op == spv::Op::OpLoad || op == spv::Op::OpImageSampleImplicitLod ||
                    op == spv::Op::OpImageSampleExplicitLod || op == spv::Op::OpImageGather || op == spv::Op::OpFunctionCall) continue;
                bool r = false, t = false;
                spv::HasResultAndType(op, &r, &t);
                if (!r || !t) continue;
                if (const uint32_t* lg = m.Glsl(v, GLSLstd450Log2))
                    if (const uint32_t c = clampOf(lg[5])) { if (found && found != c) return 0u; found = c; continue; }
                ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { todo.push_back(a); });
            }
            return found;
        };
        auto inputLoad = [&](uint32_t id) -> bool
        {
            for (int g = 0; g < 8; ++g)
            {
                if (npLoad(id)) return true;   // a noperspective varying, v * FragCoord.w
                spv::Op op; const uint32_t* i = m.At(id, &op);
                if (!i) return false;
                if (op == spv::Op::OpBitcast || op == spv::Op::OpCopyObject || op == spv::Op::OpCompositeExtract) { id = i[3]; continue; }
                if (op == spv::Op::OpExtInst && i[3] == m.glsl && (i[4] == GLSLstd450FClamp || i[4] == GLSLstd450NClamp ||
                    ((i[4] == GLSLstd450NMin || i[4] == GLSLstd450NMax) && m.fconst.count(i[6])))) { id = i[5]; continue; }
                if (op != spv::Op::OpLoad) return false;
                const uint32_t* p = m.Op(i[3], spv::Op::OpAccessChain);
                auto vc = m.varClass.find(p ? p[3] : i[3]);
                return vc != m.varClass.end() && vc->second == (uint32_t)spv::StorageClass::Input;
            }
            return false;
        };
        // The varying under x (clamps, bitcasts, at most one phi); phi/phiIn: that phi and the incoming carrying it.
        auto screenX = [&](uint32_t id, uint32_t& phi, uint32_t& phiIn) -> bool
        {
            phi = phiIn = 0;
            for (int g = 0; g < 8; ++g)
            {
                if (inputLoad(id)) return true;
                spv::Op op; const uint32_t* i = m.At(id, &op);
                if (!i) return false;
                if (op == spv::Op::OpBitcast || op == spv::Op::OpCopyObject) { id = i[3]; continue; }
                if (op == spv::Op::OpExtInst && i[3] == m.glsl && (i[4] == GLSLstd450FClamp || i[4] == GLSLstd450NClamp ||
                    ((i[4] == GLSLstd450NMin || i[4] == GLSLstd450NMax) && m.fconst.count(i[6])))) { id = i[5]; continue; }
                if (op != spv::Op::OpPhi || phi) return false;
                for (uint32_t k = 3; k + 1 < (i[0] >> 16); k += 2)
                    if (inputLoad(i[k])) { if (phiIn) return false; phiIn = i[k]; }
                if (!phiIn) return false;
                phi = id;
                return true;
            }
            return false;
        };
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const size_t ccAt = sampleCoord(k, spv::Dim::Dim3D);
            if (ccAt == SIZE_MAX || shifted.count(ccAt) || rewrite.count(ccAt)) continue;
            const uint32_t* cc = code + m.ins[ccAt].at;
            if ((cc[0] >> 16) < 6) continue;
            const uint32_t* mul = m.Op(cc[3], spv::Op::OpFMul);
            if (!mul) continue;
            uint32_t phi = 0, phiIn = 0, kId = 0;
            for (int o = 0; o < 2 && !kId; ++o)
                if (uniOnly(mul[4 - o], 0) && screenX(mul[3 + o], phi, phiIn)) kId = mul[4 - o];
            const uint32_t dc = kId ? sliceDepth(cc[5]) : 0;
            if (!dc) continue;
            // The phi's incoming must be usable after the merge: defined in the function's first block.
            spv::Op pop = spv::Op::OpNop; const uint32_t* pi = phi ? m.At(phi, &pop) : nullptr;
            if (phi)
            {
                size_t fn = ccAt;
                while (fn > 0 && m.ins[fn].op != spv::Op::OpFunction) --fn;
                if (!tBoolF || !pi || fn + 1 >= m.ins.size() || m.ins[m.def[phiIn]].block != m.ins[fn + 1].block) continue;
            }
            Emit& e = before[ccAt];
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            const uint32_t az = next++, lo = next++, P = next++, wp = next++, mx = next++, cp = next++, df = next++, corr = next++, hh = next++, hk = next++, x2 = next++;
            e.Op(spv::Op::OpExtInst, { m.tFloat, az, m.glsl, GLSLstd450FAbs, dc });
            e.Op(spv::Op::OpExtInst, { m.tFloat, lo, m.glsl, GLSLstd450FMax, az, cTiny });
            e.Op(spv::Op::OpFDiv, { m.tFloat, P, cOne, lo });
            e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, P });
            e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
            e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, P });
            e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
            e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
            e.Op(spv::Op::OpFMul, { m.tFloat, hh, corr, cHalf });
            e.Op(spv::Op::OpFMul, { m.tFloat, hk, hh, kId });
            e.Op(spv::Op::OpFSub, { m.tFloat, x2, cc[3], hk });
            uint32_t xn = x2;
            if (phi)
            {
                const uint32_t eq = next++, xs = next++;
                e.Op(pi[1] == m.tFloat ? spv::Op::OpFOrdEqual : spv::Op::OpIEqual, { tBoolF, eq, phi, phiIn });
                e.Op(spv::Op::OpSelect, { m.tFloat, xs, eq, x2, cc[3] });
                xn = xs;
            }
            rewrite[ccAt].push_back({ 3, xn });
            shifted.insert(ccAt);
            ++froxels;
        }
        counts.reproj += froxels;
    }
    // A compute reader at the invocation's pixel: each tap's x (normalised, screen-aligned) moves by the same h,
    // D the depth in log2(|clamp(D)|) with uniform bounds, less a uniform factor (a depth over far).
    if (froxelLookups && !fragment)
    {
        auto sliceView = [&](uint32_t z) -> uint32_t
        {
            std::vector<uint32_t> todo{ z };
            std::unordered_set<uint32_t> seen;
            uint32_t found = 0;
            while (!todo.empty() && seen.size() < 512)
            {
                const uint32_t v = todo.back(); todo.pop_back();
                if (!seen.insert(v).second) continue;
                spv::Op op; const uint32_t* i = m.At(v, &op);
                if (!i || op == spv::Op::OpPhi || op == spv::Op::OpLoad || op == spv::Op::OpImageSampleImplicitLod ||
                    op == spv::Op::OpImageSampleExplicitLod || op == spv::Op::OpImageGather || op == spv::Op::OpFunctionCall) continue;
                bool r = false, t = false;
                spv::HasResultAndType(op, &r, &t);
                if (!r || !t) continue;
                if (const uint32_t* lg = m.Glsl(v, GLSLstd450Log2))
                {
                    uint32_t a = m.Strip(lg[5]);
                    if (const uint32_t* ab = m.Glsl(a, GLSLstd450FAbs)) a = m.Strip(ab[5]);
                    const uint32_t* c = m.Glsl(a, GLSLstd450FClamp);
                    if (!c || !uniOnly(m.Strip(c[6]), 0) || !uniOnly(m.Strip(c[7]), 0)) return 0u;
                    uint32_t dv = m.Strip(c[5]);
                    if (const uint32_t* mul = m.Op(dv, spv::Op::OpFMul))
                        for (int o = 0; o < 2; ++o)
                            if (uniOnly(m.Strip(mul[3 + o]), 0) && !uniOnly(m.Strip(mul[4 - o]), 0)) { dv = m.Strip(mul[4 - o]); break; }
                    if (found && found != dv) return 0u;
                    found = dv;
                    continue;
                }
                ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { todo.push_back(a); });
            }
            return found;
        };
        // A builder of such a volume (it writes a 3D image) reads its own history in volume space: left alone.
        bool builder = false;
        for (size_t k = firstFunc; k < m.ins.size() && !builder; ++k)
        {
            if (m.ins[k].op != spv::Op::OpImageWrite) continue;
            spv::Op lop; const uint32_t* l = m.At(code[m.ins[k].at + 1], &lop);
            const uint32_t* it = l ? m.Op(l[1], spv::Op::OpTypeImage) : nullptr;
            builder = it && it[3] == (uint32_t)spv::Dim::Dim3D;
        }
        unsigned taps = 0;
        for (size_t k = firstFunc; !builder && k < m.ins.size(); ++k)
        {
            const size_t ccAt = sampleCoord(k, spv::Dim::Dim3D);
            if (ccAt == SIZE_MAX || shifted.count(ccAt) || rewrite.count(ccAt)) continue;
            const uint32_t* cc = code + m.ins[ccAt].at;
            if ((cc[0] >> 16) < 6 || !(m.T(m.Strip(cc[3])) & 1)) continue;
            const uint32_t dv = sliceView(cc[5]);
            if (!dv || m.def[dv] >= ccAt || m.ins[m.def[dv]].block != m.ins[ccAt].block) continue;
            Emit& e = before[ccAt];
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            const uint32_t az = next++, lo = next++, P = next++, wp = next++, mx = next++, cp = next++, df = next++, corr = next++, hh = next++, x2 = next++;
            e.Op(spv::Op::OpExtInst, { m.tFloat, az, m.glsl, GLSLstd450FAbs, dv });
            e.Op(spv::Op::OpExtInst, { m.tFloat, lo, m.glsl, GLSLstd450FMax, az, cTiny });
            e.Op(spv::Op::OpFDiv, { m.tFloat, P, cOne, lo });
            e.Op(spv::Op::OpFMul, { m.tFloat, wp, wMin, P });
            e.Op(spv::Op::OpExtInst, { m.tFloat, mx, m.glsl, GLSLstd450FMax, cOne, wp });
            e.Op(spv::Op::OpFMul, { m.tFloat, cp, conv, P });
            e.Op(spv::Op::OpFSub, { m.tFloat, df, mx, cp });
            e.Op(spv::Op::OpFMul, { m.tFloat, corr, sep, df });
            e.Op(spv::Op::OpFMul, { m.tFloat, hh, corr, cHalf });
            e.Op(spv::Op::OpFSub, { m.tFloat, x2, cc[3], hh });
            rewrite[ccAt].push_back({ 3, x2 });
            shifted.insert(ccAt);
            ++taps;
        }
        froxels += taps;
        counts.reproj += taps;
    }
    // Corner rays: a volume builder placing each cell at pos + mix(mix(c0, c1, u*k), mix(c2, c3, u*k), s) * d through
    // function variables. The right run's cell is the left frustum's at u - h(d): the product loses dR/du * h(d) * d.
    if (cornerRays && !fragment)
    {
        std::unordered_map<uint32_t, std::vector<size_t>> stores;   // function variable -> its whole stores
        std::unordered_set<uint32_t> opaque;                        // also written through a chain or by a call
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const uint32_t* i = code + m.ins[k].at;
            if (m.ins[k].op == spv::Op::OpStore)
            {
                auto vc = m.varClass.find(i[1]);
                if (vc != m.varClass.end() && vc->second == (uint32_t)spv::StorageClass::Function) stores[i[1]].push_back(k);
                else if (const uint32_t* p = m.Op(i[1], spv::Op::OpAccessChain)) opaque.insert(p[3]);
            }
            else if (m.ins[k].op == spv::Op::OpFunctionCall)
                for (uint16_t a = 4; a < m.ins[k].n; ++a) opaque.insert(i[a]);
        }
        auto fnEntry = [&](size_t at) -> uint32_t
        {
            while (at > 0 && m.ins[at].op != spv::Op::OpFunction) --at;
            while (at < m.ins.size() && m.ins[at].op != spv::Op::OpLabel) ++at;
            return at < m.ins.size() ? m.ins[at].block : 0;
        };
        // A value through function variables: a load's last store before it in its block, else the only store when
        // that one sits in the function's first block.
        auto val = [&](uint32_t id) -> uint32_t
        {
            for (int g = 0; g < 16; ++g)
            {
                id = m.Strip(id);
                const uint32_t* l = m.Op(id, spv::Op::OpLoad);
                auto st = l && !opaque.count(l[3]) ? stores.find(l[3]) : stores.end();
                if (st == stores.end()) return id;
                const size_t at = m.def[id];
                uint32_t v = 0;
                for (auto s = st->second.rbegin(); s != st->second.rend() && !v; ++s)
                    if (*s < at && m.ins[*s].block == m.ins[at].block) v = code[m.ins[*s].at + 2];
                if (!v && st->second.size() == 1 && st->second[0] < at && m.ins[st->second[0]].block == fnEntry(at))
                    v = code[m.ins[st->second[0]].at + 2];
                if (!v) return id;
                id = v;
            }
            return id;
        };
        // Reaches an invocation-id builtin (no targets) or one of `targets`: operands, variables' stores, call args.
        auto reaches = [&](uint32_t from, const std::unordered_set<uint32_t>& targets) -> bool
        {
            std::vector<uint32_t> todo{ from };
            std::unordered_set<uint32_t> seen;
            while (!todo.empty() && seen.size() < 2048)
            {
                const uint32_t v = m.Strip(todo.back()); todo.pop_back();
                if (!seen.insert(v).second) continue;
                if (targets.empty() ? m.idVars.count(v) != 0 : targets.count(v) != 0) return true;
                auto st = stores.find(v);
                if (st != stores.end()) { for (size_t s : st->second) todo.push_back(code[m.ins[s].at + 2]); continue; }
                spv::Op op; const uint32_t* i = m.At(v, &op);
                if (!i) continue;
                if (op == spv::Op::OpLoad || op == spv::Op::OpAccessChain || op == spv::Op::OpInBoundsAccessChain) { todo.push_back(i[3]); continue; }
                if (op == spv::Op::OpFunctionCall)
                {
                    for (uint16_t a = 4; a < (uint16_t)(i[0] >> 16); ++a) todo.push_back(i[a]);
                    continue;
                }
                bool r = false, t = false;
                spv::HasResultAndType(op, &r, &t);
                if (r && t) ForIds(i, (uint16_t)(i[0] >> 16), op, [&](uint32_t a) { todo.push_back(a); });
            }
            return false;
        };
        const std::unordered_set<uint32_t> pixel;
        // A screen x: component 0 of a vector built from the invocation id, loaded through a chain or extracted.
        auto screenX = [&](uint32_t id) -> bool
        {
            const uint32_t v = val(id);
            if (const uint32_t* l = m.Op(v, spv::Op::OpLoad))
            {
                const uint32_t* p = m.Op(l[3], spv::Op::OpAccessChain);
                auto c = p && (p[0] >> 16) == 5 ? m.iconst.find(p[4]) : m.iconst.end();
                return c != m.iconst.end() && c->second == 0 && !opaque.count(p[3]) && reaches(p[3], pixel);
            }
            const uint32_t* x = m.Op(v, spv::Op::OpCompositeExtract);
            return x && (x[0] >> 16) == 5 && x[4] == 0 && reaches(x[3], pixel);
        };
        auto uniVec = [&](uint32_t id) -> bool
        {
            const uint32_t* s = m.Op(val(id), spv::Op::OpVectorShuffle);
            return s ? m.FromBuffer(val(s[3])) && m.FromBuffer(val(s[4])) : m.FromBuffer(val(id));
        };
        auto splat = [&](uint32_t id) -> uint32_t
        {
            const uint32_t* c = m.Op(val(id), spv::Op::OpCompositeConstruct);
            if (!c || (c[0] >> 16) < 5) return 0;
            for (uint16_t a = 4; a < (uint16_t)(c[0] >> 16); ++a) if (c[a] != c[3]) return 0;
            return c[3];
        };
        // Ids an emit at `at` may read: defined earlier in its block, or in its function's first block.
        auto usable = [&](uint32_t id, size_t at) -> bool
        {
            auto d = m.def.find(id);
            return d != m.def.end() && d->second < at && (m.ins[d->second].block == m.ins[at].block || m.ins[d->second].block == fnEntry(at));
        };
        std::unordered_set<uint32_t> placed;
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpVectorTimesScalar || replace.count(k) || rewrite.count(k) || before.count(k)) continue;
            const uint32_t* vt = code + m.ins[k].at;
            const uint32_t* outer = m.Glsl(val(vt[3]), GLSLstd450FMix);
            if (!outer || !splat(outer[7])) continue;
            uint32_t ua = 0, ub = 0, kk = 0;
            for (int o = 0; o < 2 && !kk; ++o)
            {
                const uint32_t* in = m.Glsl(val(outer[5 + o]), GLSLstd450FMix);
                const uint32_t t = in ? splat(in[7]) : 0;
                const uint32_t* mul = t ? m.Op(val(t), spv::Op::OpFMul) : nullptr;
                if (!mul || !uniVec(in[5]) || !uniVec(in[6])) continue;
                for (int q = 0; q < 2 && !kk; ++q)
                    if (uniOnly(val(mul[4 - q]), 0) && screenX(mul[3 + q])) kk = mul[4 - q];
                if (kk) { ua = in[5]; ub = in[6]; }
            }
            if (!kk || !usable(ua, k) || !usable(ub, k) || !usable(kk, k)) continue;
            Emit& e = replace[k];
            uint32_t sep, conv, wMin;
            stereo(e, sep, conv, wMin);
            const uint32_t raw = next++, du = next++, ad = next++, md = next++, df = next++, sk = next++, sh = next++, c = next++, corr = next++;
            e.Op(spv::Op::OpVectorTimesScalar, { vt[1], raw, vt[3], vt[4] });
            e.Op(spv::Op::OpFSub, { vt[1], du, ub, ua });
            e.Op(spv::Op::OpExtInst, { m.tFloat, ad, m.glsl, GLSLstd450FAbs, vt[4] });
            e.Op(spv::Op::OpExtInst, { m.tFloat, md, m.glsl, GLSLstd450FMax, ad, wMin });
            e.Op(spv::Op::OpFSub, { m.tFloat, df, md, conv });
            e.Op(spv::Op::OpFMul, { m.tFloat, sk, sep, kk });
            e.Op(spv::Op::OpFMul, { m.tFloat, sh, sk, cHalf });
            e.Op(spv::Op::OpFMul, { m.tFloat, c, sh, df });
            e.Op(spv::Op::OpVectorTimesScalar, { vt[1], corr, du, c });
            e.Op(spv::Op::OpFSub, { vt[1], vt[2], raw, corr });
            placed.insert(vt[2]);
            ++counts.rays;
        }
        // History: a placed cell's projection divided by its w, scaled into volume uv by a uniform vec2, read by a 3D
        // sample; the right run's history holds that point h(w) further along u.
        auto vecSize = [&](uint32_t id) -> uint32_t
        {
            spv::Op op; const uint32_t* i = m.At(id, &op);
            bool r = false, t = false;
            if (i) spv::HasResultAndType(op, &r, &t);
            const uint32_t* vt = i && t ? m.Op(i[1], spv::Op::OpTypeVector) : nullptr;
            return vt ? vt[3] : 0;
        };
        auto lane = [&](uint32_t id, uint32_t l, uint32_t& src) -> uint32_t
        {
            for (int g = 0; g < 8; ++g)
            {
                id = val(id);
                const uint32_t* s = m.Op(id, spv::Op::OpVectorShuffle);
                if (!s || 5 + l >= (s[0] >> 16)) break;
                const uint32_t pick = s[5 + l], na = vecSize(s[3]);
                if (!na || pick == 0xFFFFFFFFu) return UINT32_MAX;
                if (pick < na) { id = s[3]; l = pick; } else { id = s[4]; l = pick - na; }
            }
            src = id;
            return l;
        };
        std::unordered_map<uint32_t, uint32_t> divW;   // xyz / w of a placed cell's projection -> w
        for (size_t k = firstFunc; k < m.ins.size() && !placed.empty(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFDiv) continue;
            const uint32_t* dv = code + m.ins[k].at;
            const uint32_t w = splat(dv[4]);
            if (w && vecSize(dv[2]) >= 2 && reaches(dv[3], placed)) divW[dv[2]] = w;
        }
        std::vector<uint32_t> coords3D;
        for (size_t k = firstFunc; k < m.ins.size() && !divW.empty(); ++k)
        {
            const spv::Op op = m.ins[k].op;
            if (op != spv::Op::OpImageSampleExplicitLod && op != spv::Op::OpImageSampleImplicitLod) continue;
            const uint32_t* s = code + m.ins[k].at;
            spv::Op lop; const uint32_t* l = m.At(s[3], &lop);
            const uint32_t* st = l ? m.Op(l[1], spv::Op::OpTypeSampledImage) : nullptr;
            const uint32_t* it = st ? m.Op(st[2], spv::Op::OpTypeImage) : nullptr;
            if (it && it[3] == (uint32_t)spv::Dim::Dim3D) coords3D.push_back(s[4]);
        }
        for (size_t k = firstFunc; k < m.ins.size() && !coords3D.empty(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpFMul || replace.count(k) || rewrite.count(k) || before.count(k)) continue;
            const uint32_t* f = code + m.ins[k].at;
            if (vecSize(f[2]) != 2) continue;
            for (int o = 0; o < 2; ++o)
            {
                uint32_t s0 = 0, s1 = 0;
                if (lane(f[3 + o], 0, s0) != 0 || lane(f[3 + o], 1, s1) != 1 || s0 != s1 || !uniVec(f[4 - o])) continue;
                auto dw = divW.find(s0);
                if (dw == divW.end() || !usable(dw->second, k)) continue;
                const std::unordered_set<uint32_t> self{ f[2] };
                bool sampled = false;
                for (uint32_t c3 : coords3D) if (!sampled && reaches(c3, self)) sampled = true;
                if (!sampled) continue;
                Emit& e = replace[k];
                uint32_t sep, conv, wMin;
                stereo(e, sep, conv, wMin);
                const uint32_t raw = next++, x = next++, ad = next++, md = next++, df = next++, lo = next++, q = next++, h = next++, hh = next++, x2 = next++;
                e.Op(spv::Op::OpFMul, { f[1], raw, f[3], f[4] });
                e.Op(spv::Op::OpCompositeExtract, { m.tFloat, x, raw, 0 });
                e.Op(spv::Op::OpExtInst, { m.tFloat, ad, m.glsl, GLSLstd450FAbs, dw->second });
                e.Op(spv::Op::OpExtInst, { m.tFloat, md, m.glsl, GLSLstd450FMax, ad, wMin });
                e.Op(spv::Op::OpFSub, { m.tFloat, df, md, conv });
                e.Op(spv::Op::OpExtInst, { m.tFloat, lo, m.glsl, GLSLstd450FMax, ad, cTiny });
                e.Op(spv::Op::OpFDiv, { m.tFloat, q, df, lo });
                e.Op(spv::Op::OpFMul, { m.tFloat, h, sep, q });
                e.Op(spv::Op::OpFMul, { m.tFloat, hh, h, cHalf });
                e.Op(spv::Op::OpFAdd, { m.tFloat, x2, x, hh });
                e.Op(spv::Op::OpCompositeInsert, { f[1], f[2], x2, raw, 0 });
                ++counts.reproj;
                break;
            }
        }
    }
    if (!counts.ndc && !counts.rays && !camStores && !froxels) return false;

    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t tS = next++, tPtrS = next++;
    if (!tPtrUF) { tPtrUF = next++; types.Op(spv::Op::OpTypePointer, { tPtrUF, (uint32_t)spv::StorageClass::Uniform, m.tFloat }); }
    types.Op(spv::Op::OpConstant, { tInt, cI0, 0 });
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI2, 2 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    types.Op(spv::Op::OpConstant, { m.tFloat, cOne, Bits(1.f) });
    types.Op(spv::Op::OpConstant, { m.tFloat, cFar, Bits(1e6f) });
    types.Op(spv::Op::OpConstant, { m.tFloat, cHalf, Bits(0.5f) });
    types.Op(spv::Op::OpConstant, { m.tFloat, cTiny, Bits(1e-4f) });
    if (cTinyDx) types.Op(spv::Op::OpConstant, { m.tFloat, cTinyDx, Bits(1e-20f) });   // a ray moves ~1e-3 per pixel
    for (const NewConst& c : rayConsts) types.Op(spv::Op::OpConstant, { c.type, c.id, c.value });
    if (newPtrPF) types.Op(spv::Op::OpTypePointer, { tPtrPF, (uint32_t)spv::StorageClass::PushConstant, m.tFloat });
    if (vFrag)
    {
        uint32_t tVec4 = 0;
        for (const Ins& in : m.ins) if (in.op == spv::Op::OpTypeVector && code[in.at + 2] == m.tFloat && code[in.at + 3] == 4) tVec4 = code[in.at + 1];
        if (newPtrInVec4) types.Op(spv::Op::OpTypePointer, { tPtrInVec4, (uint32_t)spv::StorageClass::Input, tVec4 });
        types.Op(spv::Op::OpVariable, { tPtrInVec4, vFrag, (uint32_t)spv::StorageClass::Input });
        deco.Op(spv::Op::OpDecorate, { vFrag, (uint32_t)spv::Decoration::BuiltIn, (uint32_t)spv::BuiltIn::FragCoord });
    }
    if (cTile)
    {
        types.Op(spv::Op::OpConstant, { m.tFloat, cTile, Bits(16.f) });
        types.Op(spv::Op::OpConstant, { m.tFloat, cTileInv, Bits(0.0625f) });
        types.Op(spv::Op::OpConstant, { m.tFloat, cZero, Bits(0.f) });
    }
    // sep, conv, wMin, eye, fsSep
    types.Op(spv::Op::OpTypeStruct, { tS, m.tFloat, m.tFloat, m.tFloat, m.tFloat, m.tFloat });
    types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
    types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
    deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
    for (uint32_t k = 0; k < 5; ++k) deco.Op(spv::Op::OpMemberDecorate, { tS, k, (uint32_t)spv::Decoration::Offset, k * 4 });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });
    if (vCam)
    {
        // set 8 binding 1: 16 dwords, the camera-window counters with the camera's w row at [6], [7].
        const uint32_t tArr = next++, tCS = next++, tPtrCS = next++;
        types.Op(spv::Op::OpConstant, { tInt, cI6, 6 });
        types.Op(spv::Op::OpConstant, { tInt, cI7, 7 });
        types.Op(spv::Op::OpConstant, { tInt, cI16, 16 });
        types.Op(spv::Op::OpTypeArray, { tArr, m.tFloat, cI16 });
        types.Op(spv::Op::OpTypeStruct, { tCS, tArr });
        types.Op(spv::Op::OpTypePointer, { tPtrCS, (uint32_t)spv::StorageClass::StorageBuffer, tCS });
        if (newPtrCF) types.Op(spv::Op::OpTypePointer, { tPtrCF, (uint32_t)spv::StorageClass::StorageBuffer, m.tFloat });
        types.Op(spv::Op::OpVariable, { tPtrCS, vCam, (uint32_t)spv::StorageClass::StorageBuffer });
        deco.Op(spv::Op::OpDecorate, { tArr, (uint32_t)spv::Decoration::ArrayStride, 4 });
        deco.Op(spv::Op::OpDecorate, { tCS, (uint32_t)spv::Decoration::Block });
        deco.Op(spv::Op::OpMemberDecorate, { tCS, 0, (uint32_t)spv::Decoration::Offset, 0 });
        deco.Op(spv::Op::OpDecorate, { vCam, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
        deco.Op(spv::Op::OpDecorate, { vCam, (uint32_t)spv::Decoration::Binding, 1 });
    }
    // The stereo loads' pointer type, unknown until tPtrUF was settled.
    auto fixPtr = [&](Emit& e)
    {
        for (size_t p = 0; p < e.v.size(); p += e.v[p] >> 16)
            if ((spv::Op)(e.v[p] & 0xFFFF) == spv::Op::OpAccessChain && e.v[p + 1] == 0) e.v[p + 1] = tPtrUF;
    };
    for (auto& b : before) fixPtr(b.second);
    for (auto& r : replace) fixPtr(r.second);

    out.clear();
    out.reserve(words + 256);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        // The half width's size query, in a module that never queried one.
        if (k == 0 && queriesSize && !hasImageQuery)
        {
            out.push_back((2u << 16) | (uint32_t)spv::Op::OpCapability);
            out.push_back((uint32_t)spv::Capability::ImageQuery);
        }
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        auto b = before.find(k);
        if (b != before.end()) out.insert(out.end(), b->second.v.begin(), b->second.v.end());
        auto r = replace.find(k);
        if (r != replace.end()) { out.insert(out.end(), r->second.v.begin(), r->second.v.end()); continue; }
        // SPIR-V 1.4+ lists every global an entry point uses, the stereo block too (spirv-val requires it).
        if (in.op == spv::Op::OpEntryPoint && code[1] >= 0x00010400)
        {
            const size_t head = out.size();
            out.push_back(0);
            out.insert(out.end(), i + 1, i + in.n);
            out.push_back(vStereo);
            if (vCam) out.push_back(vCam);
            if (vFrag) out.push_back(vFrag);
            out[head] = ((uint32_t)(out.size() - head) << 16) | (uint32_t)spv::Op::OpEntryPoint;
            continue;
        }
        const size_t at = out.size();
        out.insert(out.end(), i, i + in.n);
        auto rw = rewrite.find(k);
        if (rw != rewrite.end()) for (const auto& p : rw->second) out[at + p.first] = p.second;
    }
    out[3] = next;
    return true;
}

}} // namespace wiz3d::vk
