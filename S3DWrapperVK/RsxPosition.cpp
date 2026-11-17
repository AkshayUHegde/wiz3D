/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// RPCS3: an RSX fragment program rebuilding a position from WPOS (P = col*wpos.x + ..., then P/P.w) puts the right
// eye's pixel through the game's one camera, so P moves to the left's pixel here (rsxPositionTwins).

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>
#include <spirv/unified1/GLSL.std.450.h>
#include <stdint.h>
#include <string.h>
#include <algorithm>
#include <array>
#include <vector>
#include <unordered_map>

namespace wiz3d { namespace vk {

namespace {

struct Ins { size_t at; uint16_t n; spv::Op op; uint32_t block; };

// What a float holds in terms of wpos.x. Buf: component c of the uniform load id (c 255: id is a scalar).
// Prod: wpos.x times that Buf, plus anything free of wpos.x.
enum class Kind : uint8_t { Free, Unit, Buf, Prod, Dirty };
struct Sym { Kind k = Kind::Free; uint32_t id = 0; uint8_t c = 0; };
using Vec = std::array<Sym, 4>;

inline bool Dep(const Sym& s) { return s.k == Kind::Unit || s.k == Kind::Prod || s.k == Kind::Dirty; }

Sym Mul(const Sym& a, const Sym& b)
{
    Sym r;
    if (a.k == Kind::Unit && b.k == Kind::Buf) { r = b; r.k = Kind::Prod; return r; }
    if (b.k == Kind::Unit && a.k == Kind::Buf) { r = a; r.k = Kind::Prod; return r; }
    if (Dep(a) || Dep(b)) r.k = Kind::Dirty;
    return r;
}

Sym Add(const Sym& a, const Sym& b)
{
    Sym r;
    if (Dep(a) && Dep(b)) { r.k = Kind::Dirty; return r; }
    if (Dep(a)) return a;
    if (Dep(b)) return b;
    return r;
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

// The distinct attributes an RSX vertex program reads: `read_location(N)` once per input, N a constant; 0 otherwise.
unsigned CountRsxVertexInputs(const uint32_t* code, size_t words)
{
    if (words < 5 || code[0] != spv::MagicNumber) return 0;
    uint32_t fn = 0;
    std::vector<uint32_t> seen;
    for (size_t i = 5; i < words; )
    {
        const uint16_t n = (uint16_t)(code[i] >> 16);
        if (!n || i + n > words) return 0;
        const spv::Op op = (spv::Op)(code[i] & 0xFFFF);
        if (op == spv::Op::OpName && n >= 3 && !strncmp((const char*)(code + i + 2), "read_location(", 14)) fn = code[i + 1];
        if (op == spv::Op::OpFunctionCall && n == 5 && fn && code[i + 3] == fn && std::find(seen.begin(), seen.end(), code[i + 4]) == seen.end()) seen.push_back(code[i + 4]);
        i += n;
    }
    return (unsigned)seen.size();
}

bool PatchRsxPositionFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* divides)
{
    *divides = 0;
    if (words < 5 || code[0] != spv::MagicNumber) return false;
    std::vector<Ins> ins;
    uint32_t block = 0;
    for (size_t i = 5; i < words; )
    {
        const uint16_t n = (uint16_t)(code[i] >> 16);
        if (!n || i + n > words) return false;
        const spv::Op op = (spv::Op)(code[i] & 0xFFFF);
        if (op == spv::Op::OpLabel) ++block;
        ins.push_back({ i, n, op, block });
        i += n;
    }
    std::unordered_map<uint32_t, size_t> def;
    std::unordered_map<uint32_t, uint32_t> varClass, iconst, comps;
    std::unordered_map<uint64_t, uint32_t> ptrOf;
    bool fragment = false;
    uint32_t glsl = 0, tFloat = 0, tVec4 = 0, tInt = 0, fragVar = 0;
    size_t firstType = SIZE_MAX, firstFunc = SIZE_MAX;
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const Ins& in = ins[k];
        const uint32_t* i = code + in.at;
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(in.op, &hasRes, &hasType);
        if (hasRes) def[hasType ? i[2] : i[1]] = k;
        switch (in.op)
        {
        case spv::Op::OpEntryPoint: fragment = i[1] == (uint32_t)spv::ExecutionModel::Fragment; break;
        case spv::Op::OpExtInstImport: if (!strcmp((const char*)(i + 2), "GLSL.std.450")) glsl = i[1]; break;
        case spv::Op::OpDecorate:
            if (in.n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::FragCoord) fragVar = i[1];
            break;
        case spv::Op::OpTypeFloat: if (i[2] == 32 && !tFloat) { tFloat = i[1]; comps[i[1]] = 1; } break;
        case spv::Op::OpTypeInt: if (i[2] == 32 && (!tInt || i[3] == 1)) tInt = i[1]; break;
        case spv::Op::OpTypeVector:
            comps[i[1]] = i[3];
            if (i[2] == tFloat && i[3] == 4) tVec4 = i[1];
            break;
        case spv::Op::OpTypePointer: if (!ptrOf.count(((uint64_t)i[2] << 32) | i[3])) ptrOf[((uint64_t)i[2] << 32) | i[3]] = i[1]; break;
        case spv::Op::OpConstant: if (i[1] != tFloat && in.n == 4) iconst[i[2]] = i[3]; break;
        case spv::Op::OpVariable: varClass[i[2]] = i[3]; break;
        case spv::Op::OpFunction: if (firstFunc == SIZE_MAX) firstFunc = k; break;
        default: break;
        }
        const bool typeSection = (in.op >= spv::Op::OpTypeVoid && in.op <= spv::Op::OpTypeForwardPointer) ||
                                 (in.op >= spv::Op::OpConstantTrue && in.op <= spv::Op::OpSpecConstantOp) ||
                                 (in.op == spv::Op::OpVariable && firstFunc == SIZE_MAX);
        if (typeSection && firstType == SIZE_MAX) firstType = k;
    }
    if (!fragment || !glsl || !tFloat || !tVec4 || !tInt || !fragVar || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;

    // RPCS3's get_wpos(): no parameters, reads gl_FragCoord and takes abs(wpos_scale), the game's units per pixel.
    uint32_t wposFn = 0;
    size_t absAt = SIZE_MAX;
    for (size_t k = firstFunc; k < ins.size() && !wposFn; ++k)
    {
        if (ins[k].op != spv::Op::OpFunction || code[ins[k].at + 1] != tVec4) continue;
        if (k + 1 >= ins.size() || ins[k + 1].op == spv::Op::OpFunctionParameter) continue;
        bool frag = false;
        size_t fabs = SIZE_MAX;
        for (size_t j = k + 1; j < ins.size() && ins[j].op != spv::Op::OpFunctionEnd; ++j)
        {
            const uint32_t* f = code + ins[j].at;
            if (ins[j].op == spv::Op::OpLoad && f[3] == fragVar) frag = true;
            if (ins[j].op == spv::Op::OpExtInst && f[3] == glsl && f[4] == GLSLstd450FAbs && fabs == SIZE_MAX) fabs = j;
        }
        if (frag && fabs != SIZE_MAX) { wposFn = code[ins[k].at + 2]; absAt = fabs; }
    }
    if (!wposFn) return false;

    // In program order; branches only lose matches, as the block check below refuses a column loaded elsewhere.
    struct Site { size_t div; std::array<Sym, 4> col; };
    std::vector<Site> sites;
    std::unordered_map<uint32_t, Vec> val, vars;
    auto get = [&](uint32_t id) { auto v = val.find(id); return v == val.end() ? Vec{} : v->second; };
    auto widthOf = [&](uint32_t type) { auto c = comps.find(type); return c == comps.end() ? 0u : c->second; };
    auto typeOf = [&](uint32_t id) -> uint32_t
    {
        auto d = def.find(id);
        if (d == def.end()) return 0;
        bool r = false, t = false;
        spv::HasResultAndType(ins[d->second].op, &r, &t);
        return t ? code[ins[d->second].at + 1] : 0;
    };
    auto opAt = [&](uint32_t id, spv::Op want) -> const uint32_t*
    {
        auto d = def.find(id);
        return d != def.end() && ins[d->second].op == want ? code + ins[d->second].at : nullptr;
    };
    // A pointer as (variable, component), component 255 when it is the whole variable.
    auto place = [&](uint32_t p, uint32_t& var, uint32_t& c)
    {
        var = p; c = 255;
        if (const uint32_t* a = opAt(p, spv::Op::OpAccessChain))
        {
            var = a[3];
            if ((a[0] >> 16) != 5) { c = 254; return; }
            auto v = iconst.find(a[4]);
            c = v == iconst.end() ? 254 : v->second;
        }
    };
    for (size_t k = firstFunc; k < ins.size(); ++k)
    {
        const Ins& in = ins[k];
        const uint32_t* i = code + in.at;
        switch (in.op)
        {
        case spv::Op::OpFunction: vars.clear(); break;
        case spv::Op::OpStore:
        {
            uint32_t var, c;
            place(i[1], var, c);
            auto cls = varClass.find(var);
            if (cls == varClass.end() || (cls->second != (uint32_t)spv::StorageClass::Function && cls->second != (uint32_t)spv::StorageClass::Private)) break;
            if (c == 255) vars[var] = get(i[2]);
            else if (c < 4) vars[var][c] = get(i[2])[0];
            else if (c == 254) { Vec& v = vars[var]; for (Sym& s : v) s.k = Kind::Dirty; }
            break;
        }
        case spv::Op::OpLoad:
        {
            uint32_t var, c;
            place(i[3], var, c);
            auto cls = varClass.find(var);
            if (cls == varClass.end()) break;
            if (cls->second == (uint32_t)spv::StorageClass::Function || cls->second == (uint32_t)spv::StorageClass::Private)
            {
                auto v = vars.find(var);
                if (v == vars.end()) break;
                if (c == 255) val[i[2]] = v->second;
                else if (c < 4) { Vec r; r[0] = v->second[c]; val[i[2]] = r; }
            }
            else if (cls->second == (uint32_t)spv::StorageClass::Uniform || cls->second == (uint32_t)spv::StorageClass::StorageBuffer)
            {
                const uint32_t w = widthOf(i[1]);
                if (!w || w > 4) break;
                Vec r;
                for (uint32_t q = 0; q < w; ++q) { r[q].k = Kind::Buf; r[q].id = i[2]; r[q].c = i[1] == tFloat ? 255 : (uint8_t)q; }
                val[i[2]] = r;
            }
            break;
        }
        case spv::Op::OpFunctionCall:
            // A texel read at a coordinate holding wpos is free of it: the depth a position scales by.
            if (i[3] == wposFn) { Vec r; r[0].k = Kind::Unit; val[i[2]] = r; }
            break;
        case spv::Op::OpVectorShuffle:
        {
            const Vec a = get(i[3]), b = get(i[4]);
            const uint32_t na = widthOf(typeOf(i[3]));
            Vec r;
            for (uint16_t q = 5; q < in.n && q - 5 < 4; ++q)
            {
                const uint32_t s = i[q];
                if (s < na) r[q - 5] = a[s];
                else if (s - na < 4) r[q - 5] = b[s - na];
            }
            val[i[2]] = r;
            break;
        }
        case spv::Op::OpCompositeExtract:
            if (in.n == 5 && i[4] < 4) { Vec r; r[0] = get(i[3])[i[4]]; val[i[2]] = r; }
            break;
        case spv::Op::OpCompositeConstruct:
        {
            Vec r;
            uint32_t at = 0;
            for (uint16_t q = 3; q < in.n; ++q)
            {
                const Vec a = get(i[q]);
                const uint32_t w = widthOf(typeOf(i[q]));
                for (uint32_t e = 0; e < w && at < 4; ++e) r[at++] = a[e];
            }
            val[i[2]] = r;
            break;
        }
        case spv::Op::OpFMul:
        {
            const Vec a = get(i[3]), b = get(i[4]);
            Vec r;
            for (int q = 0; q < 4; ++q) r[q] = Mul(a[q], b[q]);
            val[i[2]] = r;
            break;
        }
        case spv::Op::OpVectorTimesScalar:
        {
            const Vec a = get(i[3]);
            const Sym s = get(i[4])[0];
            Vec r;
            for (int q = 0; q < 4; ++q) r[q] = Mul(a[q], s);
            val[i[2]] = r;
            break;
        }
        case spv::Op::OpFAdd:
        {
            const Vec a = get(i[3]), b = get(i[4]);
            Vec r;
            for (int q = 0; q < 4; ++q) r[q] = Add(a[q], b[q]);
            val[i[2]] = r;
            break;
        }
        case spv::Op::OpFSub:
        {
            const Vec a = get(i[3]), b = get(i[4]);
            Vec r;
            for (int q = 0; q < 4; ++q)
            {
                if (Dep(b[q])) r[q].k = Kind::Dirty;
                else if (Dep(a[q])) r[q] = a[q];
            }
            val[i[2]] = r;
            break;
        }
        case spv::Op::OpDot:
        {
            const Vec a = get(i[3]), b = get(i[4]);
            Sym s;
            for (int q = 0; q < 4; ++q) s = Add(s, Mul(a[q], b[q]));
            Vec r; r[0] = s;
            val[i[2]] = r;
            break;
        }
        case spv::Op::OpExtInst:
        {
            Vec r;
            if (i[3] == glsl && i[4] == GLSLstd450Fma)
            {
                const Vec a = get(i[5]), b = get(i[6]), c = get(i[7]);
                for (int q = 0; q < 4; ++q) r[q] = Add(Mul(a[q], b[q]), c[q]);
            }
            else
            {
                for (uint16_t q = 5; q < in.n; ++q)
                {
                    const Vec a = get(i[q]);
                    for (int e = 0; e < 4; ++e) if (Dep(a[e])) r[e].k = Kind::Dirty;
                }
            }
            val[i[2]] = r;
            break;
        }
        case spv::Op::OpFDiv:
        {
            const Vec a = get(i[3]), b = get(i[4]);
            Vec r;
            for (int q = 0; q < 4; ++q) if (Dep(a[q]) || Dep(b[q])) r[q].k = Kind::Dirty;
            val[i[2]] = r;
            // num / vec4(num.w), and the whole position's x row holding wpos.x.
            if (i[1] != tVec4) break;
            const uint32_t* nl = opAt(i[3], spv::Op::OpLoad);
            const uint32_t* dc = opAt(i[4], spv::Op::OpCompositeConstruct);
            const uint32_t* ds = opAt(i[4], spv::Op::OpVectorShuffle);
            bool own = false;
            if (nl && dc && (dc[0] >> 16) == 7 && dc[3] == dc[4] && dc[3] == dc[5] && dc[3] == dc[6])
            {
                const uint32_t* wl = opAt(dc[3], spv::Op::OpLoad);
                uint32_t var = 0, c = 0;
                if (wl) place(wl[3], var, c);
                own = var == nl[3] && c == 3;
            }
            else if (nl && ds && (ds[0] >> 16) == 9 && ds[5] == 3 && ds[6] == 3 && ds[7] == 3 && ds[8] == 3)
            {
                const uint32_t* sl = opAt(ds[3], spv::Op::OpLoad);
                own = sl && sl[3] == nl[3];
            }
            if (!own) break;
            int rows = 0;
            bool ok = true;
            for (int q = 0; q < 4; ++q)
            {
                if (a[q].k == Kind::Prod)
                {
                    if (q < 3) ++rows;
                    auto d = def.find(a[q].id);
                    ok = ok && d != def.end() && ins[d->second].block == in.block;
                }
                else if (a[q].k != Kind::Free) ok = false;
            }
            if (ok && rows == 3) sites.push_back({ k, a });
            break;
        }
        case spv::Op::OpFNegate: case spv::Op::OpSelect: case spv::Op::OpFMod: case spv::Op::OpMatrixTimesVector: case spv::Op::OpVectorTimesMatrix:
        {
            bool any = false;
            for (uint16_t q = 3; q < in.n; ++q) { const Vec a = get(i[q]); for (const Sym& s : a) any = any || Dep(s); }
            Vec r;
            if (any) for (Sym& s : r) s.k = Kind::Dirty;
            val[i[2]] = r;
            break;
        }
        default: break;
        }
    }
    if (sites.empty()) return false;

    uint32_t next = code[3];
    const uint32_t cI1 = next++, cI2 = next++, cI4 = next++, cI5 = next++, cZero = next++, cOne = next++, vStereo = next++, vScale = next++, tS = next++, tPtrS = next++;
    Emit deco, types;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI2, 2 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    types.Op(spv::Op::OpConstant, { tInt, cI5, 5 });
    types.Op(spv::Op::OpConstant, { tFloat, cZero, Bits(0.f) });
    types.Op(spv::Op::OpConstant, { tFloat, cOne, Bits(1.f) });
    // sep, conv, wMin, eye, fsSep, half the viewport's width
    types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat, tFloat, tFloat, tFloat });
    types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
    uint32_t tPtrUF = 0, tPtrPF = 0;
    auto pu = ptrOf.find(((uint64_t)spv::StorageClass::Uniform << 32) | tFloat);
    if (pu != ptrOf.end()) tPtrUF = pu->second; else { tPtrUF = next++; types.Op(spv::Op::OpTypePointer, { tPtrUF, (uint32_t)spv::StorageClass::Uniform, tFloat }); }
    auto pp = ptrOf.find(((uint64_t)spv::StorageClass::Private << 32) | tFloat);
    if (pp != ptrOf.end()) tPtrPF = pp->second; else { tPtrPF = next++; types.Op(spv::Op::OpTypePointer, { tPtrPF, (uint32_t)spv::StorageClass::Private, tFloat }); }
    types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
    types.Op(spv::Op::OpVariable, { tPtrPF, vScale, (uint32_t)spv::StorageClass::Private });
    deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
    for (uint32_t q = 0; q < 6; ++q) deco.Op(spv::Op::OpMemberDecorate, { tS, q, (uint32_t)spv::Decoration::Offset, q * 4 });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });

    // x_R = x_L + fsSep*(max(w,wMin) - conv)/w in ndc, P.w = 1/w: D = fsSep*(max(1, wMin*P.w) - conv*P.w)*halfWidth*
    // |wpos_scale| in game units and P_L = P - col*D, with no divide (a sky's P.w = 0 cannot NaN the left eye).
    std::unordered_map<size_t, Emit> before;
    std::unordered_map<size_t, std::pair<uint32_t, uint32_t>> rewrite;
    for (const Site& s : sites)
    {
        const uint32_t* f = code + ins[s.div].at;
        Emit& e = before[s.div];
        uint32_t c[4];
        for (int q = 0; q < 4; ++q)
        {
            if (s.col[q].k != Kind::Prod) { c[q] = cZero; continue; }
            if (s.col[q].c == 255) { c[q] = s.col[q].id; continue; }
            c[q] = next++;
            e.Op(spv::Op::OpCompositeExtract, { tFloat, c[q], s.col[q].id, s.col[q].c });
        }
        const uint32_t col = next++, iw = next++, pS = next++, sep = next++, pC = next++, conv = next++, pM = next++, wMin = next++, pH = next++, hw = next++, ks = next++;
        const uint32_t mw = next++, nr = next++, cw = next++, q1 = next++, sd = next++, px = next++, D = next++, corr = next++, num = next++, w2 = next++, den = next++;
        e.Op(spv::Op::OpCompositeConstruct, { tVec4, col, c[0], c[1], c[2], c[3] });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, iw, f[3], 3 });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
        e.Op(spv::Op::OpLoad, { tFloat, sep, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pM, vStereo, cI2 });
        e.Op(spv::Op::OpLoad, { tFloat, wMin, pM });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pH, vStereo, cI5 });
        e.Op(spv::Op::OpLoad, { tFloat, hw, pH });
        e.Op(spv::Op::OpLoad, { tFloat, ks, vScale });
        e.Op(spv::Op::OpFMul, { tFloat, mw, wMin, iw });
        e.Op(spv::Op::OpExtInst, { tFloat, nr, glsl, GLSLstd450FMax, cOne, mw });
        e.Op(spv::Op::OpFMul, { tFloat, cw, conv, iw });
        e.Op(spv::Op::OpFSub, { tFloat, q1, nr, cw });
        e.Op(spv::Op::OpFMul, { tFloat, sd, sep, q1 });
        e.Op(spv::Op::OpFMul, { tFloat, px, sd, hw });
        e.Op(spv::Op::OpFMul, { tFloat, D, px, ks });
        e.Op(spv::Op::OpVectorTimesScalar, { tVec4, corr, col, D });
        e.Op(spv::Op::OpFSub, { tVec4, num, f[3], corr });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, w2, num, 3 });
        e.Op(spv::Op::OpCompositeConstruct, { tVec4, den, w2, w2, w2, w2 });
        rewrite[s.div] = { num, den };
    }

    out.clear();
    out.reserve(words + 128 * sites.size() + 96);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const Ins& in = ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        auto b = before.find(k);
        if (b != before.end()) out.insert(out.end(), b->second.v.begin(), b->second.v.end());
        // SPIR-V 1.4+ lists every global an entry point uses.
        if (in.op == spv::Op::OpEntryPoint && code[1] >= 0x00010400)
        {
            const size_t head = out.size();
            out.push_back(0);
            out.insert(out.end(), i + 1, i + in.n);
            out.push_back(vStereo);
            out.push_back(vScale);
            out[head] = ((uint32_t)(out.size() - head) << 16) | (uint32_t)spv::Op::OpEntryPoint;
            continue;
        }
        const size_t at = out.size();
        out.insert(out.end(), i, i + in.n);
        auto r = rewrite.find(k);
        if (r != rewrite.end()) { out[at + 3] = r->second.first; out[at + 4] = r->second.second; }
        if (k == absAt) { Emit e; e.Op(spv::Op::OpStore, { vScale, i[2] }); out.insert(out.end(), e.v.begin(), e.v.end()); }
    }
    out[3] = next;
    *divides = (unsigned)sites.size();
    return true;
}

}} // namespace wiz3d::vk
