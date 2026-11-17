/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// GT Sport (VKShadGT): velocity is FragCoord's NDC minus last frame's clip position; R's FragCoord carries the stereo
// shift and the varying does not, so the twin takes the shift off the output (and the glass and decal twins below).

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>
#include <stdint.h>
#include <string.h>
#include <functional>
#include <utility>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace wiz3d { namespace vk {

namespace {

struct Ins { size_t at; uint16_t n; spv::Op op; };

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

bool PatchVelocityFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, bool skipUnitW, float unitWEps)
{
    if (words < 5 || code[0] != spv::MagicNumber) return false;
    std::vector<Ins> ins;
    for (size_t i = 5; i < words; )
    {
        const uint16_t n = (uint16_t)(code[i] >> 16);
        if (!n || i + n > words) return false;
        ins.push_back({ i, n, (spv::Op)(code[i] & 0xFFFF) });
        i += n;
    }
    std::unordered_map<uint32_t, size_t> def;
    std::unordered_map<uint32_t, uint32_t> varClass, iconst, location;
    std::unordered_map<uint64_t, uint32_t> ptrOf;
    bool fragment = false;
    uint32_t entryFn = 0, fragVar = 0, tFloat = 0, tVec4 = 0, tInt = 0, tBool = 0;
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
        case spv::Op::OpEntryPoint: fragment = i[1] == (uint32_t)spv::ExecutionModel::Fragment; entryFn = i[2]; break;
        case spv::Op::OpDecorate:
            if (in.n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::FragCoord) fragVar = i[1];
            if (in.n >= 4 && i[2] == (uint32_t)spv::Decoration::Location) location[i[1]] = i[3];
            break;
        case spv::Op::OpTypeFloat: if (i[2] == 32 && !tFloat) tFloat = i[1]; break;
        case spv::Op::OpTypeBool: tBool = i[1]; break;
        case spv::Op::OpTypeInt: if (i[2] == 32 && !tInt) tInt = i[1]; break;
        case spv::Op::OpTypeVector: if (i[2] == tFloat && i[3] == 4) tVec4 = i[1]; break;
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
    if (!fragment || !entryFn || !fragVar || !tFloat || !tVec4 || !tInt || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;

    uint32_t velo = 0;
    for (const auto& l : location)
    {
        auto c = varClass.find(l.first);
        if (l.second == 1 && c != varClass.end() && c->second == (uint32_t)spv::StorageClass::Output) velo = l.first;
    }
    auto vd = def.find(velo);
    if (!velo || vd == def.end() || !ptrOf.count(((uint64_t)spv::StorageClass::Output << 32) | tVec4) ||
        code[ins[vd->second].at + 1] != ptrOf[((uint64_t)spv::StorageClass::Output << 32) | tVec4]) return false;

    // Velocity's x must come from FragCoord.x: walk back from its store, skipping literals an instruction carries.
    auto chainOf = [&](uint32_t p, uint32_t& base, uint32_t& comp)
    {
        base = p; comp = UINT32_MAX;
        auto d = def.find(p);
        if (d == def.end() || ins[d->second].op != spv::Op::OpAccessChain || ins[d->second].n < 5) return;
        const uint32_t* a = code + ins[d->second].at;
        base = a[3];
        auto c = iconst.find(a[4]);
        if (c != iconst.end()) comp = c->second;
    };
    uint32_t xValue = 0;
    for (const Ins& in : ins)
    {
        if (in.op != spv::Op::OpStore) continue;
        uint32_t base, comp;
        chainOf(code[in.at + 1], base, comp);
        if (base == velo && comp == 0) xValue = code[in.at + 2];
    }
    if (!xValue) return false;
    bool fromFragX = false;
    std::unordered_set<uint32_t> seen;
    std::vector<uint32_t> st = { xValue };
    while (!st.empty() && !fromFragX)
    {
        const uint32_t v = st.back();
        st.pop_back();
        if (!seen.insert(v).second) continue;
        auto d = def.find(v);
        if (d == def.end()) continue;
        const Ins& in = ins[d->second];
        const uint32_t* i = code + in.at;
        switch (in.op)
        {
        case spv::Op::OpLoad:
        {
            uint32_t base, comp;
            chainOf(i[3], base, comp);
            if (base == fragVar && comp == 0) fromFragX = true;
            break;
        }
        case spv::Op::OpCompositeExtract:
        {
            auto s = def.find(i[3]);
            if (s != def.end() && ins[s->second].op == spv::Op::OpLoad && code[ins[s->second].at + 3] == fragVar && in.n >= 5 && i[4] == 0) fromFragX = true;
            else st.push_back(i[3]);
            break;
        }
        case spv::Op::OpExtInst: for (uint16_t q = 5; q < in.n; ++q) st.push_back(i[q]); break;
        case spv::Op::OpCompositeInsert: st.push_back(i[3]); st.push_back(i[4]); break;
        case spv::Op::OpVectorShuffle: st.push_back(i[3]); st.push_back(i[4]); break;
        case spv::Op::OpVariable: case spv::Op::OpConstant: case spv::Op::OpConstantComposite: case spv::Op::OpFunction: case spv::Op::OpAccessChain: break;
        default: for (uint16_t q = 3; q < in.n; ++q) st.push_back(i[q]); break;
        }
    }
    if (!fromFragX) return false;

    const uint32_t* fnAt = code + ins[def[entryFn]].at;
    const uint32_t tVoid = fnAt[1], tFn = fnAt[4];

    uint32_t next = code[3];
    const uint32_t cI0 = next++, cI1 = next++, cI2 = next++, cI3 = next++, cZero = next++, cOne = next++, cEps = next++, cNegEps = next++;
    const uint32_t tS = next++, tPtrS = next++, vStereo = next++, newMain = next++;
    Emit deco, types, fn;
    types.Op(spv::Op::OpConstant, { tInt, cI0, 0 });
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI2, 2 });
    types.Op(spv::Op::OpConstant, { tInt, cI3, 3 });
    types.Op(spv::Op::OpConstant, { tFloat, cZero, Bits(0.f) });
    types.Op(spv::Op::OpConstant, { tFloat, cOne, Bits(1.f) });
    types.Op(spv::Op::OpConstant, { tFloat, cEps, Bits(unitWEps) });
    types.Op(spv::Op::OpConstant, { tFloat, cNegEps, Bits(-unitWEps) });
    if (!tBool) { tBool = next++; types.Op(spv::Op::OpTypeBool, { tBool }); }
    types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat });   // sep, conv, wMin
    types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
    auto pointer = [&](spv::StorageClass c)
    {
        auto p = ptrOf.find(((uint64_t)c << 32) | tFloat);
        if (p != ptrOf.end()) return p->second;
        const uint32_t t = next++;
        types.Op(spv::Op::OpTypePointer, { t, (uint32_t)c, tFloat });
        return t;
    };
    const uint32_t tPtrUF = pointer(spv::StorageClass::Uniform), tPtrIF = pointer(spv::StorageClass::Input), tPtrOF = pointer(spv::StorageClass::Output);
    types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
    deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
    for (uint32_t q = 0; q < 3; ++q) deco.Op(spv::Op::OpMemberDecorate, { tS, q, (uint32_t)spv::Decoration::Offset, q * 4 });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });

    // NDC shift = sep*(max(w, wMin) - conv)/w with FragCoord.w = 1/w, so sep*(max(1, wMin*fw) - conv*fw); 0 in the
    // unit-w band the VS leaves alone. No divide: a NaN here would reach the left eye too.
    const uint32_t label = next++, call = next++, pFw = next++, fw = next++, pV = next++, v = next++, pS = next++, sep = next++, pC = next++, conv = next++, pM = next++, wMin = next++;
    const uint32_t mw = next++, gt = next++, nr = next++, cw = next++, q1 = next++, s = next++, nv = next++;
    fn.Op(spv::Op::OpFunction, { tVoid, newMain, (uint32_t)spv::FunctionControlMask::MaskNone, tFn });
    fn.Op(spv::Op::OpLabel, { label });
    fn.Op(spv::Op::OpFunctionCall, { tVoid, call, entryFn });
    fn.Op(spv::Op::OpAccessChain, { tPtrIF, pFw, fragVar, cI3 });
    fn.Op(spv::Op::OpLoad, { tFloat, fw, pFw });
    fn.Op(spv::Op::OpAccessChain, { tPtrOF, pV, velo, cI0 });
    fn.Op(spv::Op::OpLoad, { tFloat, v, pV });
    fn.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI0 });
    fn.Op(spv::Op::OpLoad, { tFloat, sep, pS });
    fn.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
    fn.Op(spv::Op::OpLoad, { tFloat, conv, pC });
    fn.Op(spv::Op::OpAccessChain, { tPtrUF, pM, vStereo, cI2 });
    fn.Op(spv::Op::OpLoad, { tFloat, wMin, pM });
    fn.Op(spv::Op::OpFMul, { tFloat, mw, wMin, fw });
    fn.Op(spv::Op::OpFOrdGreaterThan, { tBool, gt, mw, cOne });
    fn.Op(spv::Op::OpSelect, { tFloat, nr, gt, mw, cOne });
    fn.Op(spv::Op::OpFMul, { tFloat, cw, conv, fw });
    fn.Op(spv::Op::OpFSub, { tFloat, q1, nr, cw });
    fn.Op(spv::Op::OpFMul, { tFloat, s, sep, q1 });
    uint32_t applied = s;
    if (skipUnitW)
    {
        const uint32_t d1 = next++, lo = next++, hi = next++, in = next++, a = next++;
        fn.Op(spv::Op::OpFSub, { tFloat, d1, fw, cOne });
        fn.Op(spv::Op::OpFOrdLessThan, { tBool, lo, d1, cEps });
        fn.Op(spv::Op::OpFOrdGreaterThan, { tBool, hi, d1, cNegEps });
        fn.Op(spv::Op::OpLogicalAnd, { tBool, in, lo, hi });
        fn.Op(spv::Op::OpSelect, { tFloat, a, in, cZero, s });
        applied = a;
    }
    fn.Op(spv::Op::OpFSub, { tFloat, nv, v, applied });
    fn.Op(spv::Op::OpStore, { pV, nv });
    fn.Op(spv::Op::OpReturn, {});
    fn.Op(spv::Op::OpFunctionEnd, {});

    out.clear();
    out.reserve(words + fn.v.size() + types.v.size() + deco.v.size() + 8);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const Ins& in = ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (in.op == spv::Op::OpEntryPoint)
        {
            const size_t head = out.size();
            out.insert(out.end(), i, i + in.n);
            out[head + 2] = newMain;
            // SPIR-V 1.4+ lists every global an entry point uses.
            if (code[1] >= 0x00010400) out.push_back(vStereo);
            out[head] = ((uint32_t)(out.size() - head) << 16) | (uint32_t)spv::Op::OpEntryPoint;
            continue;
        }
        const size_t at = out.size();
        out.insert(out.end(), i, i + in.n);
        if ((in.op == spv::Op::OpExecutionMode || in.op == spv::Op::OpExecutionModeId) && i[1] == entryFn) out[at + 1] = newMain;
    }
    out.insert(out.end(), fn.v.begin(), fn.v.end());
    out[3] = next;
    return true;
}

// Glass reads the scene behind it at (M0.p, M1.p)/M3.p, M the left camera's screen matrix: x gets the VS's shift in UV,
// x + sep/2*(max(q, wMin) - conv), for a 2D colour read whose X, Y, q are each c0*p0 + c1*p1 + c2*p2 + c3 over one p.
bool PatchGlassFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* sitesOut)
{
    if (sitesOut) *sitesOut = 0;
    if (words < 5 || code[0] != spv::MagicNumber) return false;
    std::vector<Ins> ins;
    for (size_t i = 5; i < words; )
    {
        const uint16_t n = (uint16_t)(code[i] >> 16);
        if (!n || i + n > words) return false;
        ins.push_back({ i, n, (spv::Op)(code[i] & 0xFFFF) });
        i += n;
    }
    std::unordered_map<uint32_t, size_t> def;
    std::unordered_map<uint32_t, uint32_t> varClass, konst;
    std::unordered_map<uint64_t, uint32_t> ptrOf;
    bool fragment = false;
    uint32_t entryFn = 0, tFloat = 0, tInt = 0, tBool = 0;
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
        case spv::Op::OpEntryPoint: fragment = i[1] == (uint32_t)spv::ExecutionModel::Fragment; entryFn = i[2]; break;
        case spv::Op::OpTypeFloat: if (i[2] == 32 && !tFloat) tFloat = i[1]; break;
        case spv::Op::OpTypeBool: tBool = i[1]; break;
        case spv::Op::OpTypeInt: if (i[2] == 32 && !tInt) tInt = i[1]; break;
        case spv::Op::OpTypePointer: if (!ptrOf.count(((uint64_t)i[2] << 32) | i[3])) ptrOf[((uint64_t)i[2] << 32) | i[3]] = i[1]; break;
        case spv::Op::OpConstant: if (in.n == 4) konst[i[2]] = i[3]; break;
        case spv::Op::OpVariable: varClass[i[2]] = i[3]; break;
        case spv::Op::OpFunction: if (firstFunc == SIZE_MAX) firstFunc = k; break;
        default: break;
        }
        const bool typeSection = (in.op >= spv::Op::OpTypeVoid && in.op <= spv::Op::OpTypeForwardPointer) ||
                                 (in.op >= spv::Op::OpConstantTrue && in.op <= spv::Op::OpSpecConstantOp) ||
                                 (in.op == spv::Op::OpVariable && firstFunc == SIZE_MAX);
        if (typeSection && firstType == SIZE_MAX) firstType = k;
    }
    if (!fragment || !entryFn || !tFloat || !tInt || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;

    auto Def = [&](uint32_t id) -> const Ins* { auto d = def.find(id); return d == def.end() ? nullptr : &ins[d->second]; };
    // shadPS4 writes a coordinate as (1 & SampleID) | (~1 & bits(x)).
    auto Strip = [&](uint32_t v)
    {
        for (int k = 0; k < 6; ++k)
        {
            const Ins* d = Def(v);
            if (!d) break;
            const uint32_t* i = code + d->at;
            if (d->op == spv::Op::OpBitcast || d->op == spv::Op::OpCopyObject) { v = i[3]; continue; }
            if (d->op != spv::Op::OpBitwiseOr) break;
            uint32_t kept = 0;
            for (int o = 0; o < 2; ++o)
            {
                const Ins* a = Def(i[3 + o]);
                if (!a || a->op != spv::Op::OpBitwiseAnd) continue;
                const uint32_t* j = code + a->at;
                for (int q = 0; q < 2; ++q)
                {
                    auto c = konst.find(j[3 + q]);
                    if (c != konst.end() && c->second == 0xFFFFFFFEu) kept = j[3 + (q ^ 1)];
                }
            }
            if (!kept) break;
            v = kept;
        }
        return v;
    };
    struct Divide { uint32_t mul, inv, num, q, slot; };
    auto DivideOf = [&](uint32_t v, Divide& r)
    {
        const Ins* d = Def(v);
        if (!d || d->op != spv::Op::OpFMul) return false;
        const uint32_t* i = code + d->at;
        for (uint32_t o = 0; o < 2; ++o)
        {
            const Ins* f = Def(i[3 + o]);
            if (!f || f->op != spv::Op::OpFDiv) continue;
            const uint32_t* j = code + f->at;
            auto c = konst.find(j[3]);
            if (c == konst.end() || c->second != Bits(1.f)) continue;
            r = { v, i[3 + o], i[3 + (o ^ 1)], j[4], 3 + (o ^ 1) };
            return true;
        }
        return false;
    };
    // A constant: a buffer load, as bits or as float. Its variable, else 0.
    auto BufferLoad = [&](uint32_t v) -> uint32_t
    {
        for (int k = 0; k < 3; ++k)
        {
            const Ins* d = Def(v);
            if (!d) return 0;
            const uint32_t* i = code + d->at;
            if (d->op == spv::Op::OpBitcast || d->op == spv::Op::OpCopyObject) { v = i[3]; continue; }
            if (d->op != spv::Op::OpLoad) return 0;
            const Ins* a = Def(i[3]);
            if (!a || a->op != spv::Op::OpAccessChain) return 0;
            const uint32_t base = code[a->at + 3];
            auto c = varClass.find(base);
            if (c == varClass.end()) return 0;
            const uint32_t cls = c->second;
            return cls == (uint32_t)spv::StorageClass::StorageBuffer || cls == (uint32_t)spv::StorageClass::Uniform || cls == (uint32_t)spv::StorageClass::PushConstant ? base : 0;
        }
        return 0;
    };
    struct Row { uint32_t base = 0, p[3] = {}; bool operator==(const Row& o) const { return base == o.base && p[0] == o.p[0] && p[1] == o.p[1] && p[2] == o.p[2]; } };
    auto RowOf = [&](uint32_t v, Row& r)
    {
        std::vector<uint32_t> st = { v }, leaves;
        while (!st.empty())
        {
            const uint32_t x = st.back();
            st.pop_back();
            const Ins* d = Def(x);
            if (d && d->op == spv::Op::OpFAdd && leaves.size() + st.size() < 6) { st.push_back(code[d->at + 3]); st.push_back(code[d->at + 4]); }
            else leaves.push_back(x);
        }
        if (leaves.size() != 4) return false;
        uint32_t trans = 0, terms = 0;
        for (uint32_t x : leaves)
        {
            if (const uint32_t c = BufferLoad(x))
            {
                if (trans) return false;
                trans = c;
                continue;
            }
            const Ins* d = Def(x);
            if (!d || d->op != spv::Op::OpFMul || terms == 3) return false;
            const uint32_t a = code[d->at + 3], b = code[d->at + 4], ca = BufferLoad(a), cb = BufferLoad(b);
            if (!ca == !cb) return false;
            if (r.base && r.base != (ca ? ca : cb)) return false;
            r.base = ca ? ca : cb;
            r.p[terms++] = ca ? b : a;
        }
        if (terms != 3 || trans != r.base || r.p[0] == r.p[1] || r.p[0] == r.p[2] || r.p[1] == r.p[2]) return false;
        if (r.p[0] > r.p[1]) std::swap(r.p[0], r.p[1]);
        if (r.p[1] > r.p[2]) std::swap(r.p[1], r.p[2]);
        if (r.p[0] > r.p[1]) std::swap(r.p[0], r.p[1]);
        return true;
    };

    struct Target { uint32_t num, q, slot; };
    std::unordered_map<uint32_t, Target> targets;   // the x divide's result id
    for (const Ins& in : ins)
    {
        if (in.op != spv::Op::OpImageSampleImplicitLod && in.op != spv::Op::OpImageSampleExplicitLod) continue;
        const uint32_t* i = code + in.at;
        const Ins* si = Def(i[3]);
        const Ins* coord = Def(i[4]);
        if (!si || !coord || coord->op != spv::Op::OpCompositeConstruct || coord->n != 5) continue;
        const Ins* ty = Def(code[si->at + 1]);
        if (ty && ty->op == spv::Op::OpTypeSampledImage) ty = Def(code[ty->at + 2]);
        if (!ty || ty->op != spv::Op::OpTypeImage) continue;
        const uint32_t* t = code + ty->at;
        if (t[3] != (uint32_t)spv::Dim::Dim2D || t[4] == 1 || t[5] != 0 || t[6] != 0) continue;
        Divide dx, dy;
        if (!DivideOf(Strip(code[coord->at + 3]), dx) || !DivideOf(Strip(code[coord->at + 4]), dy) || dx.inv != dy.inv) continue;
        Row rx, ry, rq;
        if (!RowOf(dx.num, rx) || !RowOf(dy.num, ry) || !RowOf(dx.q, rq) || !(rx == ry) || !(rx == rq)) continue;
        targets[dx.mul] = { dx.num, dx.q, dx.slot };
    }
    if (targets.empty()) return false;

    uint32_t next = code[3];
    const uint32_t cI0 = next++, cI1 = next++, cI2 = next++, cHalf = next++, tS = next++, tPtrS = next++, vStereo = next++;
    Emit deco, types;
    types.Op(spv::Op::OpConstant, { tInt, cI0, 0 });
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI2, 2 });
    types.Op(spv::Op::OpConstant, { tFloat, cHalf, Bits(.5f) });
    if (!tBool) { tBool = next++; types.Op(spv::Op::OpTypeBool, { tBool }); }
    types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat });   // sep, conv, wMin
    types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
    uint32_t tPtrUF;
    auto pf = ptrOf.find(((uint64_t)spv::StorageClass::Uniform << 32) | tFloat);
    if (pf != ptrOf.end()) tPtrUF = pf->second;
    else { tPtrUF = next++; types.Op(spv::Op::OpTypePointer, { tPtrUF, (uint32_t)spv::StorageClass::Uniform, tFloat }); }
    types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
    deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
    for (uint32_t q = 0; q < 3; ++q) deco.Op(spv::Op::OpMemberDecorate, { tS, q, (uint32_t)spv::Decoration::Offset, q * 4 });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });

    out.clear();
    out.reserve(words + types.v.size() + deco.v.size() + targets.size() * 64 + 8);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const Ins& in = ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (in.op == spv::Op::OpEntryPoint && code[1] >= 0x00010400)
        {
            // SPIR-V 1.4+ lists every global an entry point uses.
            const size_t head = out.size();
            out.insert(out.end(), i, i + in.n);
            out.push_back(vStereo);
            out[head] = ((uint32_t)(out.size() - head) << 16) | (uint32_t)spv::Op::OpEntryPoint;
            continue;
        }
        if (in.op == spv::Op::OpFMul)
        {
            auto tg = targets.find(i[2]);
            if (tg != targets.end())
            {
                const Target& g = tg->second;
                const uint32_t pS = next++, sep = next++, pC = next++, conv = next++, pM = next++, wMin = next++, gt = next++, mq = next++, dq = next++, hs = next++, sh = next++, nx = next++;
                Emit e;
                e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI0 });
                e.Op(spv::Op::OpLoad, { tFloat, sep, pS });
                e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
                e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
                e.Op(spv::Op::OpAccessChain, { tPtrUF, pM, vStereo, cI2 });
                e.Op(spv::Op::OpLoad, { tFloat, wMin, pM });
                e.Op(spv::Op::OpFOrdGreaterThan, { tBool, gt, g.q, wMin });
                e.Op(spv::Op::OpSelect, { tFloat, mq, gt, g.q, wMin });
                e.Op(spv::Op::OpFSub, { tFloat, dq, mq, conv });
                e.Op(spv::Op::OpFMul, { tFloat, hs, sep, cHalf });
                e.Op(spv::Op::OpFMul, { tFloat, sh, hs, dq });
                e.Op(spv::Op::OpFAdd, { tFloat, nx, g.num, sh });
                out.insert(out.end(), e.v.begin(), e.v.end());
                const size_t at = out.size();
                out.insert(out.end(), i, i + in.n);
                out[at + g.slot] = nx;
                continue;
            }
        }
        out.insert(out.end(), i, i + in.n);
    }
    out[3] = next;
    if (sitesOut) *sitesOut = (unsigned)targets.size();
    return true;
}

// A box decal (depth read at screen x, then M*(x, y, z, 1) through its matrices, drawn where a position falls in the
// box): past the first W, x is L's pixel, x - D (D = fsSep*(max(1, wMin*W) - conv*W)); needs a discard.
bool PatchDecalFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* rowsOut, bool* fromVarying)
{
    if (rowsOut) *rowsOut = 0;
    if (fromVarying) *fromVarying = false;
    if (words < 5 || code[0] != spv::MagicNumber) return false;
    std::vector<Ins> ins;
    for (size_t i = 5; i < words; )
    {
        const uint16_t n = (uint16_t)(code[i] >> 16);
        if (!n || i + n > words) return false;
        ins.push_back({ i, n, (spv::Op)(code[i] & 0xFFFF) });
        i += n;
    }
    std::unordered_map<uint32_t, size_t> def;
    std::unordered_map<uint32_t, uint32_t> varClass, konst, ptrClass;
    std::unordered_map<uint64_t, uint32_t> ptrOf;
    std::unordered_set<uint32_t> screenVars, locationVars;
    bool fragment = false, discards = false;
    uint32_t entryFn = 0, tFloat = 0, tInt = 0, tBool = 0;
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
        case spv::Op::OpEntryPoint: fragment = i[1] == (uint32_t)spv::ExecutionModel::Fragment; entryFn = i[2]; break;
        case spv::Op::OpDecorate:
            if (in.n >= 4 && i[2] == (uint32_t)spv::Decoration::Location) { screenVars.insert(i[1]); locationVars.insert(i[1]); }
            if (in.n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::FragCoord) screenVars.insert(i[1]);
            break;
        case spv::Op::OpTypeFloat: if (i[2] == 32 && !tFloat) tFloat = i[1]; break;
        case spv::Op::OpTypeBool: tBool = i[1]; break;
        case spv::Op::OpTypeInt: if (i[2] == 32 && !tInt) tInt = i[1]; break;
        case spv::Op::OpTypePointer:
            ptrClass[i[1]] = i[2];
            if (!ptrOf.count(((uint64_t)i[2] << 32) | i[3])) ptrOf[((uint64_t)i[2] << 32) | i[3]] = i[1];
            break;
        case spv::Op::OpConstant: if (in.n == 4) konst[i[2]] = i[3]; break;
        case spv::Op::OpVariable: varClass[i[2]] = i[3]; break;
        case spv::Op::OpFunction: if (firstFunc == SIZE_MAX) firstFunc = k; break;
        case spv::Op::OpKill: case spv::Op::OpTerminateInvocation: case spv::Op::OpDemoteToHelperInvocation: discards = true; break;
        default: break;
        }
        const bool typeSection = (in.op >= spv::Op::OpTypeVoid && in.op <= spv::Op::OpTypeForwardPointer) ||
                                 (in.op >= spv::Op::OpConstantTrue && in.op <= spv::Op::OpSpecConstantOp) ||
                                 (in.op == spv::Op::OpVariable && firstFunc == SIZE_MAX);
        if (typeSection && firstType == SIZE_MAX) firstType = k;
    }
    if (!fragment || !discards || !entryFn || !tFloat || !tInt || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;

    auto Def = [&](uint32_t id) -> const Ins* { auto d = def.find(id); return d == def.end() ? nullptr : &ins[d->second]; };
    // A constant: a buffer load, directly or through shadPS4's direct_table branch (a phi of two loads).
    std::function<bool(uint32_t, int)> IsConst = [&](uint32_t v, int depth) -> bool
    {
        const Ins* d = Def(v);
        if (!d || depth > 4) return false;
        const uint32_t* i = code + d->at;
        if (d->op == spv::Op::OpBitcast || d->op == spv::Op::OpCopyObject) return IsConst(i[3], depth + 1);
        if (d->op == spv::Op::OpLoad)
        {
            const Ins* p = Def(i[3]);
            if (!p) return false;
            auto c = ptrClass.find(code[p->at + 1]);
            if (c == ptrClass.end()) return false;
            return c->second == (uint32_t)spv::StorageClass::StorageBuffer || c->second == (uint32_t)spv::StorageClass::Uniform ||
                   c->second == (uint32_t)spv::StorageClass::PushConstant || c->second == (uint32_t)spv::StorageClass::PhysicalStorageBuffer;
        }
        if (d->op == spv::Op::OpPhi)
        {
            for (uint16_t q = 3; q + 1 < d->n; q += 2) if (!IsConst(i[q], depth + 1)) return false;
            return d->n > 3;
        }
        return false;
    };
    std::function<bool(uint32_t, int)> FromImage = [&](uint32_t v, int depth) -> bool
    {
        const Ins* d = Def(v);
        if (!d || depth > 14) return false;
        const uint32_t* i = code + d->at;
        switch (d->op)
        {
        case spv::Op::OpImageSampleImplicitLod: case spv::Op::OpImageSampleExplicitLod: case spv::Op::OpImageFetch: return true;
        case spv::Op::OpCompositeExtract: case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpFNegate: return FromImage(i[3], depth + 1);
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFDiv: return FromImage(i[3], depth + 1) || FromImage(i[4], depth + 1);
        default: return false;
        }
    };
    // Which components of a varying or FragCoord a value comes from: x/w has x and w, y/w has y and w
    // (gl_BaryCoordEXT has no Location and does not count).
    bool sawLocation = false;
    std::function<uint32_t(uint32_t, int)> Comps = [&](uint32_t v, int depth) -> uint32_t
    {
        const Ins* d = Def(v);
        if (!d || depth > 16) return 0;
        const uint32_t* i = code + d->at;
        switch (d->op)
        {
        case spv::Op::OpLoad:
        {
            const Ins* p = Def(i[3]);
            if (!p || p->op != spv::Op::OpAccessChain || !screenVars.count(code[p->at + 3])) return 0;
            auto c = konst.find(code[p->at + p->n - 1]);
            if (c == konst.end() || c->second >= 4) return 0;
            if (locationVars.count(code[p->at + 3])) sawLocation = true;
            return 1u << c->second;
        }
        case spv::Op::OpCompositeExtract:
        {
            const Ins* s = Def(i[3]);
            if (s && s->op == spv::Op::OpLoad && screenVars.count(code[s->at + 3]) && d->n >= 5 && i[4] < 4)
            {
                if (locationVars.count(code[s->at + 3])) sawLocation = true;
                return 1u << i[4];
            }
            return Comps(i[3], depth + 1);
        }
        case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpFNegate: return Comps(i[3], depth + 1);
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFDiv: return Comps(i[3], depth + 1) | Comps(i[4], depth + 1);
        case spv::Op::OpExtInst: { uint32_t m = 0; for (uint16_t q = 5; q < d->n; ++q) m |= Comps(i[q], depth + 1); return m; }
        default: return 0;
        }
    };
    struct Term { uint32_t operand, coef; size_t mul; };
    struct Row { uint32_t root; size_t at; std::vector<Term> terms; };
    auto RowOf = [&](uint32_t v, size_t at, Row& r)
    {
        std::vector<uint32_t> st = { v }, leaves;
        while (!st.empty())
        {
            const uint32_t x = st.back();
            st.pop_back();
            const Ins* d = Def(x);
            if (d && d->op == spv::Op::OpFAdd && leaves.size() + st.size() < 6) { st.push_back(code[d->at + 3]); st.push_back(code[d->at + 4]); }
            else leaves.push_back(x);
        }
        if (leaves.size() != 4) return false;
        int trans = 0;
        r = { v, at, {} };
        for (uint32_t x : leaves)
        {
            if (IsConst(x, 0)) { ++trans; continue; }
            const Ins* d = Def(x);
            if (!d || d->op != spv::Op::OpFMul) return false;
            const uint32_t a = code[d->at + 3], b = code[d->at + 4];
            const bool ca = IsConst(a, 0), cb = IsConst(b, 0);
            if (ca == cb) return false;
            r.terms.push_back({ ca ? b : a, ca ? a : b, def[x] });
        }
        return trans == 1 && r.terms.size() == 3;
    };
    // Rows over one (x, y, z), z from an image read: M*(x, y, z, 1).
    std::vector<Row> rows;
    uint32_t X = 0, Z = 0;
    for (size_t k = firstFunc; k < ins.size(); ++k)
    {
        if (ins[k].op != spv::Op::OpFAdd) continue;
        Row r;
        if (!RowOf(code[ins[k].at + 2], k, r)) continue;
        uint32_t z = 0, x = 0, y = 0;
        int images = 0;
        for (const Term& t : r.terms) if (FromImage(t.operand, 0)) { z = t.operand; ++images; }
        if (images != 1) continue;
        bool loc = false;
        for (const Term& t : r.terms)
        {
            if (t.operand == z) continue;
            sawLocation = false;
            const uint32_t m = Comps(t.operand, 0);
            if ((m & 3) == 1) { x = t.operand; loc = sawLocation; }
            else if ((m & 3) == 2) y = t.operand;
        }
        if (!x || !y) continue;
        if (X && (x != X || z != Z)) return false;   // two rebuilds from different pixels: not this shape
        X = x; Z = z;
        if (fromVarying) *fromVarying = loc;
        rows.push_back(std::move(r));
    }
    if (rows.size() < 4 || rows.size() % 4) return false;

    // The first W, 1/W of the first matrix, and the code following it: depth is known there, and it must reach
    // everything below (not inside a branch).
    size_t k0 = SIZE_MAX;
    uint32_t W = 0, cW = 0;
    for (const Row& r : rows)
    {
        for (const Ins& in : ins)
        {
            if (in.op != spv::Op::OpFDiv || code[in.at + 4] != r.root) continue;
            auto c = konst.find(code[in.at + 3]);
            if (c != konst.end() && c->second == Bits(1.f)) { k0 = r.at; W = r.root; break; }
        }
        if (W) break;
    }
    if (!W) return false;
    size_t fnEnd = SIZE_MAX;
    {
        std::vector<uint32_t> open;
        bool inEntry = false;
        for (size_t k = firstFunc; k < ins.size(); ++k)
        {
            const uint32_t* i = code + ins[k].at;
            if (ins[k].op == spv::Op::OpFunction) inEntry = i[2] == entryFn;
            if (ins[k].op == spv::Op::OpFunctionEnd && inEntry) { fnEnd = k; break; }
            if (!inEntry) { if (k == k0) return false; continue; }
            if (ins[k].op == spv::Op::OpSelectionMerge || ins[k].op == spv::Op::OpLoopMerge) open.push_back(i[1]);
            if (ins[k].op == spv::Op::OpLabel) { while (!open.empty() && open.back() == i[1]) open.pop_back(); }
            if (k == k0 && !open.empty()) return false;
        }
    }
    if (fnEnd == SIZE_MAX) return false;
    // Rows that already hold R's x when W is known take the correction after their own root.
    std::unordered_map<size_t, std::pair<uint32_t, uint32_t>> correct;   // root's instruction -> (root, its x column)
    // The same, for a root W has not reached yet: its correction waits for D.
    std::vector<std::pair<uint32_t, uint32_t>> atW;
    for (const Row& r : rows)
    {
        const Term* tx = nullptr;
        for (const Term& t : r.terms) if (t.operand == X) tx = &t;
        if (!tx) return false;
        if (tx->mul > k0) continue;
        if (r.root == W) { cW = tx->coef; continue; }
        if (r.at > k0) { correct[r.at] = { r.root, tx->coef }; continue; }
        // Used before D exists: that use stays on R's pixel, so this is not a shape to patch.
        for (size_t k = r.at + 1; k <= k0; ++k)
            for (uint16_t a = 1; a < ins[k].n; ++a) if (code[ins[k].at + a] == r.root) return false;
        atW.push_back({ r.root, tx->coef });
    }
    if (!cW) return false;

    uint32_t next = code[3];
    const uint32_t cI1 = next++, cI2 = next++, cI4 = next++, cOne = next++, tS = next++, tPtrS = next++, vStereo = next++;
    Emit deco, types;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI2, 2 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    types.Op(spv::Op::OpConstant, { tFloat, cOne, Bits(1.f) });
    if (!tBool) { tBool = next++; types.Op(spv::Op::OpTypeBool, { tBool }); }
    types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat, tFloat, tFloat });   // sep, conv, wMin, eye, fsSep
    types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
    uint32_t tPtrUF;
    auto pf = ptrOf.find(((uint64_t)spv::StorageClass::Uniform << 32) | tFloat);
    if (pf != ptrOf.end()) tPtrUF = pf->second;
    else { tPtrUF = next++; types.Op(spv::Op::OpTypePointer, { tPtrUF, (uint32_t)spv::StorageClass::Uniform, tFloat }); }
    types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
    deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
    for (uint32_t q = 0; q < 5; ++q) deco.Op(spv::Op::OpMemberDecorate, { tS, q, (uint32_t)spv::Decoration::Offset, q * 4 });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });

    const uint32_t pS = next++, fsSep = next++, pC = next++, conv = next++, pM = next++, wMin = next++, mw = next++, gt = next++, nr = next++, cw = next++, q1 = next++, D = next++, xL = next++, wc = next++, Wn = next++;
    std::unordered_map<uint32_t, uint32_t> repl;
    out.clear();
    out.reserve(words + types.v.size() + deco.v.size() + rows.size() * 16 + 64);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const Ins& in = ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (in.op == spv::Op::OpEntryPoint && code[1] >= 0x00010400)
        {
            // SPIR-V 1.4+ lists every global an entry point uses.
            const size_t head = out.size();
            out.insert(out.end(), i, i + in.n);
            out.push_back(vStereo);
            out[head] = ((uint32_t)(out.size() - head) << 16) | (uint32_t)spv::Op::OpEntryPoint;
            continue;
        }
        const size_t pos = out.size();
        out.insert(out.end(), i, i + in.n);
        if (k > k0 && k < fnEnd && !repl.empty())
        {
            uint16_t from = 3, to = in.n;
            switch (in.op)
            {
            case spv::Op::OpStore: from = 1; break;
            case spv::Op::OpExtInst: from = 5; break;
            case spv::Op::OpCompositeExtract: to = 4; break;
            case spv::Op::OpCompositeInsert: case spv::Op::OpVectorShuffle: to = in.n < 5 ? in.n : 5; break;
            default:
            {
                bool hasRes = false, hasType = false;
                spv::HasResultAndType(in.op, &hasRes, &hasType);
                if (!hasRes || !hasType) to = 0;
                break;
            }
            }
            for (uint16_t a = from; a < to; ++a) { auto r = repl.find(out[pos + a]); if (r != repl.end()) out[pos + a] = r->second; }
        }
        if (k == k0)
        {
            Emit e;
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
            e.Op(spv::Op::OpLoad, { tFloat, fsSep, pS });
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
            e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pM, vStereo, cI2 });
            e.Op(spv::Op::OpLoad, { tFloat, wMin, pM });
            e.Op(spv::Op::OpFMul, { tFloat, mw, wMin, W });
            e.Op(spv::Op::OpFOrdGreaterThan, { tBool, gt, mw, cOne });
            e.Op(spv::Op::OpSelect, { tFloat, nr, gt, mw, cOne });
            e.Op(spv::Op::OpFMul, { tFloat, cw, conv, W });
            e.Op(spv::Op::OpFSub, { tFloat, q1, nr, cw });
            e.Op(spv::Op::OpFMul, { tFloat, D, fsSep, q1 });
            e.Op(spv::Op::OpFSub, { tFloat, xL, X, D });
            e.Op(spv::Op::OpFMul, { tFloat, wc, cW, D });
            e.Op(spv::Op::OpFSub, { tFloat, Wn, W, wc });
            repl[X] = xL;
            repl[W] = Wn;
            for (const auto& a : atW)
            {
                const uint32_t m = next++, nv = next++;
                e.Op(spv::Op::OpFMul, { tFloat, m, a.second, D });
                e.Op(spv::Op::OpFSub, { tFloat, nv, a.first, m });
                repl[a.first] = nv;
            }
            out.insert(out.end(), e.v.begin(), e.v.end());
        }
        auto c = correct.find(k);
        if (c != correct.end())
        {
            const uint32_t m = next++, nv = next++;
            Emit e;
            e.Op(spv::Op::OpFMul, { tFloat, m, c->second.second, D });
            e.Op(spv::Op::OpFSub, { tFloat, nv, c->second.first, m });
            out.insert(out.end(), e.v.begin(), e.v.end());
            repl[c->second.first] = nv;
        }
    }
    out[3] = next;
    if (rowsOut) *rowsOut = (unsigned)rows.size();
    return true;
}

}} // namespace wiz3d::vk
