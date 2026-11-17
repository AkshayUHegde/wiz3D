/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The vertex patch (ShaderAnalysis/modify.cpp method 2): Position.x += sep * (w - conv) from set kStereoSet after the
// stage's own body, by a wrapping entry point (before each OpEmitVertex in geometry shaders).

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include <spirv/unified1/spirv.hpp11>
#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <array>
#include <functional>

namespace wiz3d { namespace vk {

namespace {

struct Inst
{
    size_t   at;        // word offset in the module
    uint16_t count;
    spv::Op  op;
    const uint32_t* w;  // words of the instruction (w[0] is the opcode word)
};

struct Module
{
    std::vector<uint32_t> words;
    std::vector<Inst>     insts;
    uint32_t bound = 0;
    uint32_t version = 0;

    bool Parse(const uint32_t* code, size_t n)
    {
        if (n < 5 || code[0] != spv::MagicNumber) return false;
        words.assign(code, code + n);
        version = words[1];
        bound = words[3];
        size_t i = 5;
        while (i < n)
        {
            const uint32_t w0 = words[i];
            const uint16_t count = (uint16_t)(w0 >> 16);
            if (count == 0 || i + count > n) return false;
            insts.push_back({ i, count, (spv::Op)(w0 & 0xFFFF), nullptr });
            i += count;
        }
        for (auto& in : insts) in.w = &words[in.at];
        return true;
    }
};

// Assembles one instruction at a time into a word vector.
struct Builder
{
    std::vector<uint32_t> out;
    void Emit(spv::Op op, std::initializer_list<uint32_t> ops)
    {
        out.push_back(((uint32_t)(ops.size() + 1) << 16) | (uint32_t)op);
        for (uint32_t o : ops) out.push_back(o);
    }
};

inline uint32_t FloatBits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// Ids the shift sequence needs; filled from the module or freshly declared.
struct Ids
{
    uint32_t tFloat = 0, tVec4 = 0, tInt = 0, tUint = 0, tBool = 0;
    uint32_t tPtrOutVec4 = 0, tPtrUniFloat = 0;
    uint32_t cFloat0 = 0, cFloat1 = 0, cInt0 = 0, cInt1 = 0, cInt2 = 0, cMember = 0;
    uint32_t cEps = 0, cNegEps = 0;
    uint32_t cVmZ = 0, cVmSep = 0, cVmConv = 0;   // VKViewModelClipZ and its scales; 0 when off
    uint32_t vParams = 0;
    uint32_t posVar = 0;      // gl_PerVertex block variable (case a)
    uint32_t posDirect = 0;   // standalone Position variable (case b)
};

// sep * (w - conv), guarded for w == 1 when configured: what Position.x gains at this w. z: Position.z, given where
// the first-person weapon can be told by it (VKViewModelClipZ), which then takes its own scaled sep and conv.
uint32_t EmitShiftAmount(Builder& b, const Ids& id, uint32_t w, uint32_t& next, uint32_t z = 0)
{
    auto NewId = [&]() { return next++; };
    const uint32_t pSep = NewId(), sepAll = NewId(), pConv = NewId(), convAll = NewId();
    b.Emit(spv::Op::OpAccessChain, { id.tPtrUniFloat, pSep, id.vParams, id.cInt0 });
    b.Emit(spv::Op::OpLoad, { id.tFloat, sepAll, pSep });
    b.Emit(spv::Op::OpAccessChain, { id.tPtrUniFloat, pConv, id.vParams, id.cInt1 });
    b.Emit(spv::Op::OpLoad, { id.tFloat, convAll, pConv });
    uint32_t sep = sepAll, conv = convAll;
    if (z && id.cVmZ)
    {
        const uint32_t vm = NewId(), sVm = NewId(), cVm = NewId();
        sep = NewId(); conv = NewId();
        b.Emit(spv::Op::OpFOrdGreaterThan, { id.tBool, vm, z, id.cVmZ });
        b.Emit(spv::Op::OpFMul, { id.tFloat, sVm, sepAll, id.cVmSep });
        b.Emit(spv::Op::OpFMul, { id.tFloat, cVm, convAll, id.cVmConv });
        b.Emit(spv::Op::OpSelect, { id.tFloat, sep, vm, sVm, sepAll });
        b.Emit(spv::Op::OpSelect, { id.tFloat, conv, vm, cVm, convAll });
    }
    // w floored at wMin (VKNearClamp): everything nearer gets wMin's parallax; wMin = 0 is a no-op.
    const uint32_t pMin = NewId(), wMin = NewId(), below = NewId(), wc = NewId();
    b.Emit(spv::Op::OpAccessChain, { id.tPtrUniFloat, pMin, id.vParams, id.cInt2 });
    b.Emit(spv::Op::OpLoad, { id.tFloat, wMin, pMin });
    b.Emit(spv::Op::OpFOrdLessThan, { id.tBool, below, w, wMin });
    b.Emit(spv::Op::OpSelect, { id.tFloat, wc, below, wMin, w });
    const uint32_t diff = NewId(), shift = NewId();
    b.Emit(spv::Op::OpFSub, { id.tFloat, diff, wc, conv });
    b.Emit(spv::Op::OpFMul, { id.tFloat, shift, sep, diff });
    uint32_t applied = shift;
    if (Vk().VKSkipUnitW)
    {
        // |w - 1| < VKUnitWEpsilon, not w == 1: full-screen triangles can arrive with w a few ulps off 1.
        const uint32_t d = NewId(), lt = NewId(), gt = NewId(), near_ = NewId(), sel = NewId();
        b.Emit(spv::Op::OpFSub, { id.tFloat, d, w, id.cFloat1 });
        b.Emit(spv::Op::OpFOrdLessThan, { id.tBool, lt, d, id.cEps });
        b.Emit(spv::Op::OpFOrdGreaterThan, { id.tBool, gt, d, id.cNegEps });
        b.Emit(spv::Op::OpLogicalAnd, { id.tBool, near_, lt, gt });
        b.Emit(spv::Op::OpSelect, { id.tFloat, sel, near_, id.cFloat0, shift });
        applied = sel;
    }
    return applied;
}

// Position.x += sep * (w - conv), guarded for w == 1 when configured. xComp / wComp: the components holding x and w;
// wVar: the output holding w when it is not the one shifted.
void EmitShift(Builder& b, const Ids& id, uint32_t& next, uint32_t wComp = 3, uint32_t xComp = 0, uint32_t wVar = 0)
{
    auto NewId = [&]() { return next++; };
    uint32_t pPos;
    if (id.posVar) { pPos = NewId(); b.Emit(spv::Op::OpAccessChain, { id.tPtrOutVec4, pPos, id.posVar, id.cMember }); }
    else pPos = id.posDirect;
    const uint32_t pos = NewId(), x = NewId(), w = NewId();
    b.Emit(spv::Op::OpLoad, { id.tVec4, pos, pPos });
    b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, x, pos, xComp });
    uint32_t wSrc = pos;
    if (wVar) { wSrc = NewId(); b.Emit(spv::Op::OpLoad, { id.tVec4, wSrc, wVar }); }
    b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, w, wSrc, wComp });
    uint32_t z = 0;
    if (id.cVmZ && xComp == 0 && wComp == 3 && !wVar) { z = NewId(); b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, z, pos, 2 }); }
    const uint32_t applied = EmitShiftAmount(b, id, w, next, z);
    const uint32_t nx = NewId(), npos = NewId();
    b.Emit(spv::Op::OpFAdd, { id.tFloat, nx, x, applied });
    b.Emit(spv::Op::OpCompositeInsert, { id.tVec4, npos, nx, pos, xComp });
    b.Emit(spv::Op::OpStore, { pPos, npos });
}

// A projective screen varying (k*(x + w), ..., w): its x gains k times Position's shift.
void EmitProjShift(Builder& b, const Ids& id, uint32_t var, uint32_t kConst, uint32_t& next)
{
    auto NewId = [&]() { return next++; };
    uint32_t pPos;
    if (id.posVar) { pPos = NewId(); b.Emit(spv::Op::OpAccessChain, { id.tPtrOutVec4, pPos, id.posVar, id.cMember }); }
    else pPos = id.posDirect;
    const uint32_t pos = NewId(), w = NewId();
    b.Emit(spv::Op::OpLoad, { id.tVec4, pos, pPos });
    b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, w, pos, 3 });
    const uint32_t applied = EmitShiftAmount(b, id, w, next);
    const uint32_t s = NewId(), v = NewId(), x = NewId(), nx = NewId(), nv = NewId();
    b.Emit(spv::Op::OpFMul, { id.tFloat, s, applied, kConst });
    b.Emit(spv::Op::OpLoad, { id.tVec4, v, var });
    b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, x, v, 0 });
    b.Emit(spv::Op::OpFAdd, { id.tFloat, nx, x, s });
    b.Emit(spv::Op::OpCompositeInsert, { id.tVec4, nv, nx, v, 0 });
    b.Emit(spv::Op::OpStore, { var, nv });
}

// A screen-position varying (Position.xy / w): its x gains Position's shift over Position's w, nothing at w == 0.
// timesW: the shader stored it again times Position.w (a translator's noperspective emulation), so it gains the shift.
// scale: a screen uv fma(x / w, k, c) gains k times it (the constant k's id; 0 for 1).
void EmitNdcShift(Builder& b, const Ids& id, uint32_t var, uint32_t& next, bool timesW = false, uint32_t scale = 0)
{
    auto NewId = [&]() { return next++; };
    uint32_t pPos;
    if (id.posVar) { pPos = NewId(); b.Emit(spv::Op::OpAccessChain, { id.tPtrOutVec4, pPos, id.posVar, id.cMember }); }
    else pPos = id.posDirect;
    const uint32_t pos = NewId(), w = NewId();
    b.Emit(spv::Op::OpLoad, { id.tVec4, pos, pPos });
    b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, w, pos, 3 });
    const uint32_t applied = EmitShiftAmount(b, id, w, next);
    uint32_t dx = applied;
    if (!timesW)
    {
        const uint32_t q = NewId(), nz = NewId();
        dx = NewId();
        b.Emit(spv::Op::OpFDiv, { id.tFloat, q, applied, w });
        b.Emit(spv::Op::OpFOrdNotEqual, { id.tBool, nz, w, id.cFloat0 });
        b.Emit(spv::Op::OpSelect, { id.tFloat, dx, nz, q, id.cFloat0 });
    }
    if (scale) { const uint32_t s = NewId(); b.Emit(spv::Op::OpFMul, { id.tFloat, s, dx, scale }); dx = s; }
    const uint32_t v = NewId(), x = NewId(), nx = NewId(), nv = NewId();
    b.Emit(spv::Op::OpLoad, { id.tVec4, v, var });
    b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, x, v, 0 });
    b.Emit(spv::Op::OpFAdd, { id.tFloat, nx, x, dx });
    b.Emit(spv::Op::OpCompositeInsert, { id.tVec4, nv, nx, v, 0 });
    b.Emit(spv::Op::OpStore, { var, nv });
}

// A sprite drawn at w == 1 with z its depth moves by sep*(1 - conv/w), 1/w = n32*z + n33 from the camera-window
// shaders' stored inverse (set 8 binding 1 [6], [7]); screenVar (u, v, depth, ...) gives z, and its u moves too.
bool EmitSpriteShift(Builder& b, const Ids& id, Builder& types, Builder& decos, uint32_t& next, uint32_t& countVar, uint32_t screenVar)
{
    if (!id.tUint) return false;
    const uint32_t c8 = next++, tArr = next++, tC = next++, tPtrC = next++, tPtrCU = next++;
    countVar = next++;
    types.Emit(spv::Op::OpConstant, { id.tUint, c8, 16 });
    types.Emit(spv::Op::OpTypeArray, { tArr, id.tUint, c8 });
    types.Emit(spv::Op::OpTypeStruct, { tC, tArr });
    types.Emit(spv::Op::OpTypePointer, { tPtrC, (uint32_t)spv::StorageClass::StorageBuffer, tC });
    types.Emit(spv::Op::OpTypePointer, { tPtrCU, (uint32_t)spv::StorageClass::StorageBuffer, id.tUint });
    types.Emit(spv::Op::OpVariable, { tPtrC, countVar, (uint32_t)spv::StorageClass::StorageBuffer });
    decos.Emit(spv::Op::OpDecorate, { tArr, (uint32_t)spv::Decoration::ArrayStride, 4 });
    decos.Emit(spv::Op::OpDecorate, { tC, (uint32_t)spv::Decoration::Block });
    decos.Emit(spv::Op::OpMemberDecorate, { tC, 0, (uint32_t)spv::Decoration::Offset, 0 });
    decos.Emit(spv::Op::OpMemberDecorate, { tC, 0, (uint32_t)spv::Decoration::NonWritable });
    decos.Emit(spv::Op::OpDecorate, { countVar, (uint32_t)spv::Decoration::DescriptorSet, kStereoSet });
    decos.Emit(spv::Op::OpDecorate, { countVar, (uint32_t)spv::Decoration::Binding, 1 });
    const uint32_t cU6 = next++, cU7 = next++, cU0 = next++, edge = next++, two = next++;
    types.Emit(spv::Op::OpConstant, { id.tUint, cU6, 6 });
    types.Emit(spv::Op::OpConstant, { id.tUint, cU7, 7 });
    types.Emit(spv::Op::OpConstant, { id.tUint, cU0, 0 });
    types.Emit(spv::Op::OpConstant, { id.tFloat, edge, FloatBits(0.999f) });
    types.Emit(spv::Op::OpConstant, { id.tFloat, two, FloatBits(2.0f) });

    auto F = [&](spv::Op op, uint32_t a, uint32_t c) { const uint32_t r = next++; b.Emit(op, { id.tFloat, r, a, c }); return r; };
    auto Bo = [&](spv::Op op, uint32_t a, uint32_t c) { const uint32_t r = next++; b.Emit(op, { id.tBool, r, a, c }); return r; };
    auto SelF = [&](uint32_t c, uint32_t a, uint32_t e) { const uint32_t r = next++; b.Emit(spv::Op::OpSelect, { id.tFloat, r, c, a, e }); return r; };
    auto Ext = [&](uint32_t v, uint32_t k) { const uint32_t r = next++; b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, r, v, k }); return r; };
    auto Near = [&](uint32_t a, uint32_t c) { const uint32_t d = F(spv::Op::OpFSub, a, c);
        return Bo(spv::Op::OpLogicalAnd, Bo(spv::Op::OpFOrdLessThan, d, id.cEps), Bo(spv::Op::OpFOrdGreaterThan, d, id.cNegEps)); };

    uint32_t pPos;
    if (id.posVar) { pPos = next++; b.Emit(spv::Op::OpAccessChain, { id.tPtrOutVec4, pPos, id.posVar, id.cMember }); }
    else pPos = id.posDirect;
    const uint32_t pos = next++;
    b.Emit(spv::Op::OpLoad, { id.tVec4, pos, pPos });
    uint32_t screen = 0;
    if (screenVar) { screen = next++; b.Emit(spv::Op::OpLoad, { id.tVec4, screen, screenVar }); }
    const uint32_t x = Ext(pos, 0), z = screenVar ? Ext(screen, 2) : Ext(pos, 2), w = Ext(pos, 3);
    const uint32_t pSep = next++, sep = next++, pConv = next++, conv = next++;
    b.Emit(spv::Op::OpAccessChain, { id.tPtrUniFloat, pSep, id.vParams, id.cInt0 });
    b.Emit(spv::Op::OpLoad, { id.tFloat, sep, pSep });
    b.Emit(spv::Op::OpAccessChain, { id.tPtrUniFloat, pConv, id.vParams, id.cInt1 });
    b.Emit(spv::Op::OpLoad, { id.tFloat, conv, pConv });
    // [6], [7]: the camera inverse's row 3.
    uint32_t n[2];
    for (int c = 0; c < 2; ++c)
    {
        const uint32_t p = next++, u = next++;
        b.Emit(spv::Op::OpAccessChain, { tPtrCU, p, countVar, cU0, c ? cU7 : cU6 });
        b.Emit(spv::Op::OpLoad, { id.tUint, u, p });
        n[c] = next++;
        b.Emit(spv::Op::OpBitcast, { id.tFloat, n[c], u });
    }
    const uint32_t invW = F(spv::Op::OpFAdd, F(spv::Op::OpFMul, n[0], z), n[1]);
    // A light past half a unit (invW in (0, 2)), at w == 1, strictly inside the screen (full-screen quads stay put).
    const uint32_t ax = SelF(Bo(spv::Op::OpFOrdLessThan, x, id.cFloat0), F(spv::Op::OpFSub, id.cFloat0, x), x);
    uint32_t ok = Bo(spv::Op::OpFOrdNotEqual, sep, id.cFloat0);
    ok = Bo(spv::Op::OpLogicalAnd, ok, Near(w, id.cFloat1));
    ok = Bo(spv::Op::OpLogicalAnd, ok, Bo(spv::Op::OpFOrdGreaterThan, invW, id.cFloat0));
    ok = Bo(spv::Op::OpLogicalAnd, ok, Bo(spv::Op::OpFOrdLessThan, invW, two));
    ok = Bo(spv::Op::OpLogicalAnd, ok, Bo(spv::Op::OpFOrdLessThan, ax, edge));
    const uint32_t dx0 = F(spv::Op::OpFMul, sep, F(spv::Op::OpFSub, id.cFloat1, F(spv::Op::OpFMul, conv, invW)));
    const uint32_t dx = SelF(ok, dx0, id.cFloat0);
    const uint32_t nx = F(spv::Op::OpFAdd, x, dx), npos = next++;
    b.Emit(spv::Op::OpCompositeInsert, { id.tVec4, npos, nx, pos, 0 });
    b.Emit(spv::Op::OpStore, { pPos, npos });
    if (screenVar)
    {
        const uint32_t half = next++, nscreen = next++;
        types.Emit(spv::Op::OpConstant, { id.tFloat, half, FloatBits(0.5f) });
        const uint32_t nu = F(spv::Op::OpFAdd, Ext(screen, 0), F(spv::Op::OpFMul, dx, half));
        b.Emit(spv::Op::OpCompositeInsert, { id.tVec4, nscreen, nu, screen, 0 });
        b.Emit(spv::Op::OpStore, { screenVar, nscreen });
    }
    return true;
}

} // namespace

bool PatchSpirv(const uint32_t* code, size_t n, const SpirvRules& rules, std::vector<uint32_t>& out, const char** reason,
                spv::ExecutionModel* model, SpirvCounts* counts)
{
    const bool spriteShift = rules.spriteShift, forceVaryings = rules.forceVaryings, screenUv = rules.screenUv;
    const int spriteVarying = rules.spriteVarying;
    *reason = "";
    if (counts) *counts = SpirvCounts();
    Module m;
    if (!m.Parse(code, n)) { *reason = "parse"; return false; }

    // --- Locate the entry point and the sections ---------------------------
    uint32_t entryFn = 0;
    size_t entryInst = SIZE_MAX;
    size_t firstTypeInst = SIZE_MAX, firstFuncInst = SIZE_MAX;
    uint32_t voidType = 0, fnType = 0;
    spv::ExecutionModel em = spv::ExecutionModel::Max;
    for (size_t i = 0; i < m.insts.size(); ++i)
    {
        const Inst& in = m.insts[i];
        if (in.op == spv::Op::OpEntryPoint && !entryFn)
        {
            const auto e = (spv::ExecutionModel)in.w[1];
            if (e == spv::ExecutionModel::Vertex || e == spv::ExecutionModel::TessellationEvaluation || e == spv::ExecutionModel::Geometry ||
                (Vk().VKMeshDraws && e == spv::ExecutionModel::MeshEXT))
            {
                entryFn = in.w[2];
                entryInst = i;
                em = e;
            }
        }
        const bool isTypeSection =
            ((uint32_t)in.op >= (uint32_t)spv::Op::OpTypeVoid && (uint32_t)in.op <= (uint32_t)spv::Op::OpTypeForwardPointer) ||
            ((uint32_t)in.op >= (uint32_t)spv::Op::OpConstantTrue && (uint32_t)in.op <= (uint32_t)spv::Op::OpSpecConstantOp) ||
            in.op == spv::Op::OpVariable || in.op == spv::Op::OpUndef;
        if (isTypeSection && firstTypeInst == SIZE_MAX && firstFuncInst == SIZE_MAX) firstTypeInst = i;
        if (in.op == spv::Op::OpFunction && firstFuncInst == SIZE_MAX) firstFuncInst = i;
    }
    if (model) *model = em;
    if (!entryFn) { *reason = "noPositionStage"; return false; }
    if (firstTypeInst == SIZE_MAX || firstFuncInst == SIZE_MAX) { *reason = "noSections"; return false; }

    // --- Existing types and constants we can reuse -------------------------
    Ids id;
    uint32_t posStruct = 0; int posMember = -1;
    for (const Inst& in : m.insts)
    {
        switch (in.op)
        {
        case spv::Op::OpTypeFloat:  if (in.w[2] == 32 && !id.tFloat) id.tFloat = in.w[1]; break;
        case spv::Op::OpTypeInt:    if (in.w[2] == 32) { if (in.w[3] == 1 && !id.tInt) id.tInt = in.w[1]; if (in.w[3] == 0 && !id.tUint) id.tUint = in.w[1]; } break;
        case spv::Op::OpTypeBool:   if (!id.tBool) id.tBool = in.w[1]; break;
        case spv::Op::OpMemberDecorate:
            if (in.w[3] == (uint32_t)spv::Decoration::BuiltIn && in.w[4] == (uint32_t)spv::BuiltIn::Position)
            { posStruct = in.w[1]; posMember = (int)in.w[2]; }
            break;
        case spv::Op::OpDecorate:
            if (in.w[2] == (uint32_t)spv::Decoration::BuiltIn && in.w[3] == (uint32_t)spv::BuiltIn::Position)
                id.posDirect = in.w[1];
            break;
        case spv::Op::OpFunction:
            if (in.w[2] == entryFn) { voidType = in.w[1]; fnType = in.w[4]; }
            break;
        default: break;
        }
    }
    if (!id.tFloat) { *reason = "noFloat"; return false; }
    if (!voidType || !fnType) { *reason = "noEntryFunction"; return false; }

    for (const Inst& in : m.insts)
    {
        switch (in.op)
        {
        case spv::Op::OpTypeVector:  if (in.w[2] == id.tFloat && in.w[3] == 4 && !id.tVec4) id.tVec4 = in.w[1]; break;
        case spv::Op::OpTypePointer:
            if (in.w[2] == (uint32_t)spv::StorageClass::Uniform && in.w[3] == id.tFloat && !id.tPtrUniFloat) id.tPtrUniFloat = in.w[1];
            break;
        case spv::Op::OpConstant:
            if (in.w[1] == id.tFloat) { if (in.w[3] == 0 && !id.cFloat0) id.cFloat0 = in.w[2]; if (in.w[3] == FloatBits(1.f) && !id.cFloat1) id.cFloat1 = in.w[2]; }
            if (in.w[1] == id.tInt)   { if (in.w[3] == 0 && !id.cInt0) id.cInt0 = in.w[2]; if (in.w[3] == 1 && !id.cInt1) id.cInt1 = in.w[2]; if (in.w[3] == 2 && !id.cInt2) id.cInt2 = in.w[2]; }
            break;
        default: break;
        }
    }
    if (!id.tVec4) { *reason = "noVec4"; return false; }
    for (const Inst& in : m.insts)
        if (in.op == spv::Op::OpTypePointer && in.w[2] == (uint32_t)spv::StorageClass::Output && in.w[3] == id.tVec4) { id.tPtrOutVec4 = in.w[1]; break; }

    // The Position carrier: a gl_PerVertex block member (glslang) or a direct BuiltIn variable (DXC); only the Output
    // class counts, as a geometry shader's gl_in is decorated too.
    if (posStruct)
    {
        uint32_t ptrBlock = 0;
        for (const Inst& in : m.insts)
            if (in.op == spv::Op::OpTypePointer && in.w[2] == (uint32_t)spv::StorageClass::Output && in.w[3] == posStruct) { ptrBlock = in.w[1]; break; }
        if (ptrBlock)
            for (const Inst& in : m.insts)
                if (in.op == spv::Op::OpVariable && in.w[1] == ptrBlock && in.w[3] == (uint32_t)spv::StorageClass::Output) { id.posVar = in.w[2]; break; }
    }
    if (id.posDirect)
    {
        bool isOutput = false;
        for (const Inst& in : m.insts)
            if (in.op == spv::Op::OpVariable && in.w[2] == id.posDirect) { isOutput = in.w[3] == (uint32_t)spv::StorageClass::Output; break; }
        if (!isOutput) id.posDirect = 0;
    }
    if (!id.posVar && !id.posDirect) { *reason = "noPosition"; return false; }

    // Varyings carrying the same clip position as gl_Position keep the unshifted value, so an FS reading the screen at
    // them samples the left eye's spot; they are found by register provenance only (VKScreenReadVaryings, forced).
    std::vector<uint32_t> posVaryings;
    // A mesh stage writes Position per vertex through an array: it shifts at those stores, with no varying analysis.
    const bool mesh = em == spv::ExecutionModel::MeshEXT;
    const bool screenRead = Vk().VKScreenReadVaryings || forceVaryings;
    if (screenRead && !mesh)
    {
        std::unordered_map<uint32_t, uint32_t> regOf;   // result id -> base variable
        auto Src = [&](uint32_t v) -> uint32_t { auto it = regOf.find(v); return it == regOf.end() ? 0u : it->second; };
        uint32_t posValReg = 0, posChain = 0;
        std::vector<std::pair<uint32_t, uint32_t>> stores;   // (output variable, source register)
        for (const Inst& in : m.insts)
        {
            switch (in.op)
            {
            case spv::Op::OpAccessChain:
                // Base of the chain; for a GPR read that is the register.
                if (in.count >= 4) regOf[in.w[2]] = in.w[3];
                if (in.count >= 4 && id.posVar && in.w[3] == id.posVar) posChain = in.w[2];
                break;
            case spv::Op::OpLoad:     if (in.count >= 4) { uint32_t s = Src(in.w[3]); if (s) regOf[in.w[2]] = s; } break;
            case spv::Op::OpBitcast:  if (in.count >= 4) { uint32_t s = Src(in.w[3]); if (s) regOf[in.w[2]] = s; } break;
            case spv::Op::OpCompositeConstruct:
                if (in.count >= 4)
                    for (uint16_t k = 3; k < in.count; ++k)
                    { uint32_t s = Src(in.w[k]); if (s) { regOf[in.w[2]] = s; break; } }
                break;
            case spv::Op::OpStore:
                if (in.count >= 3)
                {
                    const uint32_t tgt = in.w[1], val = in.w[2];
                    if ((posChain && tgt == posChain) || (id.posDirect && tgt == id.posDirect)) posValReg = Src(val);
                    else stores.push_back({ tgt, Src(val) });
                }
                break;
            default: break;
            }
        }
        if (posValReg)
            for (const auto& st : stores)
            {
                if (st.second != posValReg || st.first == id.posDirect) continue;
                // Only a vec4 Output variable: the shift reads .w and writes .x.
                for (const Inst& v : m.insts)
                    if (v.op == spv::Op::OpVariable && v.w[2] == st.first &&
                        v.w[3] == (uint32_t)spv::StorageClass::Output && v.w[1] == id.tPtrOutVec4)
                    {
                        if (std::find(posVaryings.begin(), posVaryings.end(), st.first) == posVaryings.end())
                            posVaryings.push_back(st.first);
                        break;
                    }
            }
        // Outputs stored one component at a time, AccessChain(var, c) + Store: a varying copying the clip position
        // stores the very ids gl_Position gets.
        if (posVaryings.empty() && screenRead)
        {
            std::unordered_map<uint32_t, uint32_t> konst;                    // integer constant -> value
            std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> elem; // chain -> (variable, component)
            std::unordered_map<uint32_t, uint32_t> outVec4;                  // Output vec4 variable -> pointer type
            std::unordered_map<uint32_t, std::array<uint32_t, 4>> comp;      // variable -> value per component
            std::unordered_set<uint32_t> outVec4Ptr;
            for (const Inst& in : m.insts)
            {
                switch (in.op)
                {
                case spv::Op::OpConstant:
                    if (in.count == 4 && (in.w[1] == id.tInt || in.w[1] == id.tUint)) konst[in.w[2]] = in.w[3];
                    break;
                case spv::Op::OpTypePointer:
                    if (in.w[2] == (uint32_t)spv::StorageClass::Output && in.w[3] == id.tVec4) outVec4Ptr.insert(in.w[1]);
                    break;
                case spv::Op::OpVariable:
                    if (in.w[3] == (uint32_t)spv::StorageClass::Output && outVec4Ptr.count(in.w[1])) outVec4[in.w[2]] = in.w[1];
                    break;
                case spv::Op::OpAccessChain: case spv::Op::OpInBoundsAccessChain:
                {
                    auto last = in.count >= 5 ? konst.find(in.w[in.count - 1]) : konst.end();
                    if (last == konst.end()) break;
                    if (in.count == 5) elem[in.w[2]] = { in.w[3], last->second };
                    else if (in.count == 6 && id.posVar && in.w[3] == id.posVar)
                    {
                        auto mem = konst.find(in.w[4]);
                        if (mem != konst.end() && (int)mem->second == posMember) elem[in.w[2]] = { id.posVar, last->second };
                    }
                    break;
                }
                case spv::Op::OpStore:
                {
                    auto el = elem.find(in.w[1]);
                    if (el == elem.end() || el->second.second > 3) break;
                    auto& a = comp.emplace(el->second.first, std::array<uint32_t, 4>{ 0, 0, 0, 0 }).first->second;
                    uint32_t& c = a[el->second.second];
                    c = c ? ~0u : in.w[2];
                    break;
                }
                default: break;
                }
            }
            auto known = [](uint32_t v) { return v && v != ~0u; };
            auto match = [&](const std::unordered_map<uint32_t, std::array<uint32_t, 4>>& cm)
            {
                auto pit = cm.find(id.posDirect ? id.posDirect : id.posVar);
                if (pit == cm.end()) return;
                const auto& p = pit->second;
                if (!known(p[0]) || !known(p[1]) || !known(p[3])) return;
                for (const auto& c : cm)
                    if (c.first != pit->first && outVec4.count(c.first) && c.second[0] == p[0] && c.second[1] == p[1] && c.second[3] == p[3] &&
                        std::find(posVaryings.begin(), posVaryings.end(), c.first) == posVaryings.end())
                        posVaryings.push_back(c.first);
            };
            match(comp);
            // VKComputeRemap 4: gl_Position stored in several blocks ((0, 0, z) on each culling path) is matched by its
            // last store in program order, the main path's.
            if (posVaryings.empty() && Vk().VKComputeRemap >= 4)
            {
                std::unordered_map<uint32_t, std::array<uint32_t, 4>> last;
                for (const Inst& in : m.insts)
                {
                    if (in.op != spv::Op::OpStore) continue;
                    auto el = elem.find(in.w[1]);
                    if (el == elem.end() || el->second.second > 3) continue;
                    last.emplace(el->second.first, std::array<uint32_t, 4>{ 0, 0, 0, 0 }).first->second[el->second.second] = in.w[2];
                }
                match(last);
            }
        }
    }

    const unsigned clipVaryings = (unsigned)posVaryings.size();
    if (!forceVaryings) posVaryings.clear();

    // Last frame's clip position: x and y jittered as J*W where gl_Position takes J*w, with W stored in an output too
    // (in the same output, or across components of two).
    struct PrevClip { uint32_t var, xComp, wVar, wComp; };
    std::vector<PrevClip> prevClipVaryings;
    if (Vk().VKShiftPrevClipVaryings && em == spv::ExecutionModel::Vertex)
    {
        uint32_t glsl = 0;
        std::unordered_map<uint32_t, size_t> def;     // result id -> instruction
        std::unordered_map<uint32_t, uint32_t> konst;  // integer constant -> value
        for (size_t i = 0; i < m.insts.size(); ++i)
        {
            const Inst& in = m.insts[i];
            switch (in.op)
            {
            case spv::Op::OpExtInstImport:
                if (in.count >= 3 && !strcmp((const char*)&in.w[2], "GLSL.std.450")) glsl = in.w[1];
                break;
            case spv::Op::OpConstant:
                if (in.count == 4 && (in.w[1] == id.tInt || in.w[1] == id.tUint)) konst[in.w[2]] = in.w[3];
                break;
            case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpSelect: case spv::Op::OpExtInst:
            case spv::Op::OpCompositeExtract: case spv::Op::OpCompositeConstruct: case spv::Op::OpAccessChain:
            case spv::Op::OpFAdd: case spv::Op::OpFMul:
                if (in.count >= 3) def[in.w[2]] = i;
                break;
            default: break;
            }
        }
        auto Def = [&](uint32_t v) -> const Inst* { auto it = def.find(v); return it == def.end() ? nullptr : &m.insts[it->second]; };
        // Through translator wrappers: bitcasts, Select(exec, value, previous), extracts of a construct.
        auto Strip = [&](uint32_t v) {
            for (int k = 0; k < 32; ++k)
            {
                const Inst* in = Def(v);
                if (!in) break;
                if ((in->op == spv::Op::OpBitcast || in->op == spv::Op::OpCopyObject) && in->count == 4) v = in->w[3];
                else if (in->op == spv::Op::OpSelect && in->count == 6) v = in->w[4];
                else if (in->op == spv::Op::OpCompositeExtract && in->count == 5)
                {
                    const Inst* c = Def(in->w[3]);
                    if (!c || c->op != spv::Op::OpCompositeConstruct || 3u + in->w[4] >= c->count) break;
                    v = c->w[3 + in->w[4]];
                }
                else break;
            }
            return v;
        };
        auto Comp = [&](uint32_t vec, uint32_t k) -> uint32_t {
            const Inst* c = Def(vec);
            return (c && c->op == spv::Op::OpCompositeConstruct && c->count == 7) ? Strip(c->w[3 + k]) : 0u;
        };
        // Fma(a, b, c), or FAdd(FMul(a, b), c) either way round, with one factor wv: the other factor, else 0.
        auto FmaJ = [&](uint32_t v, uint32_t wv) -> uint32_t {
            const Inst* f = Def(v);
            if (f && glsl && f->op == spv::Op::OpExtInst && f->count == 8 && f->w[3] == glsl && f->w[4] == 50 /* Fma */)
            {
                const uint32_t a = Strip(f->w[5]), c = Strip(f->w[6]);
                return a == wv ? c : c == wv ? a : 0u;
            }
            for (int o = 0; f && f->op == spv::Op::OpFAdd && f->count == 5 && o < 2; ++o)
            {
                const Inst* mul = Def(Strip(f->w[3 + o]));
                if (!mul || mul->op != spv::Op::OpFMul || mul->count != 5) continue;
                const uint32_t a = Strip(mul->w[3]), c = Strip(mul->w[4]);
                if (a == wv) return c;
                if (c == wv) return a;
            }
            return 0u;
        };
        std::unordered_set<uint32_t> outVec4Vars;
        for (const Inst& in : m.insts)
            if (in.op == spv::Op::OpVariable && in.count >= 4 && in.w[1] == id.tPtrOutVec4 && in.w[3] == (uint32_t)spv::StorageClass::Output)
                outVec4Vars.insert(in.w[2]);
        // Each component's stored value: whole-vector stores, or AccessChain(var, c) + Store one component at a time.
        std::array<uint32_t, 4> posVals = { 0, 0, 0, 0 };
        std::vector<std::pair<uint32_t, std::array<uint32_t, 4>>> outVals;   // (Output vec4, value per component)
        auto valsOf = [&](uint32_t var) -> std::array<uint32_t, 4>& {
            for (auto& ov : outVals) if (ov.first == var) return ov.second;
            outVals.push_back({ var, { 0, 0, 0, 0 } });
            return outVals.back().second;
        };
        for (const Inst& in : m.insts)
        {
            if (in.op != spv::Op::OpStore || in.count < 3) continue;
            const uint32_t tgt = in.w[1];
            if (id.posDirect && tgt == id.posDirect) { for (uint32_t k = 0; k < 4; ++k) posVals[k] = Comp(in.w[2], k); continue; }
            const Inst* ch = Def(tgt);
            if (id.posVar && ch && ch->op == spv::Op::OpAccessChain && ch->w[3] == id.posVar && (ch->count == 5 || ch->count == 6))
            {
                auto mem = konst.find(ch->w[4]);
                if (mem == konst.end() || (int)mem->second != posMember) continue;
                if (ch->count == 5) for (uint32_t k = 0; k < 4; ++k) posVals[k] = Comp(in.w[2], k);
                else { auto c = konst.find(ch->w[5]); if (c != konst.end() && c->second < 4) posVals[c->second] = Strip(in.w[2]); }
                continue;
            }
            if (outVec4Vars.count(tgt))
            {
                std::array<uint32_t, 4> a;
                for (uint32_t k = 0; k < 4; ++k) a[k] = Comp(in.w[2], k);
                outVals.push_back({ tgt, a });   // one entry per store, as before
                continue;
            }
            if (ch && ch->op == spv::Op::OpAccessChain && ch->count == 5 && (outVec4Vars.count(ch->w[3]) || ch->w[3] == id.posDirect))
            {
                auto c = konst.find(ch->w[4]);
                if (c == konst.end() || c->second >= 4) continue;
                if (ch->w[3] == id.posDirect) posVals[c->second] = Strip(in.w[2]);
                else valsOf(ch->w[3])[c->second] = Strip(in.w[2]);
            }
        }
        const uint32_t pw = posVals[3];
        const uint32_t jx = pw ? FmaJ(posVals[0], pw) : 0u, jy = pw ? FmaJ(posVals[1], pw) : 0u;
        if (jx && jy)
            for (const auto& st : outVals)
                for (uint32_t kx = 0; kx < 4; ++kx)
                {
                    const uint32_t cx = st.second[kx];
                    const uint32_t wv = cx ? FmaJ(cx, jx) : 0u;   // x = J*W + ...: W
                    if (!wv || wv == pw) continue;                // W = gl_Position.w is the current position
                    bool yToo = false;
                    for (uint32_t ky = 0; ky < 4 && !yToo; ++ky)
                        if (ky != kx) { const uint32_t cy = st.second[ky]; yToo = cy && FmaJ(cy, jy) == wv; }
                    if (!yToo) continue;
                    PrevClip pc = { st.first, kx, 0, 0 };
                    bool stored = false;
                    for (const auto& sw : outVals)
                    {
                        for (uint32_t kw = 0; kw < 4 && !stored; ++kw)
                            if (sw.second[kw] == wv) { pc.wVar = sw.first; pc.wComp = kw; stored = true; }
                        if (stored) break;
                    }
                    bool dup = false;
                    for (const PrevClip& e : prevClipVaryings) dup |= e.var == pc.var && e.xComp == pc.xComp;
                    if (stored && !dup) prevClipVaryings.push_back(pc);
                }
    }

    // Screen-position varyings: x = Position.x / w and y = Position.y / w, stored one component at a time, which a
    // light-volume FS reads depth and the G-buffer at.
    std::vector<uint32_t> ndcVaryings;
    std::unordered_set<uint32_t> ndcTimesW;   // ndc varyings stored again as v * Position.w (noperspective emulation)
    std::unordered_map<uint32_t, uint32_t> ndcScale;   // a screen uv fma(x / w, k, c) beside them -> k's constant id
    if (Vk().VKShiftNdcVaryings && em != spv::ExecutionModel::Geometry && !mesh)
    {
        std::unordered_map<uint32_t, uint32_t> konst;                    // integer constant -> value
        std::unordered_set<uint32_t> one;                                // float constants equal to 1
        std::unordered_map<uint32_t, const Inst*> quot;                  // FMul / FDiv result -> instruction
        std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> elem; // chain -> (variable, component)
        std::unordered_set<uint32_t> outVec4Ptr, outVec4;
        std::unordered_map<uint32_t, std::array<uint32_t, 4>> comp;      // variable -> value per component, ~0: twice
        std::unordered_map<uint32_t, uint32_t> loadOf, chainBase, wholeStore;   // load->ptr, chain->base, var->value
        std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> vecTimes, extractOf, fmaOf;   // fma a*b+c -> a, b
        std::unordered_set<uint32_t> fconst;
        uint32_t glsl = 0;
        for (const Inst& in : m.insts)
        {
            switch (in.op)
            {
            case spv::Op::OpExtInstImport:
                if (in.count >= 3 && !strcmp((const char*)&in.w[2], "GLSL.std.450")) glsl = in.w[1];
                break;
            case spv::Op::OpExtInst:
                if (glsl && in.count == 8 && in.w[3] == glsl && in.w[4] == 50 /* Fma */) fmaOf[in.w[2]] = { in.w[5], in.w[6] };
                break;
            case spv::Op::OpLoad: if (in.count >= 4) loadOf[in.w[2]] = in.w[3]; break;
            case spv::Op::OpVectorTimesScalar: if (in.count == 5) vecTimes[in.w[2]] = { in.w[3], in.w[4] }; break;
            case spv::Op::OpCompositeExtract: if (in.count == 5) extractOf[in.w[2]] = { in.w[3], in.w[4] }; break;
            case spv::Op::OpConstant:
                if (in.count == 4 && (in.w[1] == id.tInt || in.w[1] == id.tUint)) konst[in.w[2]] = in.w[3];
                if (in.count == 4 && in.w[1] == id.tFloat && in.w[3] == FloatBits(1.f)) one.insert(in.w[2]);
                if (in.count == 4 && in.w[1] == id.tFloat) fconst.insert(in.w[2]);
                break;
            case spv::Op::OpTypePointer:
                if (in.w[2] == (uint32_t)spv::StorageClass::Output && in.w[3] == id.tVec4) outVec4Ptr.insert(in.w[1]);
                break;
            case spv::Op::OpVariable:
                if (in.w[3] == (uint32_t)spv::StorageClass::Output && outVec4Ptr.count(in.w[1])) outVec4.insert(in.w[2]);
                break;
            case spv::Op::OpFMul: case spv::Op::OpFDiv:
                if (in.count == 5) quot[in.w[2]] = &in;
                break;
            case spv::Op::OpAccessChain: case spv::Op::OpInBoundsAccessChain:
            {
                if (in.count >= 4) chainBase[in.w[2]] = in.w[3];
                auto last = in.count >= 5 ? konst.find(in.w[in.count - 1]) : konst.end();
                if (last == konst.end()) break;
                if (in.count == 5) elem[in.w[2]] = { in.w[3], last->second };
                else if (in.count == 6 && id.posVar && in.w[3] == id.posVar)
                {
                    auto mem = konst.find(in.w[4]);
                    if (mem != konst.end() && (int)mem->second == posMember) elem[in.w[2]] = { id.posVar, last->second };
                }
                break;
            }
            case spv::Op::OpStore:
            {
                if (outVec4.count(in.w[1])) wholeStore[in.w[1]] = in.w[2];
                auto el = elem.find(in.w[1]);
                if (el == elem.end() || el->second.second > 3) break;
                auto& a = comp.emplace(el->second.first, std::array<uint32_t, 4>{ 0, 0, 0, 0 }).first->second;
                uint32_t& c = a[el->second.second];
                c = c ? ~0u : in.w[2];
                break;
            }
            default: break;
            }
        }
        // v == x / w, as FDiv(x, w) or FMul(x, FDiv(1, w)) either way round.
        auto OverW = [&](uint32_t v, uint32_t x, uint32_t w) {
            auto it = quot.find(v);
            if (it == quot.end()) return false;
            const Inst* q = it->second;
            if (q->op == spv::Op::OpFDiv) return q->w[3] == x && q->w[4] == w;
            const uint32_t r = q->w[3] == x ? q->w[4] : q->w[4] == x ? q->w[3] : 0u;
            auto rd = r ? quot.find(r) : quot.end();
            return rd != quot.end() && rd->second->op == spv::Op::OpFDiv && one.count(rd->second->w[3]) && rd->second->w[4] == w;
        };
        auto known = [](uint32_t v) { return v && v != ~0u; };
        auto pit = comp.find(id.posDirect ? id.posDirect : id.posVar);
        if (pit != comp.end() && known(pit->second[0]) && known(pit->second[1]) && known(pit->second[3]))
        {
            const auto& p = pit->second;
            for (const auto& c : comp)
                if (c.first != pit->first && outVec4.count(c.first) && known(c.second[0]) && known(c.second[1]) &&
                    OverW(c.second[0], p[0], p[3]) && OverW(c.second[1], p[1], p[3]))
                    ndcVaryings.push_back(c.first);
            // Beside them, a screen uv fma(x / w, k, c) (the screen-uv rule stands aside here): x gains k times
            // the shift. k is the constant id, or 0 when v is not that form.
            auto scaled = [&](uint32_t v, uint32_t x) -> uint32_t {
                auto f = fmaOf.find(v);
                if (f == fmaOf.end()) return 0;
                if (OverW(f->second.first, x, p[3]) && fconst.count(f->second.second)) return f->second.second;
                if (OverW(f->second.second, x, p[3]) && fconst.count(f->second.first)) return f->second.first;
                return 0;
            };
            if (!ndcVaryings.empty())
                for (const auto& c : comp)
                    if (c.first != pit->first && outVec4.count(c.first) && known(c.second[0]) && known(c.second[1]) &&
                        scaled(c.second[0], p[0]) && scaled(c.second[1], p[1]))
                    {
                        ndcVaryings.push_back(c.first);
                        ndcScale[c.first] = scaled(c.second[0], p[0]);
                    }
        }
        // Stored again as Load(v) * Position.w: the varying then holds clip x, not x / w.
        for (uint32_t v : ndcVaryings)
        {
            auto st = wholeStore.find(v);
            auto vt = st != wholeStore.end() ? vecTimes.find(st->second) : vecTimes.end();
            if (vt == vecTimes.end()) continue;
            auto lv = loadOf.find(vt->second.first);
            auto ex = extractOf.find(vt->second.second);
            if (lv == loadOf.end() || lv->second != v || ex == extractOf.end() || ex->second.second != 3) continue;
            auto lp = loadOf.find(ex->second.first);
            if (lp == loadOf.end()) continue;
            auto cb = chainBase.find(lp->second);
            if (lp->second == id.posDirect || (id.posVar && cb != chainBase.end() && cb->second == id.posVar)) ndcTimesW.insert(v);
        }
    }

    // Projective screen varyings: v.x = k*(x + w), v.w = w of Position's own x and w, e.g. ((x+w)/2, (w-y)/2, z, w),
    // which an FS reads the G-buffer at as v.xy/v.w.
    std::vector<std::pair<uint32_t, float>> projVaryings;   // (output variable, k)
    if (Vk().VKShiftNdcVaryings && em == spv::ExecutionModel::Vertex)
    {
        std::unordered_map<uint32_t, size_t> def;
        std::unordered_map<uint32_t, uint32_t> varClass, konst, fconst;
        std::unordered_set<uint32_t> typeIds;   // a value instruction's word 1 is one of these
        for (size_t i = 0; i < m.insts.size(); ++i)
        {
            const Inst& in = m.insts[i];
            if (in.op >= spv::Op::OpTypeVoid && in.op <= spv::Op::OpTypeForwardPointer && in.count >= 2) { typeIds.insert(in.w[1]); def[in.w[1]] = i; continue; }
            if (in.count >= 3 && typeIds.count(in.w[1])) def[in.w[2]] = i;
            if (in.op == spv::Op::OpVariable && in.count >= 4) varClass[in.w[2]] = in.w[3];
            if (in.op == spv::Op::OpConstant && in.count == 4) { if (in.w[1] == id.tFloat) fconst[in.w[2]] = in.w[3]; else konst[in.w[2]] = in.w[3]; }
        }
        auto Def = [&](uint32_t v) -> const Inst* { auto it = def.find(v); return it == def.end() ? nullptr : &m.insts[it->second]; };
        // A Function-storage register a pointer addresses: component, ~0u the whole register, ~1u a dynamic index.
        auto slotOf = [&](uint32_t ptr, uint32_t& var, uint32_t& c) -> bool {
            var = ptr; c = ~0u;
            const Inst* p = Def(ptr);
            if (p && (p->op == spv::Op::OpAccessChain || p->op == spv::Op::OpInBoundsAccessChain))
            {
                var = p->w[3];
                auto k = p->count == 5 ? konst.find(p->w[4]) : konst.end();
                c = k != konst.end() ? k->second : ~1u;
            }
            auto vc = varClass.find(var);
            return vc != varClass.end() && vc->second == (uint32_t)spv::StorageClass::Function;
        };
        struct St { size_t k; uint32_t comp, value; };
        std::unordered_map<uint32_t, std::vector<St>> regStores;
        for (size_t i = 0; i < m.insts.size(); ++i)
        {
            const Inst& in = m.insts[i];
            uint32_t var, c;
            if (in.op == spv::Op::OpStore && in.count >= 3 && slotOf(in.w[1], var, c)) regStores[var].push_back({ i, c, in.w[2] });
        }
        // Component c (~0u: the value) of `v` as the instruction computing it, through bitcasts, constructs, registers.
        std::function<uint32_t(uint32_t, uint32_t, int)> leaf = [&](uint32_t v, uint32_t c, int depth) -> uint32_t {
            const Inst* in = Def(v);
            if (!in || depth > 40) return 0;
            if ((in->op == spv::Op::OpBitcast || in->op == spv::Op::OpCopyObject) && in->count == 4) return leaf(in->w[3], c, depth + 1);
            if (in->op == spv::Op::OpCompositeExtract && in->count == 5 && c == ~0u) return leaf(in->w[3], in->w[4], depth + 1);
            if (in->op == spv::Op::OpCompositeConstruct && c != ~0u) return 3u + c < in->count ? leaf(in->w[3 + c], ~0u, depth + 1) : 0;
            if (in->op == spv::Op::OpVectorShuffle && c != ~0u)
            {
                const Inst* a = Def(in->w[3]);
                const Inst* at = a ? Def(a->w[1]) : nullptr;
                const uint32_t n1 = at && at->op == spv::Op::OpTypeVector ? at->w[3] : 0;
                if (!n1 || 5u + c >= in->count) return 0;
                const uint32_t s = in->w[5 + c];
                return s < n1 ? leaf(in->w[3], s, depth + 1) : leaf(in->w[4], s - n1, depth + 1);
            }
            if (in->op == spv::Op::OpLoad && in->count >= 4)
            {
                uint32_t var, sc;
                if (!slotOf(in->w[3], var, sc)) return c == ~0u ? v : 0;   // a uniform or input read: itself
                if (sc != ~0u && c != ~0u) return 0;
                const uint32_t want = sc != ~0u ? sc : c;
                auto it = regStores.find(var);
                if (it == regStores.end()) return 0;
                const size_t at = def[v];
                for (auto s = it->second.rbegin(); s != it->second.rend(); ++s)
                {
                    if (s->k >= at) continue;
                    if (s->comp == ~1u) return 0;
                    if (s->comp == want) return leaf(s->value, ~0u, depth + 1);
                    if (s->comp == ~0u) return leaf(s->value, want, depth + 1);
                    if (want == ~0u) return 0;
                }
                return 0;
            }
            return c == ~0u ? v : 0;
        };
        // Each component's last store to gl_Position and every vec4 output, whole vectors or one component at a time:
        // (value, its component, ~0u for a scalar).
        using Src = std::pair<uint32_t, uint32_t>;
        std::array<Src, 4> pos = {};
        std::vector<std::pair<uint32_t, std::array<Src, 4>>> outs;
        auto outOf = [&](uint32_t var) -> std::array<Src, 4>* {
            auto vc = varClass.find(var);
            const Inst* d = Def(var);
            if (vc == varClass.end() || vc->second != (uint32_t)spv::StorageClass::Output || !d || d->w[1] != id.tPtrOutVec4) return nullptr;
            for (auto& o : outs) if (o.first == var) return &o.second;
            outs.push_back({ var, {} });
            return &outs.back().second;
        };
        auto compOf = [&](uint32_t c) -> uint32_t { auto k = konst.find(c); return k != konst.end() && k->second < 4 ? k->second : ~0u; };
        auto posMem = [&](uint32_t c) { auto k = konst.find(c); return k != konst.end() && (int)k->second == posMember; };
        for (const Inst& in : m.insts)
        {
            if (in.op != spv::Op::OpStore || in.count < 3) continue;
            const uint32_t tgt = in.w[1], val = in.w[2];
            const Inst* ch = Def(tgt);
            const bool chain = ch && (ch->op == spv::Op::OpAccessChain || ch->op == spv::Op::OpInBoundsAccessChain);
            if ((id.posDirect && tgt == id.posDirect) || (id.posVar && chain && ch->count == 5 && ch->w[3] == id.posVar && posMem(ch->w[4])))
            { for (uint32_t c = 0; c < 4; ++c) pos[c] = { val, c }; continue; }
            if (chain && ((id.posVar && ch->count == 6 && ch->w[3] == id.posVar && posMem(ch->w[4])) || (id.posDirect && ch->count == 5 && ch->w[3] == id.posDirect)))
            { const uint32_t c = compOf(ch->w[ch->count - 1]); if (c != ~0u) pos[c] = { val, ~0u }; continue; }
            if (auto* o = outOf(tgt)) { for (uint32_t c = 0; c < 4; ++c) (*o)[c] = { val, c }; continue; }
            if (chain && ch->count == 5)
                if (auto* o = outOf(ch->w[3])) { const uint32_t c = compOf(ch->w[4]); if (c != ~0u) (*o)[c] = { val, ~0u }; }
        }
        auto srcLeaf = [&](const Src& s) { return s.first ? leaf(s.first, s.second, 0) : 0u; };
        uint32_t glsl = 0;
        for (const Inst& in : m.insts)
            if (in.op == spv::Op::OpExtInstImport && in.count >= 3 && !strcmp((const char*)&in.w[2], "GLSL.std.450")) glsl = in.w[1];
        // Float constants, and integer ones as their bits: translated registers are ints behind bitcasts.
        auto fval = [&](uint32_t v, float& f) {
            auto it = fconst.find(v);
            if (it == fconst.end()) { it = konst.find(v); if (it == konst.end()) return false; }
            memcpy(&f, &it->second, 4);
            return true;
        };
        // mul_nonIEEE helpers: a float function of two float pointers whose body multiplies their loads.
        std::unordered_set<uint32_t> mulFns;
        for (size_t i = 0; i < m.insts.size(); ++i)
        {
            if (m.insts[i].op != spv::Op::OpFunction || m.insts[i].w[1] != id.tFloat) continue;
            std::vector<uint32_t> par;
            size_t j = i + 1;
            for (; j < m.insts.size() && m.insts[j].op == spv::Op::OpFunctionParameter; ++j) par.push_back(m.insts[j].w[2]);
            for (; par.size() == 2 && j < m.insts.size() && m.insts[j].op != spv::Op::OpFunctionEnd; ++j)
            {
                if (m.insts[j].op != spv::Op::OpFMul) continue;
                const Inst* a = Def(m.insts[j].w[3]); const Inst* b = Def(m.insts[j].w[4]);
                if (a && b && a->op == spv::Op::OpLoad && b->op == spv::Op::OpLoad &&
                    ((a->w[3] == par[0] && b->w[3] == par[1]) || (a->w[3] == par[1] && b->w[3] == par[0]))) { mulFns.insert(m.insts[i].w[2]); break; }
            }
        }
        // A value as sum(coef * leaf) + c through adds, negations and products or quotients by constants (mul_nonIEEE
        // calls too); a*0 + b moves drop the a. False on anything else.
        struct Lin { std::vector<std::pair<uint32_t, float>> t; float c = 0.f; };
        std::function<bool(uint32_t, Lin&, float, int)> lin = [&](uint32_t v, Lin& L, float s, int depth) -> bool {
            const uint32_t lf = leaf(v, ~0u, 0);
            float k;
            if (!lf || depth > 24) return false;
            if (fval(lf, k)) { L.c += s * k; return true; }
            const Inst* e = Def(lf);
            if (!e) return false;
            uint32_t p[2] = { 0, 0 };
            if (e->op == spv::Op::OpFAdd && e->count == 5) return lin(e->w[3], L, s, depth + 1) && lin(e->w[4], L, s, depth + 1);
            if (e->op == spv::Op::OpFSub && e->count == 5) return lin(e->w[3], L, s, depth + 1) && lin(e->w[4], L, -s, depth + 1);
            if (e->op == spv::Op::OpFNegate && e->count == 4) return lin(e->w[3], L, -s, depth + 1);
            if (e->op == spv::Op::OpFDiv && e->count == 5 && fval(leaf(e->w[4], ~0u, 0), k) && k != 0.f) return lin(e->w[3], L, s / k, depth + 1);
            // Fma(a, b, c) with a constant factor (a projection built that way): its product, then c.
            if (glsl && e->op == spv::Op::OpExtInst && e->count == 8 && e->w[3] == glsl && e->w[4] == 50 /* Fma */)
                for (int a = 0; a < 2; ++a)
                    if (fval(leaf(e->w[5 + a], ~0u, 0), k)) return (k == 0.f || lin(e->w[6 - a], L, s * k, depth + 1)) && lin(e->w[7], L, s, depth + 1);
            if (e->op == spv::Op::OpFMul && e->count == 5) { p[0] = e->w[3]; p[1] = e->w[4]; }
            else if (e->op == spv::Op::OpFunctionCall && e->count == 6 && mulFns.count(e->w[3]))
            {
                // The arguments: the last stores to the parameter variables before the call.
                for (int a = 0; a < 2; ++a)
                {
                    auto it = regStores.find(e->w[4 + a]);
                    if (it == regStores.end()) return false;
                    for (auto st = it->second.rbegin(); st != it->second.rend() && !p[a]; ++st) if (st->k < def[lf]) p[a] = st->value;
                    if (!p[a]) return false;
                }
            }
            if (p[0])
                for (int a = 0; a < 2; ++a)
                    if (fval(leaf(p[a], ~0u, 0), k)) return k == 0.f || lin(p[1 - a], L, s * k, depth + 1);
            for (auto& term : L.t) if (term.first == lf) { term.second += s; return true; }
            L.t.push_back({ lf, s });
            return L.t.size() <= 8;
        };
        auto coef = [](const Lin& L, uint32_t leafId) { for (const auto& t : L.t) if (t.first == leafId) return t.second; return 0.f; };
        auto approx = [](float a, float b) { return fabsf(a - b) <= 1e-6f * (std::max)(1.f, fabsf(b)); };
        // Position's x and w as linear forms (w may be a sum); x is unknown when culling moves leave a constant.
        Lin lw, lx;
        const uint32_t px = srcLeaf(pos[0]), pw = srcLeaf(pos[3]);
        const bool hasW = pw && lin(pw, lw, 1.f, 0) && !lw.t.empty();
        const bool hasX = px && lin(px, lx, 1.f, 0) && !lx.t.empty();
        for (const auto& os : outs)
        {
            Lin vw, vx;
            const uint32_t ow = srcLeaf(os.second[3]), ox = srcLeaf(os.second[0]);
            if (!hasW || !ow || !ox || !lin(ow, vw, 1.f, 0) || !lin(ox, vx, 1.f, 0)) continue;
            bool ok = vw.t.size() == lw.t.size() && approx(vw.c, lw.c);   // v.w is Position's w, term for term
            for (const auto& t : lw.t) ok = ok && approx(coef(vw, t.first), t.second);
            if (!ok) continue;
            // v.x - k*w must be k*x, or with x unknown one term of coefficient k.
            const float k = coef(vx, lw.t[0].first) / lw.t[0].second;
            if (k == 0.f || !(fabsf(k) < 1e6f)) continue;
            Lin rest;
            rest.c = vx.c - k * lw.c;
            for (const auto& t : vx.t)
            {
                const float r = t.second - k * coef(lw, t.first);
                if (!approx(r, 0.f)) rest.t.push_back({ t.first, r });
            }
            if (hasX)
            {
                ok = rest.t.size() == lx.t.size() && approx(rest.c, k * lx.c);
                for (const auto& t : lx.t) ok = ok && approx(coef(rest, t.first), k * t.second);
            }
            else ok = rest.t.size() == 1 && approx(rest.t[0].second, k) && approx(rest.c, 0.f);
            if (!ok) continue;
            bool dup = false;
            for (const auto& pv : projVaryings) dup |= pv.first == os.first;
            if (!dup) projVaryings.push_back({ os.first, k });
        }
    }

    // Position's x moved inside main, beyond the wrapper: screenUv: k*(x/w) + c or (k/w)*x + c (an outer multiply is a
    // tiled dither, kept); skyRays: a forced twin with no clip copy, whose pinned triangle takes rays from x - fsSep*w.
    std::unordered_map<size_t, uint32_t> movedX;   // FMul -> word holding x
    std::unordered_map<size_t, uint32_t> movedW;   // FMul -> the w its x is divided by (VKScreenUvVaryings)
    uint32_t posX = 0, posW = 0;
    const bool skyRays = forceVaryings && posVaryings.empty() && em == spv::ExecutionModel::Vertex;
    // skyTwins: a sky VS storing x into its velocity varying moves that store too, or the right velocity carries fsSep.
    const bool skyStores = skyRays && Traits().skyTwins;
    // anyUv: also w stored twice (a floor in its own block) and 1/w of a phi of w's bits: fma(x * (1/w), 0.5, 0.5).
    const bool anyUv = screenUv && Vk().VKScreenUvVaryings && !skyRays;
    if ((screenUv || skyRays) && ndcVaryings.empty() && em == spv::ExecutionModel::Vertex && id.posDirect)
    {
        std::unordered_map<uint32_t, uint32_t> konst, compOf;   // integer constant -> value; chain -> component
        std::unordered_map<uint32_t, const Inst*> arith;        // FMul / FDiv result
        std::unordered_set<uint32_t> one, outChain, stored, summed;
        std::unordered_set<uint32_t> outVars;
        std::unordered_set<uint32_t> posXs, posWs, fmaArg;      // values stored to Position.x / .w; stored Fma factors
        std::unordered_map<uint32_t, const Inst*> moves;        // Bitcast / CopyObject / Phi result
        uint32_t glsl = 0;
        for (const Inst& in : m.insts)
        {
            switch (in.op)
            {
            case spv::Op::OpExtInstImport:
                if (in.count >= 3 && !strcmp((const char*)&in.w[2], "GLSL.std.450")) glsl = in.w[1];
                break;
            case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpPhi:
                if (in.count >= 4) moves[in.w[2]] = &in;
                break;
            case spv::Op::OpConstant:
                if (in.count == 4 && (in.w[1] == id.tInt || in.w[1] == id.tUint)) konst[in.w[2]] = in.w[3];
                if (in.count == 4 && in.w[1] == id.tFloat && in.w[3] == FloatBits(1.f)) one.insert(in.w[2]);
                break;
            case spv::Op::OpVariable:
                if (in.w[3] == (uint32_t)spv::StorageClass::Output && in.w[2] != id.posDirect) outVars.insert(in.w[2]);
                break;
            case spv::Op::OpAccessChain:
                if (in.count == 5 && in.w[3] == id.posDirect) { auto c = konst.find(in.w[4]); if (c != konst.end()) compOf[in.w[2]] = c->second; }
                if (in.count == 5 && outVars.count(in.w[3])) outChain.insert(in.w[2]);
                break;
            case spv::Op::OpFMul: case spv::Op::OpFDiv:
                if (in.count == 5) arith[in.w[2]] = &in;
                break;
            case spv::Op::OpStore:
            {
                auto c = compOf.find(in.w[1]);
                if (c != compOf.end() && c->second == 0) { posX = posX ? ~0u : in.w[2]; posXs.insert(in.w[2]); }
                if (c != compOf.end() && c->second == 3) { posW = posW ? ~0u : in.w[2]; posWs.insert(in.w[2]); }
                if (outChain.count(in.w[1])) stored.insert(in.w[2]);
                break;
            }
            default: break;
            }
        }
        for (const Inst& in : m.insts)
        {
            if (in.op == spv::Op::OpFAdd && in.count == 5 && stored.count(in.w[2])) { summed.insert(in.w[3]); summed.insert(in.w[4]); }
            if (anyUv && glsl && in.op == spv::Op::OpExtInst && in.count == 8 && in.w[3] == glsl && in.w[4] == 50 /* Fma */ && stored.count(in.w[2]))
            { fmaArg.insert(in.w[5]); fmaArg.insert(in.w[6]); }
        }
        std::unordered_set<uint32_t> scaled;   // v of a summed k*v
        for (const auto& a : arith)
            if (a.second->op == spv::Op::OpFMul && summed.count(a.first)) { scaled.insert(a.second->w[3]); scaled.insert(a.second->w[4]); }
        // v reaches a value stored to Position.w through bitcasts and phis (any incoming): phi(phi(.., bits of w), c).
        auto isW = [&](uint32_t v) {
            std::vector<uint32_t> todo{ v };
            std::unordered_set<uint32_t> seen;
            while (!todo.empty() && seen.size() < 32)
            {
                const uint32_t x = todo.back(); todo.pop_back();
                if (!seen.insert(x).second) continue;
                if (posWs.count(x)) return true;
                auto mv = moves.find(x);
                if (mv == moves.end()) continue;
                const Inst* p = mv->second;
                if (p->op != spv::Op::OpPhi) todo.push_back(p->w[3]);
                else for (uint32_t k = 3; k + 1 < p->count; k += 2) todo.push_back(p->w[k]);
            }
            return false;
        };
        // The w of v = 1/w, 0 if v is not one.
        auto invOf = [&](uint32_t v) -> uint32_t {
            auto it = arith.find(v);
            if (it == arith.end() || it->second->op != spv::Op::OpFDiv || !one.count(it->second->w[3])) return 0;
            const uint32_t d = it->second->w[4];
            return d == posW || (anyUv && isW(d)) ? d : 0;
        };
        auto scaledInvOf = [&](uint32_t v) -> uint32_t {
            auto it = arith.find(v);
            if (it == arith.end() || it->second->op != spv::Op::OpFMul) return 0;
            const uint32_t a = invOf(it->second->w[3]);
            return a ? a : invOf(it->second->w[4]);
        };
        const bool single = posX && posX != ~0u && posW && posW != ~0u;
        if (single || (anyUv && !posXs.empty() && !posWs.empty()))
            for (size_t i = 0; i < m.insts.size(); ++i)
            {
                const Inst& in = m.insts[i];
                if (skyStores && in.op == spv::Op::OpStore && in.count >= 3 && in.w[2] == posX && outChain.count(in.w[1])) { movedX[i] = 2; continue; }
                if (in.op != spv::Op::OpFMul || in.count != 5) continue;
                for (int o = 0; o < 2; ++o)
                {
                    if (in.w[3 + o] != posX && !(anyUv && posXs.count(in.w[3 + o]))) continue;
                    const uint32_t other = in.w[4 - o], r = in.w[2];
                    uint32_t wv = 0;
                    if (!skyRays && invOf(other) && scaled.count(r)) wv = invOf(other);
                    else if (!skyRays && scaledInvOf(other) && summed.count(r)) wv = scaledInvOf(other);
                    else if (anyUv && invOf(other) && fmaArg.count(r)) wv = invOf(other);
                    if (skyRays || wv) movedX[i] = 3 + o;
                    if (anyUv && wv) movedW[i] = wv;
                    break;
                }
            }
    }
    const bool skyMoved = skyRays && !movedX.empty();

    // Mesh: chains into one vertex of the BuiltIn Position array (vertex index, component; 4 = the whole vec4); a mesh
    // stage must not read its outputs, so x and w pair by their stored values, same vertex id, same block.
    struct MeshPos { uint32_t vtx, comp; };
    // after: x re-stored after the w store; else the x store itself moves.
    struct MeshPair { uint32_t xPtr, xVal, wVal; bool after; };
    std::unordered_map<uint32_t, MeshPos> meshPos;
    std::unordered_map<size_t, MeshPair> meshPairs;   // instruction index -> its pair
    if (mesh)
    {
        if (!id.posDirect) { *reason = "noMeshPositionArray"; return false; }
        std::unordered_map<uint32_t, uint32_t> konst;
        std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> lastX, lastW;   // vertex id -> (pointer, value)
        for (size_t i = 0; i < m.insts.size(); ++i)
        {
            const Inst& in = m.insts[i];
            if (in.op == spv::Op::OpLabel) { lastX.clear(); lastW.clear(); }
            if (in.op == spv::Op::OpConstant && in.count == 4 && (in.w[1] == id.tInt || in.w[1] == id.tUint)) konst[in.w[2]] = in.w[3];
            if ((in.op == spv::Op::OpAccessChain || in.op == spv::Op::OpInBoundsAccessChain) && in.w[3] == id.posDirect)
            {
                if (in.count == 5) meshPos[in.w[2]] = { in.w[4], 4 };
                else if (in.count == 6) { auto c = konst.find(in.w[5]); if (c != konst.end() && c->second < 4) meshPos[in.w[2]] = { in.w[4], c->second }; }
            }
            if (in.op != spv::Op::OpStore) continue;
            auto p = meshPos.find(in.w[1]);
            if (p == meshPos.end()) continue;
            const uint32_t v = p->second.vtx;
            if (p->second.comp == 4) { meshPairs[i] = { 0, 0, 0, false }; continue; }
            if (p->second.comp == 0)
            {
                auto w = lastW.find(v);
                if (w != lastW.end()) meshPairs[i] = { in.w[1], in.w[2], w->second.second, false };
                else lastX[v] = { in.w[1], in.w[2] };
            }
            if (p->second.comp == 3)
            {
                auto x = lastX.find(v);
                if (x != lastX.end()) { meshPairs[i] = { x->second.first, x->second.second, in.w[2], true }; lastX.erase(x); }
                else lastW[v] = { in.w[1], in.w[2] };
            }
        }
        if (meshPairs.empty()) { *reason = "noMeshPositionPair"; return false; }
    }
    unsigned meshShifted = 0;

    // --- New ids ------------------------------------------------------------
    uint32_t next = m.bound;
    auto NewId = [&]() { return next++; };
    Builder types, decos, func;

    if (!id.tInt)  { id.tInt = NewId();  types.Emit(spv::Op::OpTypeInt, { id.tInt, 32, 1 }); }
    if (!id.tUint) { id.tUint = NewId(); types.Emit(spv::Op::OpTypeInt, { id.tUint, 32, 0 }); }
    if (!id.tBool) { id.tBool = NewId(); types.Emit(spv::Op::OpTypeBool, { id.tBool }); }
    if (!id.tPtrOutVec4)  { id.tPtrOutVec4 = NewId();  types.Emit(spv::Op::OpTypePointer, { id.tPtrOutVec4, (uint32_t)spv::StorageClass::Output, id.tVec4 }); }
    if (!id.tPtrUniFloat) { id.tPtrUniFloat = NewId(); types.Emit(spv::Op::OpTypePointer, { id.tPtrUniFloat, (uint32_t)spv::StorageClass::Uniform, id.tFloat }); }
    if (!id.cFloat0) { id.cFloat0 = NewId(); types.Emit(spv::Op::OpConstant, { id.tFloat, id.cFloat0, 0 }); }
    if (!id.cFloat1) { id.cFloat1 = NewId(); types.Emit(spv::Op::OpConstant, { id.tFloat, id.cFloat1, FloatBits(1.f) }); }
    std::vector<uint32_t> projK(projVaryings.size());
    for (size_t i = 0; i < projVaryings.size(); ++i) { projK[i] = NewId(); types.Emit(spv::Op::OpConstant, { id.tFloat, projK[i], FloatBits(projVaryings[i].second) }); }
    id.cEps = NewId();    types.Emit(spv::Op::OpConstant, { id.tFloat, id.cEps, FloatBits(Vk().VKUnitWEpsilon) });
    id.cNegEps = NewId(); types.Emit(spv::Op::OpConstant, { id.tFloat, id.cNegEps, FloatBits(-Vk().VKUnitWEpsilon) });
    if (Vk().VKViewModelClipZ > 0.f)
    {
        id.cVmZ = NewId();    types.Emit(spv::Op::OpConstant, { id.tFloat, id.cVmZ, FloatBits(Vk().VKViewModelClipZ) });
        id.cVmSep = NewId();  types.Emit(spv::Op::OpConstant, { id.tFloat, id.cVmSep, FloatBits(Vk().VKViewModelSep) });
        id.cVmConv = NewId(); types.Emit(spv::Op::OpConstant, { id.tFloat, id.cVmConv, FloatBits(Vk().VKViewModelConv) });
    }
    if (!id.cInt0) { id.cInt0 = NewId(); types.Emit(spv::Op::OpConstant, { id.tInt, id.cInt0, 0 }); }
    if (!id.cInt1) { id.cInt1 = NewId(); types.Emit(spv::Op::OpConstant, { id.tInt, id.cInt1, 1 }); }
    if (!id.cInt2) { id.cInt2 = NewId(); types.Emit(spv::Op::OpConstant, { id.tInt, id.cInt2, 2 }); }
    if (id.posVar)
    {
        if (posMember == 0) id.cMember = id.cInt0;
        else if (posMember == 1) id.cMember = id.cInt1;
        else { id.cMember = NewId(); types.Emit(spv::Op::OpConstant, { id.tInt, id.cMember, (uint32_t)posMember }); }
    }

    // Uniform block { float sep; float conv; float wMin; } at kStereoSet, binding 0.
    const uint32_t tParams = NewId(), tPtrUniParams = NewId();
    id.vParams = NewId();
    // The sky twin also reads fsSep (member 4): its pinned draw's sep is 0.
    uint32_t cInt4 = 0;
    if (skyMoved)
    {
        cInt4 = NewId();
        types.Emit(spv::Op::OpConstant, { id.tInt, cInt4, 4 });
        types.Emit(spv::Op::OpTypeStruct, { tParams, id.tFloat, id.tFloat, id.tFloat, id.tFloat, id.tFloat });
        decos.Emit(spv::Op::OpMemberDecorate, { tParams, 3, (uint32_t)spv::Decoration::Offset, 12 });
        decos.Emit(spv::Op::OpMemberDecorate, { tParams, 4, (uint32_t)spv::Decoration::Offset, 16 });
    }
    else types.Emit(spv::Op::OpTypeStruct, { tParams, id.tFloat, id.tFloat, id.tFloat });
    types.Emit(spv::Op::OpTypePointer, { tPtrUniParams, (uint32_t)spv::StorageClass::Uniform, tParams });
    types.Emit(spv::Op::OpVariable, { tPtrUniParams, id.vParams, (uint32_t)spv::StorageClass::Uniform });
    decos.Emit(spv::Op::OpDecorate, { tParams, (uint32_t)spv::Decoration::Block });
    decos.Emit(spv::Op::OpMemberDecorate, { tParams, 0, (uint32_t)spv::Decoration::Offset, 0 });
    decos.Emit(spv::Op::OpMemberDecorate, { tParams, 1, (uint32_t)spv::Decoration::Offset, 4 });
    decos.Emit(spv::Op::OpMemberDecorate, { tParams, 2, (uint32_t)spv::Decoration::Offset, 8 });
    decos.Emit(spv::Op::OpDecorate, { id.vParams, (uint32_t)spv::Decoration::DescriptorSet, kStereoSet });
    decos.Emit(spv::Op::OpDecorate, { id.vParams, (uint32_t)spv::Decoration::Binding, 0 });

    const bool geometry = em == spv::ExecutionModel::Geometry;
    uint32_t countVar = 0;   // set 8 binding 1, declared by EmitSpriteShift
    uint32_t newMain = 0;
    if (!geometry && !mesh)
    {
        // New entry function: call the original, then shift Position.
        newMain = NewId();
        const uint32_t label = NewId(), callRes = NewId();
        func.Emit(spv::Op::OpFunction, { voidType, newMain, (uint32_t)spv::FunctionControlMask::MaskNone, fnType });
        func.Emit(spv::Op::OpLabel, { label });
        func.Emit(spv::Op::OpFunctionCall, { voidType, callRes, entryFn });
        EmitShift(func, id, next);
        // Same displacement for any varying carrying this clip position, so
        // the fragment shader's screen coordinate follows the geometry.
        for (uint32_t vy : posVaryings)
        {
            Ids vid = id;
            vid.posVar = 0;
            vid.posDirect = vy;
            EmitShift(func, vid, next);
        }
        for (const PrevClip& pc : prevClipVaryings)
        {
            Ids vid = id;
            vid.posVar = 0;
            vid.posDirect = pc.var;
            EmitShift(func, vid, next, pc.wComp, pc.xComp, pc.wVar == pc.var ? 0 : pc.wVar);
        }
        for (uint32_t vy : ndcVaryings)
        {
            auto sc = ndcScale.find(vy);
            EmitNdcShift(func, id, vy, next, ndcTimesW.count(vy) != 0, sc != ndcScale.end() ? sc->second : 0u);
        }
        for (size_t i = 0; i < projVaryings.size(); ++i)
            EmitProjShift(func, id, projVaryings[i].first, projK[i], next);
        uint32_t spriteVar = 0;
        if (spriteShift && spriteVarying >= 0)
        {
            std::unordered_set<uint32_t> atLocation;
            for (const Inst& in : m.insts)
                if (in.op == spv::Op::OpDecorate && in.count >= 4 && in.w[2] == (uint32_t)spv::Decoration::Location && in.w[3] == (uint32_t)spriteVarying) atLocation.insert(in.w[1]);
            for (const Inst& in : m.insts)
                if (in.op == spv::Op::OpVariable && in.w[1] == id.tPtrOutVec4 && in.w[3] == (uint32_t)spv::StorageClass::Output && atLocation.count(in.w[2])) spriteVar = in.w[2];
            if (!spriteVar) { *reason = "noSpriteVarying"; return false; }
        }
        if (spriteShift &&(em != spv::ExecutionModel::Vertex || m.version < 0x00010300 ||
                            !EmitSpriteShift(func, id, types, decos, next, countVar, spriteVar))) { *reason = "noSpriteShift"; return false; }
        func.Emit(spv::Op::OpReturn, {});
        func.Emit(spv::Op::OpFunctionEnd, {});
    }

    // --- Assemble -----------------------------------------------------------
    out.clear();
    out.reserve(n + types.out.size() + decos.out.size() + func.out.size() + 64);
    out.insert(out.end(), m.words.begin(), m.words.begin() + 5);
    const bool listAllGlobals = m.version >= 0x00010400;   // SPIR-V 1.4: interface lists every global
    size_t emits = 0;
    for (size_t i = 0; i < m.insts.size(); ++i)
    {
        const Inst& in = m.insts[i];
        if (i == firstTypeInst) out.insert(out.end(), decos.out.begin(), decos.out.end());
        if (i == firstFuncInst) out.insert(out.end(), types.out.begin(), types.out.end());
        if (i == entryInst)
        {
            std::vector<uint32_t> ep(in.w, in.w + in.count);
            if (newMain) ep[2] = newMain;
            if (listAllGlobals) ep.push_back(id.vParams);
            if (listAllGlobals && countVar) ep.push_back(countVar);
            ep[0] = ((uint32_t)ep.size() << 16) | (uint32_t)spv::Op::OpEntryPoint;
            out.insert(out.end(), ep.begin(), ep.end());
            continue;
        }
        if (newMain && (in.op == spv::Op::OpExecutionMode || in.op == spv::Op::OpExecutionModeId) && in.w[1] == entryFn)
        {
            std::vector<uint32_t> emode(in.w, in.w + in.count);
            emode[1] = newMain;
            out.insert(out.end(), emode.begin(), emode.end());
            continue;
        }
        if (geometry && (in.op == spv::Op::OpEmitVertex || in.op == spv::Op::OpEmitStreamVertex))
        {
            Builder b;
            EmitShift(b, id, next);
            for (uint32_t vy : posVaryings)
            {
                Ids vid = id;
                vid.posVar = 0;
                vid.posDirect = vy;
                EmitShift(b, vid, next);
            }
            out.insert(out.end(), b.out.begin(), b.out.end());
            ++emits;
        }
        auto mp = mesh ? meshPairs.find(i) : meshPairs.end();
        if (mp != meshPairs.end() && !mp->second.xPtr)
        {
            // A whole vertex: x moves before the store.
            Builder b;
            const uint32_t x = next++, w = next++, nx = next++, nv = next++;
            b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, x, in.w[2], 0 });
            b.Emit(spv::Op::OpCompositeExtract, { id.tFloat, w, in.w[2], 3 });
            const uint32_t applied = EmitShiftAmount(b, id, w, next);
            b.Emit(spv::Op::OpFAdd, { id.tFloat, nx, x, applied });
            b.Emit(spv::Op::OpCompositeInsert, { id.tVec4, nv, nx, in.w[2], 0 });
            out.insert(out.end(), b.out.begin(), b.out.end());
            const size_t at = out.size();
            out.insert(out.end(), in.w, in.w + in.count);
            out[at + 2] = nv;
            ++meshShifted;
            continue;
        }
        if (mp != meshPairs.end())
        {
            // x + sep*(w - conv) from the two stored values, through x's own pointer: after the w store (x came first),
            // or as the x store itself (w came first). One invocation's own vertex, no barrier needed.
            const MeshPair& pr = mp->second;
            if (pr.after) out.insert(out.end(), in.w, in.w + in.count);
            Builder b;
            const uint32_t nx = next++;
            const uint32_t applied = EmitShiftAmount(b, id, pr.wVal, next);
            b.Emit(spv::Op::OpFAdd, { id.tFloat, nx, pr.xVal, applied });
            b.Emit(spv::Op::OpStore, { pr.xPtr, nx });
            out.insert(out.end(), b.out.begin(), b.out.end());
            ++meshShifted;
            continue;
        }
        auto mv = movedX.find(i);
        if (mv != movedX.end())
        {
            Builder b;
            const uint32_t nx = next++;
            if (skyMoved)
            {
                const uint32_t pSep = next++, sep = next++, sw = next++;
                b.Emit(spv::Op::OpAccessChain, { id.tPtrUniFloat, pSep, id.vParams, cInt4 });
                b.Emit(spv::Op::OpLoad, { id.tFloat, sep, pSep });
                b.Emit(spv::Op::OpFMul, { id.tFloat, sw, sep, posW });
                b.Emit(spv::Op::OpFSub, { id.tFloat, nx, posX, sw });
            }
            else
            {
                // x and w as this product reads them: Position's own single stores, or under VKScreenUvVaryings the
                // stored x it multiplies and the w it divides by.
                auto mw = movedW.find(i);
                const uint32_t applied = EmitShiftAmount(b, id, mw != movedW.end() ? mw->second : posW, next);
                b.Emit(spv::Op::OpFAdd, { id.tFloat, nx, in.w[mv->second], applied });
            }
            out.insert(out.end(), b.out.begin(), b.out.end());
            const size_t at = out.size();
            out.insert(out.end(), in.w, in.w + in.count);
            out[at + mv->second] = nx;
            continue;
        }
        out.insert(out.end(), in.w, in.w + in.count);
    }
    if (geometry && !emits) { *reason = "noEmit"; return false; }
    if (mesh && !meshShifted) { *reason = "noMeshPositionStore"; return false; }
    out.insert(out.end(), func.out.begin(), func.out.end());
    out[3] = next;
    if (counts)
    {
        counts->posVaryings = clipVaryings;
        counts->prevClip = (unsigned)prevClipVaryings.size();
        counts->ndcVaryings = (unsigned)(ndcVaryings.size() + projVaryings.size());
        counts->screenUvMoved = skyMoved ? 0u : (unsigned)movedX.size();
    }
    return true;
}


// (set << 32 | binding) of every storage buffer the module can write, for the dispatch-replay gate: HLSL-derived
// SPIR-V marks read-only buffers NonWritable on the variable or on every block member.
void AnalyzeStorageWrites(const uint32_t* code, size_t words, std::vector<uint64_t>& out)
{
    Module m;
    if (!m.Parse(code, words)) return;
    std::unordered_map<uint32_t, uint32_t> setOf, bindingOf, ptrPointee, ptrClass;
    std::unordered_map<uint32_t, bool> bufferBlock, nonWritableVar;
    std::unordered_map<uint32_t, uint32_t> structMembers, nonWritableMembers, arrayElem;
    std::unordered_set<uint32_t> storageTexel;   // OpTypeImage Dim Buffer, Sampled 2
    for (const Inst& in : m.insts)
        switch (in.op)
        {
        case spv::Op::OpDecorate:
            if (in.w[2] == (uint32_t)spv::Decoration::DescriptorSet) setOf[in.w[1]] = in.w[3];
            else if (in.w[2] == (uint32_t)spv::Decoration::Binding) bindingOf[in.w[1]] = in.w[3];
            else if (in.w[2] == (uint32_t)spv::Decoration::BufferBlock) bufferBlock[in.w[1]] = true;
            else if (in.w[2] == (uint32_t)spv::Decoration::NonWritable) nonWritableVar[in.w[1]] = true;
            break;
        case spv::Op::OpMemberDecorate:
            if (in.w[3] == (uint32_t)spv::Decoration::NonWritable) ++nonWritableMembers[in.w[1]];
            break;
        case spv::Op::OpTypeStruct: structMembers[in.w[1]] = in.count - 2; break;
        case spv::Op::OpTypePointer: ptrPointee[in.w[1]] = in.w[3]; ptrClass[in.w[1]] = in.w[2]; break;
        case spv::Op::OpTypeImage:
            if (in.count > 7 && in.w[3] == (uint32_t)spv::Dim::Buffer && in.w[7] == 2) storageTexel.insert(in.w[1]);
            break;
        case spv::Op::OpTypeArray: case spv::Op::OpTypeRuntimeArray: arrayElem[in.w[1]] = in.w[2]; break;
        default: break;
        }
    for (const Inst& in : m.insts)
    {
        if (in.op != spv::Op::OpVariable) continue;
        const uint32_t var = in.w[2], ptr = in.w[1];
        auto cls = ptrClass.find(ptr);
        if (cls == ptrClass.end()) continue;
        const uint32_t pointee = ptrPointee.count(ptr) ? ptrPointee[ptr] : 0;
        const bool ssbo = cls->second == (uint32_t)spv::StorageClass::StorageBuffer ||
                          (cls->second == (uint32_t)spv::StorageClass::Uniform && bufferBlock.count(pointee));
        // Storage texel buffers write through OpImageWrite (formatted buffers bound as texel buffers).
        const uint32_t elem = arrayElem.count(pointee) ? arrayElem[pointee] : pointee;
        const bool texel = Vk().VKBufferTwins && cls->second == (uint32_t)spv::StorageClass::UniformConstant && storageTexel.count(elem);
        if (texel && !nonWritableVar.count(var))
        {
            auto s = setOf.find(var);
            auto b = bindingOf.find(var);
            if (s != setOf.end() && b != bindingOf.end()) out.push_back(((uint64_t)s->second << 32) | b->second);
            continue;
        }
        if (!ssbo || nonWritableVar.count(var)) continue;
        auto mem = structMembers.find(pointee);
        if (mem != structMembers.end() && mem->second && nonWritableMembers[pointee] >= mem->second) continue;
        auto s = setOf.find(var);
        auto b = bindingOf.find(var);
        if (s == setOf.end() || b == bindingOf.end()) continue;
        out.push_back(((uint64_t)s->second << 32) | b->second);
    }
}

// Every matrix in a uniform block at its real byte offset and stride, which the 16-byte register scan cannot see in
// a scalar_block_layout block.
void AnalyzeBlockMatrices(const uint32_t* code, size_t words, std::vector<BlockMatrix>& out)
{
    Module m;
    if (!m.Parse(code, words)) return;
    std::unordered_map<uint32_t, uint32_t> setOf, bindingOf, ptrPointee, ptrClass;
    std::unordered_map<uint32_t, std::vector<uint32_t>> structMembers;
    std::unordered_map<uint64_t, uint32_t> memberOffset, memberStride;
    std::unordered_map<uint64_t, bool>     memberRowMajor;
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> matrixOf;   // type -> (columnType, columns)
    std::unordered_map<uint32_t, uint32_t> vectorComps, arrayElem;
    for (const Inst& in : m.insts)
        switch (in.op)
        {
        case spv::Op::OpDecorate:
            if (in.w[2] == (uint32_t)spv::Decoration::DescriptorSet) setOf[in.w[1]] = in.w[3];
            else if (in.w[2] == (uint32_t)spv::Decoration::Binding) bindingOf[in.w[1]] = in.w[3];
            break;
        case spv::Op::OpMemberDecorate:
        {
            const uint64_t k = ((uint64_t)in.w[1] << 32) | in.w[2];
            if (in.w[3] == (uint32_t)spv::Decoration::Offset) memberOffset[k] = in.w[4];
            else if (in.w[3] == (uint32_t)spv::Decoration::MatrixStride) memberStride[k] = in.w[4];
            else if (in.w[3] == (uint32_t)spv::Decoration::RowMajor) memberRowMajor[k] = true;
            break;
        }
        case spv::Op::OpTypeStruct: structMembers[in.w[1]].assign(in.w + 2, in.w + in.count); break;
        case spv::Op::OpTypeMatrix: matrixOf[in.w[1]] = { in.w[2], in.w[3] }; break;
        case spv::Op::OpTypeVector: vectorComps[in.w[1]] = in.w[3]; break;
        case spv::Op::OpTypeArray: case spv::Op::OpTypeRuntimeArray: arrayElem[in.w[1]] = in.w[2]; break;
        case spv::Op::OpTypePointer: ptrPointee[in.w[1]] = in.w[3]; ptrClass[in.w[1]] = in.w[2]; break;
        default: break;
        }

    // Walks a member's type, carrying the stride and majorness decorated on it.
    struct Walker
    {
        decltype(structMembers)& sm; decltype(memberOffset)& mo; decltype(memberStride)& ms;
        decltype(memberRowMajor)& rm; decltype(matrixOf)& mx; decltype(vectorComps)& vc;
        decltype(arrayElem)& ae; std::vector<BlockMatrix>& out; uint32_t set, binding; bool ssbo;
        void Walk(uint32_t type, uint32_t base, uint32_t stride, bool rowMajor, int depth)
        {
            if (depth > 8) return;
            auto mit = mx.find(type);
            if (mit != mx.end())
            {
                auto v = vc.find(mit->second.first);
                if (v == vc.end() || !stride) return;   // no MatrixStride: not a block member
                BlockMatrix b;
                b.set = set; b.binding = binding; b.offset = base; b.stride = stride;
                b.cols = (uint8_t)mit->second.second; b.rows = (uint8_t)v->second;
                b.colMajor = !rowMajor; b.ssbo = ssbo;
                out.push_back(b);
                return;
            }
            // Vectors are reported as cols=1: the reconstruction's camera
            // position is a packed vec3 and is just as invisible as the basis.
            auto vit = vc.find(type);
            if (vit != vc.end())
            {
                if (vit->second < 3 || vit->second > 4) return;
                BlockMatrix b;
                b.set = set; b.binding = binding; b.offset = base; b.stride = 0;
                b.cols = 1; b.rows = (uint8_t)vit->second; b.colMajor = true; b.ssbo = ssbo;
                out.push_back(b);
                return;
            }
            auto ait = ae.find(type);
            if (ait != ae.end()) { Walk(ait->second, base, stride, rowMajor, depth + 1); return; }
            auto sit = sm.find(type);
            if (sit == sm.end()) return;
            for (uint32_t i = 0; i < sit->second.size(); ++i)
            {
                const uint64_t k = ((uint64_t)type << 32) | i;
                auto o = mo.find(k);
                if (o == mo.end()) continue;
                auto s = ms.find(k);
                Walk(sit->second[i], base + o->second, s != ms.end() ? s->second : 0,
                     rm.count(k) != 0, depth + 1);
            }
        }
    };

    for (const Inst& in : m.insts)
    {
        if (in.op != spv::Op::OpVariable) continue;
        auto cls = ptrClass.find(in.w[1]);
        if (cls == ptrClass.end()) continue;
        if (cls->second != (uint32_t)spv::StorageClass::Uniform &&
            cls->second != (uint32_t)spv::StorageClass::StorageBuffer) continue;
        auto s = setOf.find(in.w[2]);
        auto b = bindingOf.find(in.w[2]);
        auto p = ptrPointee.find(in.w[1]);
        if (s == setOf.end() || b == bindingOf.end() || p == ptrPointee.end()) continue;
        Walker w{ structMembers, memberOffset, memberStride, memberRowMajor, matrixOf,
                  vectorComps, arrayElem, out, s->second, b->second,
                  cls->second == (uint32_t)spv::StorageClass::StorageBuffer };
        w.Walk(p->second, 0, 0, false, 0);
    }
}

// The uniform-block members gl_Position depends on, walked back from its store to the uniform loads: the camera-block
// detector tells a view matrix from its inverse by what vertex shaders multiply by.
bool AnalyzePositionChain(const uint32_t* code, size_t words, std::vector<PositionUse>& out)
{
    Module m;
    if (!m.Parse(code, words)) return false;
    std::unordered_map<uint32_t, const Inst*> byId;        // result id -> instruction
    std::unordered_map<uint32_t, uint32_t> setOf, bindingOf, arrayStride;
    std::unordered_map<uint64_t, uint32_t> memberOffset;   // (struct << 32 | member) -> Offset
    std::unordered_map<uint32_t, uint32_t> constVal;       // OpConstant (32-bit) value
    std::unordered_map<uint32_t, uint32_t> ptrPointee, ptrClass;
    std::unordered_map<uint32_t, std::vector<uint32_t>> structMembers;
    std::unordered_map<uint32_t, bool> isMatrixType;
    uint32_t posStruct = 0; int posMember = -1; uint32_t posDirect = 0, posVar = 0;
    for (const Inst& in : m.insts)
    {
        switch (in.op)
        {
        case spv::Op::OpDecorate:
            if (in.w[2] == (uint32_t)spv::Decoration::DescriptorSet) setOf[in.w[1]] = in.w[3];
            else if (in.w[2] == (uint32_t)spv::Decoration::Binding) bindingOf[in.w[1]] = in.w[3];
            else if (in.w[2] == (uint32_t)spv::Decoration::ArrayStride) arrayStride[in.w[1]] = in.w[3];
            else if (in.w[2] == (uint32_t)spv::Decoration::BuiltIn && in.w[3] == (uint32_t)spv::BuiltIn::Position) posDirect = in.w[1];
            break;
        case spv::Op::OpMemberDecorate:
            if (in.w[3] == (uint32_t)spv::Decoration::Offset) memberOffset[((uint64_t)in.w[1] << 32) | in.w[2]] = in.w[4];
            else if (in.w[3] == (uint32_t)spv::Decoration::BuiltIn && in.w[4] == (uint32_t)spv::BuiltIn::Position) { posStruct = in.w[1]; posMember = (int)in.w[2]; }
            break;
        case spv::Op::OpConstant: if (in.count >= 4) constVal[in.w[2]] = in.w[3]; break;
        case spv::Op::OpTypePointer: ptrPointee[in.w[1]] = in.w[3]; ptrClass[in.w[1]] = in.w[2]; break;
        case spv::Op::OpTypeStruct: { std::vector<uint32_t> mem(in.w + 2, in.w + in.count); structMembers[in.w[1]] = mem; break; }
        case spv::Op::OpTypeMatrix: isMatrixType[in.w[1]] = true; break;
        case spv::Op::OpTypeArray: case spv::Op::OpTypeRuntimeArray: break;
        default: break;
        }
        // Result ids: OpVariable/OpLoad/arith carry (type, result) at w[1], w[2].
        switch (in.op)
        {
        case spv::Op::OpVariable: case spv::Op::OpLoad: case spv::Op::OpAccessChain: case spv::Op::OpInBoundsAccessChain:
        case spv::Op::OpVectorTimesMatrix: case spv::Op::OpMatrixTimesVector: case spv::Op::OpMatrixTimesMatrix:
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFNegate: case spv::Op::OpVectorTimesScalar:
        case spv::Op::OpCompositeConstruct: case spv::Op::OpCompositeExtract: case spv::Op::OpVectorShuffle: case spv::Op::OpCompositeInsert:
        case spv::Op::OpDot: case spv::Op::OpPhi: case spv::Op::OpSelect: case spv::Op::OpExtInst: case spv::Op::OpFDiv:
        // Translated shaders bit-cast uniform registers between int and float.
        case spv::Op::OpBitcast: case spv::Op::OpIAdd: case spv::Op::OpISub: case spv::Op::OpIMul:
            byId[in.w[2]] = &in; break;
        default: break;
        }
    }
    // Values stored into each Function-storage variable, for shaders that route everything through register
    // variables: every store counts (an over-approximation, safe for "does Position depend on this offset").
    std::unordered_map<uint32_t, std::vector<uint32_t>> gprStores;
    if (Vk().VKGprPositionChain)
        for (const Inst& in : m.insts)
        {
            if (in.op != spv::Op::OpStore || in.count < 3) continue;
            uint32_t base = in.w[1];
            auto ac = byId.find(base);
            if (ac != byId.end() && (ac->second->op == spv::Op::OpAccessChain || ac->second->op == spv::Op::OpInBoundsAccessChain))
                base = ac->second->w[3];
            auto v = byId.find(base);
            if (v == byId.end() || v->second->op != spv::Op::OpVariable) continue;
            if (v->second->w[3] != (uint32_t)spv::StorageClass::Function) continue;
            gprStores[base].push_back(in.w[2]);
        }

    // gl_PerVertex block variable
    if (posStruct)
        for (const Inst& in : m.insts)
            if (in.op == spv::Op::OpVariable && in.w[3] == (uint32_t)spv::StorageClass::Output)
            {
                auto pt = ptrPointee.find(in.w[1]);
                if (pt != ptrPointee.end() && pt->second == posStruct) { posVar = in.w[2]; break; }
            }
    if (!posVar && !posDirect) return false;

    // Stores into Position
    std::vector<uint32_t> roots;
    for (const Inst& in : m.insts)
    {
        if (in.op != spv::Op::OpStore) continue;
        const uint32_t ptr = in.w[1];
        if (posDirect && ptr == posDirect) { roots.push_back(in.w[2]); continue; }
        auto it = byId.find(ptr);
        if (it == byId.end()) continue;
        const Inst* ac = it->second;
        if ((ac->op == spv::Op::OpAccessChain || ac->op == spv::Op::OpInBoundsAccessChain) && ac->count >= 5 && ac->w[3] == posVar)
        {
            auto cv = constVal.find(ac->w[4]);
            if (cv != constVal.end() && (int)cv->second == posMember) roots.push_back(in.w[2]);
        }
    }
    if (roots.empty()) return false;

    std::unordered_map<uint32_t, bool> visited;
    std::unordered_set<uint32_t> visitedGpr;   // expand each GPR's stores once
    std::vector<std::pair<uint32_t, int>> stack;
    for (uint32_t r : roots) stack.push_back({ r, 0 });
    while (!stack.empty())
    {
        auto [id, depth] = stack.back(); stack.pop_back();
        if (depth > 48 || visited[id]) continue;
        visited[id] = true;
        auto it = byId.find(id);
        if (it == byId.end()) continue;
        const Inst* in = it->second;
        if (in->op == spv::Op::OpLoad)
        {
            // Resolve the pointer to a uniform block member.
            auto pit = byId.find(in->w[3]);
            if (pit == byId.end()) continue;
            const Inst* ac = pit->second;
            if (ac->op != spv::Op::OpAccessChain && ac->op != spv::Op::OpInBoundsAccessChain) continue;
            auto vit = byId.find(ac->w[3]);
            if (vit == byId.end() || vit->second->op != spv::Op::OpVariable) continue;
            const Inst* var = vit->second;
            const uint32_t cls = var->w[3];
            // A GPR read: keep walking through the values written into it.
            if (cls == (uint32_t)spv::StorageClass::Function)
            {
                auto gs = gprStores.find(var->w[2]);
                if (gs != gprStores.end() && !visitedGpr.count(var->w[2]))
                {
                    visitedGpr.insert(var->w[2]);
                    for (uint32_t v : gs->second) stack.push_back({ v, depth + 1 });
                }
                continue;
            }
            if (cls != (uint32_t)spv::StorageClass::Uniform && cls != (uint32_t)spv::StorageClass::StorageBuffer) continue;
            uint32_t type = ptrPointee.count(var->w[1]) ? ptrPointee[var->w[1]] : 0;
            uint32_t offset = 0; bool ok = true;
            for (uint32_t k = 4; k < ac->count && ok; ++k)
            {
                auto sm = structMembers.find(type);
                if (sm != structMembers.end())
                {
                    auto cv = constVal.find(ac->w[k]);
                    if (cv == constVal.end() || cv->second >= sm->second.size()) { ok = false; break; }
                    auto mo = memberOffset.find(((uint64_t)type << 32) | cv->second);
                    offset += mo != memberOffset.end() ? mo->second : 0;
                    type = sm->second[cv->second];
                }
                else if (arrayStride.count(type))
                {
                    auto cv = constVal.find(ac->w[k]);
                    if (cv == constVal.end()) { ok = false; break; }
                    offset += cv->second * arrayStride[type];
                    // element type: OpTypeArray w[2]
                    for (const Inst& t : m.insts) if ((t.op == spv::Op::OpTypeArray || t.op == spv::Op::OpTypeRuntimeArray) && t.w[1] == type) { type = t.w[2]; break; }
                }
                else break;   // matrix column / vector component: same member
            }
            if (!ok) continue;
            PositionUse u;
            u.set = setOf.count(var->w[2]) ? setOf[var->w[2]] : 0;
            u.binding = bindingOf.count(var->w[2]) ? bindingOf[var->w[2]] : 0;
            u.offset = offset;
            u.matrix = isMatrixType.count(in->w[1]) != 0;
            u.depth = depth;
            bool dup = false;
            for (const auto& e : out) if (e.set == u.set && e.binding == u.binding && e.offset == u.offset) dup = true;
            // matrixOverScalar: a scalar read of P[1][1] before P * v must not hide the matrix use.
            if (dup && u.matrix && Traits().matrixOverScalar)
            {
                auto e = std::find_if(out.begin(), out.end(), [&](const PositionUse& x) { return x.set == u.set && x.binding == u.binding && x.offset == u.offset; });
                if (!e->matrix) { out.erase(e); dup = false; }
            }
            if (!dup) out.push_back(u);
            continue;
        }
        // Arithmetic: follow operands.
        switch (in->op)
        {
        case spv::Op::OpVectorTimesMatrix: case spv::Op::OpMatrixTimesVector: case spv::Op::OpMatrixTimesMatrix:
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFDiv: case spv::Op::OpVectorTimesScalar: case spv::Op::OpDot:
            stack.push_back({ in->w[3], depth + 1 }); stack.push_back({ in->w[4], depth + 1 }); break;
        case spv::Op::OpFNegate: case spv::Op::OpCompositeExtract:
            stack.push_back({ in->w[3], depth + 1 }); break;
        case spv::Op::OpBitcast:
            if (Vk().VKChainThroughBitcast) stack.push_back({ in->w[3], depth + 1 });
            break;
        case spv::Op::OpIAdd: case spv::Op::OpISub: case spv::Op::OpIMul:
            if (Vk().VKChainThroughBitcast) { stack.push_back({ in->w[3], depth + 1 }); stack.push_back({ in->w[4], depth + 1 }); }
            break;
        case spv::Op::OpCompositeConstruct: case spv::Op::OpCompositeInsert:
            for (uint32_t k = 3; k < in->count; ++k) stack.push_back({ in->w[k], depth + 1 }); break;
        case spv::Op::OpVectorShuffle: stack.push_back({ in->w[3], depth + 1 }); stack.push_back({ in->w[4], depth + 1 }); break;
        case spv::Op::OpSelect: stack.push_back({ in->w[4], depth + 1 }); stack.push_back({ in->w[5], depth + 1 }); break;
        case spv::Op::OpPhi: for (uint32_t k = 3; k < in->count; k += 2) stack.push_back({ in->w[k], depth + 1 }); break;
        // Extended instructions (fma, ...): every operand.
        case spv::Op::OpExtInst: for (uint32_t k = 5; k < in->count; ++k) stack.push_back({ in->w[k], depth + 1 }); break;
        default: break;
        }
    }
    // VKGprPositionChain: with no matrix type, a matrix is four consecutive 16-byte uniform registers the position
    // depends on.
    if (Vk().VKGprPositionChain)
        for (auto& u : out)
        {
            if (u.matrix || (u.offset & 15)) continue;
            unsigned run = 0;
            for (unsigned k = 0; k < 4; ++k)
            {
                const uint32_t off = u.offset + k * 16;
                for (const auto& e : out)
                    if (e.set == u.set && e.binding == u.binding && e.offset == off) { ++run; break; }
            }
            if (run >= 4) u.matrix = true;
        }
    return !out.empty();
}

}} // namespace wiz3d::vk
