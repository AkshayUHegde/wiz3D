/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Right-eye camera for fragment shaders that load matrices as raw uint dwords (shadPS4 binds every guest buffer so).
// A depth-fed 16-dword window that passes the in-shader inverse test is rewritten for x' = x + fsSep*(w - conv).

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <functional>
#include <tuple>
#include <algorithm>
#include "FsPatch.h"

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

// One buffer the matrices are read from: data[k + off] through the same base, member index and offset value.
struct Key
{
    uint32_t base, idx0, off, ptrType;
    bool operator<(const Key& o) const { return std::tie(base, idx0, off, ptrType) < std::tie(o.base, o.idx0, o.off, o.ptrType); }
};

} // namespace

// A fragment shader that reads a texture at x/w of one interpolated varying through at most a few scalars and no
// matrix: its VS must hand it the right eye's clip position, which a matrix * clip-varying reconstruction must not get.
bool ReadsScreenAtClipVarying(const uint32_t* code, size_t words)
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
    std::unordered_map<uint32_t, uint32_t> loc, varClass, uconst;
    bool fragment = false;
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(ins[k].op, &hasRes, &hasType);
        if (hasRes) def[hasType ? i[2] : i[1]] = k;
        if (ins[k].op == spv::Op::OpEntryPoint && i[1] == (uint32_t)spv::ExecutionModel::Fragment) fragment = true;
        if (ins[k].op == spv::Op::OpDecorate && ins[k].n >= 4 && i[2] == (uint32_t)spv::Decoration::Location) loc[i[1]] = i[3];
        if (ins[k].op == spv::Op::OpVariable) varClass[i[2]] = i[3];
        if (ins[k].op == spv::Op::OpConstant && ins[k].n == 4) uconst[i[2]] = i[3];
    }
    if (!fragment) return false;
    // Per value: the (location, component) inputs it derives from, and how many buffer loads feed it.
    struct Anc { std::set<uint32_t> lc; unsigned loads = 0; bool done = false; };
    std::unordered_map<uint32_t, Anc> memo;
    std::function<const Anc&(uint32_t, int)> anc = [&](uint32_t id, int depth) -> const Anc&
    {
        Anc& a = memo[id];
        if (a.done || depth > 80) return a;
        a.done = true;
        auto it = def.find(id);
        if (it == def.end()) return a;
        const uint32_t* i = code + ins[it->second].at;
        const uint16_t n = ins[it->second].n;
        Anc r;
        switch (ins[it->second].op)
        {
        case spv::Op::OpLoad:
        {
            auto pd = def.find(i[3]);
            if (pd == def.end()) break;
            const Ins& ac = ins[pd->second];
            const uint32_t* p = code + ac.at;
            const uint32_t base = (ac.op == spv::Op::OpAccessChain || ac.op == spv::Op::OpInBoundsAccessChain) ? p[3] : i[3];
            auto vc = varClass.find(base);
            if (vc == varClass.end()) break;
            if (vc->second == (uint32_t)spv::StorageClass::Input)
            {
                auto l = loc.find(base);
                auto c = ac.op == spv::Op::OpAccessChain && ac.n >= 5 ? uconst.find(p[ac.n - 1]) : uconst.end();
                if (l != loc.end() && c != uconst.end() && c->second < 4) r.lc.insert(l->second * 4 + c->second);
            }
            else if (vc->second == (uint32_t)spv::StorageClass::StorageBuffer || vc->second == (uint32_t)spv::StorageClass::Uniform ||
                     vc->second == (uint32_t)spv::StorageClass::PushConstant)
                r.loads = 1;
            break;
        }
        case spv::Op::OpCompositeExtract: case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpFNegate:
        { const Anc& s = anc(i[3], depth + 1); r.lc = s.lc; r.loads = s.loads; break; }
        case spv::Op::OpExtInst:
            for (uint16_t k = 5; k < n; ++k) { const Anc& s = anc(i[k], depth + 1); r.lc.insert(s.lc.begin(), s.lc.end()); r.loads += s.loads; }
            break;
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFDiv: case spv::Op::OpSelect:
        case spv::Op::OpCompositeConstruct: case spv::Op::OpVectorShuffle: case spv::Op::OpVectorTimesScalar: case spv::Op::OpDot:
            for (uint16_t k = 3; k < n; ++k) { const Anc& s = anc(i[k], depth + 1); r.lc.insert(s.lc.begin(), s.lc.end()); r.loads += s.loads; }
            break;
        default: break;
        }
        r.done = true;
        return memo[id] = r;
    };
    for (const Ins& in : ins)
    {
        if (in.op != spv::Op::OpImageSampleImplicitLod && in.op != spv::Op::OpImageSampleExplicitLod) continue;
        auto cd = def.find(code[in.at + 4]);
        if (cd == def.end() || ins[cd->second].op != spv::Op::OpCompositeConstruct) continue;
        const Anc& a = anc(code[ins[cd->second].at + 3], 0);
        std::set<uint32_t> locs;
        for (uint32_t v : a.lc) locs.insert(v / 4);
        if (locs.size() != 1 || a.loads > 4) continue;
        const uint32_t l = *locs.begin();
        if (a.lc.count(l * 4 + 0) && a.lc.count(l * 4 + 3)) return true;
    }
    return false;
}

bool PatchCameraWindowsFS(const uint32_t* code, size_t words, uint32_t stereoSet, const CameraWindowsRules& rules,
                          std::vector<uint32_t>& out, unsigned* windowsOut)
{
    bool counters = rules.counters;   // cleared below for modules without the StorageBuffer class
    const bool screenX = rules.screenX;
    const uint32_t screenXBase = rules.screenXBase;
    if (windowsOut) *windowsOut = 0;
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
    std::unordered_map<uint32_t, uint32_t> uconst, varClass, varPtr, ptrPointee, ptrClass;
    std::unordered_set<uint32_t> nonWritableVar, nonWritableStruct, bufferBlock;
    uint32_t tFloat = 0, tUint = 0, tBool = 0, entryFn = 0, tPtrUF = 0, tVec4 = 0, fragCoordDeco = 0;
    size_t entryIdx = SIZE_MAX, firstType = SIZE_MAX, firstFunc = SIZE_MAX;
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(ins[k].op, &hasRes, &hasType);
        if (hasRes) def[hasType ? i[2] : i[1]] = k;
        switch (ins[k].op)
        {
        case spv::Op::OpEntryPoint:
            if (i[1] == (uint32_t)spv::ExecutionModel::Fragment && !entryFn) { entryFn = i[2]; entryIdx = k; }
            break;
        case spv::Op::OpDecorate:
            if (i[2] == (uint32_t)spv::Decoration::NonWritable) nonWritableVar.insert(i[1]);
            if (i[2] == (uint32_t)spv::Decoration::BufferBlock) bufferBlock.insert(i[1]);
            if (ins[k].n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::FragCoord) fragCoordDeco = i[1];
            break;
        case spv::Op::OpMemberDecorate:
            if (i[2] == 0 && i[3] == (uint32_t)spv::Decoration::NonWritable) nonWritableStruct.insert(i[1]);
            break;
        case spv::Op::OpTypeFloat: if (i[2] == 32 && !tFloat) tFloat = i[1]; break;
        case spv::Op::OpTypeInt: if (i[2] == 32 && i[3] == 0 && !tUint) tUint = i[1]; break;
        case spv::Op::OpTypeBool: if (!tBool) tBool = i[1]; break;
        case spv::Op::OpTypeVector: if (tFloat && i[2] == tFloat && i[3] == 4 && !tVec4) tVec4 = i[1]; break;
        case spv::Op::OpTypePointer:
            ptrClass[i[1]] = i[2]; ptrPointee[i[1]] = i[3];
            if (i[2] == (uint32_t)spv::StorageClass::Uniform && tFloat && i[3] == tFloat && !tPtrUF) tPtrUF = i[1];
            break;
        case spv::Op::OpConstant: if (tUint && i[1] == tUint) uconst[i[2]] = i[3]; break;
        case spv::Op::OpVariable: if (firstFunc == SIZE_MAX) { varClass[i[2]] = i[3]; varPtr[i[2]] = i[1]; } break;
        case spv::Op::OpFunction: if (firstFunc == SIZE_MAX) firstFunc = k; break;
        default: break;
        }
        const bool typeSection = (ins[k].op >= spv::Op::OpTypeVoid && ins[k].op <= spv::Op::OpTypeForwardPointer) ||
                                 (ins[k].op >= spv::Op::OpConstantTrue && ins[k].op <= spv::Op::OpSpecConstantOp) ||
                                 (ins[k].op == spv::Op::OpVariable && firstFunc == SIZE_MAX);
        if (typeSection && firstType == SIZE_MAX) firstType = k;
    }
    if (!entryFn || !tFloat || !tUint || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;

    // The entry function's instruction range and the first point after its entry block's variables.
    size_t fnStart = SIZE_MAX, fnEnd = SIZE_MAX, entryTop = SIZE_MAX;
    for (size_t k = firstFunc; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        if (fnStart == SIZE_MAX) { if (ins[k].op == spv::Op::OpFunction && i[2] == entryFn) fnStart = k; continue; }
        if (entryTop == SIZE_MAX && ins[k].op == spv::Op::OpLabel) { entryTop = k; continue; }
        if (entryTop != SIZE_MAX && entryTop + 1 == k && (ins[k].op == spv::Op::OpVariable || ins[k].op == spv::Op::OpLine)) { entryTop = k; continue; }
        if (ins[k].op == spv::Op::OpFunctionEnd) { fnEnd = k; break; }
    }
    if (fnStart == SIZE_MAX || fnEnd == SIZE_MAX || entryTop == SIZE_MAX) return false;
    // The entry block: its label and terminator. The hit counter splits it, which nothing can branch back into.
    size_t entryLabel = SIZE_MAX, entryTerm = SIZE_MAX;
    for (size_t k = fnStart; k < fnEnd && entryTerm == SIZE_MAX; ++k)
    {
        const spv::Op op = ins[k].op;
        if (op == spv::Op::OpLabel && entryLabel == SIZE_MAX) entryLabel = k;
        else if (entryLabel != SIZE_MAX && (op == spv::Op::OpBranch || op == spv::Op::OpBranchConditional || op == spv::Op::OpSwitch ||
                 op == spv::Op::OpReturn || op == spv::Op::OpReturnValue || op == spv::Op::OpKill || op == spv::Op::OpUnreachable ||
                 op == spv::Op::OpTerminateInvocation || op == spv::Op::OpSelectionMerge || op == spv::Op::OpLoopMerge))
            entryTerm = k;
    }
    if (code[1] < 0x00010300) counters = false;   // StorageBuffer class

    // Read-only buffers only: a window the shader also writes could change between our load and its own.
    auto readable = [&](uint32_t var) -> bool
    {
        auto c = varClass.find(var);
        if (c == varClass.end()) return false;
        auto pt = ptrPointee.find(varPtr[var]);
        const uint32_t st = pt == ptrPointee.end() ? 0 : pt->second;
        const bool nw = nonWritableVar.count(var) || nonWritableStruct.count(st);
        if (c->second == (uint32_t)spv::StorageClass::StorageBuffer) return nw;
        if (c->second == (uint32_t)spv::StorageClass::Uniform) return nw || !bufferBlock.count(st);
        return false;
    };

    // uint loads of base[idx0][k + off], grouped by buffer and offset value.
    std::map<Key, std::map<uint32_t, std::vector<size_t>>> loads;
    for (size_t k = fnStart; k < fnEnd; ++k)
    {
        if (ins[k].op != spv::Op::OpLoad) continue;
        const uint32_t* i = code + ins[k].at;
        if (i[1] != tUint) continue;
        auto pd = def.find(i[3]);
        if (pd == def.end()) continue;
        const Ins& ac = ins[pd->second];
        if ((ac.op != spv::Op::OpAccessChain && ac.op != spv::Op::OpInBoundsAccessChain) || ac.n != 6) continue;
        const uint32_t* a = code + ac.at;
        if (!readable(a[3])) continue;
        uint32_t kk = 0, off = 0;
        auto c = uconst.find(a[5]);
        if (c != uconst.end()) kk = c->second;
        else
        {
            // k + off, or off alone for dword 0 (shadPS4 emits IAdd(0, off); glslc folds it away).
            kk = 0; off = a[5];
            auto ad = def.find(a[5]);
            if (ad != def.end() && ins[ad->second].op == spv::Op::OpIAdd)
            {
                const uint32_t* add = code + ins[ad->second].at;
                auto c3 = uconst.find(add[3]), c4 = uconst.find(add[4]);
                if (c3 != uconst.end()) { kk = c3->second; off = add[4]; }
                else if (c4 != uconst.end()) { kk = c4->second; off = add[3]; }
            }
        }
        loads[{ a[3], a[4], off, a[1] }][kk].push_back(k);
    }

    // Windows: 16 dwords at a 16-byte boundary with both ends loaded (all in the bound range) and half or more used;
    // a shader needing only clip x and w reads rows 0 and 3. Overlapping ones are all kept, the test picks.
    struct Win { Key key; uint32_t b; };
    std::vector<Win> wins;
    std::map<Key, size_t> insertAfter;      // instruction index the key's code follows
    for (const auto& kv : loads)
    {
        const Key& key = kv.first;
        size_t at = entryTop;
        if (key.off)
        {
            auto od = def.find(key.off);
            if (od == def.end() || od->second <= fnStart || od->second >= fnEnd) continue;
            const spv::Op op = ins[od->second].op;
            if (op == spv::Op::OpPhi || op == spv::Op::OpFunctionParameter || op == spv::Op::OpVariable) continue;
            at = od->second;
        }
        bool any = false;
        for (const auto& e : kv.second)
        {
            const uint32_t b = e.first;
            if (b % 4 || !kv.second.count(b + 15)) continue;
            unsigned used = 0;
            for (uint32_t j = 0; j < 16; ++j) used += kv.second.count(b + j) ? 1u : 0u;
            if (used >= 8) { wins.push_back({ key, b }); any = true; }
        }
        if (any) insertAfter[key] = at;
    }
    // Thirty-two covers every camera layout seen (lamp lights reach 27); more are blur kernels, whose cost per
    // fragment would buy nothing.
    if (wins.empty() || wins.size() > 32) return false;

    // A window counts only when a dword multiplies a value from an image read (a rebuild from the eye's depth);
    // one rebuilding from its clip varying is already right with the varying unshifted.
    std::function<bool(uint32_t, int)> fromImage = [&](uint32_t id, int depth) -> bool
    {
        auto it = def.find(id);
        if (it == def.end() || depth > 12) return false;
        const uint32_t* i = code + ins[it->second].at;
        const uint16_t n = ins[it->second].n;
        switch (ins[it->second].op)
        {
        case spv::Op::OpImageSampleImplicitLod: case spv::Op::OpImageSampleExplicitLod: case spv::Op::OpImageFetch:
        case spv::Op::OpImageRead: case spv::Op::OpImageGather:
            return true;
        case spv::Op::OpCompositeExtract: case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpFNegate:
            return fromImage(i[3], depth + 1);
        case spv::Op::OpExtInst:
            for (uint16_t k = 5; k < n; ++k) if (fromImage(i[k], depth + 1)) return true;
            return false;
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFDiv: case spv::Op::OpSelect:
        case spv::Op::OpCompositeConstruct: case spv::Op::OpVectorShuffle:
            for (uint16_t k = 3; k < n; ++k) if (fromImage(i[k], depth + 1)) return true;
            return false;
        default: return false;
        }
    };
    // shadPS4 bitcasts a load where it is used, not next to it.
    std::unordered_map<uint32_t, std::pair<Key, uint32_t>> loadRes, loadOf;   // uint load / its float -> (key, dword)
    for (const auto& kv : loads)
        for (const auto& e : kv.second)
            for (size_t li : e.second) loadRes[code[ins[li].at + 2]] = { kv.first, e.first };
    for (size_t k = fnStart; k < fnEnd; ++k)
        if (ins[k].op == spv::Op::OpBitcast)
        {
            auto lr = loadRes.find(code[ins[k].at + 3]);
            if (lr != loadRes.end()) loadOf[code[ins[k].at + 2]] = lr->second;
        }
    std::set<std::pair<Key, uint32_t>> depthMul;
    for (size_t k = fnStart; k < fnEnd; ++k)
    {
        if (ins[k].op != spv::Op::OpFMul) continue;
        const uint32_t* i = code + ins[k].at;
        for (int s = 0; s < 2; ++s)
        {
            auto lo = loadOf.find(i[3 + s]);
            if (lo == loadOf.end()) continue;
            if (!fromImage(i[4 - s], 0)) continue;
            depthMul.insert(lo->second);
        }
    }
    auto depthFed = [&](const Win& w) { for (uint32_t j = 0; j < 16; ++j) if (depthMul.count({ w.key, w.b + j })) return true; return false; };
    // Only shaders rebuilding from a depth read are patched: the test costs every fragment.
    bool anyFed = false;
    for (const Win& w : wins) anyFed = anyFed || depthFed(w);
    if (!anyFed) return false;

    uint32_t next = code[3];
    Emit deco, types;
    auto newConstF = [&](float f) { const uint32_t id = next++; types.Op(spv::Op::OpConstant, { tFloat, id, Bits(f) }); return id; };
    std::unordered_map<uint32_t, uint32_t> uconstOf;   // value -> id
    for (const auto& c : uconst) if (!uconstOf.count(c.second)) uconstOf[c.second] = c.first;
    auto constU = [&](uint32_t v)
    {
        auto it = uconstOf.find(v);
        if (it != uconstOf.end()) return it->second;
        const uint32_t id = next++;
        types.Op(spv::Op::OpConstant, { tUint, id, v });
        uconstOf[v] = id;
        return id;
    };
    if (!tBool) { tBool = next++; types.Op(spv::Op::OpTypeBool, { tBool }); }
    const uint32_t tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat, tFloat, tFloat });   // sep, conv, wMin, eye, fsSep
    types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
    if (!tPtrUF) { tPtrUF = next++; types.Op(spv::Op::OpTypePointer, { tPtrUF, (uint32_t)spv::StorageClass::Uniform, tFloat }); }
    types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
    deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
    for (uint32_t k = 0; k < 5; ++k) deco.Op(spv::Op::OpMemberDecorate, { tS, k, (uint32_t)spv::Decoration::Offset, k * 4 });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });
    const uint32_t cU1 = constU(1), cU4 = constU(4);
    const uint32_t f0 = newConstF(0.f), f1 = newConstF(1.f);
    const uint32_t fCos = newConstF(1e-3f);      // cos^2 between axes: within ~1.8 degrees of perpendicular
    const uint32_t fLo = newConstF(0.1225f), fHi = newConstF(0.8464f);   // (H/W)^2 in 0.35^2..0.92^2: not square
    const uint32_t fRow3 = newConstF(1e-6f);     // inverse row 3 x,y vs z,w, squared: 1e-3 relative

    if (screenX)
    {
        // screenXBase: the camera inverse among several depth-fed windows (the decal's projection shares its screen x).
        const Win* fw = nullptr;
        for (const Win& w : wins)
        {
            if (!depthFed(w)) continue;
            if (screenXBase != ~0u) { if (w.b == screenXBase) fw = &w; continue; }
            if (fw) return false;
            fw = &w;
        }
        if (!fw) return false;
        auto operandOf = [&](std::initializer_list<uint32_t> dwords) -> uint32_t
        {
            uint32_t v = 0;
            for (size_t k = fnStart; k < fnEnd; ++k)
            {
                if (ins[k].op != spv::Op::OpFMul) continue;
                const uint32_t* i = code + ins[k].at;
                for (int s = 0; s < 2; ++s)
                {
                    auto lo = loadOf.find(i[3 + s]);
                    if (lo == loadOf.end() || lo->second.first.base != fw->key.base || lo->second.first.idx0 != fw->key.idx0 ||
                        lo->second.first.off != fw->key.off || lo->second.first.ptrType != fw->key.ptrType) continue;
                    bool hit = false;
                    for (uint32_t dw : dwords) hit = hit || lo->second.second == fw->b + dw;
                    if (!hit) continue;
                    if (v && v != i[4 - s]) return 0;
                    v = i[4 - s];
                }
            }
            return v;
        };
        const uint32_t X = operandOf({ 0, 4, 8, 12 }), D = operandOf({ 2, 6, 10, 14 });
        if (!X || !D || X == D || !fromImage(D, 0) || fromImage(X, 0)) return false;
        auto dx = def.find(X), dd = def.find(D);
        if (dx == def.end() || dd == def.end()) return false;
        const size_t at = dx->second > dd->second ? dx->second : dd->second;
        if (dx->second < entryLabel || at >= entryTerm || insertAfter[fw->key] > at) return false;
        Emit e;
        auto F = [&](spv::Op op, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(op, { tFloat, r, a, b }); return r; };
        auto Load = [&](uint32_t k) -> uint32_t
        {
            uint32_t idx = constU(k);
            if (fw->key.off) { const uint32_t r = next++; e.Op(spv::Op::OpIAdd, { tUint, r, idx, fw->key.off }); idx = r; }
            const uint32_t p = next++, u = next++, f = next++;
            e.Op(spv::Op::OpAccessChain, { fw->key.ptrType, p, fw->key.base, fw->key.idx0, idx });
            e.Op(spv::Op::OpLoad, { tUint, u, p });
            e.Op(spv::Op::OpBitcast, { tFloat, f, u });
            return f;
        };
        const uint32_t pS = next++, s = next++, pC = next++, conv = next++;
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cU4 });
        e.Op(spv::Op::OpLoad, { tFloat, s, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cU1 });
        e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
        const uint32_t W = F(spv::Op::OpFAdd, F(spv::Op::OpFMul, Load(fw->b + 14), D), Load(fw->b + 15));
        const uint32_t xL = F(spv::Op::OpFAdd, F(spv::Op::OpFSub, X, s), F(spv::Op::OpFMul, F(spv::Op::OpFMul, s, conv), W));
        out.clear();
        out.reserve(words + 256);
        out.insert(out.end(), code, code + 5);
        const bool listAll = code[1] >= 0x00010400;
        for (size_t k = 0; k < ins.size(); ++k)
        {
            const uint32_t* i = code + ins[k].at;
            if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
            if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
            if (k == entryIdx && listAll)
            {
                std::vector<uint32_t> ep(i, i + ins[k].n);
                ep.push_back(vStereo);
                ep[0] = ((uint32_t)ep.size() << 16) | (uint32_t)spv::Op::OpEntryPoint;
                out.insert(out.end(), ep.begin(), ep.end());
                continue;
            }
            const size_t pos = out.size();
            out.insert(out.end(), i, i + ins[k].n);
            if (k > at && k < fnEnd)
            {
                uint16_t from = 3, to = ins[k].n;
                switch (ins[k].op)
                {
                case spv::Op::OpStore: from = 1; break;
                case spv::Op::OpExtInst: from = 5; break;
                case spv::Op::OpCompositeExtract: to = 4; break;
                case spv::Op::OpCompositeInsert: case spv::Op::OpVectorShuffle: to = ins[k].n < 5 ? ins[k].n : 5; break;
                default:
                {
                    bool hasRes = false, hasType = false;
                    spv::HasResultAndType(ins[k].op, &hasRes, &hasType);
                    if (!hasRes || !hasType) to = 0;
                    break;
                }
                }
                for (uint16_t a = from; a < to; ++a) if (out[pos + a] == X) out[pos + a] = xL;
            }
            else if (k < at && k > dx->second && screenXBase == ~0u)   // the decal's own depth read keeps the eye's x
            {
                for (uint16_t a = 1; a < ins[k].n; ++a) if (i[a] == X) return false;
            }
            if (k == at) out.insert(out.end(), e.v.begin(), e.v.end());
        }
        out[3] = next;
        if (windowsOut) *windowsOut = 1;
        return true;
    }

    // Code inserted after each key's offset value; original loads it covers become copies of the patched value.
    std::map<size_t, Emit> codeAfter;
    std::unordered_map<size_t, uint32_t> loadRepl;    // original load instruction -> uint value id
    std::map<Key, std::vector<size_t>> winsOf;
    for (size_t w = 0; w < wins.size(); ++w) winsOf[wins[w].key].push_back(w);
    struct Hit { uint32_t ok, n32, n33; };   // the test, inverse row 3 (1/w = n32*z + n33)
    std::vector<Hit> hits;
    size_t counterAt = 0;                              // the latest entry-block insertion point
    uint32_t activeAny = 0;
    for (const auto& kw : winsOf)
    {
        const Key& key = kw.first;
        const bool inEntry = insertAfter[key] >= entryLabel && insertAfter[key] < entryTerm;
        Emit& e = codeAfter[insertAfter[key]];
        auto F = [&](spv::Op op, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(op, { tFloat, r, a, b }); return r; };
        auto B = [&](spv::Op op, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(op, { tBool, r, a, b }); return r; };
        auto Sel = [&](uint32_t c, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(spv::Op::OpSelect, { tFloat, r, c, a, b }); return r; };
        auto Mul = [&](uint32_t a, uint32_t b) { return F(spv::Op::OpFMul, a, b); };
        auto Add = [&](uint32_t a, uint32_t b) { return F(spv::Op::OpFAdd, a, b); };
        auto Sub = [&](uint32_t a, uint32_t b) { return F(spv::Op::OpFSub, a, b); };
        auto Lt = [&](uint32_t a, uint32_t b) { return B(spv::Op::OpFOrdLessThan, a, b); };
        auto And = [&](uint32_t a, uint32_t b) { return B(spv::Op::OpLogicalAnd, a, b); };
        auto Dot3 = [&](uint32_t a0, uint32_t a1, uint32_t a2, uint32_t b0, uint32_t b1, uint32_t b2) { return Add(Add(Mul(a0, b0), Mul(a1, b1)), Mul(a2, b2)); };

        // The eye's shift: fsSep (0 in the left eye) and conv.
        const uint32_t pS = next++, s = next++, pC = next++, conv = next++;
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cU4 });
        e.Op(spv::Op::OpLoad, { tFloat, s, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cU1 });
        e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
        const uint32_t active = B(spv::Op::OpFOrdNotEqual, s, f0);
        const uint32_t sc = Mul(s, conv);
        if (inEntry && insertAfter[key] >= counterAt) { counterAt = insertAfter[key]; activeAny = active; }

        std::map<uint32_t, uint32_t> val;   // dword -> float as loaded
        auto Load = [&](uint32_t k) -> uint32_t
        {
            auto it = val.find(k);
            if (it != val.end()) return it->second;
            uint32_t idx = constU(k);
            if (key.off) { const uint32_t r = next++; e.Op(spv::Op::OpIAdd, { tUint, r, idx, key.off }); idx = r; }
            const uint32_t p = next++, u = next++, f = next++;
            e.Op(spv::Op::OpAccessChain, { key.ptrType, p, key.base, key.idx0, idx });
            e.Op(spv::Op::OpLoad, { tUint, u, p });
            e.Op(spv::Op::OpBitcast, { tFloat, f, u });
            return val[k] = f;
        };
        std::map<uint32_t, uint32_t> fin;   // dword -> final float
        for (size_t wi : kw.second)
        {
            if (!depthFed(wins[wi])) continue;   // its dwords keep their original loads
            const uint32_t b = wins[wi].b;
            uint32_t f[16];
            for (uint32_t j = 0; j < 16; ++j) f[j] = Load(b + j);
            auto M = [&](int r, int c) { return f[r * 4 + c]; };
            // world = N * clip, row-major: row 3 is (0, 0, 1/B, -A/B), columns 0/1 are right/P00 and up/P11.
            const uint32_t ref = Add(Mul(M(3,2), M(3,2)), Mul(M(3,3), M(3,3)));
            const uint32_t lc0 = Dot3(M(0,0), M(1,0), M(2,0), M(0,0), M(1,0), M(2,0));
            const uint32_t lc1 = Dot3(M(0,1), M(1,1), M(2,1), M(0,1), M(1,1), M(2,1));
            const uint32_t dc = Dot3(M(0,0), M(1,0), M(2,0), M(0,1), M(1,1), M(2,1));
            const uint32_t lim = Mul(fRow3, ref);
            uint32_t ok = Lt(f0, ref);
            ok = And(ok, Lt(Mul(M(3,0), M(3,0)), lim));
            ok = And(ok, Lt(Mul(M(3,1), M(3,1)), lim));
            ok = And(ok, Lt(Mul(dc, dc), Mul(fCos, Mul(lc0, lc1))));
            ok = And(ok, Lt(Mul(fLo, lc0), lc1));
            ok = And(ok, Lt(lc1, Mul(fHi, lc0)));
            ok = And(ok, active);
            // N' = N * S^-1, S = I + e0 a^T with a = s*e3 - s*conv*N[3]: N'_ij = N_ij - N_i0 a_j / (1 + a_0).
            uint32_t a[4];
            for (int c = 0; c < 4; ++c) a[c] = Sub(f0, Mul(sc, M(3, c)));
            a[3] = Add(a[3], s);
            const uint32_t den = Add(f1, a[0]);
            uint32_t patched[16];
            for (int r = 0; r < 4; ++r)
            {
                const uint32_t kr = F(spv::Op::OpFDiv, M(r, 0), den);
                for (int c = 0; c < 4; ++c) patched[r * 4 + c] = Sub(M(r, c), Mul(kr, a[c]));
            }
            if (inEntry) hits.push_back({ ok, M(3, 2), M(3, 3) });
            // Overlapping windows: the one whose test passed wins its dwords.
            for (uint32_t j = 0; j < 16; ++j)
            {
                auto it = fin.find(b + j);
                fin[b + j] = Sel(ok, patched[j], it == fin.end() ? f[j] : it->second);
            }
            if (windowsOut) ++*windowsOut;
        }
        // The shader's own loads of these dwords now read the patched values.
        const auto& byK = loads[key];
        for (const auto& kv : fin)
        {
            const uint32_t u = next++;
            e.Op(spv::Op::OpBitcast, { tUint, u, kv.second });
            auto lk = byK.find(kv.first);
            if (lk != byK.end()) for (size_t li : lk->second) loadRepl[li] = u;
        }
    }

    // Hit counter (set 8 binding 1, uint[8]): one right-eye fragment in 64x64 adds 1 per window, [1] camera, [2] not.
    uint32_t vCount = 0, vFragCoord = 0, mergeLabel = 0, entryLabelId = 0;
    bool newFragCoord = false;
    if (counters && !hits.empty() && entryLabel != SIZE_MAX && entryTerm != SIZE_MAX)
    {
        if (fragCoordDeco && varClass.count(fragCoordDeco) && varClass[fragCoordDeco] == (uint32_t)spv::StorageClass::Input) vFragCoord = fragCoordDeco;
        if (!vFragCoord)
        {
            if (!tVec4) { tVec4 = next++; types.Op(spv::Op::OpTypeVector, { tVec4, tFloat, 4 }); }
            const uint32_t tPtrIn = next++;
            vFragCoord = next++;
            types.Op(spv::Op::OpTypePointer, { tPtrIn, (uint32_t)spv::StorageClass::Input, tVec4 });
            types.Op(spv::Op::OpVariable, { tPtrIn, vFragCoord, (uint32_t)spv::StorageClass::Input });
            deco.Op(spv::Op::OpDecorate, { vFragCoord, (uint32_t)spv::Decoration::BuiltIn, (uint32_t)spv::BuiltIn::FragCoord });
            newFragCoord = true;
        }
        if (!tVec4) { auto pt = ptrPointee.find(varPtr[vFragCoord]); tVec4 = pt == ptrPointee.end() ? 0 : pt->second; }
        const uint32_t tArr = next++, tC = next++, tPtrC = next++, tPtrCU = next++;
        vCount = next++;
        types.Op(spv::Op::OpTypeArray, { tArr, tUint, constU(16) });
        types.Op(spv::Op::OpTypeStruct, { tC, tArr });
        types.Op(spv::Op::OpTypePointer, { tPtrC, (uint32_t)spv::StorageClass::StorageBuffer, tC });
        types.Op(spv::Op::OpTypePointer, { tPtrCU, (uint32_t)spv::StorageClass::StorageBuffer, tUint });
        types.Op(spv::Op::OpVariable, { tPtrC, vCount, (uint32_t)spv::StorageClass::StorageBuffer });
        deco.Op(spv::Op::OpDecorate, { tArr, (uint32_t)spv::Decoration::ArrayStride, 4 });
        deco.Op(spv::Op::OpDecorate, { tC, (uint32_t)spv::Decoration::Block });
        deco.Op(spv::Op::OpMemberDecorate, { tC, 0, (uint32_t)spv::Decoration::Offset, 0 });
        deco.Op(spv::Op::OpDecorate, { vCount, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
        deco.Op(spv::Op::OpDecorate, { vCount, (uint32_t)spv::Decoration::Binding, 1 });
        Emit& e = codeAfter[counterAt];
        auto U = [&](spv::Op op, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(op, { tUint, r, a, b }); return r; };
        auto Bo = [&](spv::Op op, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(op, { tBool, r, a, b }); return r; };
        auto SelU = [&](uint32_t c, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(spv::Op::OpSelect, { tUint, r, c, a, b }); return r; };
        const uint32_t fc = next++, fx = next++, fy = next++, ux = next++, uy = next++;
        e.Op(spv::Op::OpLoad, { tVec4, fc, vFragCoord });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, fx, fc, 0 });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, fy, fc, 1 });
        e.Op(spv::Op::OpConvertFToU, { tUint, ux, fx });
        e.Op(spv::Op::OpConvertFToU, { tUint, uy, fy });
        const uint32_t c0 = constU(0), c63 = constU(63);
        const uint32_t onGrid = Bo(spv::Op::OpLogicalAnd, Bo(spv::Op::OpIEqual, U(spv::Op::OpBitwiseAnd, ux, c63), c0),
                                                         Bo(spv::Op::OpIEqual, U(spv::Op::OpBitwiseAnd, uy, c63), c0));
        const uint32_t sample = Bo(spv::Op::OpLogicalAnd, onGrid, activeAny);
        std::vector<uint32_t> slot;
        for (const Hit& h : hits)
            slot.push_back(SelU(h.ok, constU(1), constU(2)));
        const uint32_t then = next++;
        mergeLabel = next++;
        entryLabelId = code[ins[entryLabel].at + 1];
        e.Op(spv::Op::OpSelectionMerge, { mergeLabel, (uint32_t)spv::SelectionControlMask::MaskNone });
        e.Op(spv::Op::OpBranchConditional, { sample, then, mergeLabel });
        e.Op(spv::Op::OpLabel, { then });
        for (uint32_t sl : slot)
        {
            const uint32_t p = next++, r = next++;
            e.Op(spv::Op::OpAccessChain, { tPtrCU, p, vCount, c0, sl });
            e.Op(spv::Op::OpAtomicIAdd, { tUint, r, p, constU((uint32_t)spv::Scope::Device), c0, constU(1) });
        }
        // [6], [7]: the camera inverse's row 3 z/w, which the sprite VS turns its depth into w with (EmitSpriteShift).
        for (const Hit& h : hits)
            for (uint32_t c = 0; c < 2; ++c)
            {
                const uint32_t p = next++, old = next++, nb = next++, v = next++;
                e.Op(spv::Op::OpAccessChain, { tPtrCU, p, vCount, c0, constU(6 + c) });
                e.Op(spv::Op::OpLoad, { tUint, old, p });
                e.Op(spv::Op::OpBitcast, { tUint, nb, c ? h.n33 : h.n32 });
                e.Op(spv::Op::OpSelect, { tUint, v, h.ok, nb, old });
                e.Op(spv::Op::OpStore, { p, v });
            }
        e.Op(spv::Op::OpBranch, { mergeLabel });
        e.Op(spv::Op::OpLabel, { mergeLabel });
    }

    out.clear();
    out.reserve(words + 4096);
    out.insert(out.end(), code, code + 5);
    const bool listAllGlobals = code[1] >= 0x00010400;   // SPIR-V 1.4: the interface lists every global
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (k == entryIdx && (listAllGlobals || newFragCoord))
        {
            std::vector<uint32_t> ep(i, i + ins[k].n);
            if (listAllGlobals) { ep.push_back(vStereo); if (vCount) ep.push_back(vCount); }
            if (newFragCoord) ep.push_back(vFragCoord);
            ep[0] = ((uint32_t)ep.size() << 16) | (uint32_t)spv::Op::OpEntryPoint;
            out.insert(out.end(), ep.begin(), ep.end());
            continue;
        }
        // The entry block now ends in the counter's merge block: phis naming it as a parent follow.
        if (mergeLabel && ins[k].op == spv::Op::OpPhi && k > fnStart && k < fnEnd)
        {
            std::vector<uint32_t> phi(i, i + ins[k].n);
            for (size_t a = 4; a < phi.size(); a += 2) if (phi[a] == entryLabelId) phi[a] = mergeLabel;
            out.insert(out.end(), phi.begin(), phi.end());
            continue;
        }
        auto lr = loadRepl.find(k);
        if (lr != loadRepl.end())
        {
            const uint32_t cp[4] = { (4u << 16) | (uint32_t)spv::Op::OpCopyObject, i[1], i[2], lr->second };
            out.insert(out.end(), cp, cp + 4);
            continue;
        }
        out.insert(out.end(), i, i + ins[k].n);
        auto ca = codeAfter.find(k);
        if (ca != codeAfter.end()) out.insert(out.end(), ca->second.v.begin(), ca->second.v.end());
    }
    out[3] = next;
    return true;
}

// A light shaft's FS (the shadPS4 family's CameraShaftFS list): the sprite's stores its depth in set 8 binding 1 [8];
// blur, the pass blurring it toward a centre at dwords 68, 69, moves that centre's u as the sprite moved.
bool PatchShaftFS(const uint32_t* code, size_t words, bool blur, uint32_t stereoSet, std::vector<uint32_t>& out)
{
    // SPIR-V 1.4+: the StorageBuffer class, and every global on the entry point.
    if (words < 5 || code[0] != spv::MagicNumber || code[1] < 0x00010400) return false;
    std::vector<Ins> ins;
    for (size_t i = 5; i < words; )
    {
        const uint16_t n = (uint16_t)(code[i] >> 16);
        if (!n || i + n > words) return false;
        ins.push_back({ i, n, (spv::Op)(code[i] & 0xFFFF) });
        i += n;
    }
    std::unordered_map<uint32_t, size_t> def;
    std::unordered_map<uint32_t, uint32_t> uconst, uconstOf, varClass;
    std::unordered_set<uint32_t> location0;
    uint32_t tFloat = 0, tUint = 0, tBool = 0, tPtrUF = 0;
    size_t entryIdx = SIZE_MAX, firstType = SIZE_MAX, firstFunc = SIZE_MAX;
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(ins[k].op, &hasRes, &hasType);
        if (hasRes) def[hasType ? i[2] : i[1]] = k;
        switch (ins[k].op)
        {
        case spv::Op::OpEntryPoint: if (i[1] == (uint32_t)spv::ExecutionModel::Fragment && entryIdx == SIZE_MAX) entryIdx = k; break;
        case spv::Op::OpDecorate: if (ins[k].n >= 4 && i[2] == (uint32_t)spv::Decoration::Location && i[3] == 0) location0.insert(i[1]); break;
        case spv::Op::OpTypeFloat: if (i[2] == 32 && !tFloat) tFloat = i[1]; break;
        case spv::Op::OpTypeInt: if (i[2] == 32 && i[3] == 0 && !tUint) tUint = i[1]; break;
        case spv::Op::OpTypeBool: if (!tBool) tBool = i[1]; break;
        case spv::Op::OpTypePointer: if (i[2] == (uint32_t)spv::StorageClass::Uniform && tFloat && i[3] == tFloat && !tPtrUF) tPtrUF = i[1]; break;
        case spv::Op::OpConstant: if (tUint && i[1] == tUint) { uconst[i[2]] = i[3]; if (!uconstOf.count(i[3])) uconstOf[i[3]] = i[2]; } break;
        case spv::Op::OpVariable: if (firstFunc == SIZE_MAX) varClass[i[2]] = i[3]; break;
        case spv::Op::OpFunction: if (firstFunc == SIZE_MAX) firstFunc = k; break;
        default: break;
        }
        const bool typeSection = (ins[k].op >= spv::Op::OpTypeVoid && ins[k].op <= spv::Op::OpTypeForwardPointer) ||
                                 (ins[k].op >= spv::Op::OpConstantTrue && ins[k].op <= spv::Op::OpSpecConstantOp) ||
                                 (ins[k].op == spv::Op::OpVariable && firstFunc == SIZE_MAX);
        if (typeSection && firstType == SIZE_MAX) firstType = k;
    }
    if (entryIdx == SIZE_MAX || !tFloat || !tUint || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;

    // The access chain of a load, when it has base[a][b] form.
    auto chainOf = [&](uint32_t ptr) -> const uint32_t*
    {
        auto pd = def.find(ptr);
        if (pd == def.end() || ins[pd->second].op != spv::Op::OpAccessChain || ins[pd->second].n != 6) return nullptr;
        return code + ins[pd->second].at;
    };
    // The blur: the one float cast of dword 68. The sprite: its first read of attr0[0].z.
    size_t at = SIZE_MAX;
    for (size_t k = firstFunc; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        if (blur)
        {
            if (ins[k].op != spv::Op::OpBitcast || i[1] != tFloat) continue;
            auto ld = def.find(i[3]);
            if (ld == def.end() || ins[ld->second].op != spv::Op::OpLoad) continue;
            const uint32_t* a = chainOf(code[ins[ld->second].at + 3]);
            if (!a) continue;
            auto ad = def.find(a[5]);
            if (ad == def.end() || ins[ad->second].op != spv::Op::OpIAdd) continue;
            const uint32_t* add = code + ins[ad->second].at;
            auto c3 = uconst.find(add[3]), c4 = uconst.find(add[4]);
            if (!((c3 != uconst.end() && c3->second == 68) || (c4 != uconst.end() && c4->second == 68))) continue;
            if (at != SIZE_MAX) return false;
            at = k;
        }
        else
        {
            if (ins[k].op != spv::Op::OpLoad || i[1] != tFloat) continue;
            const uint32_t* a = chainOf(i[3]);
            if (!a || !location0.count(a[3])) continue;
            auto vc = varClass.find(a[3]);
            auto c0 = uconst.find(a[4]), c1 = uconst.find(a[5]);
            if (vc == varClass.end() || vc->second != (uint32_t)spv::StorageClass::Input || c0 == uconst.end() || c1 == uconst.end() || c0->second != 0 || c1->second != 2) continue;
            at = k;
            break;
        }
    }
    if (at == SIZE_MAX) return false;

    uint32_t next = code[3];
    Emit types, deco, e;
    auto constU = [&](uint32_t v)
    {
        auto it = uconstOf.find(v);
        if (it != uconstOf.end()) return it->second;
        const uint32_t id = next++;
        types.Op(spv::Op::OpConstant, { tUint, id, v });
        uconstOf[v] = id;
        return id;
    };
    const uint32_t tArr = next++, tC = next++, tPtrC = next++, tPtrCU = next++, vCount = next++;
    types.Op(spv::Op::OpTypeArray, { tArr, tUint, constU(16) });
    types.Op(spv::Op::OpTypeStruct, { tC, tArr });
    types.Op(spv::Op::OpTypePointer, { tPtrC, (uint32_t)spv::StorageClass::StorageBuffer, tC });
    types.Op(spv::Op::OpTypePointer, { tPtrCU, (uint32_t)spv::StorageClass::StorageBuffer, tUint });
    types.Op(spv::Op::OpVariable, { tPtrC, vCount, (uint32_t)spv::StorageClass::StorageBuffer });
    deco.Op(spv::Op::OpDecorate, { tArr, (uint32_t)spv::Decoration::ArrayStride, 4 });
    deco.Op(spv::Op::OpDecorate, { tC, (uint32_t)spv::Decoration::Block });
    deco.Op(spv::Op::OpMemberDecorate, { tC, 0, (uint32_t)spv::Decoration::Offset, 0 });
    deco.Op(spv::Op::OpDecorate, { vCount, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vCount, (uint32_t)spv::Decoration::Binding, 1 });
    auto slot = [&](uint32_t s) { const uint32_t p = next++; e.Op(spv::Op::OpAccessChain, { tPtrCU, p, vCount, constU(0), constU(s) }); return p; };

    uint32_t vStereo = 0, raw = 0;
    if (!blur)
    {
        const uint32_t bits = next++;
        e.Op(spv::Op::OpBitcast, { tUint, bits, code[ins[at].at + 2] });
        e.Op(spv::Op::OpStore, { slot(8), bits });
    }
    else
    {
        if (!tBool) { tBool = next++; types.Op(spv::Op::OpTypeBool, { tBool }); }
        const uint32_t tS = next++, tPtrS = next++;
        vStereo = next++;
        // sep, conv, wMin, eye, fsSep
        types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat, tFloat, tFloat });
        types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
        if (!tPtrUF) { tPtrUF = next++; types.Op(spv::Op::OpTypePointer, { tPtrUF, (uint32_t)spv::StorageClass::Uniform, tFloat }); }
        types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
        deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
        for (uint32_t k = 0; k < 5; ++k) deco.Op(spv::Op::OpMemberDecorate, { tS, k, (uint32_t)spv::Decoration::Offset, k * 4 });
        deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
        deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });
        auto newConstF = [&](float f) { const uint32_t id = next++; types.Op(spv::Op::OpConstant, { tFloat, id, Bits(f) }); return id; };
        const uint32_t f0 = newConstF(0.f), f1 = newConstF(1.f), f2 = newConstF(2.f), fHalf = newConstF(0.5f);
        auto F = [&](spv::Op op, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(op, { tFloat, r, a, b }); return r; };
        auto Bo = [&](spv::Op op, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(op, { tBool, r, a, b }); return r; };
        auto Param = [&](uint32_t member) { const uint32_t p = next++, v = next++; e.Op(spv::Op::OpAccessChain, { tPtrUF, p, vStereo, constU(member) }); e.Op(spv::Op::OpLoad, { tFloat, v, p }); return v; };
        auto Slot = [&](uint32_t s) { const uint32_t p = slot(s), u = next++, v = next++; e.Op(spv::Op::OpLoad, { tUint, u, p }); e.Op(spv::Op::OpBitcast, { tFloat, v, u }); return v; };
        raw = next++;
        const uint32_t sep = Param(4), conv = Param(1);
        // EmitSpriteShift's move, in uv: nothing for the left eye, or before the sprite has stored a depth (invW is
        // then the camera's own w term).
        const uint32_t invW = F(spv::Op::OpFAdd, F(spv::Op::OpFMul, Slot(6), Slot(8)), Slot(7));
        uint32_t ok = Bo(spv::Op::OpFOrdNotEqual, sep, f0);
        ok = Bo(spv::Op::OpLogicalAnd, ok, Bo(spv::Op::OpFOrdGreaterThan, invW, f0));
        ok = Bo(spv::Op::OpLogicalAnd, ok, Bo(spv::Op::OpFOrdLessThan, invW, f2));
        const uint32_t du0 = F(spv::Op::OpFMul, fHalf, F(spv::Op::OpFMul, sep, F(spv::Op::OpFSub, f1, F(spv::Op::OpFMul, conv, invW))));
        const uint32_t du = next++;
        e.Op(spv::Op::OpSelect, { tFloat, du, ok, du0, f0 });
        e.Op(spv::Op::OpFAdd, { tFloat, code[ins[at].at + 2], raw, du });   // the cast's id now names the moved centre
    }

    out.clear();
    out.reserve(words + 512);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (k == entryIdx)
        {
            std::vector<uint32_t> ep(i, i + ins[k].n);
            ep.push_back(vCount);
            if (vStereo) ep.push_back(vStereo);
            ep[0] = ((uint32_t)ep.size() << 16) | (uint32_t)spv::Op::OpEntryPoint;
            out.insert(out.end(), ep.begin(), ep.end());
            continue;
        }
        if (k == at && blur)
        {
            const uint32_t cast[4] = { i[0], i[1], raw, i[3] };
            out.insert(out.end(), cast, cast + 4);
        }
        else out.insert(out.end(), i, i + ins[k].n);
        if (k == at) out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    return true;
}

// VKLightRaysCamera: a light-volume FS rebuilds positions as left-camera ray * d; each ray moves by its slope in left
// NDC x, R0*(w/d)/P00, times the gap between the volume's and the surface's right-eye x (light-space rays rotated).
bool PatchLightRaysFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* raysOut, unsigned* usesOut)
{
    if (raysOut) *raysOut = 0;
    if (usesOut) *usesOut = 0;
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
    std::unordered_map<uint32_t, uint32_t> uconst, fconst, varClass, varPtr, ptrPointee;
    uint32_t tFloat = 0, tUint = 0, tBool = 0, tVec4 = 0, tPtrUF = 0, glsl = 0, entryFn = 0, fragCoordDeco = 0;
    size_t entryIdx = SIZE_MAX, firstType = SIZE_MAX, firstFunc = SIZE_MAX;
    for (size_t k = 0; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(ins[k].op, &hasRes, &hasType);
        if (hasRes) def[hasType ? i[2] : i[1]] = k;
        switch (ins[k].op)
        {
        case spv::Op::OpExtInstImport: if (ins[k].n >= 3 && !strcmp((const char*)&i[2], "GLSL.std.450")) glsl = i[1]; break;
        case spv::Op::OpEntryPoint: if (i[1] == (uint32_t)spv::ExecutionModel::Fragment && !entryFn) { entryFn = i[2]; entryIdx = k; } break;
        case spv::Op::OpDecorate:
            if (ins[k].n >= 4 && i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::FragCoord) fragCoordDeco = i[1];
            break;
        case spv::Op::OpTypeFloat: if (i[2] == 32 && !tFloat) tFloat = i[1]; break;
        case spv::Op::OpTypeInt: if (i[2] == 32 && i[3] == 0 && !tUint) tUint = i[1]; break;
        case spv::Op::OpTypeBool: if (!tBool) tBool = i[1]; break;
        case spv::Op::OpTypeVector: if (tFloat && i[2] == tFloat && i[3] == 4 && !tVec4) tVec4 = i[1]; break;
        case spv::Op::OpTypePointer:
            ptrPointee[i[1]] = i[3];
            if (i[2] == (uint32_t)spv::StorageClass::Uniform && tFloat && i[3] == tFloat && !tPtrUF) tPtrUF = i[1];
            break;
        case spv::Op::OpConstant:
            if (ins[k].n == 4 && tUint && i[1] == tUint) uconst[i[2]] = i[3];
            if (ins[k].n == 4 && tFloat && i[1] == tFloat) fconst[i[2]] = i[3];
            break;
        case spv::Op::OpVariable: if (firstFunc == SIZE_MAX) { varClass[i[2]] = i[3]; varPtr[i[2]] = i[1]; } break;
        case spv::Op::OpFunction: if (firstFunc == SIZE_MAX) firstFunc = k; break;
        default: break;
        }
        const bool typeSection = (ins[k].op >= spv::Op::OpTypeVoid && ins[k].op <= spv::Op::OpTypeForwardPointer) ||
                                 (ins[k].op >= spv::Op::OpConstantTrue && ins[k].op <= spv::Op::OpSpecConstantOp) ||
                                 (ins[k].op == spv::Op::OpVariable && firstFunc == SIZE_MAX);
        if (typeSection && firstType == SIZE_MAX) firstType = k;
    }
    if (!entryFn || !tFloat || !tUint || !glsl || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;

    // The entry function and its entry block: the corrected rays are computed there, before any branch.
    size_t fnStart = SIZE_MAX, fnEnd = SIZE_MAX, entryLabel = SIZE_MAX, entryTerm = SIZE_MAX;
    for (size_t k = firstFunc; k < ins.size(); ++k)
    {
        const uint32_t* i = code + ins[k].at;
        const spv::Op op = ins[k].op;
        if (fnStart == SIZE_MAX) { if (op == spv::Op::OpFunction && i[2] == entryFn) fnStart = k; continue; }
        if (op == spv::Op::OpLabel && entryLabel == SIZE_MAX) entryLabel = k;
        else if (entryLabel != SIZE_MAX && entryTerm == SIZE_MAX &&
                 (op == spv::Op::OpBranch || op == spv::Op::OpBranchConditional || op == spv::Op::OpSwitch || op == spv::Op::OpReturn ||
                  op == spv::Op::OpReturnValue || op == spv::Op::OpKill || op == spv::Op::OpUnreachable ||
                  op == spv::Op::OpTerminateInvocation || op == spv::Op::OpSelectionMerge || op == spv::Op::OpLoopMerge))
            entryTerm = k;
        if (op == spv::Op::OpFunctionEnd) { fnEnd = k; break; }
    }
    if (fnStart == SIZE_MAX || fnEnd == SIZE_MAX || entryLabel == SIZE_MAX || entryTerm == SIZE_MAX) return false;
    auto inEntry = [&](uint32_t id) { auto it = def.find(id); return it != def.end() && it->second > entryLabel && it->second < entryTerm; };
    auto D = [&](uint32_t id) -> const Ins* { auto it = def.find(id); return it == def.end() ? nullptr : &ins[it->second]; };
    auto W = [&](const Ins* in) { return code + in->at; };
    auto isFma = [&](const Ins* in) { return in && in->op == spv::Op::OpExtInst && in->n == 8 && W(in)[3] == glsl && W(in)[4] == 50; };

    // An interpolated input component: shadPS4's Fma(v2 - v0, b2, Fma(v1 - v0, b1, v0)) over a per-vertex input, or a
    // plain Load(AccessChain(input, c)).
    auto loadOf = [&](uint32_t v, uint32_t& var, uint32_t& comp, bool perVertex) -> bool
    {
        const Ins* ld = D(v);
        if (!ld || ld->op != spv::Op::OpLoad || ld->n < 4) return false;
        const Ins* ac = D(W(ld)[3]);
        if (!ac || (ac->op != spv::Op::OpAccessChain && ac->op != spv::Op::OpInBoundsAccessChain) || ac->n != (perVertex ? 6 : 5)) return false;
        const uint32_t base = W(ac)[3];
        auto vc = varClass.find(base);
        if (vc == varClass.end() || vc->second != (uint32_t)spv::StorageClass::Input) return false;
        auto c = uconst.find(W(ac)[ac->n - 1]);
        if (c == uconst.end() || c->second > 3) return false;
        var = base; comp = c->second;
        return true;
    };
    auto interp = [&](uint32_t v, uint32_t& var, uint32_t& comp) -> bool
    {
        const Ins* f = D(v);
        if (isFma(f))
        {
            const Ins* sub = D(W(f)[5]);
            if (!sub || sub->op != spv::Op::OpFSub || sub->n != 5 || !isFma(D(W(f)[7]))) return false;
            return loadOf(W(sub)[3], var, comp, true);
        }
        return f && f->op == spv::Op::OpLoad && loadOf(v, var, comp, false);
    };

    // The screen-position varying: the x of a sample coordinate, through its scale and offset (constants or bitcast
    // buffer dwords), e.g. uv.x = buf[100] * ndc.x + buf[102].
    auto scaled = [&](uint32_t a, uint32_t b) { const Ins* ia = D(a); return (fconst.count(a) || (ia && ia->op == spv::Op::OpBitcast)) ? b : a; };
    auto screenX = [&](uint32_t u, uint32_t& xs, uint32_t& var) -> bool
    {
        for (int depth = 0; depth < 3; ++depth)
        {
            uint32_t comp = 0;
            if (interp(u, var, comp)) { xs = u; return comp == 0; }
            const Ins* in = D(u);
            if (!in) return false;
            if ((in->op == spv::Op::OpFAdd || in->op == spv::Op::OpFMul) && in->n == 5) u = scaled(W(in)[3], W(in)[4]);
            else if (isFma(in)) u = scaled(W(in)[5], W(in)[6]);
            else return false;
        }
        return false;
    };
    uint32_t xs = 0, sVar = 0, invD = 0, dVal = 0;
    size_t dIdx = SIZE_MAX, qIdx = SIZE_MAX;
    for (size_t k = entryLabel; k < entryTerm && !dVal; ++k)
    {
        if (ins[k].op != spv::Op::OpImageSampleExplicitLod && ins[k].op != spv::Op::OpImageSampleImplicitLod) continue;
        const uint32_t* s = code + ins[k].at;
        const Ins* cc = D(s[4]);
        if (!cc || cc->op != spv::Op::OpCompositeConstruct || cc->n < 5) continue;
        uint32_t x = 0, var = 0;
        if (!screenX(W(cc)[3], x, var)) continue;
        // depth = sample.x; invD = a * depth + b (FAdd(FMul) or Fma); d = 1 / invD.
        for (size_t j = k + 1; j < entryTerm && !dVal; ++j)
        {
            const uint32_t* e = code + ins[j].at;
            if (ins[j].op != spv::Op::OpCompositeExtract || ins[j].n != 5 || e[3] != s[2] || e[4] != 0) continue;
            const uint32_t dep = e[2];
            for (size_t m = j + 1; m < entryTerm && !dVal; ++m)
            {
                const uint32_t* q = code + ins[m].at;
                bool lin = false;
                if (ins[m].op == spv::Op::OpFAdd && ins[m].n == 5)
                    for (int o = 3; o <= 4 && !lin; ++o) { const Ins* fm = D(q[o]); lin = fm && fm->op == spv::Op::OpFMul && (W(fm)[3] == dep || W(fm)[4] == dep); }
                if (isFma(&ins[m]) && (q[5] == dep || q[6] == dep)) lin = true;
                if (!lin) continue;
                for (size_t r = m + 1; r < entryTerm; ++r)
                {
                    const uint32_t* dv = code + ins[r].at;
                    if (ins[r].op == spv::Op::OpFDiv && ins[r].n == 5 && dv[4] == q[2] && fconst.count(dv[3]) && fconst[dv[3]] == Bits(1.f))
                    { xs = x; sVar = var; invD = q[2]; dVal = dv[2]; dIdx = r; qIdx = m; break; }
                }
            }
        }
    }
    if (!dVal) return false;

    // A buffer dword: Bitcast(Load(AccessChain(block, member, IAdd(K, offset)))).
    struct DwordRef { uint32_t ptrType = 0, var = 0, member = 0, off = 0, k = 0, u32 = 0; };
    auto dwordRef = [&](uint32_t id, DwordRef& ref) -> bool
    {
        const Ins* bc = D(id);
        if (!bc || bc->op != spv::Op::OpBitcast || bc->n != 4) return false;
        const Ins* ld = D(W(bc)[3]);
        if (!ld || ld->op != spv::Op::OpLoad || ld->n < 4) return false;
        const Ins* ac = D(W(ld)[3]);
        if (!ac || ac->op != spv::Op::OpAccessChain || ac->n != 6) return false;
        const Ins* ia = D(W(ac)[5]);
        if (!ia || ia->op != spv::Op::OpIAdd || ia->n != 5) return false;
        uint32_t kid = W(ia)[3], off = W(ia)[4];
        if (!uconst.count(kid)) std::swap(kid, off);
        if (!uconst.count(kid)) return false;
        ref = { W(ac)[1], W(ac)[3], W(ac)[4], off, uconst[kid], W(ld)[1] };
        return true;
    };
    // The camera block: the one whose dwords 118/119 are invD's a and b.
    DwordRef cam;
    bool haveCam = false;
    {
        const uint32_t* q = code + ins[qIdx].at;
        std::vector<uint32_t> leaves;
        auto addLeaves = [&](uint32_t id) { leaves.push_back(id); const Ins* in = D(id); if (in && in->op == spv::Op::OpFMul && in->n == 5) { leaves.push_back(W(in)[3]); leaves.push_back(W(in)[4]); } };
        if (ins[qIdx].op == spv::Op::OpFAdd) { addLeaves(q[3]); addLeaves(q[4]); }
        else { leaves.push_back(q[5]); leaves.push_back(q[6]); leaves.push_back(q[7]); }
        for (uint32_t id : leaves) { DwordRef r; if (dwordRef(id, r) && (r.k == 118 || r.k == 119)) { cam = r; haveCam = true; break; } }
    }
    if (!haveCam) return false;

    // Rays: other interpolated components used as v * d, -v * d or Fma(v, d, .).
    std::vector<uint32_t> rays;
    auto addRay = [&](uint32_t v)
    {
        uint32_t var = 0, comp = 0;
        if (!interp(v, var, comp)) { const Ins* ng = D(v); if (!ng || ng->op != spv::Op::OpFNegate) return; v = W(ng)[3]; if (!interp(v, var, comp)) return; }
        if (var == sVar || !inEntry(v)) return;
        if (std::find(rays.begin(), rays.end(), v) == rays.end()) rays.push_back(v);
    };
    for (size_t k = fnStart; k < fnEnd; ++k)
    {
        const uint32_t* i = code + ins[k].at;
        if (ins[k].op == spv::Op::OpFMul && ins[k].n == 5) { if (i[3] == dVal) addRay(i[4]); else if (i[4] == dVal) addRay(i[3]); }
        else if (isFma(&ins[k])) { if (i[5] == dVal) addRay(i[6]); else if (i[6] == dVal) addRay(i[5]); }
    }
    // Each ray's partner, the dword added to (+-ray) * d: the light's camera-relative position (40-42), a camera ray;
    // the camera's light-space position (44-46) a light-space ray.
    auto rayOf = [&](uint32_t v) -> uint32_t
    {
        if (std::find(rays.begin(), rays.end(), v) != rays.end()) return v;
        const Ins* ng = D(v);
        if (ng && ng->op == spv::Op::OpFNegate && ng->n == 4 && std::find(rays.begin(), rays.end(), W(ng)[3]) != rays.end()) return W(ng)[3];
        return 0;
    };
    auto timesD = [&](uint32_t id) -> uint32_t
    {
        const Ins* m = D(id);
        if (!m || m->op != spv::Op::OpFMul || m->n != 5) return 0;
        if (W(m)[3] == dVal) return rayOf(W(m)[4]);
        if (W(m)[4] == dVal) return rayOf(W(m)[3]);
        return 0;
    };
    std::unordered_map<uint32_t, DwordRef> partner;
    for (size_t k = fnStart; k < fnEnd; ++k)
    {
        const uint32_t* i = code + ins[k].at;
        DwordRef ref;
        if (ins[k].op == spv::Op::OpFAdd && ins[k].n == 5)
        {
            for (int o = 0; o < 2; ++o) { const uint32_t r = timesD(i[3 + o]); if (r && dwordRef(i[4 - o], ref)) partner[r] = ref; }
        }
        else if (isFma(&ins[k]))
        {
            const uint32_t r = i[5] == dVal ? rayOf(i[6]) : i[6] == dVal ? rayOf(i[5]) : 0;
            if (r && dwordRef(i[7], ref)) partner[r] = ref;
        }
    }
    struct Fix { uint32_t ray, comp; bool light; };
    std::vector<Fix> fixes;
    DwordRef lightBlock;
    bool haveLight = false;
    for (uint32_t r : rays)
    {
        auto p = partner.find(r);
        uint32_t var = 0, comp = 0;
        if (p == partner.end() || !interp(r, var, comp) || comp > 2) continue;
        if (p->second.k == 40 + comp) fixes.push_back({ r, comp, false });
        else if (p->second.k == 44 + comp) { fixes.push_back({ r, comp, true }); lightBlock = p->second; haveLight = true; }
    }
    if (fixes.empty() || !inEntry(xs)) return false;
    size_t at = dIdx;
    for (const Fix& f : fixes) at = std::max(at, def[f.ray]);
    at = std::max(at, def[xs]);
    if (at >= entryTerm) return false;

    uint32_t next = code[3];
    Emit deco, types, e;
    auto newConstF = [&](float f) { const uint32_t id = next++; types.Op(spv::Op::OpConstant, { tFloat, id, Bits(f) }); return id; };
    auto newConstU = [&](uint32_t v) { for (const auto& c : uconst) if (c.second == v) return c.first; const uint32_t id = next++; types.Op(spv::Op::OpConstant, { tUint, id, v }); uconst[id] = v; return id; };
    if (!tPtrUF) { tPtrUF = next++; types.Op(spv::Op::OpTypePointer, { tPtrUF, (uint32_t)spv::StorageClass::Uniform, tFloat }); }
    const uint32_t tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat, tFloat, tFloat });   // sep, conv, wMin, eye, fsSep
    types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
    types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
    deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
    for (uint32_t k = 0; k < 5; ++k) deco.Op(spv::Op::OpMemberDecorate, { tS, k, (uint32_t)spv::Decoration::Offset, k * 4 });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });
    uint32_t vFragCoord = (fragCoordDeco && varClass.count(fragCoordDeco) && varClass[fragCoordDeco] == (uint32_t)spv::StorageClass::Input) ? fragCoordDeco : 0;
    const bool newFragCoord = !vFragCoord;
    if (newFragCoord)
    {
        if (!tVec4) { tVec4 = next++; types.Op(spv::Op::OpTypeVector, { tVec4, tFloat, 4 }); }
        const uint32_t tPtrIn = next++;
        vFragCoord = next++;
        types.Op(spv::Op::OpTypePointer, { tPtrIn, (uint32_t)spv::StorageClass::Input, tVec4 });
        types.Op(spv::Op::OpVariable, { tPtrIn, vFragCoord, (uint32_t)spv::StorageClass::Input });
        deco.Op(spv::Op::OpDecorate, { vFragCoord, (uint32_t)spv::Decoration::BuiltIn, (uint32_t)spv::BuiltIn::FragCoord });
    }
    else if (!tVec4) { auto pt = ptrPointee.find(varPtr[vFragCoord]); tVec4 = pt == ptrPointee.end() ? 0 : pt->second; }
    if (!tVec4) return false;
    const uint32_t f1 = newConstF(1.f), cU0 = newConstU(0), cU1 = newConstU(1), cU4 = newConstU(4);

    auto F = [&](spv::Op op, uint32_t a, uint32_t b) { const uint32_t r = next++; e.Op(op, { tFloat, r, a, b }); return r; };
    auto Uni = [&](uint32_t member) { const uint32_t p = next++, v = next++; e.Op(spv::Op::OpAccessChain, { tPtrUF, p, vStereo, member }); e.Op(spv::Op::OpLoad, { tFloat, v, p }); return v; };
    auto Dword = [&](const DwordRef& r, uint32_t K)
    {
        const uint32_t ia = next++, ac = next++, ld = next++, f = next++;
        e.Op(spv::Op::OpIAdd, { r.u32, ia, newConstU(K), r.off });
        e.Op(spv::Op::OpAccessChain, { r.ptrType, ac, r.var, r.member, ia });
        e.Op(spv::Op::OpLoad, { r.u32, ld, ac });
        e.Op(spv::Op::OpBitcast, { tFloat, f, ld });
        return f;
    };
    const uint32_t sep = Uni(cU0), conv = Uni(cU1), fsSep = Uni(cU4);
    const uint32_t fc = next++, fw = next++;
    e.Op(spv::Op::OpLoad, { tVec4, fc, vFragCoord });
    e.Op(spv::Op::OpCompositeExtract, { tFloat, fw, fc, 3 });
    // d is view z, not clip w (a P[59] of -1 makes it negative): w = P[59]*d + P[63].
    const uint32_t wPerD = F(spv::Op::OpFAdd, Dword(cam, 59), F(spv::Op::OpFMul, Dword(cam, 63), invD));
    const uint32_t wInv = F(spv::Op::OpFDiv, invD, wPerD);
    // k = sep*(1 - conv*FragCoord.w) - fsSep*(1 - conv/w): where the volume's left ray points minus the surface (NDC).
    const uint32_t k = F(spv::Op::OpFSub, F(spv::Op::OpFMul, sep, F(spv::Op::OpFSub, f1, F(spv::Op::OpFMul, conv, fw))),
                                          F(spv::Op::OpFMul, fsSep, F(spv::Op::OpFSub, f1, F(spv::Op::OpFMul, conv, wInv))));
    // Rx = R0 * (w/d) / P00: the left ray's change per unit of left NDC x.
    const uint32_t scale = F(spv::Op::OpFDiv, wPerD, Dword(cam, 48));
    uint32_t rx[3], lrx[3] = {};
    for (uint32_t j = 0; j < 3; ++j) rx[j] = F(spv::Op::OpFMul, Dword(cam, 32 + j), scale);
    if (haveLight)
        for (uint32_t c = 0; c < 3; ++c)
            lrx[c] = F(spv::Op::OpFAdd, F(spv::Op::OpFAdd, F(spv::Op::OpFMul, Dword(lightBlock, 24 + c), rx[0]), F(spv::Op::OpFMul, Dword(lightBlock, 28 + c), rx[1])),
                       F(spv::Op::OpFMul, Dword(lightBlock, 32 + c), rx[2]));
    std::unordered_map<uint32_t, uint32_t> repl;
    for (const Fix& f : fixes) repl[f.ray] = F(spv::Op::OpFAdd, f.ray, F(spv::Op::OpFMul, f.light ? lrx[f.comp] : rx[f.comp], k));

    // Later uses take the corrected values, in operand slots that are ids.
    unsigned uses = 0;
    auto idSlots = [&](spv::Op op, uint16_t n, std::vector<uint16_t>& slots)
    {
        slots.clear();
        switch (op)
        {
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFDiv: case spv::Op::OpDot:
        case spv::Op::OpVectorTimesScalar: case spv::Op::OpFOrdEqual: case spv::Op::OpFOrdNotEqual: case spv::Op::OpFOrdLessThan:
        case spv::Op::OpFOrdGreaterThan: case spv::Op::OpFOrdLessThanEqual: case spv::Op::OpFOrdGreaterThanEqual:
            if (n == 5) slots = { 3, 4 }; break;
        case spv::Op::OpFNegate: case spv::Op::OpBitcast: if (n == 4) slots = { 3 }; break;
        case spv::Op::OpSelect: if (n == 6) slots = { 3, 4, 5 }; break;
        case spv::Op::OpExtInst: for (uint16_t s = 5; s < n; ++s) slots.push_back(s); break;
        case spv::Op::OpCompositeConstruct: for (uint16_t s = 3; s < n; ++s) slots.push_back(s); break;
        case spv::Op::OpCompositeInsert: if (n >= 5) slots = { 3, 4 }; break;
        case spv::Op::OpStore: if (n >= 3) slots = { 2 }; break;
        case spv::Op::OpPhi: for (uint16_t s = 3; s < n; s += 2) slots.push_back(s); break;
        default: break;
        }
    };

    out.clear();
    out.reserve(words + 1024);
    out.insert(out.end(), code, code + 5);
    const bool listAllGlobals = code[1] >= 0x00010400;
    std::vector<uint16_t> slots;
    for (size_t kk = 0; kk < ins.size(); ++kk)
    {
        const uint32_t* i = code + ins[kk].at;
        if (kk == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (kk == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (kk == entryIdx && (listAllGlobals || newFragCoord))
        {
            std::vector<uint32_t> ep(i, i + ins[kk].n);
            if (listAllGlobals) ep.push_back(vStereo);
            if (newFragCoord) ep.push_back(vFragCoord);
            ep[0] = ((uint32_t)ep.size() << 16) | (uint32_t)spv::Op::OpEntryPoint;
            out.insert(out.end(), ep.begin(), ep.end());
            continue;
        }
        if (kk > at && kk < fnEnd)
        {
            idSlots(ins[kk].op, ins[kk].n, slots);
            if (!slots.empty())
            {
                std::vector<uint32_t> w(i, i + ins[kk].n);
                for (uint16_t s : slots) { auto it = repl.find(w[s]); if (it != repl.end()) { w[s] = it->second; ++uses; } }
                out.insert(out.end(), w.begin(), w.end());
                continue;
            }
        }
        out.insert(out.end(), i, i + ins[kk].n);
        if (kk == at) out.insert(out.end(), e.v.begin(), e.v.end());
    }
    if (!uses) return false;
    out[3] = next;
    if (raysOut) *raysOut = (unsigned)fixes.size();
    if (usesOut) *usesOut = uses;
    return true;
}

}} // namespace wiz3d::vk
