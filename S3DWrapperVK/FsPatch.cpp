/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The fragment and compute patchers of FsPatch.h: screen passes rebuilding positions, rays and lookups the left camera
// built, corrected for the right eye (eye*right from stereo set member 3, 0 in the left eye).

#define SPV_ENABLE_UTILITY_CODE
#include <spirv/unified1/spirv.hpp11>
#include <spirv/unified1/GLSL.std.450.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include "FsPatch.h"

namespace wiz3d { namespace vk {

namespace {

struct Ins { size_t at; uint16_t n; spv::Op op; };
inline bool TypeSection(spv::Op op, bool beforeFunctions);

struct Fs
{
    const uint32_t* w = nullptr;
    std::vector<Ins> ins;
    std::unordered_map<uint32_t, size_t> def;          // result id -> instruction index
    std::unordered_map<uint32_t, uint32_t> varClass;   // variable -> storage class

    // The module as an instruction list with every result id's defining instruction; false on anything malformed.
    bool Split(const uint32_t* code, size_t words)
    {
        if (words < 5 || code[0] != spv::MagicNumber) return false;
        w = code;
        for (size_t i = 5; i < words; )
        {
            const uint16_t n = (uint16_t)(code[i] >> 16);
            if (!n || i + n > words) return false;
            bool hasRes = false, hasType = false;
            spv::HasResultAndType((spv::Op)(code[i] & 0xFFFF), &hasRes, &hasType);
            if (hasRes) def[hasType ? code[i + 2] : code[i + 1]] = ins.size();
            ins.push_back({ i, n, (spv::Op)(code[i] & 0xFFFF) });
            i += n;
        }
        return true;
    }

    // What the patchers read from the module's header sections, gathered in one pass after Split.
    bool fragment = false, compute = false;                 // an entry point of that stage
    uint32_t tFloat = 0, tInt = 0, tPtrUF = 0;              // float32, int32 (signed preferred), Uniform float*
    uint32_t glsl = 0, vFragCoord = 0;                      // the GLSL.std.450 import, the FragCoord built-in variable
    size_t firstType = SIZE_MAX, firstFunc = SIZE_MAX, memModel = SIZE_MAX;
    std::unordered_map<uint32_t, uint32_t> vecOf;           // float vector type -> component count
    std::unordered_map<uint32_t, uint32_t> constBits;       // OpConstant -> first word, any type
    std::unordered_map<uint32_t, uint32_t> floatBits;       // ... float constants only
    std::unordered_map<uint32_t, uint32_t> binding, locationOf, sampledOf;   // decorations; sampled-image -> image
    std::unordered_set<uint32_t> floatConst, composites, inputs, uniforms, image3D;
    void Scan()
    {
        for (size_t k = 0; k < ins.size(); ++k)
        {
            const Ins& in = ins[k];
            const uint32_t* i = w + in.at;
            switch (in.op)
            {
            case spv::Op::OpExtInstImport: if (!glsl && in.n >= 4 && !strncmp((const char*)(i + 2), "GLSL.std.450", 12)) glsl = i[1]; break;
            case spv::Op::OpMemoryModel: memModel = k; break;
            case spv::Op::OpEntryPoint:
                if (i[1] == (uint32_t)spv::ExecutionModel::Fragment) fragment = true;
                if (i[1] == (uint32_t)spv::ExecutionModel::GLCompute) compute = true;
                break;
            case spv::Op::OpTypeFloat: if (i[2] == 32 && !tFloat) tFloat = i[1]; break;
            case spv::Op::OpTypeInt: if (i[2] == 32 && (!tInt || i[3] == 1)) tInt = i[1]; break;
            case spv::Op::OpTypeVector: if (i[2] == tFloat && tFloat) vecOf[i[1]] = i[3]; break;
            case spv::Op::OpTypeImage: if (i[3] == (uint32_t)spv::Dim::Dim3D) image3D.insert(i[1]); break;
            case spv::Op::OpTypeSampledImage: sampledOf[i[1]] = i[2]; break;
            case spv::Op::OpTypePointer: if (i[2] == (uint32_t)spv::StorageClass::Uniform && i[3] == tFloat && tFloat && !tPtrUF) tPtrUF = i[1]; break;
            case spv::Op::OpConstant:
                constBits[i[2]] = i[3];
                if (i[1] == tFloat && tFloat) { floatBits[i[2]] = i[3]; floatConst.insert(i[2]); }
                break;
            case spv::Op::OpConstantComposite: composites.insert(i[2]); break;
            case spv::Op::OpDecorate:
                if (i[2] == (uint32_t)spv::Decoration::Location) locationOf[i[1]] = i[3];
                else if (i[2] == (uint32_t)spv::Decoration::Binding) binding[i[1]] = i[3];
                else if (i[2] == (uint32_t)spv::Decoration::BuiltIn && i[3] == (uint32_t)spv::BuiltIn::FragCoord) vFragCoord = i[1];
                break;
            case spv::Op::OpVariable:
                varClass[i[2]] = i[3];
                if (i[3] == (uint32_t)spv::StorageClass::Input) inputs.insert(i[2]);
                else if (i[3] == (uint32_t)spv::StorageClass::Uniform) uniforms.insert(i[2]);
                break;
            case spv::Op::OpFunction: if (firstFunc == SIZE_MAX) firstFunc = k; break;
            default: break;
            }
            if (firstType == SIZE_MAX && TypeSection(in.op, firstFunc == SIZE_MAX)) firstType = k;
        }
    }

    const uint32_t* At(uint32_t id, spv::Op* op) const
    {
        auto it = def.find(id);
        if (it == def.end()) return nullptr;
        *op = ins[it->second].op;
        return w + ins[it->second].at;
    }
    // A value read straight out of a uniform/storage/push-constant block.
    bool FromBuffer(uint32_t id, int depth) const
    {
        spv::Op op; const uint32_t* i = At(id, &op);
        if (!i || depth > 10) return false;
        switch (op)
        {
        case spv::Op::OpLoad:
        {
            uint32_t p = i[3];
            spv::Op pop; const uint32_t* pi = At(p, &pop);
            if (pi && (pop == spv::Op::OpAccessChain || pop == spv::Op::OpInBoundsAccessChain)) p = pi[3];
            auto c = varClass.find(p);
            return c != varClass.end() && (c->second == (uint32_t)spv::StorageClass::Uniform ||
                c->second == (uint32_t)spv::StorageClass::StorageBuffer || c->second == (uint32_t)spv::StorageClass::PushConstant);
        }
        case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpCompositeExtract: case spv::Op::OpVectorShuffle:
            return FromBuffer(i[3], depth + 1);
        case spv::Op::OpCompositeConstruct:
            for (uint16_t k = 3; k < (uint16_t)(i[0] >> 16); ++k) if (!FromBuffer(i[k], depth + 1)) return false;
            return true;
        default: return false;
        }
    }
    // The stage-input load (a ray the vertex shader interpolates) a vector is a linear function of; MSAA variants
    // evaluate it at a sample offset, ray + ddx(ray)*a + ddy(ray)*b. 0 when there is none.
    uint32_t InputBase(uint32_t id, int depth) const
    {
        spv::Op op; const uint32_t* i = At(id, &op);
        if (!i || depth > 8) return 0;
        switch (op)
        {
        case spv::Op::OpLoad:
        {
            uint32_t p = i[3];
            spv::Op pop; const uint32_t* pi = At(p, &pop);
            if (pi && (pop == spv::Op::OpAccessChain || pop == spv::Op::OpInBoundsAccessChain)) p = pi[3];
            auto c = varClass.find(p);
            return c != varClass.end() && c->second == (uint32_t)spv::StorageClass::Input ? id : 0;
        }
        case spv::Op::OpVectorShuffle: case spv::Op::OpCompositeExtract: case spv::Op::OpCopyObject: case spv::Op::OpFNegate:
        case spv::Op::OpVectorTimesScalar: case spv::Op::OpDPdx: case spv::Op::OpDPdy:
        case spv::Op::OpDPdxFine: case spv::Op::OpDPdyFine: case spv::Op::OpDPdxCoarse: case spv::Op::OpDPdyCoarse:
            return InputBase(i[3], depth + 1);
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul:
        { const uint32_t a = InputBase(i[3], depth + 1); return a ? a : InputBase(i[4], depth + 1); }
        default: return 0;
        }
    }
    // (v.w, v.w, v.w) of a stage-input load v, as a construct of extracts or a shuffle; returns v's load.
    uint32_t WSplat(uint32_t id) const
    {
        spv::Op op; const uint32_t* i = At(id, &op);
        if (!i) return 0;
        const uint16_t n = (uint16_t)(i[0] >> 16);
        if (op == spv::Op::OpCompositeConstruct)
        {
            uint32_t l = 0;
            for (uint16_t k = 3; k < n; ++k)
            {
                spv::Op eop; const uint32_t* e = At(i[k], &eop);
                if (!e || eop != spv::Op::OpCompositeExtract || (e[0] >> 16) != 5 || e[4] != 3) return 0;
                const uint32_t b = InputBase(e[3], 0);
                if (!b || (l && b != l)) return 0;
                l = b;
            }
            return l;
        }
        if (op == spv::Op::OpVectorShuffle && n == 8 && i[3] == i[4] && i[5] == 3 && i[6] == 3 && i[7] == 3) return InputBase(i[3], 0);
        return 0;
    }
    // v.xyz / v.w of a stage-input vec4 through .xyz shuffles (a ray from volume geometry); returns v's load.
    uint32_t PerspectiveRay(uint32_t id, int depth) const
    {
        spv::Op op; const uint32_t* i = At(id, &op);
        if (!i || depth > 4) return 0;
        // .xyz of the first vector
        if (op == spv::Op::OpVectorShuffle && (i[0] >> 16) == 8 && i[5] == 0 && i[6] == 1 && i[7] == 2)
            return PerspectiveRay(i[3], depth + 1);
        if (op != spv::Op::OpFDiv) return 0;
        const uint32_t a = InputBase(i[3], 0);
        return a && a == WSplat(i[4]) ? a : 0;
    }
    // A scalar that depends on a texture read: the depth the ray is scaled by.
    bool FromImage(uint32_t id, int depth) const
    {
        spv::Op op; const uint32_t* i = At(id, &op);
        if (!i || depth > 14) return false;
        const uint16_t n = (uint16_t)(i[0] >> 16);
        switch (op)
        {
        case spv::Op::OpImageSampleImplicitLod: case spv::Op::OpImageSampleExplicitLod:
        case spv::Op::OpImageFetch: case spv::Op::OpImageRead: case spv::Op::OpImageGather:
            return true;
        case spv::Op::OpExtInst:
            for (uint16_t k = 5; k < n; ++k) if (FromImage(i[k], depth + 1)) return true;
            return false;
        case spv::Op::OpCompositeExtract: case spv::Op::OpBitcast: case spv::Op::OpCopyObject: case spv::Op::OpFNegate:
            return FromImage(i[3], depth + 1);     // extract's trailing words are literal indices, not ids
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFDiv:
        case spv::Op::OpCompositeConstruct: case spv::Op::OpVectorTimesScalar: case spv::Op::OpDot: case spv::Op::OpSelect: case spv::Op::OpPhi:
            for (uint16_t k = 3; k < n; ++k) if (FromImage(i[k], depth + 1)) return true;
            return false;
        default: return false;
        }
    }
};

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

// Types, constants and global variables: where a patch's own declarations go (decorations just ahead of the first).
inline bool TypeSection(spv::Op op, bool beforeFunctions)
{
    return (op >= spv::Op::OpTypeVoid && op <= spv::Op::OpTypeForwardPointer) || (op >= spv::Op::OpConstantTrue && op <= spv::Op::OpSpecConstantOp) ||
           (op == spv::Op::OpVariable && beforeFunctions);
}

// The stereo block at (stereoSet, binding 0): floats sep, conv, wMin, eye, fsSep [, half the viewport's width].
inline void DeclareStereo(Emit& types, Emit& deco, uint32_t tFloat, uint32_t tS, uint32_t tPtrS, uint32_t vStereo, uint32_t& tPtrUF, uint32_t& next,
                          uint32_t stereoSet, uint32_t members = 5)
{
    if (members == 6) types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat, tFloat, tFloat, tFloat });
    else types.Op(spv::Op::OpTypeStruct, { tS, tFloat, tFloat, tFloat, tFloat, tFloat });
    types.Op(spv::Op::OpTypePointer, { tPtrS, (uint32_t)spv::StorageClass::Uniform, tS });
    if (!tPtrUF) { tPtrUF = next++; types.Op(spv::Op::OpTypePointer, { tPtrUF, (uint32_t)spv::StorageClass::Uniform, tFloat }); }
    types.Op(spv::Op::OpVariable, { tPtrS, vStereo, (uint32_t)spv::StorageClass::Uniform });
    deco.Op(spv::Op::OpDecorate, { tS, (uint32_t)spv::Decoration::Block });
    for (uint32_t k = 0; k < members; ++k) deco.Op(spv::Op::OpMemberDecorate, { tS, k, (uint32_t)spv::Decoration::Offset, k * 4 });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::DescriptorSet, stereoSet });
    deco.Op(spv::Op::OpDecorate, { vStereo, (uint32_t)spv::Decoration::Binding, 0 });
}

// gl_FragCoord as a shader carries it: loads of the built-in, and function-local slots (a variable, or one constant
// member of it) stored once with it or with a vector built from its x and y.
struct FragPixel
{
    const Fs& m;
    std::unordered_map<uint64_t, int> slotStores;
    std::unordered_set<uint64_t> fragSlots;
    explicit FragPixel(const Fs& fs) : m(fs)
    {
        for (const Ins& in : m.ins)
        {
            uint64_t key;
            if (in.op != spv::Op::OpStore || !SlotOf(m.w[in.at + 1], key)) continue;
            ++slotStores[key];
            const uint32_t v = m.w[in.at + 2];
            spv::Op op; const uint32_t* c = m.At(v, &op);
            if (IsFragLoad(v) || (c && op == spv::Op::OpCompositeConstruct && (c[0] >> 16) >= 5 && FragComp(c[3], 0) && FragComp(c[4], 1)))
                fragSlots.insert(key);
        }
    }
    bool IsFragLoad(uint32_t id) const { spv::Op op; const uint32_t* l = m.At(id, &op); return l && op == spv::Op::OpLoad && l[3] == m.vFragCoord; }
    // The vector a .xy shuffle (v.xy of one vector) reads, 0 otherwise.
    uint32_t XyOf(uint32_t id) const
    {
        spv::Op op; const uint32_t* s = m.At(id, &op);
        return (s && op == spv::Op::OpVectorShuffle && (s[0] >> 16) == 7 && s[3] == s[4] && s[5] == 0 && s[6] == 1) ? s[3] : 0;
    }
    bool FragComp(uint32_t id, uint32_t comp) const
    {
        spv::Op op; const uint32_t* e = m.At(id, &op);
        if (!e || op != spv::Op::OpCompositeExtract || (e[0] >> 16) != 5 || e[4] != comp) return false;
        if (IsFragLoad(e[3])) return true;
        const uint32_t v = XyOf(e[3]);
        return v && IsFragLoad(v);
    }
    // A function-local slot, keyed (variable, index id).
    bool SlotOf(uint32_t ptr, uint64_t& key) const
    {
        spv::Op op; const uint32_t* p = m.At(ptr, &op);
        uint32_t var = ptr, idx = 0;
        if (p && (op == spv::Op::OpAccessChain || op == spv::Op::OpInBoundsAccessChain))
        {
            if ((p[0] >> 16) != 5) return false;
            var = p[3]; idx = p[4];
        }
        auto c = m.varClass.find(var);
        if (c == m.varClass.end() || c->second != (uint32_t)spv::StorageClass::Function) return false;
        key = ((uint64_t)var << 32) | idx;
        return true;
    }
    // FragCoord itself, or a load of a slot stored once with it.
    bool PixelVec(uint32_t id) const
    {
        if (IsFragLoad(id)) return true;
        spv::Op op; const uint32_t* l = m.At(id, &op);
        uint64_t key;
        if (!l || op != spv::Op::OpLoad || !SlotOf(l[3], key) || !fragSlots.count(key)) return false;
        auto s = slotStores.find(key);
        return s != slotStores.end() && s->second == 1;
    }
};

} // namespace

bool PatchReconstructionFS(const uint32_t* code, size_t words, uint32_t stereoSet, const ReconstructionRules& rules,
                           std::vector<uint32_t>& out, ReconstructionCounts* counts)
{
    const bool skewExact = rules.skewExact, fragDepth = rules.fragDepth, volumeRays = rules.volumeRays, volumeConv = rules.volumeConv,
               screenSkew = rules.screenSkew, skewedRays = rules.skewedRays, volumeLights = rules.volumeLights;
    uint32_t vFragCoord = 0;   // fragDepth: rule 1 also takes a ray scaled by the fragment's own depth, 1 / FragCoord.w
    if (counts) *counts = ReconstructionCounts();
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    vFragCoord = m.vFragCoord;
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF, glsl = m.glsl;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc, memModel = m.memModel;
    auto& vecOf = m.vecOf;          // vector type -> component count (float vectors only)
    auto& constBits = m.floatBits;  // float constant -> bits
    if (!m.fragment || !tFloat || firstType == SIZE_MAX || firstFunc == SIZE_MAX || memModel == SIZE_MAX) return false;

    // vec(dot(A, r0), dot(A, r1), dot(A, r2) [, x]) with A = (x, y, 1.0) and the rows loaded from a buffer.
    struct Ray { uint32_t v; uint32_t rows[3]; };
    std::vector<Ray> rays;
    for (const Ins& in : m.ins)
    {
        if (in.op != spv::Op::OpCompositeConstruct) continue;
        const uint32_t* i = code + in.at;
        auto vt = vecOf.find(i[1]);
        if (vt == vecOf.end() || (vt->second != 3 && vt->second != 4) || in.n < 6) continue;
        uint32_t a[3], b[3];
        bool dots = true;
        for (int c = 0; c < 3 && dots; ++c)
        {
            spv::Op op; const uint32_t* d = m.At(i[3 + c], &op);
            if (!d || op != spv::Op::OpDot) dots = false; else { a[c] = d[3]; b[c] = d[4]; }
        }
        if (!dots) continue;
        uint32_t A, rows[3];
        if (a[0] == a[1] && a[1] == a[2]) { A = a[0]; rows[0] = b[0]; rows[1] = b[1]; rows[2] = b[2]; }
        else if (b[0] == b[1] && b[1] == b[2]) { A = b[0]; rows[0] = a[0]; rows[1] = a[1]; rows[2] = a[2]; }
        else continue;
        spv::Op aop; const uint32_t* ai = m.At(A, &aop);
        if (!ai || aop != spv::Op::OpCompositeConstruct || vecOf.count(ai[1]) == 0 || vecOf[ai[1]] != 3) continue;
        auto one = constBits.find(ai[(ai[0] >> 16) - 1]);
        if (one == constBits.end() || one->second != Bits(1.f)) continue;
        if (!m.FromBuffer(rows[0], 0) || !m.FromBuffer(rows[1], 0) || !m.FromBuffer(rows[2], 0)) continue;
        rays.push_back({ i[2], { rows[0], rows[1], rows[2] } });
    }
    // The ray itself when vec3, else its .xyz shuffles.
    std::unordered_map<uint32_t, size_t> rayOf;         // vec3 value -> rays index
    for (size_t r = 0; r < rays.size(); ++r)
    {
        spv::Op op; const uint32_t* vi = m.At(rays[r].v, &op);
        if (vecOf[vi[1]] == 3) rayOf[rays[r].v] = r;
    }
    for (const Ins& in : m.ins)
    {
        if (in.op != spv::Op::OpVectorShuffle || in.n != 8) continue;
        const uint32_t* i = code + in.at;
        auto vt = vecOf.find(i[1]);
        if (vt == vecOf.end() || vt->second != 3 || i[5] != 0 || i[6] != 1 || i[7] != 2) continue;
        for (size_t r = 0; r < rays.size(); ++r) if (i[3] == rays[r].v) { rayOf[i[2]] = r; break; }
    }
    // Who consumes each value, over the ops the position can flow through to a shadow compare.
    std::unordered_map<uint32_t, std::vector<size_t>> users;
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        const uint32_t* i = code + m.ins[k].at;
        const uint16_t n = m.ins[k].n;
        switch (m.ins[k].op)
        {
        case spv::Op::OpCompositeConstruct: for (uint16_t a = 3; a < n; ++a) users[i[a]].push_back(k); break;
        case spv::Op::OpCompositeExtract: users[i[3]].push_back(k); break;
        case spv::Op::OpVectorShuffle: case spv::Op::OpCompositeInsert: users[i[3]].push_back(k); users[i[4]].push_back(k); break;
        case spv::Op::OpFAdd: case spv::Op::OpFSub: case spv::Op::OpFMul: case spv::Op::OpFDiv:
        case spv::Op::OpVectorTimesScalar: case spv::Op::OpDot: users[i[3]].push_back(k); users[i[4]].push_back(k); break;
        case spv::Op::OpExtInst: for (uint16_t a = 5; a < n; ++a) users[i[a]].push_back(k); break;
        case spv::Op::OpSelect: users[i[4]].push_back(k); users[i[5]].push_back(k); break;
        // The cascade picked by branch.
        case spv::Op::OpPhi: for (uint16_t a = 3; a < n; a += 2) users[i[a]].push_back(k); break;
        case spv::Op::OpImageSampleDrefImplicitLod: case spv::Op::OpImageSampleDrefExplicitLod:
        case spv::Op::OpImageSampleProjDrefImplicitLod: case spv::Op::OpImageSampleProjDrefExplicitLod: case spv::Op::OpImageDrefGather:
            users[i[4]].push_back(k); users[i[5]].push_back(k); break;
        default: break;
        }
    }
    // Cascade selection chains phis and a receiver-plane bias between the position and the compare: ~15 steps.
    std::unordered_map<uint32_t, bool> seen;
    std::function<bool(uint32_t, int)> reachesDref = [&](uint32_t id, int depth) -> bool
    {
        if (depth > 32 || !seen.emplace(id, true).second) return false;
        auto u = users.find(id);
        if (u == users.end()) return false;
        for (size_t k : u->second)
        {
            const spv::Op op = m.ins[k].op;
            if (op == spv::Op::OpImageSampleDrefImplicitLod || op == spv::Op::OpImageSampleDrefExplicitLod ||
                op == spv::Op::OpImageSampleProjDrefImplicitLod || op == spv::Op::OpImageSampleProjDrefExplicitLod ||
                op == spv::Op::OpImageDrefGather) return true;
            if (reachesDref(code[m.ins[k].at + 2], depth + 1)) return true;
        }
        return false;
    };
    // Rule 3's own flow map: rules 1 and 2 keep theirs, so their matches cannot change. It adds matrix products and
    // plain samples: rule 3 fires on a position that addresses a 3D texture (a per-prop occlusion volume) or a light's.
    std::unordered_map<uint32_t, uint32_t> imageDim, sampledImageOf;
    std::unordered_map<uint32_t, std::vector<size_t>> users3;
    if (volumeRays)
    {
        for (const Ins& in : m.ins)
        {
            const uint32_t* i = code + in.at;
            if (in.op == spv::Op::OpTypeImage) imageDim[i[1]] = i[3];
            else if (in.op == spv::Op::OpTypeSampledImage) sampledImageOf[i[1]] = i[2];
        }
        users3 = users;
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const uint32_t* i = code + m.ins[k].at;
            switch (m.ins[k].op)
            {
            case spv::Op::OpMatrixTimesVector: users3[i[4]].push_back(k); break;
            case spv::Op::OpVectorTimesMatrix: users3[i[3]].push_back(k); break;
            case spv::Op::OpImageSampleImplicitLod: case spv::Op::OpImageSampleExplicitLod: users3[i[4]].push_back(k); break;
            default: break;
            }
        }
    }
    auto is3DSample = [&](size_t k) -> bool
    {
        const spv::Op op = m.ins[k].op;
        if (op != spv::Op::OpImageSampleImplicitLod && op != spv::Op::OpImageSampleExplicitLod) return false;
        spv::Op sop; const uint32_t* si = m.At(code[m.ins[k].at + 3], &sop);
        if (!si) return false;
        auto st = sampledImageOf.find(si[1]);
        if (st == sampledImageOf.end()) return false;
        auto dim = imageDim.find(st->second);
        return dim != imageDim.end() && dim->second == (uint32_t)spv::Dim::Dim3D;
    };
    std::unordered_map<uint32_t, bool> seen3;
    std::function<bool(uint32_t, int)> reaches3D = [&](uint32_t id, int depth) -> bool
    {
        if (depth > 32 || !seen3.emplace(id, true).second) return false;
        auto u = users3.find(id);
        if (u == users3.end()) return false;
        for (size_t k : u->second)
        {
            if (is3DSample(k)) return true;
            if (m.ins[k].op == spv::Op::OpImageSampleImplicitLod || m.ins[k].op == spv::Op::OpImageSampleExplicitLod) continue;
            if (reaches3D(code[m.ins[k].at + 2], depth + 1)) return true;
        }
        return false;
    };
    // origin + ray*t (RAGE's camera block row 3) already gets the translation from the constants path.
    auto addsOrigin = [&](uint32_t id) -> bool
    {
        auto u = users.find(id);
        if (u == users.end()) return false;
        for (size_t k : u->second)
        {
            if (m.ins[k].op != spv::Op::OpFAdd) continue;
            const uint32_t* i = code + m.ins[k].at;
            if (m.FromBuffer(i[3] == id ? i[4] : i[3], 0)) return true;
        }
        return false;
    };
    // centre - position, the centre read from a buffer: the vector to a light the position is shaded against.
    auto lightVector = [&](uint32_t id) -> bool
    {
        auto u = users.find(id);
        if (u == users.end()) return false;
        for (size_t k : u->second)
        {
            const uint32_t* i = code + m.ins[k].at;
            if (m.ins[k].op == spv::Op::OpFSub && i[4] == id && m.FromBuffer(i[3], 0)) return true;
        }
        return false;
    };
    // A ray times a depth-derived scalar: rule 1 rebuilds the ray from buffer rows, rule 2 interpolates it into a
    // shadow compare, rule 3 (volumeRays) is v.xyz/v.w. Prod: ray == SIZE_MAX for rules 2/3, along dx; own: 1/w.
    struct Prod { size_t idx; size_t ray; uint32_t dx; uint32_t load = 0; bool own = false; };
    std::vector<Prod> prods;
    std::unordered_map<size_t, std::vector<std::pair<uint32_t, uint32_t>>> dxAfter;   // def index -> (varying, dPdx id)
    std::unordered_map<uint32_t, uint32_t> dxOf;
    uint32_t next = code[3];
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        if (m.ins[k].op != spv::Op::OpVectorTimesScalar) continue;
        const uint32_t* i = code + m.ins[k].at;
        // Only a linearised depth, c / ((1 + k) - d): volumetric lights scale a march step by blue noise, and their
        // ray comes from volume geometry that is already relative to the left camera.
        spv::Op sop; const uint32_t* si = m.At(i[4], &sop);
        if (!si || sop != spv::Op::OpFDiv) continue;
        // fragDepth: 1.0 / gl_FragCoord.w, the depth this fragment was drawn at (shifted for the right eye, as a read).
        bool ownDepth = false;
        if (fragDepth && vFragCoord)
        {
            auto one = constBits.find(si[3]);
            spv::Op eop; const uint32_t* ex = m.At(si[4], &eop);
            if (one != constBits.end() && one->second == 0x3F800000u && ex && eop == spv::Op::OpCompositeExtract && (ex[0] >> 16) == 5 && ex[4] == 3)
            {
                spv::Op lop; const uint32_t* ld = m.At(ex[3], &lop);
                ownDepth = ld && lop == spv::Op::OpLoad && ld[3] == vFragCoord;
            }
        }
        if (!ownDepth && !m.FromImage(i[4], 0)) continue;
        auto r = rayOf.find(i[3]);
        if (r != rayOf.end()) { prods.push_back({ k, r->second, 0, 0, ownDepth }); continue; }
        if (ownDepth) continue;   // only rule 1 takes the fragment's own depth
        if (volumeRays || volumeLights)
        {
            const uint32_t load = m.PerspectiveRay(i[3], 0);
            spv::Op lop; const uint32_t* li = load ? m.At(load, &lop) : nullptr;
            seen3.clear();
            if (li && vecOf.count(li[1]) && vecOf[li[1]] == 4 && !addsOrigin(i[2]) &&
                ((volumeRays && reaches3D(i[2], 0)) || (volumeLights && lightVector(i[2]))))
            {
                uint32_t& dx = dxOf[load];
                if (!dx) { dx = next++; dxAfter[m.def[load]].push_back({ load, dx }); }
                prods.push_back({ k, SIZE_MAX, dx, load });
                continue;
            }
        }
        auto vt = vecOf.find(i[1]);
        const uint32_t base = (vt != vecOf.end() && vt->second == 3) ? m.InputBase(i[3], 0) : 0;
        if (!base || addsOrigin(i[2])) continue;
        spv::Op bop; const uint32_t* bi = m.At(base, &bop);
        if (!bi || i[1] != bi[1]) continue;           // a vec3 varying, same type as the product
        seen.clear();
        if (!reachesDref(i[2], 0)) continue;
        uint32_t& dx = dxOf[base];
        if (!dx) { dx = next++; dxAfter[m.def[base]].push_back({ base, dx }); }
        prods.push_back({ k, SIZE_MAX, dx });
    }
    // Rule 4 (screenSkew): a ray varying times this pixel's depth, projected back through the left camera's buffer
    // matrix, gets x += fsSep*w; only beside a rule 2 match or with skewedRays (the right eye's skewed rays).
    std::vector<size_t> projs;
    bool rule2 = skewedRays;
    for (const Prod& pr : prods) rule2 |= pr.ray == SIZE_MAX && !pr.load;
    if (screenSkew && rule2)
    {
        std::function<bool(uint32_t, int)> fromInput = [&](uint32_t id, int depth) -> bool
        {
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 8) return false;
            switch (op)
            {
            case spv::Op::OpLoad: return m.InputBase(id, 0) == id;
            case spv::Op::OpCompositeExtract: case spv::Op::OpCopyObject: return fromInput(i[3], depth + 1);
            case spv::Op::OpVectorShuffle: return fromInput(i[3], depth + 1) || fromInput(i[4], depth + 1);
            case spv::Op::OpCompositeConstruct:
                for (uint16_t k = 3; k < (uint16_t)(i[0] >> 16); ++k) if (fromInput(i[k], depth + 1)) return true;
                return false;
            default: return false;
            }
        };
        // Offsets along the way are directions; a buffer origin added in would make it a world point, which needs more.
        std::function<bool(uint32_t, int)> depthPoint = [&](uint32_t id, int depth) -> bool
        {
            spv::Op op; const uint32_t* i = m.At(id, &op);
            if (!i || depth > 12) return false;
            switch (op)
            {
            case spv::Op::OpVectorTimesScalar:
            {
                spv::Op sop; const uint32_t* si = m.At(i[4], &sop);
                return si && sop == spv::Op::OpFDiv && m.FromImage(i[4], 0) && fromInput(i[3], 0);
            }
            case spv::Op::OpFAdd: case spv::Op::OpFSub:
                if (m.FromBuffer(i[3], 0) || m.FromBuffer(i[4], 0)) return false;
                return depthPoint(i[3], depth + 1) || depthPoint(i[4], depth + 1);
            case spv::Op::OpCompositeExtract: case spv::Op::OpCopyObject: return depthPoint(i[3], depth + 1);
            default: return false;
            }
        };
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            if (m.ins[k].op != spv::Op::OpMatrixTimesVector) continue;
            const uint32_t* i = code + m.ins[k].at;
            auto vt = vecOf.find(i[1]);
            if (vt == vecOf.end() || vt->second != 4 || !m.FromBuffer(i[3], 0)) continue;
            spv::Op vop; const uint32_t* vi = m.At(i[4], &vop);
            if (!vi || vop != spv::Op::OpCompositeConstruct) continue;
            const uint16_t vn = (uint16_t)(vi[0] >> 16);
            auto one = constBits.find(vi[vn - 1]);
            if (one == constBits.end() || one->second != Bits(1.f)) continue;
            bool point = false;
            for (uint16_t a = 3; a + 1 < vn && !point; ++a) point = depthPoint(vi[a], 0);
            if (!point) continue;
            // Clip space mapped to a uv: its x and w are added.
            std::vector<uint32_t> xs, ws;
            auto u = users.find(i[2]);
            if (u == users.end()) continue;
            for (size_t k2 : u->second)
            {
                const uint32_t* e = code + m.ins[k2].at;
                if (m.ins[k2].op != spv::Op::OpCompositeExtract || m.ins[k2].n != 5) continue;
                if (e[4] == 0) xs.push_back(e[2]); else if (e[4] == 3) ws.push_back(e[2]);
            }
            bool mapped = false;
            for (uint32_t ex : xs)
            {
                auto ux = users.find(ex);
                if (ux == users.end()) continue;
                for (size_t k2 : ux->second)
                {
                    const uint32_t* a = code + m.ins[k2].at;
                    if (m.ins[k2].op != spv::Op::OpFAdd) continue;
                    for (uint32_t ew : ws) if ((a[3] == ex && a[4] == ew) || (a[3] == ew && a[4] == ex)) mapped = true;
                }
            }
            if (mapped) projs.push_back(k);
        }
    }
    if (prods.empty() && projs.empty()) return false;

    Emit hdr, deco, types;
    // "GLSL.std.450"
    if (!glsl) { glsl = next++; hdr.Op(spv::Op::OpExtInstImport, { glsl, 0x4C534C47u, 0x6474732Eu, 0x3035342Eu, 0u }); }
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cIdx = next++, cTiny = next++, cTinyDx = next++, tS = next++, tPtrS = next++, vStereo = next++;
    const uint32_t cI1 = next++, cI2 = next++, cI4 = next++, cFar = next++, cOne = next++;
    // Member 0 only where an own-depth product needs it: modules without one keep their bytes.
    bool anyOwn = false;
    for (const Prod& pr : prods) anyOwn |= pr.own;
    const uint32_t cI0 = anyOwn ? next++ : 0;
    if (anyOwn) types.Op(spv::Op::OpConstant, { tInt, cI0, 0 });
    types.Op(spv::Op::OpConstant, { tFloat, cOne, Bits(1.f) });
    types.Op(spv::Op::OpConstant, { tInt, cIdx, 3 });
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI2, 2 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    types.Op(spv::Op::OpConstant, { tFloat, cFar, Bits(1e6f) });        // sky depth is inf; 0 * inf would be NaN
    types.Op(spv::Op::OpConstant, { tFloat, cTiny, Bits(1e-12f) });
    types.Op(spv::Op::OpConstant, { tFloat, cTinyDx, Bits(1e-30f) });   // a ray changes ~1e-4 per pixel
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);

    std::unordered_map<size_t, size_t> prodAt;
    for (size_t p = 0; p < prods.size(); ++p) prodAt[prods[p].idx] = p;
    std::unordered_map<size_t, bool> projAt;
    for (size_t k : projs) projAt[k] = true;
    out.clear();
    out.reserve(words + 64 * prods.size() + 128);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == memModel) out.insert(out.end(), hdr.v.begin(), hdr.v.end());
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (projAt.count(k))
        {
            // clip = M*v  ->  raw = M*v; clip = raw with x + fsSep*w (fsSep is 0 in the left eye).
            const uint32_t raw = next++, x = next++, w = next++, pS = next++, fs = next++, t = next++, x2 = next++;
            Emit e;
            e.Op(spv::Op::OpMatrixTimesVector, { i[1], raw, i[3], i[4] });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, x, raw, 0 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, w, raw, 3 });
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
            e.Op(spv::Op::OpLoad, { tFloat, fs, pS });
            e.Op(spv::Op::OpFMul, { tFloat, t, fs, w });
            e.Op(spv::Op::OpFAdd, { tFloat, x2, x, t });
            e.Op(spv::Op::OpCompositeInsert, { i[1], i[2], x2, raw, 0 });
            out.insert(out.end(), e.v.begin(), e.v.end());
            continue;
        }
        auto pa = prodAt.find(k);
        if (pa == prodAt.end())
        {
            out.insert(out.end(), i, i + in.n);
            // The varying's screen derivative, taken where it is loaded (uniform control flow): d(ray)/d(pixel x)
            // points along the camera's right axis in whatever space the ray is in (light space for the sun mask).
            auto da = dxAfter.find(k);
            if (da != dxAfter.end())
                for (const auto& vd : da->second) { Emit e; e.Op(spv::Op::OpDPdx, { i[1], vd.second, vd.first }); out.insert(out.end(), e.v.begin(), e.v.end()); }
            continue;
        }
        // p = ray*s  ->  raw = ray*s; p = raw + cx * (eye / |cx|), cx = the rows' x coefficients (camera right),
        // or the varying's d/dx for rule 2.
        const Prod& pr = prods[pa->second];
        const uint32_t tV3 = i[1], p = i[2];
        const uint32_t raw = next++, cx = pr.ray == SIZE_MAX ? pr.dx : next++, dd = next++, mx = next++, is = next++;
        const uint32_t pe = next++, eye = next++, kk = next++, corr = next++;
        Emit e;
        e.Op(spv::Op::OpVectorTimesScalar, { tV3, raw, i[3], i[4] });
        if (pr.load)
        {
            // Rule 3: right = d(v.xyz/v.w)/dx ~ dv.xyz*w - v.xyz*dw; p = raw + right/|right| * eye * (1 - depth/w).
            const uint32_t xyz = next++, w = next++, dxyz = next++, dw = next++, a = next++, b = next++, rd = next++;
            const uint32_t zr = next++, zs = next++, q = next++, om = next++, t1 = next++;
            e.Op(spv::Op::OpVectorShuffle, { tV3, xyz, pr.load, pr.load, 0, 1, 2 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, w, pr.load, 3 });
            e.Op(spv::Op::OpVectorShuffle, { tV3, dxyz, pr.dx, pr.dx, 0, 1, 2 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, dw, pr.dx, 3 });
            e.Op(spv::Op::OpVectorTimesScalar, { tV3, a, dxyz, w });
            e.Op(spv::Op::OpVectorTimesScalar, { tV3, b, xyz, dw });
            e.Op(spv::Op::OpFSub, { tV3, rd, a, b });
            e.Op(spv::Op::OpDot, { tFloat, dd, rd, rd });
            e.Op(spv::Op::OpExtInst, { tFloat, mx, glsl, GLSLstd450FMax, dd, cTinyDx });
            e.Op(spv::Op::OpExtInst, { tFloat, is, glsl, GLSLstd450InverseSqrt, mx });
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pe, vStereo, cIdx });
            e.Op(spv::Op::OpLoad, { tFloat, eye, pe });
            e.Op(spv::Op::OpExtInst, { tFloat, zr, glsl, GLSLstd450FClamp, i[4], cTiny, cFar });   // sky depth is inf
            if (volumeConv)
            {
                // A full-screen pass's ray (w = 1, the left camera's rows): ray*z + eye*right*(1 - z/conv).
                // z/conv as z*conv / max(conv^2, tiny): conv keeps its sign, and 0 gives 0.
                const uint32_t pC = next++, conv = next++, c2 = next++, zc = next++;
                e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
                e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
                e.Op(spv::Op::OpFMul, { tFloat, c2, conv, conv });
                e.Op(spv::Op::OpExtInst, { tFloat, zs, glsl, GLSLstd450FMax, c2, cTiny });
                e.Op(spv::Op::OpFMul, { tFloat, zc, zr, conv });
                e.Op(spv::Op::OpFDiv, { tFloat, q, zc, zs });
            }
            else
            {
                e.Op(spv::Op::OpExtInst, { tFloat, zs, glsl, GLSLstd450FMax, w, cTiny });
                e.Op(spv::Op::OpFDiv, { tFloat, q, zr, zs });
            }
            e.Op(spv::Op::OpFSub, { tFloat, om, cOne, q });
            e.Op(spv::Op::OpFMul, { tFloat, t1, is, eye });
            e.Op(spv::Op::OpFMul, { tFloat, kk, t1, om });
            e.Op(spv::Op::OpVectorTimesScalar, { tV3, corr, rd, kk });
            e.Op(spv::Op::OpFAdd, { tV3, p, raw, corr });
            out.insert(out.end(), e.v.begin(), e.v.end());
            continue;
        }
        if (pr.ray != SIZE_MAX)
        {
            const Ray& ray = rays[pr.ray];
            const uint32_t x0 = next++, x1 = next++, x2 = next++;
            e.Op(spv::Op::OpCompositeExtract, { tFloat, x0, ray.rows[0], 0 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, x1, ray.rows[1], 0 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, x2, ray.rows[2], 0 });
            e.Op(spv::Op::OpCompositeConstruct, { tV3, cx, x0, x1, x2 });
        }
        if (skewExact && pr.ray != SIZE_MAX)
        {
            // The depth read was drawn with ndc.x += fsSep*(max(w,wMin)-conv)/w: rows(ndc,1)*w is off by that x cx*w;
            // the fragment's own depth has this draw's shift, member 0 (zero on a pinned draw, where member 4 is not).
            const uint32_t pS = next++, fsSep = next++, pC = next++, conv = next++, pM = next++, wMin = next++, wc = next++, df = next++;
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, pr.own ? cI0 : cI4 });
            e.Op(spv::Op::OpLoad, { tFloat, fsSep, pS });
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
            e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pM, vStereo, cI2 });
            e.Op(spv::Op::OpLoad, { tFloat, wMin, pM });
            e.Op(spv::Op::OpExtInst, { tFloat, wc, glsl, GLSLstd450FClamp, i[4], wMin, cFar });
            e.Op(spv::Op::OpFSub, { tFloat, df, conv, wc });
            e.Op(spv::Op::OpFMul, { tFloat, kk, fsSep, df });
        }
        else
        {
            e.Op(spv::Op::OpDot, { tFloat, dd, cx, cx });
            e.Op(spv::Op::OpExtInst, { tFloat, mx, glsl, GLSLstd450FMax, dd, pr.ray == SIZE_MAX ? cTinyDx : cTiny });
            e.Op(spv::Op::OpExtInst, { tFloat, is, glsl, GLSLstd450InverseSqrt, mx });
            e.Op(spv::Op::OpAccessChain, { tPtrUF, pe, vStereo, cIdx });
            e.Op(spv::Op::OpLoad, { tFloat, eye, pe });
            e.Op(spv::Op::OpFMul, { tFloat, kk, is, eye });
        }
        e.Op(spv::Op::OpVectorTimesScalar, { tV3, corr, cx, kk });
        e.Op(spv::Op::OpFAdd, { tV3, p, raw, corr });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (counts)
    {
        counts->products = (unsigned)prods.size();
        for (const Prod& pr : prods) if (pr.load) ++counts->volume;
        counts->screen = (unsigned)projs.size();
    }
    return true;
}

// A compute pass marching far geometry (clouds) along rows.(ndc, 1) reads the right run's ray at the far-field skew.
bool PatchSkyRaysCS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* rays)
{
    if (rays) *rays = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& constBits = m.floatBits;
    if (!m.compute || !tFloat || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    // A view ray as dot(A, r0..r2), A = (ndc.x, ndc.y, 1.0), rows from a buffer: the right run reads it at x - sep.
    std::unordered_map<uint32_t, uint32_t> shifted;    // A -> its shifted id
    std::unordered_map<size_t, uint32_t> dotAt;        // OpDot index -> A
    for (const Ins& in : m.ins)
    {
        if (in.op != spv::Op::OpCompositeConstruct || in.n < 6) continue;
        const uint32_t* i = code + in.at;
        auto vt = vecOf.find(i[1]);
        if (vt == vecOf.end() || (vt->second != 3 && vt->second != 4)) continue;
        uint32_t A = 0;
        size_t dots[3];
        bool ok = true;
        for (int c = 0; c < 3 && ok; ++c)
        {
            spv::Op op; const uint32_t* d = m.At(i[3 + c], &op);
            if (!d || op != spv::Op::OpDot || (A && d[3] != A) || !m.FromBuffer(d[4], 0)) { ok = false; break; }
            A = d[3]; dots[c] = m.def[i[3 + c]];
        }
        if (!ok) continue;
        spv::Op aop; const uint32_t* ai = m.At(A, &aop);
        if (!ai || aop != spv::Op::OpCompositeConstruct || vecOf.count(ai[1]) == 0 || vecOf[ai[1]] != 3) continue;
        auto one = constBits.find(ai[(ai[0] >> 16) - 1]);
        if (one == constBits.end() || one->second != Bits(1.f)) continue;
        shifted[A] = 0;
        for (size_t k : dots) dotAt[k] = A;
    }
    if (shifted.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI0 = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI0, 0 });
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);
    for (auto& a : shifted) a.second = next++;

    out.clear();
    out.reserve(words + 64);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        auto da = dotAt.find(k);
        if (da != dotAt.end())
        {
            Emit e;
            e.Op(spv::Op::OpDot, { i[1], i[2], shifted[da->second], i[4] });
            out.insert(out.end(), e.v.begin(), e.v.end());
            continue;
        }
        out.insert(out.end(), i, i + in.n);
        auto sa = in.op == spv::Op::OpCompositeConstruct ? shifted.find(i[2]) : shifted.end();
        if (sa == shifted.end()) continue;
        // A2 = A with x - sep (0 in the left run).
        const uint32_t pS = next++, sep = next++, x = next++, x2 = next++;
        Emit e;
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI0 });
        e.Op(spv::Op::OpLoad, { tFloat, sep, pS });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, x, i[2], 0 });
        e.Op(spv::Op::OpFSub, { tFloat, x2, x, sep });
        e.Op(spv::Op::OpCompositeInsert, { i[1], sa->second, x2, i[2], 0 });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (rays) *rays = (unsigned)shifted.size();
    return true;
}

// A compute pass rebuilding positions as rows.(ndc, 1) * depth and projecting them through a camera block (screen-space
// reflections): the right run's ray, eye origin (sep*conv * column 0) and projections (x += sep*(w - conv*v.w)).
bool PatchEyeRaysCS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t projBinding, uint32_t projMembers,
                    std::vector<uint32_t>& out, unsigned* counts)
{
    if (counts) counts[0] = counts[1] = counts[2] = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& constBits = m.constBits;
    auto& binding = m.binding;
    if (!m.compute || !tFloat || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    // Rays dot(A, r0..r2), A = (ndc.x, ndc.y, 1.0), rows from a buffer: as in PatchSkyRaysCS.
    struct Ray { uint32_t rows[3]; };
    std::unordered_map<uint32_t, uint32_t> shifted;    // A -> its shifted id
    std::unordered_map<size_t, uint32_t> dotAt;        // OpDot index -> A
    std::unordered_map<uint32_t, Ray> rayOf;           // the (dot, dot, dot[, 1]) composite -> its rows
    for (const Ins& in : m.ins)
    {
        if (in.op != spv::Op::OpCompositeConstruct || in.n < 6) continue;
        const uint32_t* i = code + in.at;
        auto vt = vecOf.find(i[1]);
        if (vt == vecOf.end() || (vt->second != 3 && vt->second != 4)) continue;
        uint32_t A = 0;
        size_t dots[3];
        Ray r{};
        bool ok = true;
        for (int c = 0; c < 3 && ok; ++c)
        {
            spv::Op op; const uint32_t* d = m.At(i[3 + c], &op);
            if (!d || op != spv::Op::OpDot || (A && d[3] != A) || !m.FromBuffer(d[4], 0)) { ok = false; break; }
            A = d[3]; dots[c] = m.def[i[3 + c]]; r.rows[c] = d[4];
        }
        if (!ok) continue;
        spv::Op aop; const uint32_t* ai = m.At(A, &aop);
        if (!ai || aop != spv::Op::OpCompositeConstruct || vecOf.count(ai[1]) == 0 || vecOf[ai[1]] != 3) continue;
        auto one = constBits.find(ai[(ai[0] >> 16) - 1]);
        if (one == constBits.end() || one->second != Bits(1.f)) continue;
        shifted[A] = 0;
        for (size_t k : dots) dotAt[k] = A;
        rayOf[i[2]] = r;
    }
    if (shifted.empty()) return false;
    // ray * depth (the ray itself or its .xyz): the position, which gets the eye origin.
    std::unordered_map<size_t, Ray> posAt;
    // P * v with P loaded whole from the camera block's projection members: the clip, which gets the skew.
    std::unordered_map<size_t, uint32_t> projAt;       // index -> v
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op == spv::Op::OpVectorTimesScalar)
        {
            uint32_t v = i[3];
            spv::Op sop; const uint32_t* s = m.At(v, &sop);
            if (s && sop == spv::Op::OpVectorShuffle && s[3] == s[4] && (s[0] >> 16) == 8 && s[5] == 0 && s[6] == 1 && s[7] == 2) v = s[3];
            auto r = rayOf.find(v);
            if (r != rayOf.end() && vecOf.count(i[1]) && vecOf[i[1]] == 3) posAt[k] = r->second;
            continue;
        }
        if (in.op != spv::Op::OpMatrixTimesVector && in.op != spv::Op::OpVectorTimesMatrix) continue;
        const uint32_t mat = in.op == spv::Op::OpMatrixTimesVector ? i[3] : i[4], vec = in.op == spv::Op::OpMatrixTimesVector ? i[4] : i[3];
        if (vecOf.count(i[1]) == 0 || vecOf[i[1]] != 4) continue;
        spv::Op lop; const uint32_t* l = m.At(mat, &lop);
        if (!l || lop != spv::Op::OpLoad) continue;
        spv::Op cop; const uint32_t* ch = m.At(l[3], &cop);
        if (!ch || cop != spv::Op::OpAccessChain || (ch[0] >> 16) != 5) continue;
        auto b = binding.find(ch[3]);
        auto mem = constBits.find(ch[4]);
        if (b == binding.end() || b->second != projBinding || mem == constBits.end() || mem->second >= 32 || !(projMembers & (1u << mem->second))) continue;
        projAt[k] = vec;
    }

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI0 = next++, cI1 = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI0, 0 });
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);
    for (auto& a : shifted) a.second = next++;
    auto member = [&](Emit& e, uint32_t idx) {
        const uint32_t p = next++, v = next++;
        e.Op(spv::Op::OpAccessChain, { tPtrUF, p, vStereo, idx });
        e.Op(spv::Op::OpLoad, { tFloat, v, p });
        return v;
    };

    out.clear();
    out.reserve(words + 256);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        auto da = dotAt.find(k);
        if (da != dotAt.end())
        {
            Emit e;
            e.Op(spv::Op::OpDot, { i[1], i[2], shifted[da->second], i[4] });
            out.insert(out.end(), e.v.begin(), e.v.end());
            continue;
        }
        auto pa = posAt.find(k);
        if (pa != posAt.end())
        {
            // T = ray * depth; result = T + sep*conv * (r0.x, r1.x, r2.x): the right eye's origin, 0 in the left run.
            const uint32_t T = next++, c0 = next++, c1 = next++, c2 = next++, col = next++, sc = next++, off = next++;
            Emit e;
            e.Op(spv::Op::OpVectorTimesScalar, { i[1], T, i[3], i[4] });
            const uint32_t sep = member(e, cI0), conv = member(e, cI1);
            e.Op(spv::Op::OpFMul, { tFloat, sc, sep, conv });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, c0, pa->second.rows[0], 0 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, c1, pa->second.rows[1], 0 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, c2, pa->second.rows[2], 0 });
            e.Op(spv::Op::OpCompositeConstruct, { i[1], col, c0, c1, c2 });
            e.Op(spv::Op::OpVectorTimesScalar, { i[1], off, col, sc });
            e.Op(spv::Op::OpFAdd, { i[1], i[2], T, off });
            out.insert(out.end(), e.v.begin(), e.v.end());
            continue;
        }
        auto pj = projAt.find(k);
        if (pj != projAt.end())
        {
            // T = P * v; result.x = T.x + sep*(T.w - conv*v.w): the right camera's projection.
            const uint32_t T = next++, x = next++, w = next++, vw = next++, cw = next++, dw = next++, s = next++, x2 = next++;
            Emit e;
            e.Op(in.op, { i[1], T, i[3], i[4] });
            const uint32_t sep = member(e, cI0), conv = member(e, cI1);
            e.Op(spv::Op::OpCompositeExtract, { tFloat, x, T, 0 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, w, T, 3 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, vw, pj->second, 3 });
            e.Op(spv::Op::OpFMul, { tFloat, cw, conv, vw });
            e.Op(spv::Op::OpFSub, { tFloat, dw, w, cw });
            e.Op(spv::Op::OpFMul, { tFloat, s, sep, dw });
            e.Op(spv::Op::OpFAdd, { tFloat, x2, x, s });
            e.Op(spv::Op::OpCompositeInsert, { i[1], i[2], x2, T, 0 });
            out.insert(out.end(), e.v.begin(), e.v.end());
            continue;
        }
        out.insert(out.end(), i, i + in.n);
        auto sa = in.op == spv::Op::OpCompositeConstruct ? shifted.find(i[2]) : shifted.end();
        if (sa == shifted.end()) continue;
        // A2 = A with x - sep (0 in the left run).
        const uint32_t x = next++, x2 = next++;
        Emit e;
        const uint32_t sep = member(e, cI0);
        e.Op(spv::Op::OpCompositeExtract, { tFloat, x, i[2], 0 });
        e.Op(spv::Op::OpFSub, { tFloat, x2, x, sep });
        e.Op(spv::Op::OpCompositeInsert, { i[1], sa->second, x2, i[2], 0 });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (counts) { counts[0] = (unsigned)shifted.size(); counts[1] = (unsigned)posAt.size(); counts[2] = (unsigned)projAt.size(); }
    return true;
}

// A fragment shader projecting a DIRECTION (w = 0) through the patched camera P to use as a screen offset (water's
// reflection distortion): the right P adds fsSep*w to x, so the offset is taken back to the left's.
bool PatchDirProjectionFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t projBinding, uint32_t projMembers,
                          std::vector<uint32_t>& out, unsigned* dirs)
{
    if (dirs) *dirs = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& constBits = m.constBits;
    auto& binding = m.binding;
    auto& floatConst = m.floatConst;
    if (!m.fragment || !tFloat || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    std::unordered_set<size_t> dirAt;
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op != spv::Op::OpMatrixTimesVector && in.op != spv::Op::OpVectorTimesMatrix) continue;
        const uint32_t mat = in.op == spv::Op::OpMatrixTimesVector ? i[3] : i[4], vec = in.op == spv::Op::OpMatrixTimesVector ? i[4] : i[3];
        if (vecOf.count(i[1]) == 0 || vecOf[i[1]] != 4) continue;
        spv::Op lop; const uint32_t* l = m.At(mat, &lop);
        if (!l || lop != spv::Op::OpLoad) continue;
        spv::Op cop; const uint32_t* ch = m.At(l[3], &cop);
        if (!ch || cop != spv::Op::OpAccessChain || (ch[0] >> 16) != 5) continue;
        auto b = binding.find(ch[3]);
        auto mem = constBits.find(ch[4]);
        if (b == binding.end() || b->second != projBinding || mem == constBits.end() || mem->second >= 32 || !(projMembers & (1u << mem->second))) continue;
        // The vector's w is the constant 0: a direction.
        spv::Op vop; const uint32_t* v = m.At(vec, &vop);
        if (!v || vop != spv::Op::OpCompositeConstruct) continue;
        const uint32_t last = v[(v[0] >> 16) - 1];
        auto lc = constBits.find(last);
        if (!floatConst.count(last) || lc == constBits.end() || lc->second != 0) continue;
        dirAt.insert(k);
    }
    if (dirAt.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI4 = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);

    out.clear();
    out.reserve(words + 64);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (!dirAt.count(k)) { out.insert(out.end(), i, i + in.n); continue; }
        // T = P * (d, 0); result.x = T.x - fsSep*T.w.
        const uint32_t T = next++, p = next++, fsSep = next++, x = next++, w = next++, s = next++, x2 = next++;
        Emit e;
        e.Op(in.op, { i[1], T, i[3], i[4] });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, p, vStereo, cI4 });
        e.Op(spv::Op::OpLoad, { tFloat, fsSep, p });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, x, T, 0 });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, w, T, 3 });
        e.Op(spv::Op::OpFMul, { tFloat, s, fsSep, w });
        e.Op(spv::Op::OpFSub, { tFloat, x2, x, s });
        e.Op(spv::Op::OpCompositeInsert, { i[1], i[2], x2, T, 0 });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (dirs) *dirs = (unsigned)dirAt.size();
    return true;
}

// A position (w = 1) projected through a view-projection in a shared buffer, the left camera's in both eyes (water's
// refracted point): the clip gets the eye's shift, x + fsSep*(w - conv).
bool PatchBufferProjectionFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t bufBinding, uint32_t member, uint32_t field,
                             std::vector<uint32_t>& out, unsigned* projections)
{
    if (projections) *projections = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& constBits = m.constBits;
    auto& binding = m.binding;
    auto& floatConst = m.floatConst;
    if (!m.fragment || !tFloat || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    std::unordered_set<size_t> projAt;
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op != spv::Op::OpMatrixTimesVector && in.op != spv::Op::OpVectorTimesMatrix) continue;
        const uint32_t mat = in.op == spv::Op::OpMatrixTimesVector ? i[3] : i[4], vec = in.op == spv::Op::OpMatrixTimesVector ? i[4] : i[3];
        if (vecOf.count(i[1]) == 0 || vecOf[i[1]] != 4) continue;
        spv::Op lop; const uint32_t* l = m.At(mat, &lop);
        if (!l || lop != spv::Op::OpLoad) continue;
        // The matrix: block[0][element 0].member.field, the last two indices as given.
        spv::Op cop; const uint32_t* ch = m.At(l[3], &cop);
        if (!ch || cop != spv::Op::OpAccessChain || (ch[0] >> 16) != 8) continue;
        auto b = binding.find(ch[3]);
        auto mi = constBits.find(ch[6]), fi = constBits.find(ch[7]);
        if (b == binding.end() || b->second != bufBinding || mi == constBits.end() || fi == constBits.end() || mi->second != member || fi->second != field) continue;
        // The vector's w is the constant 1: a position.
        spv::Op vop; const uint32_t* v = m.At(vec, &vop);
        if (!v || vop != spv::Op::OpCompositeConstruct) continue;
        const uint32_t last = v[(v[0] >> 16) - 1];
        auto lc = constBits.find(last);
        if (!floatConst.count(last) || lc == constBits.end() || lc->second != 0x3F800000u) continue;
        projAt.insert(k);
    }
    if (projAt.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI1 = next++, cI4 = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);

    out.clear();
    out.reserve(words + 64);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (!projAt.count(k)) { out.insert(out.end(), i, i + in.n); continue; }
        // T = the clip; result.x = T.x + fsSep*(T.w - conv). fsSep is 0 in the left eye.
        const uint32_t T = next++, pS = next++, fsSep = next++, pC = next++, conv = next++, x = next++, w = next++, dw = next++, s = next++, x2 = next++;
        Emit e;
        e.Op(in.op, { i[1], T, i[3], i[4] });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
        e.Op(spv::Op::OpLoad, { tFloat, fsSep, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, x, T, 0 });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, w, T, 3 });
        e.Op(spv::Op::OpFSub, { tFloat, dw, w, conv });
        e.Op(spv::Op::OpFMul, { tFloat, s, fsSep, dw });
        e.Op(spv::Op::OpFAdd, { tFloat, x2, x, s });
        e.Op(spv::Op::OpCompositeInsert, { i[1], i[2], x2, T, 0 });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (projections) *projections = (unsigned)projAt.size();
    return true;
}

// A screen-aligned texture read at the fragment's own clip position (mirror glass: bias + scale*clip.xy/clip.w into
// a reflection drawn for the left's crop): the varying's x goes back by the eye's shift, x - fsSep*(w - conv).
bool PatchClipVaryingLeftFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t location, std::vector<uint32_t>& out, unsigned* loads)
{
    if (loads) *loads = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& locationOf = m.locationOf;
    auto& inputs = m.inputs;
    if (!m.fragment || !tFloat || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    std::unordered_set<size_t> loadAt;
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op != spv::Op::OpLoad || !inputs.count(i[3])) continue;
        auto loc = locationOf.find(i[3]);
        auto vt = vecOf.find(i[1]);
        if (loc == locationOf.end() || loc->second != location || vt == vecOf.end() || vt->second != 4) continue;
        loadAt.insert(k);
    }
    if (loadAt.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI1 = next++, cI4 = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);

    out.clear();
    out.reserve(words + 64);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (!loadAt.count(k)) { out.insert(out.end(), i, i + in.n); continue; }
        // T = the varying; result.x = T.x - fsSep*(T.w - conv). fsSep is 0 in the left eye.
        const uint32_t T = next++, pS = next++, fsSep = next++, pC = next++, conv = next++, x = next++, w = next++, d = next++, s = next++, x2 = next++;
        Emit e;
        e.Op(spv::Op::OpLoad, { i[1], T, i[3] });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
        e.Op(spv::Op::OpLoad, { tFloat, fsSep, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, x, T, 0 });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, w, T, 3 });
        e.Op(spv::Op::OpFSub, { tFloat, d, w, conv });
        e.Op(spv::Op::OpFMul, { tFloat, s, fsSep, d });
        e.Op(spv::Op::OpFSub, { tFloat, x2, x, s });
        e.Op(spv::Op::OpCompositeInsert, { i[1], i[2], x2, T, 0 });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (loads) *loads = (unsigned)loadAt.size();
    return true;
}

// A pinned lighting pass's view direction normalize(ray varying / w), the left camera's ray, is taken at ndc.x - s
// (s = eye*P00/conv from its camera block): ray - ddx(ray) * s / (2 * ddx(uv.x)); the position product keeps its ray.
bool PatchViewRayFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t rayLocation, uint32_t uvLocation, uint32_t projBinding, uint32_t projMember,
                    std::vector<uint32_t>& out, unsigned* dirs)
{
    if (dirs) *dirs = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF, glsl = m.glsl, vProj = 0, vUv = 0;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& locationOf = m.locationOf;
    auto& binding = m.binding;
    auto& inputs = m.inputs;
    auto& uniforms = m.uniforms;
    if (!m.fragment || !tFloat || !glsl || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    for (uint32_t v : uniforms) { auto b = binding.find(v); if (b != binding.end() && b->second == projBinding) vProj = v; }
    for (uint32_t v : inputs) { auto l = locationOf.find(v); if (l != locationOf.end() && l->second == uvLocation) vUv = v; }
    if (!vProj || !vUv) return false;
    // A vec3 taken whole from another value: the shuffle spirv-cross prints as (x).xyz.
    auto through = [&](uint32_t id) {
        for (int hop = 0; hop < 4; ++hop)
        {
            spv::Op op; const uint32_t* s = m.At(id, &op);
            if (!s || op != spv::Op::OpVectorShuffle || (s[0] >> 16) != 8 || s[5] != 0 || s[6] != 1 || s[7] != 2) break;
            id = s[3];
        }
        return id;
    };
    // normalize(X), X = load(ray varying).xyz / w.
    std::unordered_map<size_t, uint32_t> normAt;      // Normalize instruction -> the ray it reads
    std::unordered_map<uint32_t, uint32_t> rayType;   // ray id -> its vec4 load type
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op != spv::Op::OpExtInst || in.n != 6 || i[3] != glsl || i[4] != 69) continue;
        auto vt = vecOf.find(i[1]);
        if (vt == vecOf.end() || vt->second != 3) continue;
        spv::Op dop; const uint32_t* dv = m.At(through(i[5]), &dop);
        if (!dv || dop != spv::Op::OpFDiv) continue;
        spv::Op lop; const uint32_t* ld = m.At(through(dv[3]), &lop);
        if (!ld || lop != spv::Op::OpLoad || !inputs.count(ld[3])) continue;
        auto loc = locationOf.find(ld[3]);
        if (loc == locationOf.end() || loc->second != rayLocation) continue;
        normAt[k] = i[5];
        rayType[i[5]] = ld[1];
    }
    if (normAt.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI0 = next++, cI1 = next++, cI3 = next++, cIM = next++, cEps = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI0, 0 });
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI3, 3 });
    types.Op(spv::Op::OpConstant, { tInt, cIM, projMember });
    types.Op(spv::Op::OpConstant, { tFloat, cEps, 0x322BCC77u });   // 1e-8: the floor under conv^2
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);

    // The moved ray is built right after the ray itself (the entry block: derivatives need uniform control flow).
    std::unordered_map<uint32_t, uint32_t> moved;     // ray id -> ray at ndc.x - s
    out.clear();
    out.reserve(words + 128);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        auto na = normAt.find(k);
        if (na != normAt.end() && moved.count(na->second))
        {
            Emit e;
            e.Op(spv::Op::OpExtInst, { i[1], i[2], glsl, 69, moved[na->second] });
            out.insert(out.end(), e.v.begin(), e.v.end());
            continue;
        }
        out.insert(out.end(), i, i + in.n);
        bool hasRes = false, hasType = false;
        spv::HasResultAndType(in.op, &hasRes, &hasType);
        if (!hasRes || !hasType) continue;
        auto rt = rayType.find(i[2]);
        if (rt == rayType.end()) continue;
        const uint32_t ray = i[2], t3 = i[1];
        const uint32_t dRay = next++, U = next++, ux = next++, uw = next++, u = next++, du = next++, pE = next++, eye = next++, pC = next++, conv = next++, cm = next++,
                       pP = next++, p00 = next++, ep = next++, s = next++, den = next++, kk = next++, off = next++, r2 = next++, c2 = next++, epc = next++;
        Emit e;
        e.Op(spv::Op::OpDPdx, { t3, dRay, ray });
        e.Op(spv::Op::OpLoad, { rt->second, U, vUv });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, ux, U, 0 });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, uw, U, 3 });
        e.Op(spv::Op::OpFDiv, { tFloat, u, ux, uw });
        e.Op(spv::Op::OpDPdx, { tFloat, du, u });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pE, vStereo, cI3 });
        e.Op(spv::Op::OpLoad, { tFloat, eye, pE });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
        // eye*P00/conv as eye*P00*conv / max(conv^2, 1e-8): conv keeps its sign, and 0 gives 0, not a NaN.
        e.Op(spv::Op::OpFMul, { tFloat, c2, conv, conv });
        e.Op(spv::Op::OpExtInst, { tFloat, cm, glsl, 40, c2, cEps });   // FMax
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pP, vProj, cIM, cI0, cI0 });
        e.Op(spv::Op::OpLoad, { tFloat, p00, pP });
        e.Op(spv::Op::OpFMul, { tFloat, ep, eye, p00 });
        e.Op(spv::Op::OpFMul, { tFloat, epc, ep, conv });
        e.Op(spv::Op::OpFDiv, { tFloat, s, epc, cm });
        e.Op(spv::Op::OpFAdd, { tFloat, den, du, du });
        e.Op(spv::Op::OpFDiv, { tFloat, kk, s, den });
        e.Op(spv::Op::OpVectorTimesScalar, { t3, off, dRay, kk });
        e.Op(spv::Op::OpFSub, { t3, r2, ray, off });
        out.insert(out.end(), e.v.begin(), e.v.end());
        moved[ray] = r2;
    }
    out[3] = next;
    if (moved.empty()) return false;
    if (dirs) *dirs = (unsigned)normAt.size();
    return true;
}

// A pinned lighting pass's position (ray varying).xyz * depth with the left camera's ray: the right's surface point is
// ray*z + right*eye*(1 - max(z, wMin)/conv), right = normalize(ddx(ray)), rule 1's exact term without the rows.
bool PatchScreenRayPositionFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t rayLocation, std::vector<uint32_t>& out, unsigned* products)
{
    if (products) *products = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF, glsl = m.glsl;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& locationOf = m.locationOf;
    auto& inputs = m.inputs;
    if (!m.fragment || !tFloat || !glsl || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    // The vec3 operand back to the varying's load: (v).xyz, float4(v.x, v.y, v.z, 1), v.xyz / v.w, nothing else.
    std::function<bool(uint32_t, int)> fromRay = [&](uint32_t id, int depth) -> bool
    {
        spv::Op op; const uint32_t* s = m.At(id, &op);
        if (!s || depth > 8) return false;
        switch (op)
        {
        case spv::Op::OpLoad:
        {
            auto loc = locationOf.find(s[3]);
            return inputs.count(s[3]) && loc != locationOf.end() && loc->second == rayLocation;
        }
        case spv::Op::OpVectorShuffle: return (s[0] >> 16) == 8 && s[5] == 0 && s[6] == 1 && s[7] == 2 && fromRay(s[3], depth + 1);
        case spv::Op::OpCompositeConstruct:
        {
            if ((s[0] >> 16) < 6) return false;
            uint32_t src = 0;
            for (uint32_t c = 0; c < 3; ++c)
            {
                spv::Op eop; const uint32_t* ex = m.At(s[3 + c], &eop);
                if (!ex || eop != spv::Op::OpCompositeExtract || (ex[0] >> 16) != 5 || ex[4] != c || (src && ex[3] != src)) return false;
                src = ex[3];
            }
            return fromRay(src, depth + 1);
        }
        case spv::Op::OpFDiv: return fromRay(s[3], depth + 1);
        default: return false;
        }
    };
    // Derivatives need uniform control flow: only products ahead of the function's first branch.
    std::unordered_set<size_t> prodAt;
    bool branched = false;
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op == spv::Op::OpFunction) branched = false;
        if (in.op == spv::Op::OpBranchConditional || in.op == spv::Op::OpSwitch || in.op == spv::Op::OpKill) branched = true;
        if (in.op != spv::Op::OpVectorTimesScalar || branched) continue;
        auto vt = vecOf.find(i[1]);
        if (vt == vecOf.end() || vt->second != 3) continue;
        spv::Op sop; const uint32_t* si = m.At(i[4], &sop);
        if (!si || sop != spv::Op::OpFDiv) continue;          // the linearised depth, c / ((1 + k) - d)
        if (!fromRay(i[3], 0)) continue;
        prodAt.insert(k);
    }
    if (prodAt.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI1 = next++, cI2 = next++, cI3 = next++, cEps = next++, cTiny = next++, cFar = next++, cOne = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI2, 2 });
    types.Op(spv::Op::OpConstant, { tInt, cI3, 3 });
    types.Op(spv::Op::OpConstant, { tFloat, cEps, 0x322BCC77u });    // 1e-8: the floor under conv^2
    types.Op(spv::Op::OpConstant, { tFloat, cTiny, 0x0DA24260u });   // 1e-30: a ray changes ~1e-4 per pixel
    types.Op(spv::Op::OpConstant, { tFloat, cFar, 0x49742400u });    // 1e6: sky depth is inf
    types.Op(spv::Op::OpConstant, { tFloat, cOne, 0x3F800000u });
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);

    out.clear();
    out.reserve(words + 64 * prodAt.size() + 128);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (!prodAt.count(k)) { out.insert(out.end(), i, i + in.n); continue; }
        // p = ray*z -> raw = ray*z; p = raw + ddx(ray)/|ddx(ray)| * eye * (1 - clamp(z, wMin, 1e6)/conv), eye 0 in the
        // left eye; z/conv as z*conv / max(conv^2, 1e-8): conv keeps its sign, and 0 gives 0, not a NaN.
        const uint32_t t3 = i[1], p = i[2];
        const uint32_t raw = next++, dR = next++, dd = next++, mx = next++, is = next++, pE = next++, eye = next++, pC = next++, conv = next++, cm = next++,
                       pM = next++, wMin = next++, zc = next++, q = next++, om = next++, t1 = next++, kk = next++, corr = next++, c2 = next++, zcc = next++;
        Emit e;
        e.Op(spv::Op::OpVectorTimesScalar, { t3, raw, i[3], i[4] });
        e.Op(spv::Op::OpDPdx, { t3, dR, i[3] });
        e.Op(spv::Op::OpDot, { tFloat, dd, dR, dR });
        e.Op(spv::Op::OpExtInst, { tFloat, mx, glsl, 40, dd, cTiny });           // FMax
        e.Op(spv::Op::OpExtInst, { tFloat, is, glsl, 32, mx });                  // InverseSqrt
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pE, vStereo, cI3 });
        e.Op(spv::Op::OpLoad, { tFloat, eye, pE });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
        e.Op(spv::Op::OpFMul, { tFloat, c2, conv, conv });
        e.Op(spv::Op::OpExtInst, { tFloat, cm, glsl, 40, c2, cEps });            // FMax
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pM, vStereo, cI2 });
        e.Op(spv::Op::OpLoad, { tFloat, wMin, pM });
        e.Op(spv::Op::OpExtInst, { tFloat, zc, glsl, 43, i[4], wMin, cFar });    // FClamp
        e.Op(spv::Op::OpFMul, { tFloat, zcc, zc, conv });
        e.Op(spv::Op::OpFDiv, { tFloat, q, zcc, cm });
        e.Op(spv::Op::OpFSub, { tFloat, om, cOne, q });
        e.Op(spv::Op::OpFMul, { tFloat, t1, is, eye });
        e.Op(spv::Op::OpFMul, { tFloat, kk, t1, om });
        e.Op(spv::Op::OpVectorTimesScalar, { t3, corr, dR, kk });
        e.Op(spv::Op::OpFAdd, { t3, p, raw, corr });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (products) *products = (unsigned)prodAt.size();
    return true;
}

// A camera-aligned fog volume (the left eye's) sampled at (screen uv, depth slice): read at the left eye's pixel for
// the surface point, x - fsSep/2 * (1 - conv/depth), depth the linear depth behind the slice coordinate's min().
bool PatchFogVolumeLeftFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* lookups)
{
    if (lookups) *lookups = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF, glsl = m.glsl;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& sampledOf = m.sampledOf;
    auto& image3D = m.image3D;
    if (!m.fragment || !tFloat || !glsl || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    // The linear depth under a slice coordinate: the divide feeding the first min() found walking back from it.
    std::function<uint32_t(uint32_t, int)> depthOf = [&](uint32_t id, int hops) -> uint32_t {
        spv::Op op; const uint32_t* s = m.At(id, &op);
        if (!s || hops > 10) return 0;
        const uint32_t n = s[0] >> 16;
        if (op == spv::Op::OpExtInst && n == 7 && s[3] == glsl && (s[4] == 37 || s[4] == 79))
        {
            for (int a = 5; a <= 6; ++a) { spv::Op dop; const uint32_t* dv = m.At(s[a], &dop); if (dv && dop == spv::Op::OpFDiv) return s[a]; }
            return 0;
        }
        if (op == spv::Op::OpExtInst && s[3] == glsl) { for (uint32_t a = 5; a < n; ++a) if (uint32_t r = depthOf(s[a], hops + 1)) return r; return 0; }
        if (op == spv::Op::OpFMul || op == spv::Op::OpFAdd || op == spv::Op::OpFSub || op == spv::Op::OpFDiv)
        { for (uint32_t a = 3; a <= 4; ++a) if (uint32_t r = depthOf(s[a], hops + 1)) return r; }
        return 0;
    };
    std::unordered_map<size_t, uint32_t> siteDepth;   // the coordinate's OpCompositeConstruct -> linear depth
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op != spv::Op::OpImageSampleExplicitLod && in.op != spv::Op::OpImageSampleImplicitLod) continue;
        spv::Op sop; const uint32_t* si = m.At(i[3], &sop);
        if (!si || sop != spv::Op::OpSampledImage) continue;
        auto st = sampledOf.find(si[1]);
        if (st == sampledOf.end() || !image3D.count(st->second)) continue;
        spv::Op cop; const uint32_t* cc = m.At(i[4], &cop);
        if (!cc || cop != spv::Op::OpCompositeConstruct || (cc[0] >> 16) != 6) continue;
        auto vt = vecOf.find(cc[1]);
        if (vt == vecOf.end() || vt->second != 3) continue;
        const uint32_t z = depthOf(cc[5], 0);
        if (z) siteDepth[m.def[i[4]]] = z;
    }
    if (siteDepth.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI1 = next++, cI4 = next++, cF1 = next++, cHalf = next++, cNear = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    types.Op(spv::Op::OpConstant, { tFloat, cF1, 0x3F800000u });
    types.Op(spv::Op::OpConstant, { tFloat, cHalf, 0x3F000000u });
    types.Op(spv::Op::OpConstant, { tFloat, cNear, 0x3D4CCCCDu });   // 0.05, the near plane: a zero depth cannot NaN
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);

    out.clear();
    out.reserve(words + 96);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        auto sd = siteDepth.find(k);
        if (sd == siteDepth.end()) { out.insert(out.end(), i, i + in.n); continue; }
        const uint32_t pS = next++, fsSep = next++, pC = next++, conv = next++, zc = next++, q = next++, one = next++, h = next++, kk = next++, x2 = next++;
        Emit e;
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
        e.Op(spv::Op::OpLoad, { tFloat, fsSep, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, conv, pC });
        e.Op(spv::Op::OpExtInst, { tFloat, zc, glsl, 40, sd->second, cNear });
        e.Op(spv::Op::OpFDiv, { tFloat, q, conv, zc });
        e.Op(spv::Op::OpFSub, { tFloat, one, cF1, q });
        e.Op(spv::Op::OpFMul, { tFloat, h, fsSep, cHalf });
        e.Op(spv::Op::OpFMul, { tFloat, kk, h, one });
        e.Op(spv::Op::OpFSub, { tFloat, x2, i[3], kk });
        e.Op(spv::Op::OpCompositeConstruct, { i[1], i[2], x2, i[4], i[5] });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (lookups) *lookups = (unsigned)siteDepth.size();
    return true;
}

// A screen-space dither (int(xy) & 63, mod(floor(xy), 3) or fract(dot(xy, k)) of FragCoord.xy, plus any per-instance
// phase) is read at the left's pixel: x - round(fsSep*(1 - conv*FragCoord.w)*halfWidth), halfWidth stereo member 5.
bool PatchScreenDitherFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* sites)
{
    if (sites) *sites = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF, glsl = m.glsl;
    const uint32_t vFrag = m.vFragCoord;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    auto& vecOf = m.vecOf;
    auto& constants = m.composites;
    if (!m.fragment || !tFloat || !glsl || !vFrag || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    // gl_FragCoord.xy: a two-component shuffle of the loaded built-in. Returns the load's vec4 type, 0 otherwise.
    auto builtinXY = [&](uint32_t id) -> uint32_t {
        spv::Op op; const uint32_t* s = m.At(id, &op);
        if (!s || op != spv::Op::OpVectorShuffle || (s[0] >> 16) != 7 || s[5] != 0 || s[6] != 1) return 0;
        spv::Op lop; const uint32_t* l = m.At(s[3], &lop);
        // Some shaders first store 1 / w back into the loaded value: x and y are still the built-in's.
        if (l && lop == spv::Op::OpCompositeInsert && (l[0] >> 16) == 6 && l[5] >= 2) l = m.At(l[4], &lop);
        return (l && lop == spv::Op::OpLoad && l[3] == vFrag) ? l[1] : 0;
    };
    // The pixel coordinate: that, or that plus an offset (a per-instance dither phase).
    auto fragXY = [&](uint32_t id) -> uint32_t {
        if (const uint32_t t = builtinXY(id)) return t;
        spv::Op op; const uint32_t* s = m.At(id, &op);
        if (!s || op != spv::Op::OpFAdd) return 0;
        if (const uint32_t t = builtinXY(s[3])) return t;
        return builtinXY(s[4]);
    };
    // Candidates: int(xy) and floor(xy), kept only if an `& constant` (noise tile) or a mod (3x3 table) consumes them;
    // dot(xy, constant), kept only when a fract takes it (the checkerboard).
    std::unordered_map<uint32_t, size_t> convAt, floorAt, extractOf, dotAt;
    std::unordered_map<size_t, uint32_t> siteV4;   // instruction -> gl_FragCoord's vec4 type
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op == spv::Op::OpConvertFToS && fragXY(i[3])) convAt[i[2]] = k;
        else if (in.op == spv::Op::OpExtInst && in.n == 6 && i[3] == glsl && i[4] == 8 && fragXY(i[5])) floorAt[i[2]] = k;
        else if (in.op == spv::Op::OpDot && ((fragXY(i[3]) && constants.count(i[4])) || (fragXY(i[4]) && constants.count(i[3])))) dotAt[i[2]] = k;
        else if (in.op == spv::Op::OpExtInst && in.n == 6 && i[3] == glsl && i[4] == 10 && dotAt.count(i[5]))   // Fract
        {
            const uint32_t* dt = code + m.ins[dotAt[i[5]]].at;
            siteV4[dotAt[i[5]]] = fragXY(dt[3]) ? fragXY(dt[3]) : fragXY(dt[4]);
        }
        else if (in.op == spv::Op::OpCompositeExtract && floorAt.count(i[3])) extractOf[i[2]] = floorAt[i[3]];
        else if (in.op == spv::Op::OpBitwiseAnd)
        {
            for (int a = 3; a <= 4; ++a)
                if (convAt.count(i[a]) && constants.count(i[a == 3 ? 4 : 3])) siteV4[convAt[i[a]]] = fragXY(code[m.ins[convAt[i[a]]].at + 3]);
        }
        else if (in.op == spv::Op::OpFMod)
        {
            const uint32_t x = i[3];
            size_t at = SIZE_MAX;
            if (floorAt.count(x)) at = floorAt[x];
            else if (extractOf.count(x)) at = extractOf[x];
            if (at != SIZE_MAX) siteV4[at] = fragXY(code[m.ins[at].at + 5]);
        }
    }
    if (siteV4.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI1 = next++, cI4 = next++, cI5 = next++, cF1 = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    types.Op(spv::Op::OpConstant, { tInt, cI5, 5 });
    types.Op(spv::Op::OpConstant, { tFloat, cF1, 0x3F800000u });
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet, 6);

    out.clear();
    out.reserve(words + 96);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        auto st = siteV4.find(k);
        if (st == siteV4.end() || !st->second) { out.insert(out.end(), i, i + in.n); continue; }
        const bool conv = in.op == spv::Op::OpConvertFToS, dot = in.op == spv::Op::OpDot;
        const uint32_t xy = conv ? i[3] : dot ? (fragXY(i[3]) ? i[3] : i[4]) : i[5];
        spv::Op sop; const uint32_t* sh = m.At(xy, &sop);
        const uint32_t t2 = sh[1];
        const uint32_t fc = next++, fw = next++, pS = next++, fsSep = next++, pC = next++, cv = next++, pH = next++, half = next++, cw = next++, one = next++, a = next++, d = next++,
                       dr = next++, x = next++, x2 = next++, xy2 = next++;
        Emit e;
        e.Op(spv::Op::OpLoad, { st->second, fc, vFrag });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, fw, fc, 3 });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
        e.Op(spv::Op::OpLoad, { tFloat, fsSep, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, cv, pC });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pH, vStereo, cI5 });
        e.Op(spv::Op::OpLoad, { tFloat, half, pH });
        e.Op(spv::Op::OpFMul, { tFloat, cw, cv, fw });
        e.Op(spv::Op::OpFSub, { tFloat, one, cF1, cw });
        e.Op(spv::Op::OpFMul, { tFloat, a, fsSep, one });
        e.Op(spv::Op::OpFMul, { tFloat, d, a, half });
        e.Op(spv::Op::OpExtInst, { tFloat, dr, glsl, 1, d });   // Round: whole pixels, the pattern is per pixel
        e.Op(spv::Op::OpCompositeExtract, { tFloat, x, xy, 0 });
        e.Op(spv::Op::OpFSub, { tFloat, x2, x, dr });
        e.Op(spv::Op::OpCompositeInsert, { t2, xy2, x2, xy, 0 });
        if (conv) e.Op(spv::Op::OpConvertFToS, { i[1], i[2], xy2 });
        else if (dot) e.Op(spv::Op::OpDot, { i[1], i[2], xy == i[3] ? xy2 : i[3], xy == i[3] ? i[4] : xy2 });
        else e.Op(spv::Op::OpExtInst, { i[1], i[2], glsl, 8, xy2 });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (sites) *sites = (unsigned)siteV4.size();
    return true;
}

// Tile lists: an integer pixel from gl_FragCoord.xy, directly or through a function-local copy stored once, that a
// constant of 8 or more divides (or a shift of 3 or more) is read at the left's pixel; screen uvs keep this eye's.
bool PatchTileListsLeftFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* sites,
                          float vmClipZ, float vmSep, float vmConv)
{
    if (sites) *sites = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF, glsl = m.glsl;
    const uint32_t vFrag = m.vFragCoord;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    if (!m.fragment || !tFloat || !glsl || !vFrag || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    uint32_t tV4 = 0;
    for (const Ins& in : m.ins) if (in.op == spv::Op::OpLoad && code[in.at + 3] == vFrag) { tV4 = code[in.at + 1]; break; }
    if (!tV4) return false;
    const FragPixel px(m);
    // The integer pixel reaches a tile index through extracts and a local variable it is stored to and loaded from.
    auto tileIndexed = [&](uint32_t r) -> bool {
        std::unordered_set<uint32_t> vals{ r }, ptrs;
        for (int pass = 0; pass < 4; ++pass)
            for (size_t k = firstFunc; k < m.ins.size(); ++k)
            {
                const Ins& in = m.ins[k];
                const uint32_t* i = code + in.at;
                if (in.op == spv::Op::OpStore && vals.count(i[2])) ptrs.insert(i[1]);
                else if ((in.op == spv::Op::OpAccessChain || in.op == spv::Op::OpInBoundsAccessChain) && ptrs.count(i[3])) ptrs.insert(i[2]);
                else if (in.op == spv::Op::OpLoad && ptrs.count(i[3])) vals.insert(i[2]);
                else if ((in.op == spv::Op::OpCompositeExtract || in.op == spv::Op::OpBitcast) && vals.count(i[3])) vals.insert(i[2]);
            }
        for (size_t k = firstFunc; k < m.ins.size(); ++k)
        {
            const Ins& in = m.ins[k];
            const uint32_t* i = code + in.at;
            const bool div = in.op == spv::Op::OpUDiv || in.op == spv::Op::OpSDiv;
            const bool shr = in.op == spv::Op::OpShiftRightLogical || in.op == spv::Op::OpShiftRightArithmetic;
            if ((!div && !shr) || in.n < 5 || !vals.count(i[3])) continue;
            auto c = m.constBits.find(i[4]);
            if (c != m.constBits.end() && (div ? c->second >= 8 : c->second >= 3)) return true;
        }
        return false;
    };
    std::unordered_map<size_t, uint32_t> siteXY;   // conversion -> its .xy operand
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        if (in.op != spv::Op::OpConvertFToU && in.op != spv::Op::OpConvertFToS) continue;
        const uint32_t* i = code + in.at;
        const uint32_t src = px.XyOf(i[3]);
        if (src && px.PixelVec(src) && tileIndexed(i[2])) siteXY[k] = i[3];
    }
    if (siteXY.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI1 = next++, cI4 = next++, cI5 = next++, cF1 = next++, tS = next++, tPtrS = next++, vStereo = next++;
    const uint32_t cF0 = next++, cF2 = next++;
    uint32_t tBool = 0;   // the module's own bool type: a second OpTypeBool is invalid
    for (const Ins& in : m.ins) if (in.op == spv::Op::OpTypeBool) { tBool = code[in.at + 1]; break; }
    const bool newBool = !tBool;
    if (newBool) tBool = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    types.Op(spv::Op::OpConstant, { tInt, cI5, 5 });
    types.Op(spv::Op::OpConstant, { tFloat, cF1, 0x3F800000u });
    types.Op(spv::Op::OpConstant, { tFloat, cF0, 0u });
    types.Op(spv::Op::OpConstant, { tFloat, cF2, 0x40000000u });
    uint32_t cVmZ = 0, cVmSep = 0, cVmConv = 0;
    if (vmClipZ > 0.f)
    {
        uint32_t u;
        if (newBool) types.Op(spv::Op::OpTypeBool, { tBool });
        cVmZ = next++;    memcpy(&u, &vmClipZ, 4); types.Op(spv::Op::OpConstant, { tFloat, cVmZ, u });
        cVmSep = next++;  memcpy(&u, &vmSep, 4);   types.Op(spv::Op::OpConstant, { tFloat, cVmSep, u });
        cVmConv = next++; memcpy(&u, &vmConv, 4);  types.Op(spv::Op::OpConstant, { tFloat, cVmConv, u });
    }
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet, 6);

    out.clear();
    out.reserve(words + 80 * siteXY.size() + 64);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        auto st = siteXY.find(k);
        if (st == siteXY.end()) { out.insert(out.end(), i, i + in.n); continue; }
        const uint32_t xy = st->second;
        spv::Op sop; const uint32_t* sh = m.At(xy, &sop);
        const uint32_t t2 = sh[1];
        const uint32_t fc = next++, fw = next++, pS = next++, fsSep = next++, pC = next++, cv = next++, pH = next++, half = next++, cw = next++,
                       one = next++, a = next++, d = next++, dr = next++, x = next++, x2 = next++, xy2 = next++;
        Emit e;
        e.Op(spv::Op::OpLoad, { tV4, fc, vFrag });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, fw, fc, 3 });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
        e.Op(spv::Op::OpLoad, { tFloat, fsSep, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, cv, pC });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pH, vStereo, cI5 });
        e.Op(spv::Op::OpLoad, { tFloat, half, pH });
        uint32_t sepE = fsSep, convE = cv;
        if (cVmZ)
        {
            // The weapon's pixels: its clip z is FragCoord.z / FragCoord.w, as the vertex shader tested it.
            const uint32_t fz = next++, zc = next++, vm = next++, sV = next++, cV = next++;
            sepE = next++; convE = next++;
            e.Op(spv::Op::OpCompositeExtract, { tFloat, fz, fc, 2 });
            e.Op(spv::Op::OpFDiv, { tFloat, zc, fz, fw });
            e.Op(spv::Op::OpFOrdGreaterThan, { tBool, vm, zc, cVmZ });
            e.Op(spv::Op::OpFMul, { tFloat, sV, fsSep, cVmSep });
            e.Op(spv::Op::OpFMul, { tFloat, cV, cv, cVmConv });
            e.Op(spv::Op::OpSelect, { tFloat, sepE, vm, sV, fsSep });
            e.Op(spv::Op::OpSelect, { tFloat, convE, vm, cV, cv });
        }
        const uint32_t wd = next++, mw = next++, mx = next++, xc = next++;
        e.Op(spv::Op::OpFMul, { tFloat, cw, convE, fw });
        e.Op(spv::Op::OpFSub, { tFloat, one, cF1, cw });
        e.Op(spv::Op::OpFMul, { tFloat, a, sepE, one });
        e.Op(spv::Op::OpFMul, { tFloat, d, a, half });
        e.Op(spv::Op::OpExtInst, { tFloat, dr, glsl, 1, d });   // Round: whole pixels
        e.Op(spv::Op::OpCompositeExtract, { tFloat, x, xy, 0 });
        e.Op(spv::Op::OpFSub, { tFloat, x2, x, dr });
        // On the screen: a shift past its edge would read another tile row's list, or none. Never below the pixel's own
        // x: a set without a half width (the left eye's, shift 0) leaves it as it was.
        e.Op(spv::Op::OpFMul, { tFloat, wd, half, cF2 });
        e.Op(spv::Op::OpFSub, { tFloat, mw, wd, cF1 });
        e.Op(spv::Op::OpExtInst, { tFloat, mx, glsl, 40, mw, x });   // FMax
        e.Op(spv::Op::OpExtInst, { tFloat, xc, glsl, 43, x2, cF0, mx });   // FClamp
        e.Op(spv::Op::OpCompositeInsert, { t2, xy2, xc, xy, 0 });
        e.Op(in.op, { i[1], i[2], xy2 });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (sites) *sites = (unsigned)siteXY.size();
    return true;
}

// Screen march: P = rows . (point, 1), rows four vec4s of a uniform block; P is stored, and later P's xyz minus a start
// stored from (FragCoord.xy * k, ...) steps the ray. P is a uv-space clip, so the eye's shift is halved.
bool PatchScreenMarchFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* marches,
                        float vmClipZ, float vmSep, float vmConv)
{
    if (marches) *marches = 0;
    Fs m;
    if (!m.Split(code, words)) return false;
    m.Scan();
    uint32_t tFloat = m.tFloat, tInt = m.tInt, tPtrUF = m.tPtrUF, glsl = m.glsl;
    const uint32_t vFrag = m.vFragCoord;
    const size_t firstType = m.firstType, firstFunc = m.firstFunc;
    if (!m.fragment || !tFloat || !glsl || !vFrag || firstType == SIZE_MAX || firstFunc == SIZE_MAX) return false;
    uint32_t tV4 = 0;
    for (const Ins& in : m.ins) if (in.op == spv::Op::OpLoad && code[in.at + 3] == vFrag) { tV4 = code[in.at + 1]; break; }
    if (!tV4) return false;
    const FragPixel px(m);
    std::unordered_set<uint32_t> tMat4;
    std::unordered_map<uint32_t, std::vector<uint32_t>> storesTo;   // variable -> the values stored to it whole
    for (const Ins& in : m.ins)
    {
        const uint32_t* i = code + in.at;
        if (in.op == spv::Op::OpTypeMatrix && i[3] == 4 && m.vecOf.count(i[2]) && m.vecOf.at(i[2]) == 4) tMat4.insert(i[1]);
        else if (in.op == spv::Op::OpStore) storesTo[i[1]].push_back(i[2]);
    }
    auto funcVar = [&](uint32_t id) { auto c = m.varClass.find(id); return c != m.varClass.end() && c->second == (uint32_t)spv::StorageClass::Function; };
    auto storedOnce = [&](uint32_t var) -> uint32_t { auto s = storesTo.find(var); return s != storesTo.end() && s->second.size() == 1 ? s->second[0] : 0; };
    // A 4x4 matrix read out of a uniform block, directly or through a function argument's variable.
    auto blockRows = [&](uint32_t id) -> bool {
        if (funcVar(id)) id = storedOnce(id);
        spv::Op op; const uint32_t* c = m.At(id, &op);
        return c && tMat4.count(c[1]) && m.FromBuffer(id, 0);
    };
    // The projections: result id -> instruction.
    std::unordered_map<uint32_t, size_t> proj;
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.n < 4 || !m.vecOf.count(i[1]) || m.vecOf.at(i[1]) != 4) continue;
        bool rows = false;
        if (in.op == spv::Op::OpFunctionCall) for (uint16_t a = 4; a < in.n && !rows; ++a) rows = blockRows(i[a]);
        else if (in.op == spv::Op::OpMatrixTimesVector) rows = blockRows(i[3]);
        else if (in.op == spv::Op::OpVectorTimesMatrix) rows = blockRows(i[4]);
        if (rows) proj[i[2]] = k;
    }
    if (proj.empty()) return false;
    // The variable a vector is loaded from, through a shuffle of one vector.
    auto varOf = [&](uint32_t id) -> uint32_t {
        spv::Op op; const uint32_t* i = m.At(id, &op);
        if (i && op == spv::Op::OpVectorShuffle && i[3] == i[4]) i = m.At(i[3], &op);
        return i && op == spv::Op::OpLoad && funcVar(i[3]) ? i[3] : 0;
    };
    // The march's start: stored once, x = (FragCoord.xy * k).x.
    auto pixelStart = [&](uint32_t var) -> bool {
        spv::Op op; const uint32_t* c = m.At(storedOnce(var), &op);
        if (!c || op != spv::Op::OpCompositeConstruct) return false;
        const uint32_t* x = m.At(c[3], &op);
        if (x && op == spv::Op::OpCompositeExtract && (x[0] >> 16) == 5 && x[4] == 0) x = m.At(x[3], &op);
        if (!x || op != spv::Op::OpFMul) return false;
        auto pix = [&](uint32_t id) { const uint32_t s = px.XyOf(id); return s && px.PixelVec(s); };
        return pix(x[3]) || pix(x[4]);
    };
    std::unordered_set<size_t> sites;
    for (size_t k = firstFunc; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (in.op != spv::Op::OpFSub || !m.vecOf.count(i[1])) continue;
        const uint32_t a = varOf(i[3]), b = varOf(i[4]);
        if (!a || !b) continue;
        for (const uint32_t endVar : { a, b })
        {
            if (!pixelStart(endVar == a ? b : a)) continue;
            for (const uint32_t v : storesTo[endVar]) { auto p = proj.find(v); if (p != proj.end()) sites.insert(p->second); }
        }
    }
    if (sites.empty()) return false;

    uint32_t next = code[3];
    Emit deco, types;
    if (!tInt) { tInt = next++; types.Op(spv::Op::OpTypeInt, { tInt, 32, 1 }); }
    const uint32_t cI1 = next++, cI2 = next++, cI4 = next++, cHalf = next++, tS = next++, tPtrS = next++, vStereo = next++;
    types.Op(spv::Op::OpConstant, { tInt, cI1, 1 });
    types.Op(spv::Op::OpConstant, { tInt, cI2, 2 });
    types.Op(spv::Op::OpConstant, { tInt, cI4, 4 });
    types.Op(spv::Op::OpConstant, { tFloat, cHalf, 0x3F000000u });
    uint32_t cVmZ = 0, cVmSep = 0, cVmConv = 0, tBool = 0;
    if (vmClipZ > 0.f)
    {
        for (const Ins& in : m.ins) if (in.op == spv::Op::OpTypeBool) { tBool = code[in.at + 1]; break; }
        if (!tBool) { tBool = next++; types.Op(spv::Op::OpTypeBool, { tBool }); }
        cVmZ = next++;    types.Op(spv::Op::OpConstant, { tFloat, cVmZ, Bits(vmClipZ) });
        cVmSep = next++;  types.Op(spv::Op::OpConstant, { tFloat, cVmSep, Bits(vmSep) });
        cVmConv = next++; types.Op(spv::Op::OpConstant, { tFloat, cVmConv, Bits(vmConv) });
    }
    DeclareStereo(types, deco, tFloat, tS, tPtrS, vStereo, tPtrUF, next, stereoSet);

    out.clear();
    out.reserve(words + 64 * sites.size() + 64);
    out.insert(out.end(), code, code + 5);
    for (size_t k = 0; k < m.ins.size(); ++k)
    {
        const Ins& in = m.ins[k];
        const uint32_t* i = code + in.at;
        if (k == firstType) out.insert(out.end(), deco.v.begin(), deco.v.end());
        if (k == firstFunc) out.insert(out.end(), types.v.begin(), types.v.end());
        if (!sites.count(k)) { out.insert(out.end(), i, i + in.n); continue; }
        // T = the projection; result.x = T.x + fsSep/2*(max(T.w, wMin) - conv). fsSep is 0 in the left eye.
        const uint32_t T = next++, pS = next++, fsSep = next++, pC = next++, cv = next++, pW = next++, wMin = next++,
                       x = next++, w = next++, wc = next++, dw = next++, s = next++, h = next++, x2 = next++;
        out.insert(out.end(), i, i + in.n);
        out[out.size() - in.n + 2] = T;
        Emit e;
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pS, vStereo, cI4 });
        e.Op(spv::Op::OpLoad, { tFloat, fsSep, pS });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pC, vStereo, cI1 });
        e.Op(spv::Op::OpLoad, { tFloat, cv, pC });
        e.Op(spv::Op::OpAccessChain, { tPtrUF, pW, vStereo, cI2 });
        e.Op(spv::Op::OpLoad, { tFloat, wMin, pW });
        uint32_t sepE = fsSep, convE = cv;
        if (cVmZ)
        {
            // The weapon's pixels (clip z = FragCoord.z / FragCoord.w above vmClipZ) took the scaled shift.
            const uint32_t fc = next++, fz = next++, fw = next++, zc = next++, vm = next++, sV = next++, cV = next++;
            sepE = next++; convE = next++;
            e.Op(spv::Op::OpLoad, { tV4, fc, vFrag });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, fz, fc, 2 });
            e.Op(spv::Op::OpCompositeExtract, { tFloat, fw, fc, 3 });
            e.Op(spv::Op::OpFDiv, { tFloat, zc, fz, fw });
            e.Op(spv::Op::OpFOrdGreaterThan, { tBool, vm, zc, cVmZ });
            e.Op(spv::Op::OpFMul, { tFloat, sV, fsSep, cVmSep });
            e.Op(spv::Op::OpFMul, { tFloat, cV, cv, cVmConv });
            e.Op(spv::Op::OpSelect, { tFloat, sepE, vm, sV, fsSep });
            e.Op(spv::Op::OpSelect, { tFloat, convE, vm, cV, cv });
        }
        e.Op(spv::Op::OpCompositeExtract, { tFloat, x, T, 0 });
        e.Op(spv::Op::OpCompositeExtract, { tFloat, w, T, 3 });
        e.Op(spv::Op::OpExtInst, { tFloat, wc, glsl, 40, w, wMin });   // FMax
        e.Op(spv::Op::OpFSub, { tFloat, dw, wc, convE });
        e.Op(spv::Op::OpFMul, { tFloat, s, sepE, dw });
        e.Op(spv::Op::OpFMul, { tFloat, h, s, cHalf });
        e.Op(spv::Op::OpFAdd, { tFloat, x2, x, h });
        e.Op(spv::Op::OpCompositeInsert, { i[1], i[2], x2, T, 0 });
        out.insert(out.end(), e.v.begin(), e.v.end());
    }
    out[3] = next;
    if (marches) *marches = (unsigned)sites.size();
    return true;
}

}} // namespace wiz3d::vk
