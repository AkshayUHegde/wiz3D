/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The constants path: at submit, the uniform ranges of bound sets are read through the game's mapping, camera blocks
// are learned per (layout, binding), and a sibling copy rewritten for the right eye's camera is what its sets bind.

#include "Layer.h"
#include "FamilyVK.h"
#include "../S3DAPI/GlobalData.h"
#include "../S3DAPI/ReadData.h"
#include "../Shared/StereoMatrix.h"
#include <unordered_set>
#include <algorithm>
#include <stdio.h>
#include <thread>
#include <condition_variable>
#include <atomic>

namespace wiz3d { namespace vk {

// The mirror copies read the game's write-combined uniform memory, which is latency-bound and so scales with threads.
struct CopyJob { unsigned char* dst; const unsigned char* src; size_t bytes; };

class CopyPool
{
public:
    void Run(const std::vector<CopyJob>& jobs)
    {
        if (jobs.size() < 8) { for (const CopyJob& j : jobs) memcpy(j.dst, j.src, j.bytes); return; }
        std::call_once(m_once, [this] { Spawn(); });
        if (m_threads.empty()) { for (const CopyJob& j : jobs) memcpy(j.dst, j.src, j.bytes); return; }
        {
            std::lock_guard<std::mutex> g(m_mx);
            m_jobs = &jobs;
            m_next.store(0, std::memory_order_relaxed);
            m_active = m_threads.size();
            ++m_ticket;
        }
        m_cv.notify_all();
        Drain(&jobs);                       // the calling thread copies too
        std::unique_lock<std::mutex> g(m_mx);
        m_doneCv.wait(g, [this] { return m_active == 0; });
        m_jobs = nullptr;
    }
private:
    void Drain(const std::vector<CopyJob>* js)
    {
        if (!js) return;
        for (;;)
        {
            const size_t i = m_next.fetch_add(1, std::memory_order_relaxed);
            if (i >= js->size()) return;
            const CopyJob& j = (*js)[i];
            memcpy(j.dst, j.src, j.bytes);
        }
    }
    // Detached, never joined: a layer DLL cannot join at unload (loader lock), and a joinable std::thread's destructor
    // would call std::terminate when the static pool is destroyed.
    void Spawn()
    {
        const unsigned hw = std::thread::hardware_concurrency();
        // One per logical processor (workers + the calling thread): these threads stall on PCIe rather than
        // compute, so SMT siblings are useful here.
        const unsigned total = hw ? hw : 8;
        const unsigned n = total > 1 ? total - 1 : 0;
        Log("[const] copy pool: %u threads\n", n + 1);
        for (unsigned i = 0; i < n; ++i)
            m_threads.emplace_back([this] {
                uint64_t seen = 0;
                for (;;)
                {
                    const std::vector<CopyJob>* js = nullptr;
                    {
                        std::unique_lock<std::mutex> g(m_mx);
                        m_cv.wait(g, [this, seen] { return m_ticket != seen; });
                        seen = m_ticket;
                        js = m_jobs;
                    }
                    Drain(js);
                    {
                        std::lock_guard<std::mutex> g(m_mx);
                        if (--m_active == 0) m_doneCv.notify_one();
                    }
                }
            });
        for (std::thread& t : m_threads) t.detach();
    }
    std::vector<std::thread> m_threads;
    std::mutex m_mx;
    std::condition_variable m_cv, m_doneCv;
    std::once_flag m_once;
    const std::vector<CopyJob>* m_jobs = nullptr;
    std::atomic<size_t> m_next{0};
    size_t m_active = 0;
    uint64_t m_ticket = 0;
};
static CopyPool g_copyPool;

struct CmdState;
// CommandBuffer.cpp, lock held.
void CmdStateTakeUsedSets(DeviceData* d, VkCommandBuffer cb, std::vector<UsedSet>& out);

// Camera block: the right eye gets its own projection P' (the shader shift x += sep*(w - conv) exactly), and every
// matrix classified against the block's own V and P is rewritten to match (K = P^-1 P').
struct M4 { float m[16]; };    // canonical row-vector layout, m[r*4+c]

static M4 M4Identity() { M4 r = {}; r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.f; return r; }
static M4 M4Mul(const M4& a, const M4& b)
{
    M4 r;
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j)
    {
        float s = 0.f;
        for (int k = 0; k < 4; ++k) s += a.m[i * 4 + k] * b.m[k * 4 + j];
        r.m[i * 4 + j] = s;
    }
    return r;
}
static bool M4Inverse(const M4& a, M4& out)
{
    double w[4][8];
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) { w[i][j] = a.m[i * 4 + j]; w[i][j + 4] = (i == j) ? 1.0 : 0.0; }
    for (int col = 0; col < 4; ++col)
    {
        int piv = col;
        for (int r = col + 1; r < 4; ++r) if (fabs(w[r][col]) > fabs(w[piv][col])) piv = r;
        if (fabs(w[piv][col]) < 1e-12) return false;
        if (piv != col) for (int j = 0; j < 8; ++j) { double t = w[col][j]; w[col][j] = w[piv][j]; w[piv][j] = t; }
        const double inv = 1.0 / w[col][col];
        for (int j = 0; j < 8; ++j) w[col][j] *= inv;
        for (int r = 0; r < 4; ++r)
        {
            if (r == col) continue;
            const double f = w[r][col];
            if (f != 0.0) for (int j = 0; j < 8; ++j) w[r][j] -= f * w[col][j];
        }
    }
    for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) out.m[i * 4 + j] = (float)w[i][j + 4];
    return true;
}
static float M4MaxAbs(const M4& a) { float mx = 0.f; for (int i = 0; i < 16; ++i) mx = fmaxf(mx, fabsf(a.m[i])); return mx; }
static M4 ReadM4(const float* f, bool transposed) { M4 r; CanonicalizeMat4(f, transposed, r.m); return r; }
static void WriteM4(float* f, bool transposed, const M4& a)
{
    if (!transposed) { memcpy(f, a.m, sizeof(a.m)); return; }
    for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) f[c * 4 + r] = a.m[r * 4 + c];
}

// The camera tests' limits (FamilyVK.h); a family may widen them.
static const CameraLimits& Limits() { return Traits().limits; }

// Bare perspective projection: diagonal x/y scales, no off-axis terms, w = +/-z, no translation in x/y, P44 = 0
// (infinite reversed-Z has P33 = 0).
static bool IsBareProjection(const M4& p)
{
    const float* m = p.m;
    if (!AllFinite16(m)) return false;
    const float zoom = Limits().zoom;
    if (m[0] < 0.2f || m[0] > zoom || fabsf(m[5]) < 0.2f || fabsf(m[5]) > zoom) return false;
    const float ratio = fabsf(m[5]) / m[0];
    if (ratio < 1.1f || ratio > 2.7f) return false;
    if (m[1] != 0.f || m[2] != 0.f || m[3] != 0.f || m[4] != 0.f || m[6] != 0.f || m[7] != 0.f) return false;
    // The off-axis slots may carry sub-pixel jitter and our convergence skew: a jittered and an unjittered copy of
    // one camera must both pass, or a prepass and its shading pass disagree about the eye.
    if (fabsf(m[8]) > 0.1f || fabsf(m[9]) > 0.1f || fabsf(fabsf(m[11]) - 1.f) > 1e-4f) return false;
    if (fabsf(m[12]) > 0.1f || fabsf(m[13]) > 0.1f || m[15] != 0.f) return false;
    return m[14] != 0.f || m[10] != 0.f;
}

// Rigid view: orthonormal 3x3, last column (0,0,0,1).
static bool IsRigidView(const M4& v)
{
    const float* m = v.m;
    if (!AllFinite16(m)) return false;
    if (m[3] != 0.f || m[7] != 0.f || m[11] != 0.f || fabsf(m[15] - 1.f) > 1e-5f) return false;
    for (int r = 0; r < 3; ++r)
    {
        const float n = m[r * 4] * m[r * 4] + m[r * 4 + 1] * m[r * 4 + 1] + m[r * 4 + 2] * m[r * 4 + 2];
        if (fabsf(n - 1.f) > 0.01f) return false;
        for (int q = r + 1; q < 3; ++q)
            if (fabsf(m[r * 4] * m[q * 4] + m[r * 4 + 1] * m[q * 4 + 1] + m[r * 4 + 2] * m[q * 4 + 2]) > 0.01f) return false;
    }
    return true;
}

// Kinds by what a matrix does, not its shape: toClip becomes M*K, fromClip K^-1*M; view-space kinds and the
// [I | camera position] block stay the game's, as the whole shift is a clip-space one.
enum CamKind : uint8_t { K_None, K_P, K_ToClip, K_FromClip, K_ToView, K_FromView, K_CamOffset, K_RayBasis, K_UnitBasis };
static const char* KindName(CamKind k) { static const char* n[] = { "?", "P", "toClip", "fromClip", "toView", "fromView", "camOffset", "rayBasis", "unitBasis" }; return n[k]; }

// Last column (0,0,0,1): no perspective, i.e. a world/view-space transform.
static bool IsAffine(const M4& m)
{
    const float r = fabsf(m.m[3]) + fabsf(m.m[7]) + fabsf(m.m[11]) + fabsf(m.m[15] - 1.f);
    return r <= 1e-4f * (1.f + M4MaxAbs(m));
}

// 3x3 comparison, optionally against the transpose (a rotation's inverse).
static bool Basis3Near(const M4& a, const M4& b, bool transpose)
{
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k)
            if (fabsf(a.m[r * 4 + k] - (transpose ? b.m[k * 4 + r] : b.m[r * 4 + k])) > 0.002f) return false;
    return true;
}

static bool IsIdentity3(const M4& m)
{
    for (int r = 0; r < 3; ++r)
        for (int k = 0; k < 3; ++k)
            if (fabsf(m.m[r * 4 + k] - (r == k ? 1.f : 0.f)) > 1e-4f) return false;
    return true;
}

// A packed 3x3 into the 4-wide layout the tests expect.
static void UnpackBasis3x3(const unsigned char* base, uint32_t off, uint32_t stride, bool rowMajor, float* out12)
{
    const unsigned char* p = base + off;
    for (int i = 0; i < 3; ++i)
    {
        const float* v = (const float*)(p + (size_t)i * stride);
        for (int j = 0; j < 3; ++j)
        {
            if (rowMajor) out12[i * 4 + j] = v[j];   // v is row i
            else          out12[j * 4 + i] = v[j];   // v is column i
        }
    }
}

// A reconstruction basis R (ray = R * (ndc.x, ndc.y, 1)): columns tan(fovX/2), tan(fovY/2) and the unit forward,
// which must align with the camera's forward `fwd` (shape alone admits bone and instance rotations).
static bool IsRayBasis(const float* f, const float* fwd)
{
    for (int i = 0; i < 12; ++i) if (!(f[i] == f[i]) || fabsf(f[i]) > 1e30f) return false;
    const float cx[3] = { f[0], f[4], f[8] };
    const float cy[3] = { f[1], f[5], f[9] };
    const float cz[3] = { f[2], f[6], f[10] };
    auto dot = [](const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; };
    const float nx = sqrtf(dot(cx, cx)), ny = sqrtf(dot(cy, cy)), nz = sqrtf(dot(cz, cz));
    if (nz < 0.9f || nz > 1.1f) return false;
    const float tele = Limits().tele;
    if (nx < tele || nx > 3.f || ny < tele || ny > 3.f) return false;
    const float ratio = nx / ny;
    if (ratio < 1.1f || ratio > 2.7f) return false;               // same aspect gate as IsBareProjection
    if (fabsf(dot(cx, cz)) > 0.05f * nx || fabsf(dot(cy, cz)) > 0.05f * ny) return false;
    float cr[3] = { cx[1]*cy[2] - cx[2]*cy[1], cx[2]*cy[0] - cx[0]*cy[2], cx[0]*cy[1] - cx[1]*cy[0] };
    if (fabsf(dot(cr, cz)) < 0.85f * nx * ny) return false;
    // An exact identity passes every test above but is an unwritten block.
    if (fabsf(nx - 1.f) < 1e-4f && fabsf(ny - 1.f) < 1e-4f && fabsf(cz[0]) < 1e-4f && fabsf(cz[1]) < 1e-4f) return false;
    // Camera-forward alignment: without it, scalar-packed blocks give false positives the patcher would write into.
    if (!fwd) return false;
    const float align = fabsf(dot(cz, fwd));
    if (align < 0.995f * nz) return false;
    return true;
}

// unitCameraBlock: rows 0-2 a unit view basis with tanX/tanY in .w, row 3 the camera position, row 4 xy the NDC
// jitter; ray = ((ndc + jitter) * tan, -1) * basis, which IsRayBasis cannot pass (tan is not in the columns).
static bool IsUnitCameraBasis(const float* f, const float* fwd)
{
    for (int i = 0; i < 20; ++i) if (!(f[i] == f[i]) || fabsf(f[i]) > 1e7f) return false;
    auto dot = [](const float* a, const float* b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; };
    const float* r0 = f; const float* r1 = f + 4; const float* r2 = f + 8;
    if (fabsf(dot(r0, r0) - 1.f) > 0.01f || fabsf(dot(r1, r1) - 1.f) > 0.01f || fabsf(dot(r2, r2) - 1.f) > 0.01f) return false;
    if (fabsf(dot(r0, r1)) > 0.01f || fabsf(dot(r0, r2)) > 0.01f || fabsf(dot(r1, r2)) > 0.01f) return false;
    const float tx = r0[3], ty = r1[3];
    if (!(ty > Limits().teleY && tx < 5.f)) return false;
    const float ratio = tx / ty;
    if (ratio < 1.1f || ratio > 2.7f) return false;                       // same aspect gate as IsBareProjection
    if (fabsf(f[16]) > 0.05f || fabsf(f[17]) > 0.05f) return false;       // jitter is sub-pixel NDC
    // Only against a known main-camera forward: a light basis also passes the shape tests.
    if (!fwd || fabsf(dot(r2, fwd)) < 0.99f) return false;
    return true;
}

struct PSlot { unsigned reg; bool transposed; };
// Stride 16 is the ordinary register layout addressed by `reg`; anything else is scalar_block_layout, at byteOff.
struct CamSlot
{
    unsigned reg; bool transposed; CamKind kind; unsigned pIndex;
    uint32_t byteOff = 0; uint8_t stride = 16;
};
struct CameraLayout
{
    bool valid = false;
    bool hasV = false;              // a view rotation lives here, so parallax can go into it
    unsigned vReg = 0;              // where that view sits, so its forward can be published
    bool     vTransposed = false;
    std::vector<PSlot>   ps;        // every camera projection in the block (current, previous frame)
    std::vector<CamSlot> slots;
    unsigned applied = 0, skipped = 0;
};

// clipSlotStraddle, at learning: an affine camera matrix ([I | t], the view rotation or its transpose) starting 1-3
// registers into the window at `reg` means the window spans two members and is no clip matrix.
static bool StraddlesAffine(const float* f, size_t reg, size_t regs, const M4* V)
{
    for (size_t k = 1; k <= 3 && reg + k + 4 <= regs; ++k)
        for (int o = 0; o < 2; ++o)
        {
            const M4 m = ReadM4(f + (reg + k) * 4, o == 1);
            if (!AllFinite16(m.m) || !IsAffine(m)) continue;
            if (IsIdentity3(m) || (V && (Basis3Near(m, *V, false) || Basis3Near(m, *V, true)))) return true;
        }
    return false;
}

static bool LearnCamera(const float* f, size_t regs, const std::unordered_map<uint32_t, unsigned>* used, CameraLayout& out, const char** why, bool allowBasisOnly, const float* fwd)
{
    CameraLayout L;
    // Projections, the ones the vertex shaders multiply by first.
    for (int pass = 0; pass < 2; ++pass)
        for (size_t reg = 0; reg + 4 <= regs; ++reg)
        {
            const bool isUsed = used && used->count((uint32_t)reg * 16) != 0;
            if ((pass == 0) != isUsed) continue;
            bool dup = false;
            for (const PSlot& p : L.ps) if (p.reg == (unsigned)reg) dup = true;
            if (dup) continue;
            for (int o = 0; o < 2; ++o)
                if (IsBareProjection(ReadM4(f + reg * 4, o == 1)))
                {
                    L.ps.push_back({ (unsigned)reg, o == 1 });
                    L.slots.push_back({ (unsigned)reg, o == 1, K_P, (unsigned)(L.ps.size() - 1) });
                    break;
                }
        }
    if (L.ps.empty())
    {
        // A block with no projection may still hold a reconstruction basis: learned only in ranges a full-screen
        // draw consumes, as the shape test alone admits skinning and instance blocks.
        if (!allowBasisOnly) { *why = "noP"; return false; }
        for (size_t reg = 0; reg + 3 <= regs; ++reg)
            if (IsRayBasis(f + reg * 4, fwd))
            {
                L.slots.push_back({ (unsigned)reg, false, K_RayBasis, 0 });
                reg += 2;
            }
        if (L.slots.empty()) { *why = "noP"; return false; }
        L.valid = true;
        out = L;
        return true;
    }

    std::vector<M4> P, invP;
    std::vector<char> hasInv;
    for (const PSlot& p : L.ps)
    {
        M4 m = ReadM4(f + p.reg * 4, p.transposed), inv;
        const bool ok = M4Inverse(m, inv);
        if (!ok && !Limits().singularP) { *why = "singularP"; return false; }
        P.push_back(m);
        invP.push_back(ok ? inv : M4Identity());
        hasInv.push_back(ok);
    }

    // The view rotation: rigid, and not the [I | camera position] block. The
    // vertex chain's own offsets settle which one it is when several qualify.
    bool haveV = false;
    unsigned vReg = 0;
    bool vTrans = false;
    M4 V;
    for (int pass = 0; pass < 2 && !haveV; ++pass)
        for (size_t reg = 0; reg + 4 <= regs && !haveV; ++reg)
        {
            const bool isUsed = used && used->count((uint32_t)reg * 16) != 0;
            if ((pass == 0) != isUsed) continue;
            for (int o = 0; o < 2 && !haveV; ++o)
            {
                M4 m = ReadM4(f + reg * 4, o == 1);
                if (!IsAffine(m) || IsIdentity3(m) || !IsRigidView(m)) continue;
                V = m;
                haveV = true;
                vReg = (unsigned)reg;
                vTrans = (o == 1);
            }
        }
    // Not fatal: a block with no view rotation still carries convergence and
    // any clip-space products, it just cannot hold the parallax translation.
    L.hasV = haveV;
    L.vReg = vReg;
    L.vTransposed = vTrans;

    std::vector<bool> taken(regs + 4, false);
    for (const PSlot& p : L.ps)
        for (unsigned k = p.reg; k < p.reg + 4 && k < regs; ++k) taken[k] = true;
    for (size_t reg = 0; reg + 4 <= regs; ++reg)
    {
        if (taken[reg]) continue;
        if (reg + 3 <= regs && !taken[reg + 1] && !taken[reg + 2] && IsRayBasis(f + reg * 4, fwd))
        {
            L.slots.push_back({ (unsigned)reg, false, K_RayBasis, 0 });
            for (size_t k = reg; k < reg + 3; ++k) taken[k] = true;
            reg += 2;
            continue;
        }
        bool done = false;
        for (int o = 0; o < 2 && !done; ++o)
        {
            M4 m = ReadM4(f + reg * 4, o == 1);
            if (!AllFinite16(m.m)) continue;
            if (IsAffine(m))
            {
                if (!L.hasV) continue;
                // [I | camera position]: learned as camOffset, which publishes the camera position.
                if (IsIdentity3(m))
                {
                    if (m.m[12] != 0.f || m.m[13] != 0.f || m.m[14] != 0.f)
                    { L.slots.push_back({ (unsigned)reg, o == 1, K_CamOffset, 0 }); done = true; }
                    continue;
                }
                if (Basis3Near(m, V, false)) { L.slots.push_back({ (unsigned)reg, o == 1, K_ToView, 0 }); done = true; }
                else if (Basis3Near(m, V, true)) { L.slots.push_back({ (unsigned)reg, o == 1, K_FromView, 0 }); done = true; }
                continue;
            }
            if (Traits().clipSlotStraddle && StraddlesAffine(f, reg, regs, L.hasV ? &V : nullptr))
            {
                static std::atomic<int> s_log{8};
                if (s_log.fetch_sub(1, std::memory_order_relaxed) > 0)
                    Log("[cam] no clip slot at reg %zu of %zu: one of the block's affine matrices starts inside the window (clip-slot straddle)\n", reg, regs);
                break;
            }
            for (size_t pi = 0; pi < P.size() && !done; ++pi)
            {
                if (!hasInv[pi]) continue;
                if (IsAffine(M4Mul(m, invP[pi]))) { L.slots.push_back({ (unsigned)reg, o == 1, K_ToClip, (unsigned)pi }); done = true; }
                else if (IsAffine(M4Mul(P[pi], m))) { L.slots.push_back({ (unsigned)reg, o == 1, K_FromClip, (unsigned)pi }); done = true; }
            }
        }
        if (done) for (size_t k = reg; k < reg + 4 && k < regs; ++k) taken[k] = true;
    }
    L.valid = true;
    out = L;
    return true;
}

struct Constants;
static bool ApplyCamera(Constants* c, unsigned char* dst, const float* src, size_t regs, CameraLayout& L, float eyeSep, float convDist, uint64_t consumerMask, bool* wroteForward = nullptr, uint64_t chainStarts = 0);

struct Learned
{
    unsigned sightings = 0;                // only counted until the block is learned
    unsigned rayFails = 0;                 // consecutive failed re-verifies of a basis-only key
    CameraLayout cam;
    unsigned retryFrame = 0;               // frame of the last retry after the give-up
    unsigned retryLeft = 0;                // tries left in that frame (a layout's slices can hold other blocks)
    const char* lastFail = "none";         // why the last learn attempt failed
    bool evicting = false;                 // claimedKeysNeedP dropped it this submit
    uint64_t chainStarts = 0;              // clipSlotStraddle: registers where claiming position chains load a matrix
};

struct Constants
{
    std::mutex lock;                       // guards everything below; submits may come from several threads
    std::mutex collect;                    // sharedCollect: one submit collects at a time, under a shared device lock
    unsigned collectRetries = 0;           // sharedCollect: sets collected again under the exclusive lock (a first map)
    std::unordered_map<uint64_t, Learned> learned;   // key: layout handle ^ binding
    // Per report window
    unsigned ranges = 0, unmapped = 0, scans = 0, copies = 0, noSibling = 0;
    unsigned storageRanges = 0;            // of `ranges`, how many Phase 1 let through
    unsigned layerMaps = 0, unmappable = 0;   // memory we mapped ourselves, and memory no host can reach
    unsigned long long copyBytes = 0;
    unsigned siblings = 0;
    int cameraDumpBudget = 4;
    unsigned camApplied = 0, camSkipped = 0, camRanges = 0;
    unsigned projRefused = 0;              // claimed ranges whose first projection failed IsBareProjection this window
    float refusedP00 = 0.f, refusedP11 = 0.f;   // the last such projection, for the [cam] row
    unsigned typeHits[32] = {};            // memory types the constants are read out of
    unsigned rayPatched = 0;               // reconstruction bases rewritten for the right eye
    unsigned unitPatched = 0, prevPatched = 0;   // unit camera blocks / unclaimed previous-frame projections rewritten
    unsigned claimConflicts = 0;           // patched ranges also read by a draw that deliberately skipped its claim
    unsigned staleShadowed = 0;            // ranges a shadow reads from our sibling that no learned key refreshes
    std::unordered_set<VkPipeline> conflictNamed;   // skip-claim pipelines already named in the log
    // All under `lock`. unpatchedKeys; re-point keys; claimedKeysNeedP: uniform keys of a claiming draw's set not
    // learned with a P; claimedKeyRetry: such position keys; claimMirror: keys on the claim sibling, and vetoed ones.
    std::unordered_set<uint64_t> claimedKeys, skipPosKeys;
    std::unordered_set<uint64_t> repointKeys;
    std::unordered_set<uint64_t> geomKeys;
    std::unordered_set<uint64_t> posKeys;
    std::unordered_set<uint64_t> claimMirrorKeys;
    std::unordered_set<uint64_t> claimVetoKeys;              // claimed keys sharing bytes with other geometry
    // claimedForward: this submit's claimed view rotations with counts; the frame's best is published next frame.
    struct FwdVote { float f[3]; unsigned n = 0; };      // forward of a bit-exact rotation, and its count
    std::unordered_map<uint64_t, FwdVote> fwdVotes;
    float camFwdVote[3] = {};               // the frame's majority claimed forward; a basis may match it or camFwd
    bool  camFwdVoteValid = false;
    unsigned frameBestN = 0, publishedN = 0;
    uint32_t voteFrame = UINT32_MAX;
    float lastP11 = 0.f;                   // camera projection _11, for the composite crop
    unsigned invApplied = 0, invRejected = 0;   // fromClip slots that were / were not the range's own P^-1
    // Claimed ranges per projection this submit (P11 * 1000): VKMajorityP11 publishes the majority camera's scale,
    // not the last one applied, when a frame patches several projections.
    std::unordered_map<int, unsigned> p11Hist;
    float camFwd[3] = {};                  // world-space camera forward from the validated V
    bool  camFwdValid = false;
    // The reconstruction's missing constant term goes into the camera position.
    // The ordinary block carries it; the packed one is matched against it.
    float camPos[3] = {};
    bool  camPosValid = false;
    float lastSep = -1.f, lastConv = -1.f;   // preset the quarantine list belongs to
};

// A basis is the main camera's when it aligns with either forward: the last claimed view (key order, so right in some
// sessions only) or the frame's majority claimed rotation. Lock: c->lock.
static bool UnitBasisOk(const Constants* c, const float* u)
{
    return IsUnitCameraBasis(u, c->camFwdValid ? c->camFwd : nullptr) || (c->camFwdVoteValid && IsUnitCameraBasis(u, c->camFwdVote));
}
static bool RayBasisOk(const Constants* c, const float* t)
{
    return IsRayBasis(t, c->camFwdValid ? c->camFwd : nullptr) || (c->camFwdVoteValid && IsRayBasis(t, c->camFwdVote));
}

// Writes the right-eye block into dst (already a copy of src): the eye eyeSep world units along the camera's x axis,
// zero parallax at convDist.
static bool ApplyCamera(Constants* c, unsigned char* dst, const float* src, size_t regs, CameraLayout& L, float eyeSep, float convDist, uint64_t consumerMask, bool* wroteForward, uint64_t chainStarts)
{
    if (L.ps.empty())
    {
        bool hasRay = false;
        for (const CamSlot& sl : L.slots) if (sl.kind == K_RayBasis || sl.kind == K_UnitBasis) hasRay = true;
        if (!hasRay) return false;
    }
    // The whole shift lives in the projection, as on DX11 (a w-proportional skew, then a clip-space constant), so world
    // and view space stay identical between the eyes, and so does everything built from them that we cannot reach.
    M4 T = M4Identity();
    std::vector<M4> Pp(L.ps.size()), K(L.ps.size()), Kinv(L.ps.size()), P0(L.ps.size());
    std::vector<char> kOk(L.ps.size(), 1);   // singularP: 0 for a P with no inverse (its clip-space slots stay)
    for (size_t i = 0; i < L.ps.size(); ++i)
    {
        if (L.ps[i].reg + 4 > regs) return false;
        M4 P = ReadM4(src + L.ps[i].reg * 4, L.ps[i].transposed);
        if (!IsBareProjection(P))                                // a shared layout holding something else this time
        {
            if (i == 0 && consumerMask) { ++c->projRefused; c->refusedP00 = P.m[0]; c->refusedP11 = P.m[5]; }
            return false;
        }
        P0[i] = P;                                   // unpatched, for the inverse identity test below
        M4 invP;
        const bool hasInv = M4Inverse(P, invP);
        if (!hasInv && !Limits().singularP) return false;
        const float s = P.m[11] > 0.f ? 1.f : -1.f;  // w_clip = s * z_view
        // The P00 of the main view sizes the composite crop; a mirror's zoomed P must not vote for it.
        if (i == 0 && !Vk().VKMajorityP11) c->lastP11 = P.m[0];
        else if (i == 0 && consumerMask) ++c->p11Hist[(int)(P.m[0] * 1000.f + 0.5f)];
        Pp[i] = P;
        Pp[i].m[8] += (convDist != 0.f) ? P.m[0] * eyeSep / convDist * s : 0.f;   // b: the w-proportional half
        Pp[i].m[12] -= P.m[0] * eyeSep;                                           // m41: the clip-space constant
        if (!hasInv) { K[i] = Kinv[i] = M4Identity(); kOk[i] = 0; continue; }
        K[i] = M4Mul(M4Mul(invP, T), Pp[i]);
        if (!M4Inverse(K[i], Kinv[i])) return false;
    }
    // The camera forward for the basis alignment test (the rigid V's third column); claimedForward: from claimed ranges
    // only, as unclaimed blocks come in key-hash order.
    const bool publishFwd = !Traits().claimedForward || consumerMask != 0;
    for (const CamSlot& sl : L.slots)
        if (publishFwd && sl.kind == K_ToView && sl.reg + 4 <= regs)
        {
            M4 V = ReadM4(src + sl.reg * 4, sl.transposed);
            // A rotation about the world's up axis alone (identity included) is an object's, not a camera's:
            // its forward is (0,0,1) whatever the yaw.
            if (Traits().claimedForward && fabsf(V.m[10]) > 0.999f) break;
            if (IsRigidView(V))
            {
                c->camFwd[0] = V.m[2]; c->camFwd[1] = V.m[6]; c->camFwd[2] = V.m[10];
                c->camFwdValid = true;
                if (Traits().claimedForward)
                {
                    // The last writer follows key order, so it is a per-session lottery; claimed views also vote on
                    // their whole rotation, the main view's being bit-identical across its draws (camFwdVote).
                    const float r[9] = { V.m[0], V.m[1], V.m[2], V.m[4], V.m[5], V.m[6], V.m[8], V.m[9], V.m[10] };
                    // A mirrored view (reflection) is orthonormal too, but its determinant is negative: no vote.
                    const float det = r[0] * (r[4] * r[8] - r[5] * r[7]) - r[1] * (r[3] * r[8] - r[5] * r[6]) + r[2] * (r[3] * r[7] - r[4] * r[6]);
                    if (det < 0.f) break;
                    uint64_t h = 1469598103934665603ull;
                    for (int k = 0; k < 9; ++k) { uint32_t b; memcpy(&b, &r[k], 4); h = (h ^ b) * 1099511628211ull; }
                    Constants::FwdVote& v = c->fwdVotes[h];
                    if (!v.n) { v.f[0] = r[2]; v.f[1] = r[5]; v.f[2] = r[8]; }
                    ++v.n;
                }
            }
            break;
        }
    for (const CamSlot& sl : L.slots)
        if (sl.kind == K_CamOffset && sl.stride == 16 && sl.reg < regs)
        {
            const float* p = src + sl.reg * 4;
            bool ok = true;
            for (int i = 0; i < 3; ++i) if (!(p[i] == p[i]) || fabsf(p[i]) > 1e7f) ok = false;
            if (ok) { c->camPos[0] = p[0]; c->camPos[1] = p[1]; c->camPos[2] = p[2]; c->camPosValid = true; }
            break;
        }
    if (eyeSep == 0.f) return true;
    // R's z column takes the depth-proportional half of the reconstruction correction; the other half, one eye along
    // the basis x, goes to the packed origin (K_CamOffset).
    float packedShift[3] = {};
    bool packedShiftValid = false;
    for (const CamSlot& sl : L.slots)
    {
        if (sl.kind != K_RayBasis) continue;
        float t[12] = {};
        // Register-addressed and packed bases alike.
        if (sl.stride == 16)
        {
            if (sl.reg + 3 > regs) continue;
            memcpy(t, src + sl.reg * 4, sizeof(t));
        }
        else
        {
            if ((size_t)sl.byteOff + (size_t)sl.stride * 3 > regs * 16) continue;
            UnpackBasis3x3((const unsigned char*)src, sl.byteOff, sl.stride, sl.transposed, t);
        }
        if (!RayBasisOk(c, t)) continue;
        const float cx[3] = { t[0], t[4], t[8] };
        const float n = sqrtf(cx[0]*cx[0] + cx[1]*cx[1] + cx[2]*cx[2]);
        if (n < 1e-6f) continue;
        for (int i = 0; i < 3; ++i) packedShift[i] = cx[i] / n * eyeSep;
        packedShiftValid = true;
        break;
    }
    bool rayApplied = false;
    const uint64_t spanOf = Vk().VKMaskMatrixSpan ? 0xFull : 1ull;
    bool pClaimed = false;
    for (const CamSlot& sl : L.slots)
        if ((sl.kind == K_P || sl.kind == K_ToClip) && sl.reg < 64 && (consumerMask & (spanOf << sl.reg))) pClaimed = true;
    // clipSlotStraddle: the chain's matrix starts, plus their previous-frame twins at the second projection's offset.
    uint64_t chain = Traits().clipSlotStraddle ? chainStarts : 0;
    if (chain)
    {
        int p0 = -1, p1 = -1;
        for (const CamSlot& sl : L.slots)
            if (sl.kind == K_P && sl.reg < 64) { if (sl.pIndex == 0) p0 = (int)sl.reg; else if (sl.pIndex == 1) p1 = (int)sl.reg; }
        if (p0 >= 0 && p1 > p0 && ((chain >> p0) & 1)) chain |= chain << (p1 - p0);
    }
    // A 4x4 clip slot with a chain matrix starting inside it spans two matrices, admitted by chance.
    auto straddles = [chain](const CamSlot& sl) { return sl.reg < 61 && ((chain >> (sl.reg + 1)) & 7) != 0; };
    for (const CamSlot& sl : L.slots)
    {
        // Bases have no P index: only the P-indexed slots are range-checked here.
        const bool basis = sl.kind == K_RayBasis || sl.kind == K_UnitBasis;
        if (!basis && (sl.reg + 4 > regs || sl.pIndex >= L.ps.size())) continue;
        const M4 B = basis ? M4Identity() : ReadM4(src + sl.reg * 4, sl.transposed);
        M4 R;
        switch (sl.kind)
        {
        // The forward shift goes into a projection only for a register a draw claimed (consumerMask): that draw's
        // shader was left unshifted for it, while other consumers of the range must not move.
        case K_P:
        case K_ToClip:
        {
            // VKMaskMatrixSpan: any of the 4x4's registers claims it, and registers past the 64-bit mask are skipped;
            // off, the first register's bit alone (reg & 63) decides.
            if (sl.reg >= 64 && Vk().VKMaskMatrixSpan) continue;
            const bool claimed = (consumerMask & (spanOf << (sl.reg & 63))) != 0;
            // previousProjection: an unclaimed previous-frame P (feeding only velocity) is patched with a claimed one,
            // or right-eye motion vectors are off by P11*e*(1/C - 1/w).
            const bool prev = !claimed && Traits().previousProjection && sl.kind == K_P && pClaimed;
            if (!claimed && !prev) continue;
            if (sl.kind == K_ToClip && !kOk[sl.pIndex]) continue;
            if (sl.kind == K_ToClip && straddles(sl)) continue;
            R = (sl.kind == K_P) ? Pp[sl.pIndex] : M4Mul(B, K[sl.pIndex]);
            // The only thing that entitles a draw to the sep=0 set (the return value is true for any layout with a P).
            if (claimed && wroteForward) *wroteForward = true;
            if (prev) ++c->prevPatched;
            break;
        }
        case K_FromClip:
        {
            if (!kOk[sl.pIndex]) continue;
            if (straddles(sl)) continue;
            // VKValidateInverse: a key is a (layout, binding), not a shader, and shaders sharing a layout may hold
            // other data in the slot, so the admitting test (P*M affine) re-runs on this range's live bytes.
            if (Vk().VKValidateInverse)
            {
                const M4 A = M4Mul(P0[sl.pIndex], B);
                if (!IsAffine(A)) { ++c->invRejected; continue; }
                ++c->invApplied;
            }
            R = M4Mul(Kinv[sl.pIndex], B);
            break;
        }
        // ray = R * (ndc, 1), pos = ray * t: the depth-linear half of the shift, (P11*e/C)*t along cx/|cx| with
        // P11 = 1/|cx|, comes off column z; the constant e*right half is the origin's (K_CamOffset).
        case K_RayBasis:
        {
            if (convDist == 0.f) continue;
            const size_t bytes = regs * 16;
            float rb[12] = {};
            const bool packed = sl.stride != 16;
            if (packed)
            {
                if ((size_t)sl.byteOff + (size_t)sl.stride * 3 > bytes) continue;
                UnpackBasis3x3((const unsigned char*)src, sl.byteOff, sl.stride, sl.transposed, rb);
            }
            else
            {
                if (sl.reg + 3 > regs) continue;
                memcpy(rb, src + sl.reg * 4, sizeof(rb));
            }
            if (!RayBasisOk(c, rb)) continue;    // a shared layout holding something else this time
            const float cxv[3] = { rb[0], rb[4], rb[8] };
            const float n = sqrtf(cxv[0]*cxv[0] + cxv[1]*cxv[1] + cxv[2]*cxv[2]);
            if (n < 1e-6f) continue;
            const float k = -eyeSep / (convDist * n);
            if (packed)
            {
                // dst already mirrors src, so the z column is shifted in place.
                // Row-major keeps that column as the third float of each row.
                unsigned char* p = dst + sl.byteOff;
                for (int i = 0; i < 3; ++i)
                {
                    float* e = sl.transposed ? (float*)(p + (size_t)i * sl.stride) + 2
                                             : (float*)(p + (size_t)2 * sl.stride) + i;
                    *e += k * cxv[i];
                }
            }
            else
            {
                float* o = (float*)(dst + sl.reg * 16);
                o[2]  = rb[2]  + k * cxv[0];
                o[6]  = rb[6]  + k * cxv[1];
                o[10] = rb[10] + k * cxv[2];
            }
            ++c->rayPatched;
            rayApplied = true;
            continue;
        }
        // The right eye is this camera moved e along its right axis (row 0) with the projection skewed by
        // b = P11*e/C, P11 = 1/tanX of THIS block: jitter.x -= b moves the ray, row 3 += e*right the origin.
        case K_UnitBasis:
        {
            if (sl.reg + 5 > regs) continue;
            const float* u = src + sl.reg * 4;
            if (!UnitBasisOk(c, u)) continue;
            float* o = (float*)(dst + sl.reg * 16);
            if (convDist != 0.f) o[16] = u[16] - eyeSep / (convDist * u[3]);
            for (int i = 0; i < 3; ++i) o[12 + i] = u[12 + i] + eyeSep * u[i];
            ++c->unitPatched;
            rayApplied = true;
            continue;
        }
        // View, world->view and the camera position stay the game's in both eyes: the shift is a clip-space one.
        case K_CamOffset:
            // Only the packed reconstruction origin moves; the ordinary camera position must match between the eyes.
            if (sl.stride == 0 && packedShiftValid && (size_t)sl.byteOff + 12 <= regs * 16)
            {
                float* p = (float*)(dst + sl.byteOff);
                p[0] += packedShift[0]; p[1] += packedShift[1]; p[2] += packedShift[2];
                rayApplied = true;
            }
            continue;
        case K_ToView:
        case K_FromView:
        default: continue;
        }
        WriteM4((float*)(dst + sl.reg * 16), sl.transposed, R);
    }
    // Basis-only layouts report whether any slot still verified, so the caller
    // can quarantine keys that only aligned with the camera transiently.
    if (L.ps.empty()) return rayApplied;
    return true;
}

void ConstantsInit(DeviceData* d)
{
    d->constants = new Constants();
}

void ConstantsDestroy(DeviceData* d)
{
    for (auto& kv : d->buffers)
    {
        BufferInfo& b = kv.second;
        if (b.siblingMapped) d->UnmapMemory(d->device, b.siblingMem);
        if (b.sibling) d->DestroyBuffer(d->device, b.sibling, nullptr);
        if (b.siblingMem) d->FreeMemory(d->device, b.siblingMem, nullptr);
        b.sibling = VK_NULL_HANDLE; b.siblingMem = VK_NULL_HANDLE; b.siblingMapped = nullptr;
        if (b.claimSiblingMapped) d->UnmapMemory(d->device, b.claimSiblingMem);
        if (b.claimSibling) d->DestroyBuffer(d->device, b.claimSibling, nullptr);
        if (b.claimSiblingMem) d->FreeMemory(d->device, b.claimSiblingMem, nullptr);
        b.claimSibling = VK_NULL_HANDLE; b.claimSiblingMem = VK_NULL_HANDLE; b.claimSiblingMapped = nullptr;
    }
    delete d->constants;
    d->constants = nullptr;
}

// From the descriptor path, d->lock held: reads the key set, not the constants map, whose mutex a submit holds.
bool ConstantsRangeIsStereo(DeviceData* d, VkDescriptorSetLayout layout, uint32_t binding)
{
    if (!d->constants || d->cameraKeys.empty()) return false;
    const uint64_t k = LearnKey(layout, binding);
    // unpatchedKeys: a skip-claim full-screen draw's position block no draw claims reads the game's buffer, so
    // another key's patch of the same ring slice cannot move it.
    if (Traits().unpatchedKeys && d->skipPosKeys.count(k) && !d->claimedKeys.count(k)) return false;
    return d->cameraKeys.count(k) != 0;
}

bool BufferSibling(DeviceData* d, VkBuffer buf, VkBuffer* sibling)
{
    auto it = d->buffers.find(buf);
    if (it == d->buffers.end() || !it->second.sibling) return false;
    if (sibling) *sibling = it->second.sibling;
    return true;
}

bool BufferClaimSibling(DeviceData* d, VkBuffer buf, VkBuffer* sibling)
{
    auto it = d->buffers.find(buf);
    if (it == d->buffers.end() || !it->second.claimSibling) return false;
    if (sibling) *sibling = it->second.claimSibling;
    return true;
}

// The copy of a uniform block the right eye's set binds (d->lock held): the game's buffer for no camera key or no
// sibling yet, the claim sibling for a claimMirror key that has one, else the shared sibling.
RightSource ConstantsRightSource(DeviceData* d, VkDescriptorSetLayout layout, uint32_t binding, VkBuffer buffer, VkBuffer* copy)
{
    VkBuffer sib;
    if (!ConstantsRangeIsStereo(d, layout, binding) || !BufferSibling(d, buffer, &sib)) return RightSource::Game;
    if (Traits().claimMirror && !d->claimMirrorKeys.empty() && d->claimMirrorKeys.count(LearnKey(layout, binding)) && BufferClaimSibling(d, buffer, copy))
        return RightSource::ClaimSibling;
    *copy = sib;
    return RightSource::Sibling;
}

// Tracking hooks
static VKAPI_ATTR VkResult VKAPI_CALL Hook_CreateBuffer(VkDevice device, const VkBufferCreateInfo* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkBuffer* pBuffer)
{
    DeviceData* d = FindDevice(device);
    // A uniform range is mirrored with vkCmdCopyBuffer, which needs the source to allow it.
    VkBufferCreateInfo ci = *pCreateInfo;
    if (d->stereoEnabled) ci.usage |= VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VkResult r = d->CreateBuffer(device, &ci, pAllocator, pBuffer);
    if (r != VK_SUCCESS) return r;
    BufferInfo info;
    info.size = ci.size;
    info.usage = ci.usage;
    info.flags = ci.flags;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        d->buffers[*pBuffer] = info;
    }
    TwinOnCreateBuffer(d, *pBuffer);
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_DestroyBuffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* pAllocator)
{
    DeviceData* d = FindDevice(device);
    BufferInfo b;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->buffers.find(buffer);
        if (it != d->buffers.end()) { b = it->second; d->buffers.erase(it); }
        // A destroyed ring takes its siblings with it: shadow sets still naming one fall back to the game's own set
        // until it is rewritten.
        if (b.sibling)
            d->sets.ForEach([&](VkDescriptorSet, DescriptorSetInfo& info)
            {
                bool hit = false;
                for (UboRef& r : info.ubos)
                    if (r.shadowBuffer == b.sibling || (b.claimSibling && r.shadowBuffer == b.claimSibling)) { r.shadowBuffer = VK_NULL_HANDLE; hit = true; }
                if (hit) info.stereoAny = false;
            });
    }
    if (b.siblingMapped) d->UnmapMemory(device, b.siblingMem);
    if (b.sibling) d->DestroyBuffer(device, b.sibling, nullptr);
    if (b.siblingMem) { d->FreeMemory(device, b.siblingMem, nullptr); d->perf.vramAdded.fetch_sub((long long)b.size, std::memory_order_relaxed); }
    if (b.claimSiblingMapped) d->UnmapMemory(device, b.claimSiblingMem);
    if (b.claimSibling) d->DestroyBuffer(device, b.claimSibling, nullptr);
    if (b.claimSiblingMem) { d->FreeMemory(device, b.claimSiblingMem, nullptr); d->perf.vramAdded.fetch_sub((long long)b.size, std::memory_order_relaxed); }
    if (b.twin) TwinOnDestroyBuffer(d, b.twin);
    TileForgetBuffer(d, buffer, b.size);
    d->DestroyBuffer(device, buffer, pAllocator);
}

static void NoteBind(DeviceData* d, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize offset)
{
    std::lock_guard<std::shared_mutex> g(d->lock);
    auto it = d->buffers.find(buffer);
    if (it == d->buffers.end()) return;
    it->second.memory = memory;
    it->second.offset = offset;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_BindBufferMemory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memoryOffset)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->BindBufferMemory(device, buffer, memory, memoryOffset);
    if (r == VK_SUCCESS) { NoteBind(d, buffer, memory, memoryOffset); TwinOnBindBuffer(d, buffer); TileNoteBuffer(d, buffer); }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_BindBufferMemory2(VkDevice device, uint32_t bindInfoCount, const VkBindBufferMemoryInfo* pBindInfos)
{
    DeviceData* d = FindDevice(device);
    VkResult r = d->BindBufferMemory2(device, bindInfoCount, pBindInfos);
    if (r == VK_SUCCESS)
        for (uint32_t i = 0; i < bindInfoCount; ++i)
        {
            NoteBind(d, pBindInfos[i].buffer, pBindInfos[i].memory, pBindInfos[i].memoryOffset);
            TwinOnBindBuffer(d, pBindInfos[i].buffer);
            TileNoteBuffer(d, pBindInfos[i].buffer);
        }
    return r;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_MapMemory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** ppData)
{
    DeviceData* d = FindDevice(device);
    {
        // One mapping per allocation: if the layer mapped it first, the game gets the view it asked for.
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->memory.find(memory);
        if (it != d->memory.end() && it->second.layerMapped && it->second.mapped)
        {
            *ppData = (char*)it->second.mapped + offset;
            return VK_SUCCESS;
        }
    }
    VkResult r = d->MapMemory(device, memory, offset, size, flags, ppData);
    if (r == VK_SUCCESS)
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        ConstantsMapMemory(d, memory, offset, size, *ppData);
    }
    return r;
}

static VKAPI_ATTR void VKAPI_CALL Hook_UnmapMemory(VkDevice device, VkDeviceMemory memory)
{
    DeviceData* d = FindDevice(device);
    {
        // Ours to hold: the next submit still reads these ranges.
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->memory.find(memory);
        if (it != d->memory.end() && it->second.layerMapped) return;
        ConstantsUnmapMemory(d, memory);
    }
    d->UnmapMemory(device, memory);
}

void ConstantsMapMemory(DeviceData* d, VkDeviceMemory mem, VkDeviceSize off, VkDeviceSize size, void* ptr)
{
    auto it = d->memory.find(mem);
    if (it == d->memory.end()) return;
    it->second.mapped = ptr;
    it->second.mapOff = off;
    it->second.mapSize = size == VK_WHOLE_SIZE ? it->second.size - off : size;
}

void ConstantsUnmapMemory(DeviceData* d, VkDeviceMemory mem)
{
    auto it = d->memory.find(mem);
    if (it == d->memory.end()) return;
    it->second.mapped = nullptr;
    it->second.mapSize = 0;
}

// Siblings

// Lock NOT held. Host-visible, coherent, persistently mapped; the right eye's shadow sets use it at the game's offsets.
static bool CreateBufferSibling(DeviceData* d, VkBuffer buffer, bool claim = false)
{
    BufferInfo info;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        auto it = d->buffers.find(buffer);
        if (it == d->buffers.end() || (claim ? (!it->second.sibling || it->second.claimSibling) : it->second.sibling)) return false;
        info = it->second;
    }
    VkBufferCreateInfo ci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    ci.size = info.size;
    ci.usage = info.usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT;   // the GPU mirror writes here
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer sib = VK_NULL_HANDLE;
    if (d->CreateBuffer(d->device, &ci, nullptr, &sib) != VK_SUCCESS) { Log("[const] sibling buffer create failed\n"); return false; }
    VkMemoryRequirements req;
    d->GetBufferMemoryRequirements(d->device, sib, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, want | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UINT32_MAX);
    // The device-local host-visible heap is small (256 MB on NVIDIA without ReBAR); big rings go to system memory.
    if (mai.memoryTypeIndex == UINT32_MAX || req.size > 64u * 1024 * 1024)
        mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, want, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX) { d->DestroyBuffer(d->device, sib, nullptr); Log("[const] no host-visible memory type for sibling\n"); return false; }
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void* mapped = nullptr;
    if (d->AllocateMemory(d->device, &mai, nullptr, &mem) != VK_SUCCESS ||
        d->BindBufferMemory(d->device, sib, mem, 0) != VK_SUCCESS ||
        d->MapMemory(d->device, mem, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS)
    {
        if (mem) d->FreeMemory(d->device, mem, nullptr);
        d->DestroyBuffer(d->device, sib, nullptr);
        Log("[const] sibling buffer memory failed (%llu bytes)\n", (unsigned long long)req.size);
        return false;
    }
    d->perf.vramAdded.fetch_add((long long)req.size, std::memory_order_relaxed);
    std::lock_guard<std::shared_mutex> g(d->lock);
    auto it = d->buffers.find(buffer);
    if (it == d->buffers.end() || (claim && (!it->second.sibling || it->second.claimSibling)))
    { d->UnmapMemory(d->device, mem); d->DestroyBuffer(d->device, sib, nullptr); d->FreeMemory(d->device, mem, nullptr); return false; }
    if (claim)
    {
        it->second.claimSibling = sib;
        it->second.claimSiblingMem = mem;
        it->second.claimSiblingMapped = mapped;
        Log("[const] claim sibling for buffer %p (%llu bytes, claim mirror)\n", (void*)buffer, (unsigned long long)info.size);
        return true;
    }
    it->second.sibling = sib;
    it->second.siblingMem = mem;
    it->second.siblingMapped = mapped;
    ++d->constants->siblings;
    Log("[const] sibling for buffer %p (%llu bytes, usage 0x%x)\n", (void*)buffer, (unsigned long long)info.size, info.usage);
    return true;
}

// Submit-time scan
struct Work
{
    const unsigned char* host;      // game bytes of the range
    unsigned char*       sibling;   // sibling bytes at the same offset, or null
    size_t               bytes;
    uint64_t             key;
    VkBuffer             buffer;
    uint32_t             binding;
    uint64_t             consumerMask;   // registers a draw served by constants claimed here; 0 = mirror only
    uint32_t             fs;             // bound at a full-screen draw; ray bases may only live here
    bool                 storage;        // storage range: scanned and learned, never mirrored (Phase 1)
    size_t               fullBytes;      // whole binding range; bytes is the 4KB scan window
    VkBuffer             siblingBuf;     // sibling buffer handle, for the GPU mirror
    VkDeviceSize         srcOff;         // offset of the range inside both buffers
    VkPipeline           pipeline = VK_NULL_HANDLE;   // the claiming draw's pipeline; gets ApplyCamera's verdict
    VkPipeline           noClaimPipe = VK_NULL_HANDLE;   // a skip-claim draw also read this range
    uint32_t             noClaimW = 0, noClaimH = 0;     // that draw's target, for the conflict log
    bool                 shadowed = false;               // the set's shadow reads our sibling here: refresh it
    bool                 skipPos = false;                // a skip-claim full-screen draw's position binding
    bool                 repoint = false;                // a claimed position binding whose cached sets are re-pointed
    bool                 posClaim = false;               // claimedKeyRetry: a claiming draw's position binding
    VkBuffer             claimBuf = VK_NULL_HANDLE;      // claimMirror: the buffer's claim sibling, if made
    unsigned char*       claimMapped = nullptr;          // its bytes at this range's offset
};

// Memory the game left unmapped is mapped here, host-visible and coherent only (no invalidates needed);
// device-local memory stays counted as unreadable.
static bool LayerMappable(const DeviceData* d, const MemoryInfo& m)
{
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return m.typeIndex < d->memProps.memoryTypeCount && (d->memProps.memoryTypes[m.typeIndex].propertyFlags & want) == want;
}

static bool EnsureLayerMapped(DeviceData* d, VkDeviceMemory mem, MemoryInfo& m, Constants& c)
{
    if (!LayerMappable(d, m)) { ++c.unmappable; return false; }
    void* p = nullptr;
    if (d->MapMemory(d->device, mem, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS || !p) { ++c.unmappable; return false; }
    m.mapped = p;
    m.mapOff = 0;
    m.mapSize = m.size;
    m.layerMapped = true;
    ++c.layerMaps;
    return true;
}

// Collects one bound set's uniform ranges, lock held. mayMap false (sharedCollect, shared lock): a range on memory
// never mapped yet abandons the set and returns false, and the caller collects it again under the exclusive lock.
static bool CollectSet(DeviceData* d, const UsedSet& u, std::vector<Work>& work, Constants& c, bool mayMap = true)
{
    const DescriptorSetInfo* sit = d->sets.Find(u.set);
    if (!sit) return true;
    const size_t start = work.size();
    const DescriptorSetInfo& info = *sit;
    uint32_t dynIndex = 0;
    for (const UboRef& ref : info.ubos)          // sorted by (binding, element): dynamic offsets apply in that order
    {
        VkDeviceSize off = ref.offset;
        bool storage = false;
        if (ref.dynamic) { if (dynIndex < u.dynCount) off += u.dyn[dynIndex]; ++dynIndex; }
        // Storage ranges count only at bindings carrying a packed matrix (a reconstruction basis may live in a storage
        // buffer that also serves vertex, index and indirect data).
        if (!ref.uniform)
        {
            const bool packed = ref.binding < 256 &&
                (d->packedBindingMask[ref.binding >> 6] & (1ull << (ref.binding & 63))) != 0;
            if (!packed) continue;
            // Only ranges mirrored exactly: a partial copy would stomp its neighbours in a shared ring.
            if (ref.range == VK_WHOLE_SIZE || ref.range > 65536) continue;
            ++c.storageRanges;
            storage = true;
        }
        ++c.ranges;
        auto bit = d->buffers.find(ref.buffer);
        if (bit == d->buffers.end()) { ++c.unmapped; continue; }
        const BufferInfo& b = bit->second;
        auto mit = d->memory.find(b.memory);
        if (mit == d->memory.end()) { ++c.unmapped; continue; }
        // Only memory a map can succeed on is worth the exclusive retry; the rest stays unreadable either way.
        if (!mit->second.mapped && !mayMap && LayerMappable(d, mit->second)) { work.resize(start); return false; }
        if (!mit->second.mapped && !mayMap) { ++c.unmapped; ++c.unmappable; continue; }
        if (!mit->second.mapped && !EnsureLayerMapped(d, b.memory, mit->second, c)) { ++c.unmapped; continue; }
        const MemoryInfo& m = mit->second;
        if (m.typeIndex < 32) ++c.typeHits[m.typeIndex];   // which memory we read constants out of
        VkDeviceSize range = ref.range == VK_WHOLE_SIZE ? (b.size > off ? b.size - off : 0) : ref.range;
        const VkDeviceSize memOff = b.offset + off;
        if (range < 64 || memOff < m.mapOff || memOff + range > m.mapOff + m.mapSize) { ++c.unmapped; continue; }
        Work w;
        w.host = (const unsigned char*)m.mapped + (memOff - m.mapOff);
        w.sibling = b.siblingMapped ? (unsigned char*)b.siblingMapped + off : nullptr;
        // The sibling mirrors the buffer 1:1, so one offset serves both sides.
        w.siblingBuf = b.sibling;
        w.claimBuf = b.claimSibling;
        w.claimMapped = b.claimSiblingMapped ? (unsigned char*)b.claimSiblingMapped + off : nullptr;
        w.srcOff = off;
        w.bytes = (size_t)(range > 4096 ? 4096 : range);   // invVP sits near the front of any real layout
        // The copy spans the whole binding, as consumers read past the scan window; capped at 1MB so a WHOLE_SIZE ring
        // binding cannot turn one sighting into a ring-sized copy.
        w.fullBytes = (size_t)(range > (1u << 20) ? (1u << 20) : range);
        w.key = LearnKey(info.layout, ref.binding);
        w.buffer = ref.buffer;
        w.binding = ref.binding;
        w.consumerMask = u.consumerMask;
        w.fs = u.fs;
        w.pipeline = u.pipeline;
        w.noClaimPipe = u.noClaimPipe;
        w.noClaimW = u.noClaimW; w.noClaimH = u.noClaimH;
        w.skipPos = u.noClaimPipe && u.noClaimBinding == ref.binding;
        w.repoint = u.repointBinding == ref.binding;
        w.posClaim = u.posBinding == ref.binding;
        w.shadowed = ref.shadowBuffer != VK_NULL_HANDLE;
        w.storage = storage;
        work.push_back(w);
    }
    return true;
}

// GPU transport for the mirrors: vkCmdCopyBuffer keeps ReBAR bytes on the device and the patched camera block rides
// along as vkCmdUpdateBuffer; ordering comes from submission order, visibility from the barriers.
struct GpuCopy   { VkBuffer src, dst; VkBufferCopy region; };
struct GpuUpdate { VkBuffer dst; VkDeviceSize off; uint32_t size, dataOff; };

static bool AcquireXfer(DeviceData* d, uint32_t family, VkCommandBuffer* outCb, VkFence* outFence)
{
    std::lock_guard<std::mutex> g(d->xferLock);
    DeviceData::XferRing& r = d->xferRings[family];
    if (!r.pool)
    {
        VkCommandPoolCreateInfo pi = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        pi.queueFamilyIndex = family;
        if (d->CreateCommandPool(d->device, &pi, nullptr, &r.pool) != VK_SUCCESS) { r.pool = VK_NULL_HANDLE; return false; }
        const uint32_t kRing = 32;
        VkCommandBufferAllocateInfo ai = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        ai.commandPool = r.pool;
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = kRing;
        r.cbs.resize(kRing);
        if (d->AllocateCommandBuffers(d->device, &ai, r.cbs.data()) != VK_SUCCESS) { r.cbs.clear(); return false; }
        StampCommandBuffers(d, r.cbs.data(), kRing);
        r.fences.assign(kRing, VK_NULL_HANDLE);
        VkFenceCreateInfo fi = { VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        fi.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        for (uint32_t i = 0; i < kRing; ++i) d->CreateFence(d->device, &fi, nullptr, &r.fences[i]);
        Log("[const] GPU mirror ring on queue family %u\n", family);
    }
    if (r.cbs.empty()) return false;
    const size_t i = (r.next++) % r.cbs.size();
    if (!r.fences[i]) return false;
    if (d->WaitForFences(d->device, 1, &r.fences[i], VK_TRUE, 2000000000ull) != VK_SUCCESS) return false;
    d->ResetFences(d->device, 1, &r.fences[i]);
    *outCb = r.cbs[i];
    *outFence = r.fences[i];
    return true;
}

static void SubmitGpuMirror(DeviceData* d, VkQueue queue, const std::vector<GpuCopy>& copies,
                            const std::vector<GpuUpdate>& updates, const std::vector<unsigned char>& data)
{
    if (copies.empty() && updates.empty()) return;
    uint32_t family = 0;
    {
        std::shared_lock<std::shared_mutex> rd(d->lock, std::defer_lock);
        std::unique_lock<std::shared_mutex> ex(d->lock, std::defer_lock);
        if (Traits().sharedCollect) rd.lock(); else ex.lock();   // a map read only
        auto q = d->queueFamilies.find(queue);
        if (q == d->queueFamilies.end()) return;
        family = q->second;
    }
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    if (!AcquireXfer(d, family, &cb, &fence)) return;

    VkCommandBufferBeginInfo bi = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (d->BeginCommandBuffer(cb, &bi) != VK_SUCCESS) return;
    std::vector<VkBufferCopy> regions;
    for (size_t i = 0; i < copies.size(); )
    {
        size_t j = i;
        regions.clear();
        while (j < copies.size() && copies[j].src == copies[i].src && copies[j].dst == copies[i].dst)
            regions.push_back(copies[j++].region);
        d->CmdCopyBuffer(cb, copies[i].src, copies[i].dst, (uint32_t)regions.size(), regions.data());
        i = j;
    }
    VkMemoryBarrier mb = { VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    for (const GpuUpdate& u : updates) d->CmdUpdateBuffer(cb, u.dst, u.off, u.size, data.data() + u.dataOff);
    mb.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
    d->EndCommandBuffer(cb);

    VkSubmitInfo si = { VK_STRUCTURE_TYPE_SUBMIT_INFO };
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    { const auto ql = FgQueueLock(); d->QueueSubmit(queue, 1, &si, fence); }
}

void ConstantsSubmit(DeviceData* d, VkQueue queue, uint32_t cbCount, const VkCommandBuffer* cbs)
{
    Constants* c = d->constants;
    if (!c || !gInfo.DetectInverseMatrix) return;
    PerfScope pt(d->perf.constants);

    std::vector<Work> work;
    {
        std::vector<UsedSet> used;
        // sharedCollect: the device maps are only read here, under a shared lock, one submit at a time (the counters in
        // c); a set needing a first map is collected again exclusively below.
        const bool shared = Traits().sharedCollect;
        std::unique_lock<std::mutex> one(c->collect, std::defer_lock);
        std::shared_lock<std::shared_mutex> rd(d->lock, std::defer_lock);
        std::unique_lock<std::shared_mutex> ex(d->lock, std::defer_lock);
        const long long wait0 = PerfNow();
        if (shared) { one.lock(); rd.lock(); } else ex.lock();
        const long long held0 = PerfNow();
        d->perf.constWait.fetch_add(held0 - wait0, std::memory_order_relaxed);
        for (uint32_t i = 0; i < cbCount; ++i) CmdStateTakeUsedSets(d, cbs[i], used);
        // Claims merge before collecting: a set appears once from vkCmdBindDescriptorSets (mask 0) and once per draw
        // that claimed it, and every claim must survive.
        std::unordered_map<uint64_t, size_t> seen;
        std::vector<UsedSet> merged;
        merged.reserve(used.size());
        for (const UsedSet& u : used)
        {
            uint64_t h = (uint64_t)u.set;
            for (uint32_t k = 0; k < u.dynCount; ++k) h = h * 1099511628211ull ^ u.dyn[k];
            auto it = seen.find(h);
            if (it != seen.end())
            {
                UsedSet& m = merged[it->second];
                m.consumerMask |= u.consumerMask; m.fs |= u.fs;
                // The bind-time entry names no pipeline; the claim that follows does, and gets the verdict.
                if (!m.pipeline) m.pipeline = u.pipeline;
                if (!m.noClaimPipe) { m.noClaimPipe = u.noClaimPipe; m.noClaimW = u.noClaimW; m.noClaimH = u.noClaimH; m.noClaimBinding = u.noClaimBinding; }
                if (m.repointBinding == UINT32_MAX) m.repointBinding = u.repointBinding;
                if (m.posBinding == UINT32_MAX) m.posBinding = u.posBinding;
                continue;
            }
            seen.emplace(h, merged.size());
            merged.push_back(u);
        }
        std::vector<size_t> later;
        for (size_t i = 0; i < merged.size(); ++i)
            if (!CollectSet(d, merged[i], work, *c, !shared)) later.push_back(i);
        if (!later.empty())
        {
            rd.unlock();
            ex.lock();
            for (size_t i : later) CollectSet(d, merged[i], work, *c);
            c->collectRetries += (unsigned)later.size();
        }
        d->perf.constHeld.fetch_add(PerfNow() - held0, std::memory_order_relaxed);
    }

    if (work.empty()) return;
    // A range reached through several sets of one layout is patched once; the key is part of the identity, as many
    // keys point into the same ring.
    std::sort(work.begin(), work.end(), [](const Work& a, const Work& b) {
        return a.key != b.key ? a.key < b.key : a.host < b.host; });
    {
        // Duplicates merge (a bind-time entry and a draw's claim of one range), so no claim is lost to the sort order.
        size_t out = 0;
        for (size_t i = 0; i < work.size(); ++i)
        {
            if (out && work[out - 1].key == work[i].key && work[out - 1].host == work[i].host)
            {
                work[out - 1].consumerMask |= work[i].consumerMask; work[out - 1].fs |= work[i].fs;
                if (!work[out - 1].pipeline) work[out - 1].pipeline = work[i].pipeline;
                if (!work[out - 1].noClaimPipe)
                { work[out - 1].noClaimPipe = work[i].noClaimPipe; work[out - 1].noClaimW = work[i].noClaimW; work[out - 1].noClaimH = work[i].noClaimH; }
                work[out - 1].skipPos = work[out - 1].skipPos || work[i].skipPos;
                work[out - 1].repoint = work[out - 1].repoint || work[i].repoint;
                work[out - 1].posClaim = work[out - 1].posClaim || work[i].posClaim;
                work[out - 1].shadowed = work[out - 1].shadowed || work[i].shadowed;
                continue;
            }
            work[out++] = work[i];
        }
        work.resize(out);
    }
    std::vector<VkBuffer> needSibling;
    std::vector<uint64_t> newKeys, deadKeys;
    // unpatchedKeys: keys newly seen claimed, and newly seen holding a skip-claim full-screen draw's position.
    std::vector<uint64_t> newClaimed, newSkipPos, newRepoint;
    std::vector<uint64_t> newClaimMirror;          // claimMirror: keys whose sets move to the claim sibling
    std::vector<uint64_t> dropClaimMirror;         // claimMirror: keys whose sets move back to the shared sibling
    std::vector<VkBuffer> needClaimSibling;
    if (Traits().unpatchedKeys)
    {
        std::lock_guard<std::mutex> cl(c->lock);
        for (const Work& w : work)
        {
            if (w.repoint && w.consumerMask && c->repointKeys.insert(w.key).second) newRepoint.push_back(w.key);
            if (w.consumerMask) { if (c->claimedKeys.insert(w.key).second) newClaimed.push_back(w.key); }
            else if (w.skipPos && c->skipPosKeys.insert(w.key).second) newSkipPos.push_back(w.key);
        }
    }
    const unsigned kGiveUpAfter = 256;
    // Separation and convergence are independent here: the eye offset is StereoBase in world units (VKSeparationScale
    // for games not in metres), the zero-parallax distance 1/One_div_ZPS at either sign. No convergence, no offset.
    const CameraPreset* preset = gInfo.Input.GetActivePreset();
    const float convDist = (preset && preset->One_div_ZPS != 0.f) ? 1.f / preset->One_div_ZPS : 0.f;
    const float eyeSep = (gInfo.Input.StereoActive && preset && convDist != 0.f) ? preset->StereoBase * gInfo.SeparationScale * Vk().VKSeparationScale : 0.f;
    // A key dies with its preset: a preset change releases the quarantine, as deadKeys stops their ranges being
    // collected at all.
    if (fabsf(eyeSep - c->lastSep) > 1e-6f || fabsf(convDist - c->lastConv) > 1e-3f)
    {
        c->lastSep = eyeSep;
        c->lastConv = convDist;
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (!d->deadKeys.empty())
        {
            Log("[cam] preset changed (sep=%.6f conv=%.4f): %zu quarantined keys released\n",
                eyeSep, convDist, d->deadKeys.size());
            d->deadKeys.clear();
        }
    }
    {
        PerfScope pa(d->perf.constApply);
        std::lock_guard<std::mutex> g(c->lock);
        // The mirror copies and patched blocks go to the GPU; camera blocks are fetched in parallel below.
        static std::vector<CopyJob> jobs;
        static std::vector<GpuCopy> gpuCopies;
        static std::vector<GpuUpdate> gpuUpdates;
        static std::vector<unsigned char> gpuData;
        jobs.clear();
        gpuCopies.clear();
        gpuUpdates.clear();
        gpuData.clear();
        // claimMirror: the claim sibling's copies and updates; the claimed updates go after the unclaimed ones.
        static std::vector<GpuCopy> claimCopies;
        static std::vector<GpuUpdate> claimPlain, claimPatched;
        claimCopies.clear(); claimPlain.clear(); claimPatched.clear();
        if (Traits().claimMirror)
        {
            // A claimed range another camera key mirrors unclaimed (its update would land over the patch) reads its own
            // copy, unless an unclaimed key there positions geometry too: both must keep one camera.
            std::unordered_map<const unsigned char*, uint64_t> plainAt;
            std::unordered_set<const unsigned char*> positionedAt;
            for (const Work& w : work)
            {
                if (w.consumerMask || !c->learned[w.key].cam.valid) continue;
                if (w.skipPos || c->claimedKeys.count(w.key) || c->skipPosKeys.count(w.key)) positionedAt.insert(w.host);
                else plainAt.emplace(w.host, w.key);
            }
            for (const Work& w : work)
            {
                if (!w.consumerMask || !c->learned[w.key].cam.valid) continue;
                uint32_t vs = 0;
                auto vsOf = [&]() {
                    std::shared_lock<std::shared_mutex> gd(d->lock);
                    auto pit = d->pipelines.find(w.pipeline);
                    return pit != d->pipelines.end() ? pit->second.vsCrc : 0u;
                };
                // A key once found sharing bytes with other positioned geometry stays on the shared mirror for good.
                if (positionedAt.count(w.host))
                {
                    if (c->claimVetoKeys.insert(w.key).second && c->claimMirrorKeys.erase(w.key))
                    {
                        dropClaimMirror.push_back(w.key);
                        vs = vsOf();
                        Log("[cam] claim mirror key=%016llx b%u vs=%08X dropped: another key positions geometry through the same bytes\n",
                            (unsigned long long)w.key, w.binding, vs);
                    }
                    continue;
                }
                auto p = plainAt.find(w.host);
                if (p == plainAt.end() || p->second == w.key || c->claimVetoKeys.count(w.key) || !c->claimMirrorKeys.insert(w.key).second) continue;
                newClaimMirror.push_back(w.key);
                vs = vsOf();
                Log("[cam] claim mirror key=%016llx b%u vs=%08X: key=%016llx mirrors the same bytes unclaimed; the claim reads its own copy (claim mirror)\n",
                    (unsigned long long)w.key, w.binding, vs, (unsigned long long)p->second);
            }
            for (const Work& w : work)
                if (!w.claimBuf && c->claimMirrorKeys.count(w.key)) needClaimSibling.push_back(w.buffer);
        }

        // Every camera block is fetched up front, in parallel: uncached ReBAR reads scale with threads.
        static std::vector<unsigned char> snaps;
        static std::vector<size_t> snapAt;
        snapAt.assign(work.size(), (size_t)-1);
        size_t snapTotal = 0;
        for (size_t i = 0; i < work.size(); ++i)
        {
            const Learned& L = c->learned[work[i].key];
            if (!L.cam.valid || !work[i].sibling) continue;
            snapAt[i] = snapTotal;
            snapTotal += work[i].bytes;
        }
        snaps.resize(snapTotal);
        jobs.clear();
        for (size_t i = 0; i < work.size(); ++i)
            if (snapAt[i] != (size_t)-1) jobs.push_back({ snaps.data() + snapAt[i], work[i].host, work[i].bytes });
        {
            const long long r0 = PerfNow();
            g_copyPool.Run(jobs);
            d->perf.constRead.fetch_add(PerfNow() - r0, std::memory_order_relaxed);
        }

        unsigned char snap[4096];
        long long tRead = 0, tMath = 0, tKey = 0, tRec = 0;   // accumulated, one atomic add at the end
        for (size_t wi = 0; wi < work.size(); ++wi)
        {
            const Work& w = work[wi];
            const long long k0 = PerfNow();
            Learned& L = c->learned[w.key];
            tKey += PerfNow() - k0;
            const size_t regs = w.bytes / 16;
            // A key learned during this pass has no prefetched block: it fetches its own, inline.
            const unsigned char* src = snap;
            if (snapAt[wi] != (size_t)-1) src = snaps.data() + snapAt[wi];
            else if (L.cam.valid && w.sibling) { memcpy(snap, w.host, w.bytes); }
            const float* f = (const float*)src;
            if (w.consumerMask && !(L.cam.valid && !L.cam.ps.empty()))
            {
                if (Traits().claimedKeysNeedP && !w.storage) c->geomKeys.insert(w.key);
                if (Traits().claimedKeyRetry && w.posClaim) c->posKeys.insert(w.key);
            }

            if (L.cam.valid)
            {
                ++c->camRanges;
                if (!w.sibling) { ++c->noSibling; needSibling.push_back(w.buffer); continue; }
                // The sibling is what the right eye reads here, so it mirrors the game's bytes every submit; only the
                // patch is conditional.
                ++c->copies; c->copyBytes += w.fullBytes;
                if (w.consumerMask) ++d->statScopedRanges; else ++d->statScopedSkipped;
                // On the GPU the patch is an absolute write of the whole block: a later duplicate wins.
                unsigned char* dst;
                const bool viaGpu = w.siblingBuf && (w.srcOff % 4) == 0 && (w.bytes % 4) == 0;
                if (viaGpu)
                {
                    const long long c0 = PerfNow();
                    gpuCopies.push_back({ w.buffer, w.siblingBuf, { w.srcOff, w.srcOff, (VkDeviceSize)w.fullBytes } });
                    const size_t at = gpuData.size();
                    gpuData.resize(at + w.bytes);
                    memcpy(gpuData.data() + at, src, w.bytes);   // cached -> cached
                    gpuUpdates.push_back({ w.siblingBuf, w.srcOff, (uint32_t)w.bytes, (uint32_t)at });
                    if (w.claimBuf && Traits().claimMirror && c->claimMirrorKeys.count(w.key))
                    {
                        claimCopies.push_back({ w.buffer, w.claimBuf, { w.srcOff, w.srcOff, (VkDeviceSize)w.fullBytes } });
                        (w.consumerMask ? claimPatched : claimPlain).push_back({ w.claimBuf, w.srcOff, (uint32_t)w.bytes, (uint32_t)at });
                    }
                    dst = gpuData.data() + at;
                    tRec += PerfNow() - c0;
                }
                else
                {
                    dst = w.sibling;
                }
                const long long m0 = PerfNow();
                bool wrote = false;
                // claimedKeysNeedP: a ray basis with no projection where geometry positions is a bone or cloth palette.
                bool basisOnly = L.cam.ps.empty();
                if (basisOnly) { basisOnly = false; for (const CamSlot& sl : L.cam.slots) if (sl.kind == K_RayBasis) basisOnly = true; }
                if (Traits().claimedKeysNeedP && basisOnly && w.consumerMask && !w.storage)
                {
                    // Mirrored unpatched for this submit, then dropped: the right eye reads the game's bytes again.
                    if (!L.evicting)
                    {
                        L.evicting = true;
                        deadKeys.push_back(w.key);
                        uint32_t vs = 0;
                        {
                            std::shared_lock<std::shared_mutex> gd(d->lock);
                            auto pit = d->pipelines.find(w.pipeline);
                            if (pit != d->pipelines.end()) vs = pit->second.vsCrc;
                        }
                        Log("[cam] basis key=%016llx b%u dropped: vs=%08X positions through it, no projection (claimed keys need P)\n",
                            (unsigned long long)w.key, w.binding, vs);
                    }
                    continue;
                }
                if (Traits().clipSlotStraddle && w.posClaim && w.consumerMask != ~0ull) L.chainStarts |= w.consumerMask;
                const bool applied = ApplyCamera(c, dst, f, regs, L.cam, eyeSep, convDist, w.consumerMask, &wrote, L.chainStarts);
                tMath += PerfNow() - m0;
                // claimMirror off the GPU path: the claim sibling takes the game's tail and this range's patched block.
                if (!viaGpu && w.claimMapped && Traits().claimMirror && c->claimMirrorKeys.count(w.key))
                {
                    if (w.fullBytes > w.bytes) memcpy(w.claimMapped + w.bytes, w.host + w.bytes, w.fullBytes - w.bytes);
                    memcpy(w.claimMapped, dst, w.bytes);
                }
                // A draw that skipped its claim (<=6 verts, EQUAL) reads the projection this claim just patched.
                if (wrote && w.noClaimPipe)
                {
                    ++c->claimConflicts;
                    const bool name = c->conflictNamed.size() < 32 && c->conflictNamed.insert(w.noClaimPipe).second;
                    if (name)
                    {
                        std::lock_guard<std::shared_mutex> gd(d->lock);
                        auto pit = d->pipelines.find(w.noClaimPipe);
                        auto cit = d->pipelines.find(w.pipeline);
                        // Named once per pipeline.
                        if (pit != d->pipelines.end())
                            Log("[cam] conflict: skip-claim vs=%08X depthOp=%d pos s%u:b%u+%u target %ux%u reads a range claimed by vs=%08X (key=%016llx binding=%u bytes=%zu)\n",
                                pit->second.vsCrc, pit->second.depthOp, pit->second.posSet, pit->second.posBinding, pit->second.posOffset,
                                w.noClaimW, w.noClaimH, cit != d->pipelines.end() ? cit->second.vsCrc : 0u,
                                (unsigned long long)w.key, w.binding, w.fullBytes);
                    }
                }
                // The verdict for the claiming pipeline (served: its next replay binds sep=0), read only under
                // VKServeFallback; lock order c->lock -> d->lock, as in the branch below.
                if (w.pipeline && Vk().VKServeFallback)
                {
                    std::lock_guard<std::shared_mutex> gd(d->lock);
                    auto pit = d->pipelines.find(w.pipeline);
                    if (pit != d->pipelines.end())
                    {
                        if (applied && wrote) { pit->second.served = true; pit->second.missStreak = 0; ++pit->second.servedApplied; }
                        else { ++pit->second.servedSkipped; if (++pit->second.missStreak >= 8) pit->second.served = false; }
                    }
                }
                if (applied) { ++c->camApplied; ++L.cam.applied; L.rayFails = 0; }
                else
                {
                    ++c->camSkipped; ++L.cam.skipped;
                    // A basis-only key that keeps failing (a bone that once faced the camera) is quarantined; a unit
                    // camera block never is, as binding, size, tan and aspect gate it.
                    bool cameraBlock = !L.cam.slots.empty();
                    for (const CamSlot& sl : L.cam.slots) if (sl.kind != K_UnitBasis) cameraBlock = false;
                    if (L.cam.ps.empty() && eyeSep != 0.f && !cameraBlock && ++L.rayFails > 60)
                    {
                        deadKeys.push_back(w.key);
                        Log("[cam] quarantined basis key=%016llx after %u failed verifies\n",
                            (unsigned long long)w.key, L.rayFails);
                    }
                }
                continue;
            }

            // Not copied below, yet the right eye reads the sibling here: what an earlier frame left at this offset.
            if (w.shadowed && w.sibling) ++c->staleShadowed;
            // Not a camera block yet: try on first sight, then every 64th, then give up (one binding of one layout
            // keeps its content type).
            const unsigned n = L.sightings++;
            // claimedKeyRetry: a position key is kept and retried; tries made before any camera was seen start over.
            const bool keep = Traits().claimedKeyRetry && !w.fs && c->posKeys.count(w.key) != 0;
            if (n == kGiveUpAfter && !w.fs && Traits().claimedKeyRetry && !c->camFwdValid) { L.sightings = 0; continue; }
            // A full-screen-consumed range never dies: its ray basis may become learnable only once gameplay writes it.
            if (n == kGiveUpAfter && !w.fs)
            {
                if (!keep) deadKeys.push_back(w.key);
                uint32_t vs = 0;
                {
                    std::shared_lock<std::shared_mutex> gd(d->lock);
                    auto pit = d->pipelines.find(w.pipeline ? w.pipeline : w.noClaimPipe);
                    if (pit != d->pipelines.end()) vs = pit->second.vsCrc;
                }
                Log("[cam] key=%016llx b%u given up after %u sightings without learning a camera (vs=%08X claimed=%d fwd %s, last try %s, bytes=%zu)%s\n",
                    (unsigned long long)w.key, w.binding, n, vs, w.consumerMask != 0, c->camFwdValid ? "valid" : "invalid", L.lastFail, w.bytes,
                    keep ? " -- kept, retried every 60 frames (claimed-key retry)" : "");
                if (keep && w.bytes >= 192)
                {
                    const float* q = (const float*)w.host + 32;
                    Log("[cam]   r8..r11: %g %g %g %g | %g %g %g %g | %g %g %g %g | %g %g %g %g\n", q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7],
                        q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15]);
                }
            }
            // Once a second per key, staggered by key so retries never bunch into one frame.
            if (keep && n > kGiveUpAfter && d->frames != L.retryFrame && (d->frames + (unsigned)(w.key >> 7)) % 60 == 0)
            { L.retryFrame = d->frames; L.retryLeft = 4; }
            const bool retry = keep && n > kGiveUpAfter && d->frames == L.retryFrame && L.retryLeft > 0;
            if (retry) --L.retryLeft;
            if (!retry && ((n >= kGiveUpAfter && !w.fs) || (n && (n % 64) != 0))) continue;
            memcpy(snap, w.host, w.bytes);
            ++c->scans;

            // The vertex shaders' own uniform offsets settle which matrix is the projection and which the view.
            std::unordered_map<uint32_t, unsigned> used;
            std::vector<BlockMatrix> packed;
            {
                std::shared_lock<std::shared_mutex> rd(d->lock, std::defer_lock);
                std::unique_lock<std::shared_mutex> ex(d->lock, std::defer_lock);
                if (Traits().sharedCollect) rd.lock(); else ex.lock();   // map reads only
                auto uit = d->positionUse.find(w.binding);
                if (uit != d->positionUse.end()) used = uit->second;
                auto pit = d->packedMatrices.find(w.binding);
                if (pit != d->packedMatrices.end()) packed = pit->second;
            }
            const char* why = "";
            CameraLayout cl;
            // claimedKeysNeedP: where geometry positions, a basis alone is a bone that faced the camera.
            const bool needP = Traits().claimedKeysNeedP && c->geomKeys.count(w.key) != 0;
            const bool byRegister = LearnCamera(f, regs, used.empty() ? nullptr : &used, cl, &why,
                                                w.fs != 0 && !needP, c->camFwdValid ? c->camFwd : nullptr);
            // Matrices the register scan cannot see: a reconstruction basis at a scalar-layout stride.
            bool packedBasis = false;
            for (const BlockMatrix& bm : packed)
            {
                if (needP) break;
                if (bm.rows != 3 || bm.cols != 3 || !bm.colMajor) continue;
                if ((size_t)bm.offset + (size_t)bm.stride * 3 > w.bytes) continue;
                float t[12] = {};
                UnpackBasis3x3(snap, bm.offset, bm.stride, !bm.colMajor, t);
                const bool ok = RayBasisOk(c, t);
                if (!ok) continue;
                CamSlot sl{};
                sl.kind = K_RayBasis; sl.byteOff = bm.offset; sl.stride = (uint8_t)bm.stride;
                sl.transposed = !bm.colMajor;
                cl.slots.push_back(sl);
                packedBasis = true;
                Log("[cam] packed basis binding=%u offset=%u stride=%u accepted\n", w.binding, bm.offset, bm.stride);
            }
            // unitCameraBlock: it carries its own origin (row 3), so it needs neither projection nor value search.
            if (Traits().unitCameraBlock && cl.ps.empty())
                for (size_t reg = 0; reg + 5 <= regs; ++reg)
                {
                    if (!UnitBasisOk(c, f + reg * 4)) continue;
                    CamSlot sl{};
                    sl.kind = K_UnitBasis; sl.reg = (unsigned)reg;
                    cl.slots.push_back(sl);
                    Log("[cam] unit camera block binding=%u reg=%zu bytes=%zu tan=%.4f/%.4f accepted\n",
                        w.binding, reg, w.bytes, f[reg * 4 + 3], f[reg * 4 + 7]);
                    break;
                }
            // The correction's constant half, one eye of world offset, goes to a basis-only block's origin;
            // the ordinary camera position stays the game's.
            bool hasRay = false, origin = false;
            for (const CamSlot& s : cl.slots) if (s.kind == K_RayBasis) hasRay = true;
            if (hasRay && cl.ps.empty() && c->camPosValid)
                for (size_t off = 0; off + 12 <= w.bytes && !origin; off += 4)
                {
                    const float* p = (const float*)(snap + off);
                    bool match = true;
                    for (int i = 0; i < 3; ++i)
                    {
                        // Relative: world coordinates run to thousands, where an absolute epsilon never matches.
                        const float tol = 0.01f + 1e-5f * fabsf(c->camPos[i]);
                        if (fabsf(p[i] - c->camPos[i]) > tol) { match = false; break; }
                    }
                    if (!match) continue;
                    CamSlot sl{};
                    sl.kind = K_CamOffset; sl.byteOff = (uint32_t)off; sl.stride = 0;
                    cl.slots.push_back(sl);
                    origin = true;
                    Log("[cam] reconstruction origin binding=%u offset=%zu accepted\n", w.binding, off);
                }
            // Three registers of basis are usually rows 0-2 of a 4x4 whose row 3 is the origin: taken by layout when
            // the exact match missed.
            if (hasRay && !origin && cl.ps.empty() && c->camPosValid)
                for (const CamSlot& s : cl.slots)
                {
                    if (s.kind != K_RayBasis || s.stride != 16) continue;
                    const size_t off = (size_t)(s.reg + 3) * 16;
                    if (off + 12 > w.bytes) continue;
                    const float* p = (const float*)(snap + off);
                    bool ok = true;
                    for (int i = 0; i < 3; ++i)
                        if (!(p[i] == p[i]) || fabsf(p[i] - c->camPos[i]) > 5.f) { ok = false; break; }
                    if (!ok) continue;
                    CamSlot sl{};
                    sl.kind = K_CamOffset; sl.byteOff = (uint32_t)off; sl.stride = 0;
                    cl.slots.push_back(sl);
                    Log("[cam] origin from basis row binding=%u offset=%zu accepted\n", w.binding, off);
                    break;
                }
            if (!byRegister && cl.slots.empty()) { L.lastFail = why[0] ? why : "noLayout"; continue; }
            const size_t r0 = cl.ps.empty()
                ? (cl.slots[0].stride == 16 ? (size_t)cl.slots[0].reg * 16 : cl.slots[0].byteOff)
                : (size_t)cl.ps[0].reg * 16;
            const size_t cmpLen = w.bytes - r0 < 64 ? w.bytes - r0 : 64;
            // Still being written.
            if (memcmp(snap + r0, w.host + r0, cmpLen) != 0) { L.lastFail = "beingWritten"; continue; }

            // LearnCamera sets `valid` only on its own success; a layout from packed detection alone is valid too.
            cl.valid = true;
            L.cam = cl;
            d->cameraActive.store(true);
            newKeys.push_back(w.key);
            needSibling.push_back(w.buffer);
            // Learned on a retry: sets cached since the give-up still name the game's buffer, so they are re-pointed.
            if (retry && c->repointKeys.insert(w.key).second) newRepoint.push_back(w.key);
            char slots[512] = {}; size_t len = 0;
            for (const CamSlot& sl : cl.slots)
                len += (size_t)snprintf(slots + len, sizeof(slots) - len > 0 ? sizeof(slots) - len : 0, " %s@%u%s/P%u", KindName(sl.kind), sl.reg, sl.transposed ? "T" : "", sl.pIndex);
            Log("[cam] learned key=%016llx buffer %p bytes=%zu: %zu projection(s) view=%d, slots:%s\n",
                (unsigned long long)w.key, (void*)w.buffer, w.bytes, cl.ps.size(), (int)cl.hasV, slots);
            if (c->cameraDumpBudget > 0)
            {
                --c->cameraDumpBudget;
                for (size_t r2 = 0; r2 < regs; ++r2)
                    Log("      r%03zu %12.5f %12.5f %12.5f %12.5f\n", r2, f[r2 * 4], f[r2 * 4 + 1], f[r2 * 4 + 2], f[r2 * 4 + 3]);
            }
        }
        {
            const long long s0 = PerfNow();
            gpuCopies.insert(gpuCopies.end(), claimCopies.begin(), claimCopies.end());
            gpuUpdates.insert(gpuUpdates.end(), claimPlain.begin(), claimPlain.end());
            gpuUpdates.insert(gpuUpdates.end(), claimPatched.begin(), claimPatched.end());
            SubmitGpuMirror(d, queue, gpuCopies, gpuUpdates, gpuData);
            tRec += PerfNow() - s0;
        }
        d->perf.constRead.fetch_add(tRead, std::memory_order_relaxed);
        d->perf.constMath.fetch_add(tMath, std::memory_order_relaxed);
        d->perf.constKey.fetch_add(tKey, std::memory_order_relaxed);
        d->perf.constRec.fetch_add(tRec, std::memory_order_relaxed);
    }

    // The far field shifts by P11 * eyeSep / convDist: the majority P11 sizes the composite's edge crop and the shift.
    {
        std::lock_guard<std::mutex> g(c->lock);
        int best = 0; unsigned n = 0;
        for (const auto& kv : c->p11Hist) if (kv.second > n) { n = kv.second; best = kv.first; }
        if (n) c->lastP11 = best / 1000.f;
        c->p11Hist.clear();
        // The main view draws the most claimed geometry in a frame: a smaller submit later in it (reflection,
        // cubemap) cannot replace it, and the next frame starts over.
        if (c->voteFrame != d->frames) { c->frameBestN = 0; c->voteFrame = d->frames; }
        const Constants::FwdVote* top = nullptr;
        for (const auto& kv : c->fwdVotes) if (!top || kv.second.n > top->n) top = &kv.second;
        if (top && top->n >= 16 && top->n > c->frameBestN)
        {
            c->frameBestN = top->n;
            c->camFwdVote[0] = top->f[0]; c->camFwdVote[1] = top->f[1]; c->camFwdVote[2] = top->f[2];
            c->camFwdVoteValid = true;
            c->publishedN = top->n;
        }
        c->fwdVotes.clear();
    }
    if (c->lastP11 > 0.f) d->cameraP11.store(c->lastP11, std::memory_order_relaxed);
    if (!deadKeys.empty())
    {
        std::lock_guard<std::mutex> g(c->lock);
        for (uint64_t k : deadKeys) c->learned.erase(k);
    }
    if (!newKeys.empty() || !deadKeys.empty() || !newClaimed.empty() || !newSkipPos.empty() || !newRepoint.empty() || !newClaimMirror.empty() ||
        !dropClaimMirror.empty())
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        for (uint64_t k : newRepoint) d->repointKeys.insert(k);
        for (uint64_t k : newClaimMirror) d->claimMirrorKeys.insert(k);
        for (uint64_t k : dropClaimMirror) d->claimMirrorKeys.erase(k);
        for (uint64_t k : newClaimed) d->claimedKeys.insert(k);
        for (uint64_t k : newSkipPos)
            if (d->skipPosKeys.insert(k).second && !d->claimedKeys.count(k))
                Log("[cam] skip-claim position key=%016llx reads the game's buffer in the right eye (unpatched keys)\n", (unsigned long long)k);
        for (uint64_t k : newKeys) d->cameraKeys.insert(k);
        for (uint64_t k : deadKeys) { d->deadKeys.insert(k); d->cameraKeys.erase(k); }
    }
    std::sort(needSibling.begin(), needSibling.end());
    needSibling.erase(std::unique(needSibling.begin(), needSibling.end()), needSibling.end());
    bool madeSibling = false;
    for (VkBuffer b : needSibling) madeSibling |= CreateBufferSibling(d, b);
    std::sort(needClaimSibling.begin(), needClaimSibling.end());
    needClaimSibling.erase(std::unique(needClaimSibling.begin(), needClaimSibling.end()), needClaimSibling.end());
    for (VkBuffer b : needClaimSibling) madeSibling |= CreateBufferSibling(d, b, true);
    // Only a new key, a new sibling or a newly named re-point or claim key can make a cached set stale.
    if (!newKeys.empty() || madeSibling || !newRepoint.empty() || !newClaimMirror.empty() || !dropClaimMirror.empty())
    {
        unsigned forced = 0;
        const unsigned n = RepointLearnedSets(d, Vk().VKRepointLearnedSets, &forced);
        if (n) Log("[dset] frame %u: %u cached set binding(s) still on the game's camera buffer, %s\n",
                   d->frames, n, Vk().VKRepointLearnedSets ? "re-pointed at the sibling" : "left (VKRepointLearnedSets=0)");
        if (forced) Log("[dset] frame %u: %u of them re-pointed at the sibling: named position keys (re-point, claimed-key retry)\n", d->frames, forced);
    }
}

// Storage buffers the compute gate refused a dispatch over. Lock NOT held.
void DrainPendingBufferSiblings(DeviceData* d)
{
    std::vector<VkBuffer> want;
    {
        std::lock_guard<std::shared_mutex> g(d->lock);
        if (d->pendingBufferSiblings.empty()) return;
        want.swap(d->pendingBufferSiblings);
    }
    for (VkBuffer b : want) CreateBufferSibling(d, b);
}

void ConstantsLogStats(DeviceData* d)
{
    Constants* c = d->constants;
    if (!c || !gInfo.DetectInverseMatrix) return;
    std::lock_guard<std::mutex> g(c->lock);
    unsigned cams = 0, camsNoV = 0;
    for (const auto& kv : c->learned)
        if (kv.second.cam.valid) { ++cams; if (!kv.second.cam.hasV) ++camsNoV; }
    Log("[cam %u] keys=%u (noView %u) ranges seen=%u unmapped=%u scans=%u | camera=%u patched=%u notCamera=%u | copies=%u (%llu KB) noSibling=%u siblings=%u active=%d\n",
        d->frames, cams, camsNoV, c->ranges, c->unmapped, c->scans,
        c->camRanges, c->camApplied, c->camSkipped, c->copies, c->copyBytes / 1024, c->noSibling, c->siblings, (int)d->cameraActive.load());
    // scoped: ranges a draw claimed and we patched; unclaimed: camera ranges mirrored and left alone (screen passes).
    // unclaimed >> scoped means the claim is not reaching the draws.
    Log("[cam %u] scoped=%u unclaimed=%u | layerMapped=%u unmappable=%u | storage ranges=%u | rayBasis patched=%u\n",
        d->frames, d->statScopedRanges, d->statScopedSkipped, c->layerMaps, c->unmappable, c->storageRanges, c->rayPatched);
    Log("[cam %u] inverse slots: patched=%u rejected=%u (rejected = slot was not this range's P^-1)\n", d->frames, c->invApplied, c->invRejected);
    Log("[cam %u] unit camera blocks patched=%u | prevP patched=%u | claim conflicts=%u (skip-claim draw read a patched range) | stale shadowed=%u (right eye reads an unrefreshed sibling) | fwd last=(%.3f %.3f %.3f)%s vote=(%.3f %.3f %.3f)%s from %u\n",
        d->frames, c->unitPatched, c->prevPatched, c->claimConflicts, c->staleShadowed,
        c->camFwd[0], c->camFwd[1], c->camFwd[2], c->camFwdValid ? "" : " invalid",
        c->camFwdVote[0], c->camFwdVote[1], c->camFwdVote[2], c->camFwdVoteValid ? "" : " invalid", c->publishedN);
    // A claimed camera range left unpatched because its projection failed the shape test: that draw's geometry is flat.
    if (c->projRefused)
        Log("[cam %u] projection refused on %u claimed range(s), last P00=%g P11=%g\n", d->frames, c->projRefused, c->refusedP00, c->refusedP11);
    c->projRefused = 0;
    if (Traits().sharedCollect)
        Log("[cam %u] shared collect: %u set(s) collected again under the exclusive lock (first map of their memory)\n", d->frames, c->collectRetries);
    c->collectRetries = 0;
    c->unitPatched = c->prevPatched = c->claimConflicts = c->staleShadowed = 0;
    c->rayPatched = c->storageRanges = 0;
    c->invApplied = c->invRejected = 0;
    // DEVICE_LOCAL|HOST_VISIBLE without HOST_CACHED: every CPU read of these constants is an uncached round trip.
    for (uint32_t t = 0; t < 32 && t < d->memProps.memoryTypeCount; ++t)
    {
        if (!c->typeHits[t]) continue;
        const VkMemoryPropertyFlags f = d->memProps.memoryTypes[t].propertyFlags;
        Log("[cam %u] read %u ranges from memory type %u:%s%s%s%s\n", d->frames, c->typeHits[t], t,
            (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? " DEVICE_LOCAL" : "",
            (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? " HOST_VISIBLE" : "",
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? " HOST_COHERENT" : "",
            (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? " HOST_CACHED" : "");
        c->typeHits[t] = 0;
    }
    c->layerMaps = c->unmappable = 0;
    d->statScopedRanges = d->statScopedSkipped = 0;
    // Every range bound through a camera key must be copied or still wait for its sibling; a shortfall means a range
    // kept a previous frame's constants.
    if (c->copies + c->noSibling != c->camRanges)
        Log("[cam %u] INVARIANT: %u camera ranges but only %u copied + %u without a sibling - %u left stale\n",
            d->frames, c->camRanges, c->copies, c->noSibling, c->camRanges - c->copies - c->noSibling);
    c->camRanges = c->camApplied = c->camSkipped = 0;
    c->ranges = c->unmapped = c->scans = c->copies = c->noSibling = 0;
    c->copyBytes = 0;
}

// Submit hooks
static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit(VkQueue queue, uint32_t count, const VkSubmitInfo* pSubmits, VkFence fence)
{
    DeviceData* d = FindDevice(queue);
    for (uint32_t i = 0; i < count; ++i)
        ConstantsSubmit(d, queue, pSubmits[i].commandBufferCount, pSubmits[i].pCommandBuffers);
    VkResult sr;
    { const auto ql = FgQueueLock(); sr = d->QueueSubmit(queue, count, pSubmits, fence); }
    // DEVICE_LOST here means the GPU faulted on recorded work, which no CPU-side counter would show.
    if (sr != VK_SUCCESS) { static int n = 8; if (n-- > 0) Log("!! vkQueueSubmit returned %d\n", sr); }
    return sr;
}

static VKAPI_ATTR VkResult VKAPI_CALL Hook_QueueSubmit2(VkQueue queue, uint32_t count, const VkSubmitInfo2* pSubmits, VkFence fence)
{
    DeviceData* d = FindDevice(queue);
    std::vector<VkCommandBuffer> cbs;
    for (uint32_t i = 0; i < count; ++i)
    {
        cbs.clear();
        for (uint32_t k = 0; k < pSubmits[i].commandBufferInfoCount; ++k) cbs.push_back(pSubmits[i].pCommandBufferInfos[k].commandBuffer);
        ConstantsSubmit(d, queue, (uint32_t)cbs.size(), cbs.data());
    }
    VkResult sr;
    { const auto ql = FgQueueLock(); sr = d->QueueSubmit2(queue, count, pSubmits, fence); }
    return sr;
}

const HookEntry g_constantsHooks[] = {
    WIZ3D_HOOK(CreateBuffer), WIZ3D_HOOK(DestroyBuffer),
    WIZ3D_HOOK(BindBufferMemory), WIZ3D_HOOK(BindBufferMemory2),
    WIZ3D_HOOK(MapMemory), WIZ3D_HOOK(UnmapMemory),
    WIZ3D_HOOK(QueueSubmit), WIZ3D_HOOK(QueueSubmit2),
    { nullptr, nullptr },
};

}} // namespace wiz3d::vk
