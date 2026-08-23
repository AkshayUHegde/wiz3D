/* wiz3D - numeric view-projection / inverse-matrix detection shared by the
 * DX11 wrapper and the Vulkan layer.
 *
 * Deferred renderers reconstruct position in screen-space passes from an
 * inverse view-projection that no shader analysis reaches. Instead, identify
 * it numerically: cache the pre-shift (mono) values of every clean
 * view-projection the forward path sees, then test 4x4 blocks of constant data
 * for C * V == I against that ring. A hit is patched for the right eye with
 * T^-1 * P^-1, where T is the clip-space shear the forward path applies.
 *
 * Bodies are the ones tuned on Battlefield 3, Far Cry 3 and BioShock Infinite
 * in Context11Proxy.cpp; they moved here unchanged so both APIs share one
 * detector. Header-only so the DX11 project (precompiled headers) needs no
 * build change.
 */

#pragma once

#include <math.h>
#include <string.h>
#include <stddef.h>

namespace wiz3d {

// f interpreted per `transposed` into canonical row-vector layout (the
// w-producing column at m[3], m[7], m[11], m[15]).
inline void CanonicalizeMat4(const float* f, bool transposed, float* out)
{
    if (!transposed) { memcpy(out, f, 16 * sizeof(float)); return; }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            out[r * 4 + c] = f[c * 4 + r];
}

// 3x3 norms of the x and w bases. |w basis| is 1 for a clean view-projection
// (the camera forward axis is unit) and carries the object scale for a WVP.
inline void MatrixBasisNorms(const float* f, bool transposed, float& xn, float& wn)
{
    float x0, x1, x2, w0, w1, w2;
    if (transposed) { x0 = f[0]; x1 = f[1]; x2 = f[2];  w0 = f[12]; w1 = f[13]; w2 = f[14]; }
    else            { x0 = f[0]; x1 = f[4]; x2 = f[8];  w0 = f[3];  w1 = f[7];  w2 = f[11]; }
    xn = sqrtf(x0 * x0 + x1 * x1 + x2 * x2);
    wn = sqrtf(w0 * w0 + w1 * w1 + w2 * w2);
}

// |cos| between the x and w bases; a camera (or rigid WVP) keeps them
// orthogonal, a block of unrelated float4s does not.
inline bool BasesOrthogonal(const float* f, bool transposed, float xn, float wn)
{
    float x0, x1, x2, w0, w1, w2;
    if (transposed) { x0 = f[0]; x1 = f[1]; x2 = f[2];  w0 = f[12]; w1 = f[13]; w2 = f[14]; }
    else            { x0 = f[0]; x1 = f[4]; x2 = f[8];  w0 = f[3];  w1 = f[7];  w2 = f[11]; }
    const float cosXW = (x0 * w0 + x1 * w1 + x2 * w2) / (xn * wn + 1e-12f);
    return fabsf(cosXW) <= 0.2f;
}

// A clean view-projection: unit w basis, x and w bases orthogonal. Neither a
// scaled WVP nor a light projection nor a UE3 material block passes.
inline bool IsCleanViewProjection(const float* f, bool transposed, float* outXScale)
{
    float xn, wn;
    MatrixBasisNorms(f, transposed, xn, wn);
    if (!(wn > 1e-6f && fabsf(wn - 1.f) < 0.05f && BasesOrthogonal(f, transposed, xn, wn))) return false;
    if (outXScale) *outXScale = xn / wn;
    return true;
}

// One product element with a cancellation-aware tolerance: |acc - target|
// must stay under 1e-4 * (sum of |term|) + 2e-3. Returned as a ratio of that
// bound, so <= 1 passes and the frame log can report the closest miss.
inline float InverseProductDeviation(const float* c, const float* v, int i, int j)
{
    float acc = 0.f, mag = 0.f;
    for (int k = 0; k < 4; ++k)
    {
        const float t = c[i * 4 + k] * v[k * 4 + j];
        acc += t;
        mag += fabsf(t);
    }
    const float target = (i == j) ? 1.f : 0.f;
    // Bound doubled from the Frostbite-tuned 1e-4/2e-3: UE3's screen-to-world
    // folds viewport terms into the inverse and sat at bestMiss=1.00-1.01
    // (BioShock Infinite), just outside. True pairs score far below 1, junk
    // far above, so the margin is wide.
    return fabsf(acc - target) / (2e-4f * mag + 4e-3f);
}

// NaN poisons the max-accumulated deviation test: every comparison with NaN
// is false, so a NaN term can never raise `dev` and a garbage block passes
// (first Far Cry 3 "hit" was a NaN-filled matrix). Reject non-finite blocks.
inline bool AllFinite16(const float* f)
{
    for (int i = 0; i < 16; ++i)
    {
        // Bit test: an all-ones exponent is NaN or Inf whatever the float
        // model makes of x != x (the Vulkan layer saw NaN blocks pass).
        unsigned bits;
        memcpy(&bits, f + i, 4);
        if ((bits & 0x7F800000u) == 0x7F800000u) return false;
        if (fabsf(f[i]) > 1e30f) return false;
    }
    return true;
}

// Row-major 4x4 product, row-vector convention: out = a * b.
inline void MulMat4(const float* a, const float* b, float* out)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            float acc = 0.f;
            for (int k = 0; k < 4; ++k) acc += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = acc;
        }
}

// Gauss-Jordan in double; D3DXMatrixInverse is unavailable in this shared header.
inline bool InvertMat4(const float* m, float* out)
{
    double a[4][8];
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c) a[r][c] = m[r * 4 + c];
        for (int c = 0; c < 4; ++c) a[r][4 + c] = (r == c) ? 1.0 : 0.0;
    }
    for (int col = 0; col < 4; ++col)
    {
        int piv = col;
        for (int r = col + 1; r < 4; ++r)
            if (fabs(a[r][col]) > fabs(a[piv][col])) piv = r;
        if (fabs(a[piv][col]) < 1e-12) return false;
        if (piv != col)
            for (int c = 0; c < 8; ++c) { const double t = a[col][c]; a[col][c] = a[piv][c]; a[piv][c] = t; }
        const double d = a[col][col];
        for (int c = 0; c < 8; ++c) a[col][c] /= d;
        for (int r = 0; r < 4; ++r)
        {
            if (r == col) continue;
            const double f = a[r][col];
            if (f == 0.0) continue;
            for (int c = 0; c < 8; ++c) a[r][c] -= f * a[col][c];
        }
    }
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[r * 4 + c] = (float)a[r][4 + c];
    return AllFinite16(out);
}

// Cheap gate before any ring work: an unprojection cannot reintroduce a
// perspective divide on x or y, so P^-1 -- and P^-1 * L, since L's first three
// rows carry w = 0 -- has zeroes at [0][3] and [1][3]. Bounded relative to each
// row's magnitude so off-axis and TAA-jittered projections still pass.
inline bool LooksLikeUnprojection(const float* canon)
{
    for (int r = 0; r < 2; ++r)
    {
        const float* row = canon + r * 4;
        const float n = fabsf(row[0]) + fabsf(row[1]) + fabsf(row[2]) + fabsf(row[3]);
        if (!(n > 1e-20f)) return false;              // an all-zero row is not a matrix
        if (fabsf(row[3]) > 1e-2f * n) return false;  // a real w term: not an unprojection
    }
    return true;
}

// Is M a light view-projection -- what remains when P^-1 divides out of a
// composed screen-to-light matrix? A CSM light is orthographic (no w column)
// and its 3x3 is R * diag(s), whose COLUMNS are orthogonal, not its rows. A
// folded-in texture bias is diagonal, so it preserves that.
inline bool IsLightTransform(const float* M)
{
    if (!AllFinite16(M)) return false;
    if (fabsf(M[3]) > 1e-4f || fabsf(M[7]) > 1e-4f || fabsf(M[11]) > 1e-4f) return false;
    if (fabsf(M[15] - 1.f) > 1e-3f) return false;     // also rejects an all-zero product

    const float col[3][3] = {
        { M[0], M[4], M[8]  },
        { M[1], M[5], M[9]  },
        { M[2], M[6], M[10] },
    };
    float n[3];
    for (int i = 0; i < 3; ++i)
    {
        n[i] = sqrtf(col[i][0] * col[i][0] + col[i][1] * col[i][1] + col[i][2] * col[i][2]);
        if (!(n[i] > 1e-12f)) return false;
    }
    for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 3; ++j)
        {
            const float d = col[i][0] * col[j][0] + col[i][1] * col[j][1] + col[i][2] * col[j][2];
            if (fabsf(d) > 0.02f * n[i] * n[j]) return false;
        }
    return true;
}

// Full C * V == I test of one canonical candidate against one cached VP.
// Returns the max deviation ratio (<= 1 is a match).
inline float InverseDeviation(const float* cand, const float* v)
{
    float dev = InverseProductDeviation(cand, v, 0, 0);
    const float d01 = InverseProductDeviation(cand, v, 0, 1);   // two-element prefilter
    if (d01 > dev) dev = d01;
    if (dev > 1.f) return dev;
    for (int i = 0; i < 4 && dev <= 1.f; ++i)
        for (int j = 0; j < 4; ++j)
        {
            const float d = InverseProductDeviation(cand, v, i, j);
            if (d > dev) dev = d;
            if (dev > 1.f) break;
        }
    return dev;
}

// FNV-1a over a canonical 4x4: a memcmp per ring entry per insert is the one
// dedup cost that scales with a game's draw count.
inline unsigned long long HashMat4(const float* m)
{
    const unsigned char* p = reinterpret_cast<const unsigned char*>(m);
    unsigned long long h = 1469598103934665603ULL;
    for (int i = 0; i < 64; ++i)
    {
        h ^= (unsigned long long)p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

struct StereoMatrixState
{
    // Camera projection _11, cached from the last clean view-projection seen.
    float projXScale = 0.f;

    // Ring of pre-shift (mono) perspective matrices the forward path corrected,
    // canonical row-vector layout; DetectInverse tests candidates against it.
    //
    // A rigid per-object WVP passes the test that gates caching and RAGE writes
    // one per draw, so GTA V evicted the camera VP ~5x per frame (hits=0).
    // Admission stays immediate anyway: requiring a repeat stalled BF3, whose
    // moving camera writes a unique VP once per frame. Repetition drives only
    // eviction, and age overrides it -- a moved-on camera confirming a learned
    // target is the same intermittent patching by another route.
    struct CachedForwardVP
    {
        float              m[16];
        unsigned long long hash;
        unsigned           hits;       // identical re-sightings; eviction tiebreak
        unsigned           lastFrame;  // when last sighted; dominates eviction
        unsigned           serial;     // insert order, final tiebreak
        bool               valid;
    };
    // 64, not 32: BF3 writes its deferred CB after a run of per-object matrices.
    // Depth is cheap now LooksLikeUnprojection gates the walk.
    CachedForwardVP ring[64] = {};
    unsigned ringSerial = 0;
    unsigned ringMRU = 0;                // scanned first: most likely to match
    unsigned ringInsertsThisFrame = 0;   // churn indicator for the log
    unsigned evictedHotThisFrame = 0;    // evicted an entry that had re-sightings
    unsigned invStaleThisFrame = 0;      // learned inverses failing re-verify
    unsigned invStaleShapeThisFrame = 0; // ...of those, not unprojection-shaped

    // Advanced once per present; entries older than kMaxAge cannot match.
    unsigned frameId = 0;
    // Bisect switches, set from config: legacyRing restores round-robin
    // admission over 32 slots with no ageing, skipPrefilter bypasses the
    // structural gate. Both reproduce the pre-2026-08-22 detector exactly.
    bool legacyRing = false;
    bool skipPrefilter = false;
    static const unsigned kRing       = 64;
    static const unsigned kLegacyRing = 32;
    unsigned RingSlots() const { return legacyRing ? kLegacyRing : kRing; }
    // One frame of slack: a CB can be written before geometry re-caches the VP.
    static const unsigned kMaxAge = 1;

    bool EntryUsable(const CachedForwardVP& e, unsigned maxAge) const
    {
        if (!e.valid) return false;
        if (legacyRing) return true;
        return (frameId - e.lastFrame) <= maxAge;
    }

    unsigned RingValid() const
    {
        unsigned n = 0;
        for (unsigned i = 0; i < kRing; ++i) if (ring[i].valid) ++n;
        return n;
    }

    // Young enough to match; a full ring with none of these is what reads as
    // intermittent patching.
    unsigned RingFresh() const
    {
        unsigned n = 0;
        for (unsigned i = 0; i < kRing; ++i) if (EntryUsable(ring[i], kMaxAge)) ++n;
        return n;
    }

    // Right-eye clip-space x shift per unit w. The forward and inverse paths
    // describe the same transform T, so both must use this same value.
    float EyeShiftB(float eyeShift) const
    {
        return eyeShift * (projXScale != 0.f ? projXScale : 1.f);
    }

    // Camera _11. Both stereo terms are world-matrix independent, so _11 is the only
    // per-matrix unknown -- and xn/wn only recovers it when the object scale is
    // uniform. Trust the ratio when |w basis| is unit, else reuse the last matrix
    // that was, so a non-uniformly scaled mesh cannot displace itself.
    float ProjXScaleFromMatrix(const float* f, bool transposed) const
    {
        float xn, wn;
        MatrixBasisNorms(f, transposed, xn, wn);
        if (wn <= 1e-6f) return projXScale;
        const float ratio = xn / wn;
        if (fabsf(wn - 1.f) < 0.05f) return ratio;
        return (projXScale != 0.f) ? projXScale : ratio;
    }

    void CacheForwardVP(const float* f, bool transposed)
    {
        float canon[16];
        CanonicalizeMat4(f, transposed, canon);
        const unsigned long long h = HashMat4(canon);

        // Re-sighting refreshes age and raises hits: the camera VP recurs, the
        // per-object WVPs around it do not.
        const unsigned slots = RingSlots();
        for (unsigned i = 0; i < slots; ++i)
        {
            if (!ring[i].valid || ring[i].hash != h) continue;
            if (memcmp(ring[i].m, canon, sizeof(canon)) != 0) continue;   // hash collision
            ++ring[i].hits;
            ring[i].lastFrame = frameId;
            ringMRU = i;
            return;
        }

        // Free slot, then anything too old to match, then an entry nothing has
        // repeated (round-robin among those, the original behaviour). GTA V's
        // WVPs stay at hits == 1, so its flood consumes only its own slots.
        unsigned slot = 0;
        bool     found = false;
        if (legacyRing)
        {
            slot = ringSerial % kLegacyRing;    // plain round robin
            found = true;
        }
        for (unsigned i = 0; i < slots && !found; ++i)
            if (!ring[i].valid) { slot = i; found = true; }
        if (!found)
            for (unsigned i = 0; i < slots && !found; ++i)
                if (!EntryUsable(ring[i], kMaxAge)) { slot = i; found = true; }
        if (!found)
        {
            slot = 0;
            for (unsigned i = 1; i < slots; ++i)
            {
                const CachedForwardVP& a = ring[i];
                const CachedForwardVP& b = ring[slot];
                const bool aCold = (a.hits <= 1), bCold = (b.hits <= 1);
                if (aCold != bCold)           { if (aCold) slot = i; }
                else if (a.serial < b.serial) slot = i;
            }
            if (ring[slot].hits > 1) ++evictedHotThisFrame;
        }
        memcpy(ring[slot].m, canon, sizeof(canon));
        ring[slot].hash      = h;
        ring[slot].hits      = 1;
        ring[slot].lastFrame = frameId;
        ring[slot].serial    = ++ringSerial;
        ring[slot].valid     = true;
        ringMRU = slot;
        ++ringInsertsThisFrame;
    }

    // Ring slot to test at position `ri`: the most recently touched entry first,
    // then everything else. The MRU entry is the camera VP in any frame that is
    // still writing geometry, so a real match usually lands on the first probe.
    unsigned RingProbeOrder(unsigned ri) const
    {
        const unsigned slots = RingSlots();
        if (ri == 0) return ringMRU < slots ? ringMRU : 0;
        const unsigned idx = ri - 1;
        const unsigned mru = ringMRU < slots ? ringMRU : 0;
        return (idx < mru) ? idx : (idx + 1) % slots;   // skip the slot already probed
    }

    // The P^-1 the patch needs: the candidate itself for a bare inverse VP, or
    // the inverse of the ring entry that divided out for a composed one.
    struct InverseMatch
    {
        bool  composed = false;
        float basis[16] = {};
    };

    // `allowComposed` is a per-game opt-in: it accepts a wider set, and a false
    // positive patches an unrelated constant with no shape guard behind it.
    bool MatchInverse(const float* cand, bool allowComposed,
                      InverseMatch* out, float* bestMiss,
                      bool* outShapeReject = nullptr) const
    {
        // Split the two ways a match fails: the block is not unprojection-shaped
        // at all, or it is but no cached camera inverts it. They need opposite
        // fixes, and the frame log cannot tell them apart otherwise.
        if (!skipPrefilter && !LooksLikeUnprojection(cand))
        {
            if (outShapeReject) *outShapeReject = true;
            return false;
        }
        for (unsigned ri = 0; ri < RingSlots(); ++ri)
        {
            const unsigned idx = RingProbeOrder(ri);
            if (!EntryUsable(ring[idx], kMaxAge)) continue;

            const float dev = InverseDeviation(cand, ring[idx].m);
            if (dev <= 1.f)
            {
                if (out) { out->composed = false; memcpy(out->basis, cand, 16 * sizeof(float)); }
                return true;
            }
            if (bestMiss && dev < *bestMiss) *bestMiss = dev;

            // V * X should leave a light transform behind. The basis this yields
            // goes straight into the patch, so unlike the pure case it must be
            // the current frame's VP, not one frame stale.
            if (!allowComposed || !EntryUsable(ring[idx], 0)) continue;
            float product[16];
            MulMat4(ring[idx].m, cand, product);
            if (!IsLightTransform(product)) continue;
            if (!out) return true;
            if (!InvertMat4(ring[idx].m, out->basis)) continue;
            out->composed = true;
            return true;
        }
        return false;
    }

    struct DetectedInverse
    {
        unsigned reg;
        bool     transposed;
        bool     composed;
    };

    // Every match, not just the first: a deferred pass keeps the inverse VP and
    // one screen-to-light matrix per cascade in one buffer, and the caller only
    // rescans while it has no targets, so the rest were never revisited.
    // `reg` is in 16-byte registers; a hit consumes its four, so one 4x4 cannot
    // also be reported as three shifted copies of itself.
    unsigned DetectInverses(const unsigned char* data, size_t byteCount,
                            bool allowComposed, DetectedInverse* out, unsigned maxOut,
                            float& bestMiss) const
    {
        const size_t regCount = byteCount / 16;
        if (regCount < 4 || maxOut == 0) return 0;
        // A real layout keeps invVP near the front; cap the scan so one large
        // material buffer cannot make every write O(buffer size).
        const size_t lastReg = (regCount > 128 ? 128 : regCount) - 4;
        const float* fbuf = reinterpret_cast<const float*>(data);
        unsigned found = 0;
        for (size_t reg = 0; reg <= lastReg && found < maxOut; ++reg)
        {
            if (!AllFinite16(fbuf + reg * 4)) continue;
            for (int orient = 0; orient < 2; ++orient)
            {
                float cand[16];
                CanonicalizeMat4(fbuf + reg * 4, orient == 1, cand);
                InverseMatch match;
                if (!MatchInverse(cand, allowComposed, &match, &bestMiss)) continue;
                out[found].reg        = (unsigned)reg;
                out[found].transposed = (orient == 1);
                out[found].composed   = match.composed;
                ++found;
                reg += 3;      // consume this matrix; the loop's ++reg adds the 4th
                break;
            }
        }
        return found;
    }

    // Single-match convenience, kept for callers that only want the first.
    bool DetectInverse(const unsigned char* data, size_t byteCount,
                       unsigned& outReg, bool& outTransposed, float& bestMiss) const
    {
        DetectedInverse hit;
        if (DetectInverses(data, byteCount, false, &hit, 1, bestMiss) == 0) return false;
        outReg = hit.reg;
        outTransposed = hit.transposed;
        return true;
    }

    // Re-check a learned inverse each write before trusting it: pooled buffers
    // are reused for unrelated content, and the inverse patch skips every shape
    // guard, so a stale target smears the shear across arbitrary constants
    // (Far Cry 3: white/colorful flashes at ~20k patches per 60 frames). The
    // camera case re-passes trivially — the game rewrites both matrices from the
    // same camera each frame, and its VP re-enters the ring at geometry time
    // before the lighting CB is written.
    // The basis is re-derived, not remembered with the target: it is P^-1 for
    // the current camera and would be stale a frame later.
    bool VerifyInverseAgainstRing(const float* f, bool transposed,
                                  bool allowComposed = false,
                                  InverseMatch* out = nullptr,
                                  bool* outShapeReject = nullptr) const
    {
        if (!AllFinite16(f)) { if (outShapeReject) *outShapeReject = true; return false; }
        float cand[16];
        CanonicalizeMat4(f, transposed, cand);
        return MatchInverse(cand, allowComposed, out, nullptr, outShapeReject);
    }

    // Exact inverse of the forward shift x += b*w - b*conv. That shift is
    // P_r = P + w.e0^T with w = b*P[:,3] + m41*e3 (m41 = -b*conv), a rank-1
    // update of column 0, so by Sherman-Morrison the correction
    // K = P_r^-1 * P = I - q.e0^T / (1 + q0), q = b*e3 + m41*C[:,3], is a rank-1
    // update of the identity: subtract a multiple of row 0 from every row.
    // conv == 0 leaves only c3, the shear-only body this replaced -- dropping
    // the convergence term displaced every right-eye screen-space
    // reconstruction by StereoBase at every depth.
    // K multiplies on the LEFT, so the same rows correct a composed
    // screen-to-light P^-1 * L given that L's P^-1 in `basis`; `basis` defaults
    // to the matrix itself, right whenever the stored value IS the inverse VP.
    void PatchInverse(float* f, bool transposed, float eyeShift, float conv,
                      const float* basis = nullptr) const
    {
        const float b = EyeShiftB(eyeShift);
        float m[16];
        CanonicalizeMat4(f, transposed, m);

        // q = b*e3 + m41*C[:,3]; the 4th column of C is m[3], m[7], m[11], m[15].
        const float* C = basis ? basis : m;
        const float  m41   = -b * conv;
        const float  denom = 1.f + m41 * C[3];
        if (!(fabsf(denom) > 1e-4f)) return;   // also rejects a NaN denominator
        const float c0 = (m41 * C[3])       / denom;
        const float c1 = (m41 * C[7])       / denom;
        const float c2 = (m41 * C[11])      / denom;
        const float c3 = (m41 * C[15] + b)  / denom;

        const float r0[4] = { m[0], m[1], m[2], m[3] };
        for (int k = 0; k < 4; ++k)
        {
            m[k]      -= c0 * r0[k];
            m[4  + k] -= c1 * r0[k];
            m[8  + k] -= c2 * r0[k];
            m[12 + k] -= c3 * r0[k];
        }
        // A degenerate basis can still produce infinities; leave the game's
        // value alone rather than writing them into a live constant buffer.
        if (!AllFinite16(m)) return;
        CanonicalizeMat4(m, transposed, f);
    }
};

} // namespace wiz3d
