/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Numeric inverse view-projection detection: the forward view-projections the right eye shifts are cached, and blocks
// of constant data that invert one of them (C * V == I) are patched with the exact inverse of that shift.

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

// One product element against a cancellation-aware bound (relative to the sum of |term|), as a ratio: <= 1 passes,
// and the frame report shows the closest miss.
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
    // Twice the Frostbite bound: UE3 folds viewport terms into its screen-to-world (BioShock Infinite sat just
    // outside). True pairs score far below 1 and junk far above, so the margin is wide.
    return fabsf(acc - target) / (2e-4f * mag + 4e-3f);
}

// A NaN never raises the max-accumulated deviation (every comparison with it is false), so a NaN-filled block would
// pass (Far Cry 3): non-finite blocks are rejected.
inline bool AllFinite16(const float* f)
{
    for (int i = 0; i < 16; ++i)
    {
        // Bit test: an all-ones exponent is NaN or Inf whatever the float model makes of x != x.
        unsigned bits;
        memcpy(&bits, f + i, 4);
        if ((bits & 0x7F800000u) == 0x7F800000u) return false;
        if (fabsf(f[i]) > 1e30f) return false;
    }
    return true;
}

// Cheap gate before any ring work: an unprojection has zeroes at [0][3] and
// [1][3]. Relative to each row, so jittered/off-axis projections still pass.
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

    // Ring of pre-shift (mono) perspective matrices the forward path corrected, canonical row-vector layout. Admission
    // is immediate (BF3's camera VP is unique per frame); repetition drives eviction, age overrides it.
    struct CachedForwardVP
    {
        float              m[16];
        unsigned long long hash;
        unsigned           hits;       // identical re-sightings; eviction tiebreak
        unsigned           lastFrame;  // when last sighted; dominates eviction
        unsigned           serial;     // insert order, final tiebreak
        bool               valid;
    };
    // 64 deep: BF3 writes its deferred CB after a run of per-object matrices (LooksLikeUnprojection keeps the walk
    // cheap).
    static const unsigned kRing = 64;
    CachedForwardVP ring[kRing] = {};
    unsigned ringSerial = 0;
    unsigned ringMRU = 0;                // scanned first: most likely to match
    unsigned ringInsertsThisFrame = 0;   // churn indicator for the log
    unsigned evictedHotThisFrame = 0;    // evicted an entry that had re-sightings
    unsigned invStaleThisFrame = 0;      // learned inverses failing re-verify
    unsigned invStaleShapeThisFrame = 0; // ...of those, not unprojection-shaped

    // Advanced once per present; entries older than kMaxAge cannot match.
    unsigned frameId = 0;
    // In presents, and a game can present more than once per frame: at 1, entries expired mid-frame (GTA V).
    static const unsigned kMaxAge = 4;

    bool EntryUsable(const CachedForwardVP& e, unsigned maxAge) const
    {
        return e.valid && (frameId - e.lastFrame) <= maxAge;
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

    // Camera _11, the only per-matrix unknown of both stereo terms: xn/wn recovers it when |w basis| is unit, else the
    // last such matrix's is reused, so a non-uniformly scaled mesh cannot displace itself.
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
        const unsigned slots = kRing;
        for (unsigned i = 0; i < slots; ++i)
        {
            if (!ring[i].valid || ring[i].hash != h) continue;
            if (memcmp(ring[i].m, canon, sizeof(canon)) != 0) continue;   // hash collision
            ++ring[i].hits;
            ring[i].lastFrame = frameId;
            ringMRU = i;
            return;
        }

        // Free slot, then anything too old to match, then the oldest entry nothing has repeated: a flood of per-object
        // WVPs (hits == 1) consumes only its own slots.
        unsigned slot = 0;
        bool     found = false;
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

    // Ring slot to test at position `ri`: the most recently touched entry first (the camera VP in a frame still
    // writing geometry, so a real match usually lands on the first probe), then everything else.
    unsigned RingProbeOrder(unsigned ri) const
    {
        const unsigned slots = kRing;
        if (ri == 0) return ringMRU < slots ? ringMRU : 0;
        const unsigned idx = ri - 1;
        const unsigned mru = ringMRU < slots ? ringMRU : 0;
        return (idx < mru) ? idx : (idx + 1) % slots;   // skip the slot already probed
    }

    // A canonical candidate against the ring. outShapeReject tells the two ways a match fails apart (not
    // unprojection-shaped at all, or shaped but inverting no cached camera): they need opposite fixes.
    bool MatchInverse(const float* cand, float* bestMiss, bool* outShapeReject = nullptr) const
    {
        if (!LooksLikeUnprojection(cand))
        {
            if (outShapeReject) *outShapeReject = true;
            return false;
        }
        for (unsigned ri = 0; ri < kRing; ++ri)
        {
            const unsigned idx = RingProbeOrder(ri);
            if (!EntryUsable(ring[idx], kMaxAge)) continue;
            const float dev = InverseDeviation(cand, ring[idx].m);
            if (dev <= 1.f) return true;
            if (bestMiss && dev < *bestMiss) *bestMiss = dev;
        }
        return false;
    }

    struct DetectedInverse
    {
        unsigned reg;
        bool     transposed;
    };

    // Up to maxOut matches in a buffer's first 128 registers; `reg` is in 16-byte registers, and a hit consumes its
    // four.
    unsigned DetectInverses(const unsigned char* data, size_t byteCount,
                            DetectedInverse* out, unsigned maxOut, float& bestMiss) const
    {
        const size_t regCount = byteCount / 16;
        if (regCount < 4 || maxOut == 0) return 0;
        // A real layout keeps the inverse near the front, and one large material buffer must not cost O(size) a write.
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
                if (!MatchInverse(cand, &bestMiss)) continue;
                out[found].reg        = (unsigned)reg;
                out[found].transposed = (orient == 1);
                ++found;
                reg += 3;      // consume this matrix; the loop's ++reg adds the 4th
                break;
            }
        }
        return found;
    }

    // A learned inverse is re-checked at each write: pooled buffers are reused for unrelated content, and a stale
    // target smears the shear across arbitrary constants (Far Cry 3's flashes). The camera's re-passes trivially.
    bool VerifyInverseAgainstRing(const float* f, bool transposed, bool* outShapeReject = nullptr) const
    {
        if (!AllFinite16(f)) { if (outShapeReject) *outShapeReject = true; return false; }
        float cand[16];
        CanonicalizeMat4(f, transposed, cand);
        return MatchInverse(cand, nullptr, outShapeReject);
    }

    // Exact inverse of x += b*w - b*conv: P_r = P + w.e0^T is a rank-1 update of column 0, so by Sherman-Morrison the
    // correction K = P_r^-1 * P subtracts a multiple of row 0 from every row. conv == 0 leaves the shear only.
    void PatchInverse(float* f, bool transposed, float eyeShift, float conv) const
    {
        const float b = EyeShiftB(eyeShift);
        float m[16];
        CanonicalizeMat4(f, transposed, m);

        // q = b*e3 + m41*C[:,3]; the 4th column of C is m[3], m[7], m[11], m[15].
        const float* C = m;
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
