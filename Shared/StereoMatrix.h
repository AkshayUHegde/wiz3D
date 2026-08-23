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

struct StereoMatrixState
{
    // Camera projection _11, cached from the last clean view-projection seen.
    float projXScale = 0.f;

    // Ring of pre-shift (mono) perspective matrices the forward path corrected,
    // canonical row-vector layout; DetectInverse tests candidates against it.
    struct CachedForwardVP { float m[16]; bool valid; };
    CachedForwardVP ring[32] = {};
    unsigned ringNext = 0;
    unsigned ringInsertsThisFrame = 0;   // churn indicator for the log
    unsigned invStaleThisFrame = 0;      // learned inverses failing re-verify

    static const unsigned kRing = 32;

    unsigned RingValid() const
    {
        unsigned n = 0;
        for (unsigned i = 0; i < kRing; ++i) if (ring[i].valid) ++n;
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
        for (unsigned i = 0; i < kRing; ++i)
            if (ring[i].valid && memcmp(ring[i].m, canon, sizeof(canon)) == 0)
                return;
        memcpy(ring[ringNext].m, canon, sizeof(canon));
        ring[ringNext].valid = true;
        ringNext = (ringNext + 1) % kRing;
        ++ringInsertsThisFrame;
    }

    // Scan the buffer's 4x4 blocks (both storage orders) against the cached
    // forward matrices, newest first. bestMiss tracks the closest failing
    // candidate so a run of zero hits names its margin in the frame log.
    // outReg is in 16-byte registers from `data`.
    bool DetectInverse(const unsigned char* data, size_t byteCount,
                       unsigned& outReg, bool& outTransposed, float& bestMiss) const
    {
        const size_t regCount = byteCount / 16;
        if (regCount < 4) return false;
        // A real layout keeps invVP near the front; cap the scan so one large
        // material buffer cannot make every write O(buffer size).
        const size_t lastReg = (regCount > 128 ? 128 : regCount) - 4;
        const float* fbuf = reinterpret_cast<const float*>(data);
        for (size_t reg = 0; reg <= lastReg; ++reg)
        {
            if (!AllFinite16(fbuf + reg * 4)) continue;
            for (int orient = 0; orient < 2; ++orient)
            {
                float cand[16];
                CanonicalizeMat4(fbuf + reg * 4, orient == 1, cand);
                for (unsigned ri = 0; ri < kRing; ++ri)
                {
                    const unsigned idx = (ringNext + kRing - 1 - ri) % kRing;
                    if (!ring[idx].valid) continue;
                    const float dev = InverseDeviation(cand, ring[idx].m);
                    if (dev <= 1.f)
                    {
                        outReg = (unsigned)reg;
                        outTransposed = (orient == 1);
                        return true;
                    }
                    if (dev < bestMiss) bestMiss = dev;
                }
            }
        }
        return false;
    }

    // Re-check a learned inverse each write before trusting it: pooled buffers
    // are reused for unrelated content, and the inverse patch skips every shape
    // guard, so a stale target smears the shear across arbitrary constants
    // (Far Cry 3: white/colorful flashes at ~20k patches per 60 frames). The
    // camera case re-passes trivially — the game rewrites both matrices from the
    // same camera each frame, and its VP re-enters the ring at geometry time
    // before the lighting CB is written.
    bool VerifyInverseAgainstRing(const float* f, bool transposed) const
    {
        if (!AllFinite16(f)) return false;
        float cand[16];
        CanonicalizeMat4(f, transposed, cand);
        for (unsigned ri = 0; ri < kRing; ++ri)
        {
            const unsigned idx = (ringNext + kRing - 1 - ri) % kRing;
            if (!ring[idx].valid) continue;
            if (InverseDeviation(cand, ring[idx].m) <= 1.f) return true;
        }
        return false;
    }

    // Inverse (unprojection) matrices. The forward path is P' = P*T with
    // T = I except T._41 = b, so the corrected inverse is T^-1 * P^-1.
    void PatchInverse(float* f, bool transposed, float eyeShift) const
    {
        const float s = EyeShiftB(eyeShift);
        if (transposed)
        {
            // stored = Q^T, so the update is column 4 -= s * column 1.
            f[3] -= s * f[0];   f[7]  -= s * f[4];
            f[11] -= s * f[8];  f[15] -= s * f[12];
        }
        else
        {
            // T^-1 touches row 4 only: row4 -= s * row1.
            f[12] -= s * f[0];  f[13] -= s * f[1];
            f[14] -= s * f[2];  f[15] -= s * f[3];
        }
    }
};

} // namespace wiz3d
