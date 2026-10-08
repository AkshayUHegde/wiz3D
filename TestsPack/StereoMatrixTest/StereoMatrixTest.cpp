/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// Offline proof of Shared/StereoMatrix.h: PatchInverse turns P^-1 into P_r^-1, checked against a numeric inverse and
// by a round trip through the right-eye matrix. Build and run: TestsPack\StereoMatrixTest\run.bat

#include <stdio.h>
#include <math.h>
#include <string.h>

#include "../../Shared/StereoMatrix.h"

using wiz3d::StereoMatrixState;

static int g_failures = 0;
static int g_checks   = 0;

// Reference linear algebra, row-vector throughout: a point is a row and transforms compose left to right, so m[r*4+c]
// has the w-producing column at c == 3 (StereoMatrix.h's canonical layout).
static void Mul(const float* a, const float* b, float* out)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            float acc = 0.f;
            for (int k = 0; k < 4; ++k) acc += a[r * 4 + k] * b[k * 4 + c];
            out[r * 4 + c] = acc;
        }
}

static void MulPoint(const float* p, const float* m, float* out)
{
    for (int c = 0; c < 4; ++c)
    {
        float acc = 0.f;
        for (int r = 0; r < 4; ++r) acc += p[r] * m[r * 4 + c];
        out[c] = acc;
    }
}

// Gauss-Jordan with partial pivoting. Independent of the rank-1 shortcut under
// test, which is the point: the test must not reuse the code it is checking.
static bool Invert(const float* m, float* out)
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
        if (piv != col) for (int c = 0; c < 8; ++c) { double t = a[col][c]; a[col][c] = a[piv][c]; a[piv][c] = t; }
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
    return true;
}

static void Transpose(const float* m, float* out)
{
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c) out[c * 4 + r] = m[r * 4 + c];
}

// Projections, row-vector D3D style: the w-producing column is index 3.
static void PerspectiveLH(float xScale, float yScale, float zn, float zf, float* m)
{
    memset(m, 0, 16 * sizeof(float));
    m[0]  = xScale;                 // _11
    m[5]  = yScale;                 // _22
    m[10] = zf / (zf - zn);         // _33
    m[11] = 1.f;                    // _34
    m[14] = -zn * zf / (zf - zn);   // _43
}

// Reversed-Z with an infinite far plane (depth 1 at the near plane, 0 at infinity), as GTA V's: the case where the
// correction's row-2 coefficient gets large.
static void PerspectiveReversedInfinite(float xScale, float yScale, float zn, float* m)
{
    memset(m, 0, 16 * sizeof(float));
    m[0]  = xScale;
    m[5]  = yScale;
    m[10] = 0.f;
    m[11] = 1.f;
    m[14] = zn;
}

// Orthographic light view-projection as an engine builds one: a rigid light view times a diagonal ortho projection
// (a texture-space bias folded in is diagonal too). `scale` distinguishes cascades.
static void OrthoLight(float scale, float* m)
{
    const float c = 0.8f, s = 0.6f;   // light view: rotation about Y
    const float V[16] = {
          c, 0.f,   s, 0.f,
        0.f, 1.f, 0.f, 0.f,
         -s, 0.f,   c, 0.f,
       -14.f, 22.f, 9.f, 1.f,
    };
    // Ortho projection + bias: strictly diagonal plus a translation row.
    float O[16];
    memset(O, 0, sizeof O);
    O[0]  = scale;  O[5] = scale * 1.3f;  O[10] = scale * 0.4f;
    O[12] = 0.5f;   O[13] = 0.5f;         O[14] = 0.f;  O[15] = 1.f;
    Mul(V, O, m);
}

// The forward shift under test, as Context11Proxy.cpp's full-column (draw duplication) branch applies it: column 0
// += b * column 3, then _41 += m41.
static void ForwardShiftRightEye(float* f, float b, float m41)
{
    f[0] += b * f[3];   f[4]  += b * f[7];
    f[8] += b * f[11];  f[12] += b * f[15];
    f[12] += m41;
}

static void Fail(const char* what, const char* detail)
{
    ++g_failures;
    printf("  FAIL  %s: %s\n", what, detail);
}

static void CheckClose(const char* what, const float* got, const float* want, float tol)
{
    ++g_checks;
    float worst = 0.f;
    int   wr = 0, wc = 0;
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
        {
            const float scale = fabsf(want[r * 4 + c]) + 1.f;
            const float d = fabsf(got[r * 4 + c] - want[r * 4 + c]) / scale;
            if (d > worst) { worst = d; wr = r; wc = c; }
        }
    if (worst > tol)
    {
        char buf[256];
        snprintf(buf, sizeof buf, "worst relative delta %.3e at [%d][%d] (got %.6f want %.6f)",
                 worst, wr, wc, got[wr * 4 + wc], want[wr * 4 + wc]);
        Fail(what, buf);
    }
}

static void CheckBitIdentical(const char* what, const float* got, const float* want)
{
    ++g_checks;
    if (memcmp(got, want, 16 * sizeof(float)) != 0)
        Fail(what, "not bit-identical to the shear-only result");
}

// Tests

// The claim: patching P^-1 yields the exact inverse of the right-eye forward
// matrix. Everything else follows from this.
static void TestExactInverse(const char* name, const float* P, float eyeShift, float conv)
{
    StereoMatrixState st;
    st.projXScale = P[0];                     // as cached from a clean view-projection
    const float b   = st.EyeShiftB(eyeShift);
    const float m41 = -b * conv;

    float Pr[16];
    memcpy(Pr, P, sizeof Pr);
    ForwardShiftRightEye(Pr, b, m41);

    float PrInvRef[16];
    if (!Invert(Pr, PrInvRef)) { Fail(name, "reference inverse of P_r is singular"); return; }

    float patched[16];
    if (!Invert(P, patched)) { Fail(name, "reference inverse of P is singular"); return; }
    st.PatchInverse(patched, /*transposed*/false, eyeShift, conv);

    CheckClose(name, patched, PrInvRef, 2e-4f);
}

// Stronger and convention-independent, as a deferred pass does it: world points through the right-eye matrix, the
// pixel-space vector (ndc.xy, ndc.z, 1) through the patched inverse, and the original point back.
static void TestRoundTrip(const char* name, const float* P, float eyeShift, float conv)
{
    StereoMatrixState st;
    st.projXScale = P[0];
    const float b   = st.EyeShiftB(eyeShift);
    const float m41 = -b * conv;

    float Pr[16];
    memcpy(Pr, P, sizeof Pr);
    ForwardShiftRightEye(Pr, b, m41);

    float refInv[16];
    if (!Invert(Pr, refInv)) { Fail(name, "reference inverse of P_r is singular"); return; }

    float inv[16];
    if (!Invert(P, inv)) { Fail(name, "reference inverse of P is singular"); return; }
    st.PatchInverse(inv, false, eyeShift, conv);

    // Spread across the depth range: near, at convergence, and far, off-centre
    // so an x-only error cannot hide.
    const float pts[][4] = {
        {  1.5f,  0.9f,   2.0f, 1.f },
        { -3.0f,  1.2f,  10.0f, 1.f },
        {  8.0f, -4.0f,  60.0f, 1.f },
        { -25.f,  6.0f, 400.0f, 1.f },
    };
    for (int i = 0; i < (int)(sizeof pts / sizeof pts[0]); ++i)
    {
        ++g_checks;
        float clip[4];
        MulPoint(pts[i], Pr, clip);
        if (fabsf(clip[3]) < 1e-6f) { Fail(name, "degenerate w"); continue; }

        // What the pixel shader has: NDC xy/z from the raster + depth, w = 1.
        const float v[4] = { clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1.f };
        float world[4];
        MulPoint(v, inv, world);
        if (fabsf(world[3]) < 1e-9f) { Fail(name, "unprojection produced w = 0"); continue; }

        const float rx = world[0] / world[3];
        const float ry = world[1] / world[3];
        const float rz = world[2] / world[3];

        // Assertion 1 (the property under test): the patched matrix agrees with the exact inverse of P_r; both see the
        // same quantised depth, so any disagreement is algebra, not precision.
        float refWorld[4];
        MulPoint(v, refInv, refWorld);
        const float ex = refWorld[0] / refWorld[3];
        const float ey = refWorld[1] / refWorld[3];
        const float ez = refWorld[2] / refWorld[3];
        const float scale = fabsf(pts[i][0]) + fabsf(pts[i][1]) + fabsf(pts[i][2]) + 1.f;
        const float dRef = (fabsf(rx - ex) + fabsf(ry - ey) + fabsf(rz - ez)) / scale;
        if (dRef > 1e-5f)
        {
            char buf[256];
            snprintf(buf, sizeof buf,
                     "point %d disagrees with the exact P_r inverse: got (%.4f %.4f %.4f)"
                     " vs (%.4f %.4f %.4f), rel %.3e", i, rx, ry, rz, ex, ey, ez, dRef);
            Fail(name, buf);
        }

        // Assertion 2 (sanity): the round trip recovers the original point, loosely: a float32 non-reversed depth
        // loses this much at the far end, where z_ndc is near 1 (the reversed-Z case holds far tighter).
        const float err = fabsf(rx - pts[i][0]) + fabsf(ry - pts[i][1]) + fabsf(rz - pts[i][2]);
        if (err / scale > 1e-3f)
        {
            char buf[256];
            snprintf(buf, sizeof buf,
                     "point %d: got (%.4f %.4f %.4f) want (%.4f %.4f %.4f), rel err %.3e",
                     i, rx, ry, rz, pts[i][0], pts[i][1], pts[i][2], err / scale);
            Fail(name, buf);
        }
    }
}

// conv == 0 must give exactly the shear-only inverse, which the replay path's m[2][0] shift needs.
static void TestLegacyEquivalence(const float* P, float eyeShift)
{
    StereoMatrixState st;
    st.projXScale = P[0];
    const float s = st.EyeShiftB(eyeShift);

    float inv[16];
    Invert(P, inv);

    float legacy[16];
    memcpy(legacy, inv, sizeof legacy);
    legacy[12] -= s * legacy[0];  legacy[13] -= s * legacy[1];
    legacy[14] -= s * legacy[2];  legacy[15] -= s * legacy[3];

    float got[16];
    memcpy(got, inv, sizeof got);
    st.PatchInverse(got, false, eyeShift, /*conv*/0.f);
    CheckBitIdentical("legacy equivalence (conv = 0, non-transposed)", got, legacy);

    // Transposed storage must reach the same matrix by the same route.
    float invT[16], legacyT[16], gotT[16];
    Transpose(inv, invT);
    Transpose(legacy, legacyT);
    memcpy(gotT, invT, sizeof gotT);
    st.PatchInverse(gotT, true, eyeShift, 0.f);
    CheckBitIdentical("legacy equivalence (conv = 0, transposed)", gotT, legacyT);
}

// Transposed storage, full arithmetic.
static void TestTransposed(const char* name, const float* P, float eyeShift, float conv)
{
    StereoMatrixState st;
    st.projXScale = P[0];
    const float b   = st.EyeShiftB(eyeShift);
    const float m41 = -b * conv;

    float Pr[16], PrInvRef[16], inv[16];
    memcpy(Pr, P, sizeof Pr);
    ForwardShiftRightEye(Pr, b, m41);
    if (!Invert(Pr, PrInvRef) || !Invert(P, inv)) { Fail(name, "singular"); return; }

    float stored[16];
    Transpose(inv, stored);
    st.PatchInverse(stored, /*transposed*/true, eyeShift, conv);

    float got[16];
    Transpose(stored, got);   // back to canonical for comparison
    CheckClose(name, got, PrInvRef, 2e-4f);
}

// A basis that makes the denominator vanish must leave the game's constants
// untouched rather than writing infinities into a live buffer.
static void TestDegenerateGuard()
{
    ++g_checks;
    StereoMatrixState st;
    st.projXScale = 1.f;

    float m[16];
    memset(m, 0, sizeof m);
    for (int i = 0; i < 4; ++i) m[i * 4 + i] = 1.f;
    // 1 + m41 * C[0][3] == 0  =>  C[0][3] = -1 / m41.
    const float eyeShift = 0.016f, conv = 10.f;
    const float b = st.EyeShiftB(eyeShift), m41 = -b * conv;
    m[3] = -1.f / m41;

    float before[16];
    memcpy(before, m, sizeof before);
    st.PatchInverse(m, false, eyeShift, conv);
    if (memcmp(before, m, sizeof before) != 0)
        Fail("degenerate denominator guard", "matrix was modified");
}

// Rigid world-view-projection, as an engine writes per object: it passes the unit-w / orthogonal-bases test that
// gates ring admission.
static void MakeRigidWVP(const float* P, float angle, float tx, float tz, float* out)
{
    const float c = cosf(angle), s = sinf(angle);
    float W[16] = {
          c, 0.f,   s, 0.f,
        0.f, 1.f, 0.f, 0.f,
         -s, 0.f,   c, 0.f,
         tx, 0.f,  tz, 1.f,
    };
    Mul(W, P, out);
}

// One camera view-projection reused across the frame among a flood of unique per-object WVPs (GTA V): the VP must
// still be in the ring at the end.
static void TestRingAdmission(const float* P)
{
    StereoMatrixState st;

    // Admission is immediate: a moving camera never writes the same matrix twice (Battlefield 3), so requiring a
    // second sighting would admit nothing.
    ++g_checks;
    st.CacheForwardVP(P, false);
    if (st.RingValid() != 1) Fail("ring admission", "a fresh view-projection was not admitted");

    // Now the flood. 1000 unique rigid WVPs, each written once, with the camera
    // VP rewritten between them as a real frame does.
    for (int i = 0; i < 1000; ++i)
    {
        float wvp[16];
        MakeRigidWVP(P, 0.013f * i, 3.f * i, -1.7f * i, wvp);
        st.CacheForwardVP(wvp, false);
        st.CacheForwardVP(P, false);
    }

    ++g_checks;
    float inv[16];
    Invert(P, inv);
    if (!st.VerifyInverseAgainstRing(inv, false))
        Fail("ring admission", "the camera VP was evicted by single-sighting WVPs");

    // End to end: an inverse sitting in a constant buffer must still be found.
    ++g_checks;
    float cb[4 * 16];
    memset(cb, 0, sizeof cb);
    for (int i = 0; i < 16; ++i) cb[2 * 4 + i] = inv[i];   // at register 2
    StereoMatrixState::DetectedInverse hit = {};
    float bestMiss = 1e30f;
    if (st.DetectInverses(reinterpret_cast<const unsigned char*>(cb), sizeof cb, &hit, 1, bestMiss) != 1 ||
        hit.reg != 2 || hit.transposed)
    {
        char buf[128];
        snprintf(buf, sizeof buf, "detect failed (reg %u, transposed %d, bestMiss %.2f)",
                 hit.reg, (int)hit.transposed, bestMiss);
        Fail("ring admission", buf);
    }
}

// A repeated WVP in the ring must not match the inverse of a different matrix: the pairing stays exact, or an
// unrelated constant gets patched.
static void TestNoCrossMatch(const float* P)
{
    ++g_checks;
    StereoMatrixState st;
    float wvp[16];
    MakeRigidWVP(P, 0.4f, 12.f, -30.f, wvp);
    st.CacheForwardVP(wvp, false);
    st.CacheForwardVP(wvp, false);      // admitted: it repeats

    float invP[16];
    Invert(P, invP);                    // the inverse of a DIFFERENT matrix
    if (st.VerifyInverseAgainstRing(invP, false))
        Fail("no cross match", "P^-1 matched a rigid WVP in the ring");
}

// A cascade-resolve buffer: the inverse view-projection plus one screen-to-light matrix per cascade. Only the bare
// inverse matches; a screen-to-light matrix inverts no cached camera.
static void TestDetection(const float* P)
{
    StereoMatrixState st;
    st.projXScale = P[0];
    st.CacheForwardVP(P, false);

    float PInv[16];
    Invert(P, PInv);

    // Register 0: the bare inverse. Registers 4, 8, 12, 16: four cascades.
    float cb[20 * 4];
    memset(cb, 0, sizeof cb);
    memcpy(cb, PInv, sizeof PInv);
    float L[4][16], X[4][16];
    for (int i = 0; i < 4; ++i)
    {
        OrthoLight(0.02f / (1 << i), L[i]);
        Mul(PInv, L[i], X[i]);
        memcpy(cb + (4 + 4 * i) * 4, X[i], sizeof X[i]);
    }

    ++g_checks;
    wiz3d::StereoMatrixState::DetectedInverse hits[8];
    float bestMiss = 1e30f;
    const unsigned n = st.DetectInverses(reinterpret_cast<const unsigned char*>(cb), sizeof cb, hits, 8, bestMiss);
    if (n != 1 || hits[0].reg != 0)
    {
        char buf[128];
        snprintf(buf, sizeof buf, "%u hits, first at reg %u (want the bare inverse at reg 0 only)", n,
                 n ? hits[0].reg : 0u);
        Fail("detection", buf);
    }
}

// The pre-filter is a performance gate: it must never reject a matrix the detector would match, off-axis and
// jittered projections included.
static void TestUnprojectionPrefilter()
{
    struct { const char* name; float m[16]; } cases[5];
    int n = 0;

    PerspectiveLH(1.2071f, 2.1445f, 0.1f, 1000.f, cases[n].m); cases[n].name = "standard";      ++n;
    PerspectiveReversedInfinite(1.2071f, 2.1445f, 0.15f, cases[n].m); cases[n].name = "reversed-Z"; ++n;
    PerspectiveLH(0.6f, 1.0667f, 0.05f, 12000.f, cases[n].m); cases[n].name = "wide FOV";       ++n;

    // TAA jitter: a sub-pixel offset in the x/y shear terms.
    PerspectiveLH(1.2071f, 2.1445f, 0.1f, 1000.f, cases[n].m);
    cases[n].m[8] += 0.0009f; cases[n].m[9] -= 0.0016f;
    cases[n].name = "TAA-jittered"; ++n;

    // Off-axis: our own right-eye shift, which is a much larger shear.
    PerspectiveLH(1.2071f, 2.1445f, 0.1f, 1000.f, cases[n].m);
    ForwardShiftRightEye(cases[n].m, 0.019f, -0.19f);
    cases[n].name = "off-axis (right eye)"; ++n;

    for (int i = 0; i < n; ++i)
    {
        ++g_checks;
        float inv[16];
        if (!Invert(cases[i].m, inv)) { Fail("prefilter", "singular"); continue; }
        if (!wiz3d::LooksLikeUnprojection(inv))
            Fail("prefilter rejected a real unprojection", cases[i].name);

        // Transposed storage is canonicalised before the filter sees it, so the canonical form must pass.
        ++g_checks;
        float invT[16], canon[16];
        Transpose(inv, invT);
        wiz3d::CanonicalizeMat4(invT, true, canon);
        if (!wiz3d::LooksLikeUnprojection(canon))
            Fail("prefilter rejected a canonicalised transposed unprojection", cases[i].name);
    }

    // Rejection rate on random blocks: the perf claim, as the scan cost drops only if nearly all of them are rejected.
    ++g_checks;
    unsigned kept = 0;
    const unsigned trials = 20000;
    unsigned seed = 12345;
    for (unsigned t = 0; t < trials; ++t)
    {
        float blk[16];
        for (int i = 0; i < 16; ++i)
        {
            seed = seed * 1664525u + 1013904223u;
            blk[i] = ((seed >> 8) & 0xFFFF) / 32768.f - 1.f;
        }
        if (wiz3d::LooksLikeUnprojection(blk)) ++kept;
    }
    printf("  prefilter keeps %u of %u random blocks (%.2f%%)\n",
           kept, trials, 100.f * kept / trials);
    if (kept > trials / 100)
        Fail("prefilter", "keeps more than 1% of random blocks; scan cost will not drop");
}

// A moving camera writes a different view-projection once per frame (Battlefield 3): the frame's inverse must verify
// in that frame, or some writes are patched and others not (geometry flashing between two positions).
static void TestMovingCamera()
{
    StereoMatrixState st;
    float P[16], inv[16];

    for (unsigned frame = 1; frame <= 400; ++frame)
    {
        st.frameId = frame;
        // A camera that translates and rotates a little every frame.
        float V[16];
        PerspectiveLH(1.2071f, 2.1445f, 0.1f, 1000.f, P);
        const float a = 0.004f * frame;
        const float c = cosf(a), s = sinf(a);
        const float W[16] = {
              c, 0.f,   s, 0.f,
            0.f, 1.f, 0.f, 0.f,
             -s, 0.f,   c, 0.f,
            0.9f * frame, 0.2f * frame, -0.5f * frame, 1.f,
        };
        Mul(W, P, V);
        st.CacheForwardVP(V, false);      // written ONCE per frame

        // A handful of per-object WVPs after it, as a real frame has.
        for (int i = 0; i < 40; ++i)
        {
            float wvp[16];
            MakeRigidWVP(V, 0.11f * i + a, 2.f * i, -3.f * i, wvp);
            st.CacheForwardVP(wvp, false);
        }

        // The deferred pass writes its inverse late in the same frame.
        ++g_checks;
        Invert(V, inv);
        if (!st.VerifyInverseAgainstRing(inv, false))
        {
            char buf[128];
            snprintf(buf, sizeof buf, "frame %u: this frame's inverse did not verify", frame);
            Fail("moving camera", buf);
            break;
        }
    }

    // And a view-projection from an old camera must NOT keep confirming targets,
    // or a stale buffer verifies against a camera that no longer exists.
    ++g_checks;
    float oldV[16], oldInv[16];
    st.frameId = 1000;
    PerspectiveLH(1.2071f, 2.1445f, 0.1f, 1000.f, oldV);
    oldV[12] = 123.f;   // a distinctive camera
    st.CacheForwardVP(oldV, false);
    Invert(oldV, oldInv);
    if (!st.VerifyInverseAgainstRing(oldInv, false))
        Fail("moving camera", "the just-written camera failed to verify");
    // Past the ageing window (which counts PRESENTS, and is wide enough for a
    // game that presents twice per frame), never re-written.
    st.frameId = 1000 + StereoMatrixState::kMaxAge + 1;
    ++g_checks;
    if (st.VerifyInverseAgainstRing(oldInv, false))
        Fail("moving camera", "a stale camera still confirms targets");
}

// Matrices from a Battlefield 3 session, in register order with the storage order the detector reported: the
// pre-filter must accept every one.
static void TestRealBF3Matrices()
{
    struct Case { const char* name; bool transposed; float m[16]; };
    const Case cases[] = {
        { "BF3 reg=1 transposed=1", true, {
             0.9255f,  0.0000f, -0.0000f,   0.0000f,
             0.0000f,  0.5206f, -0.0000f,   0.0000f,
             0.0000f,  0.0000f, -0.0000f,  -1.0000f,
             0.0000f,  0.0000f, -16.6666f, 16.6667f } },
        { "BF3 reg=1 transposed=0", false, {
            -0.0114f, -0.0014f,  0.4143f,   0.0000f,
            -0.0251f,  0.2203f,  0.0001f,   0.0000f,
             0.0000f,  0.0000f,  0.0000f, -16.6666f,
             0.9932f,  0.1130f,  0.0276f,  16.6667f } },
        { "BF3 reg=0 transposed=1 (screen-to-world)", true, {
            -0.9254f,  0.0000f, 31738.1328f, -31776.2188f,
             0.0000f,  0.5206f, -1196.3318f,   1197.7672f,
            -0.0046f,  0.0000f, 22344.6934f, -22370.5039f,
             0.0000f,  0.0000f,    -16.6467f,     16.6667f } },
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; ++i)
    {
        ++g_checks;
        float canon[16];
        wiz3d::CanonicalizeMat4(cases[i].m, cases[i].transposed, canon);
        if (!wiz3d::LooksLikeUnprojection(canon))
            Fail("prefilter rejected a matrix BF3 really uses", cases[i].name);
    }
}

int main()
{
    // Representative of the shipped model: StereoBase 0.16, One_div_ZPS 0.1, so
    // eyeShift = 0.016 and conv = 10 world units.
    const float eyeShift = 0.16f * 0.1f;
    const float conv     = 10.f;

    float std60[16], revZ[16], wide[16];
    PerspectiveLH(1.2071f, 2.1445f, 0.1f, 1000.f, std60);   // ~60 deg vertical, 16:9
    PerspectiveReversedInfinite(1.2071f, 2.1445f, 0.15f, revZ);
    PerspectiveLH(0.6f, 1.0667f, 0.05f, 12000.f, wide);     // wide FOV, huge far plane

    printf("StereoMatrix inverse-correction tests\n");

    TestExactInverse("exact inverse, standard projection",      std60, eyeShift, conv);
    TestExactInverse("exact inverse, reversed-Z infinite far",  revZ,  eyeShift, conv);
    TestExactInverse("exact inverse, wide FOV",                 wide,  eyeShift, conv);
    TestExactInverse("exact inverse, tight convergence",        std60, eyeShift, 2.f);
    TestExactInverse("exact inverse, large separation",         std60, 0.08f,    conv);

    TestRoundTrip("round trip, standard projection",     std60, eyeShift, conv);
    TestRoundTrip("round trip, reversed-Z infinite far", revZ,  eyeShift, conv);
    TestRoundTrip("round trip, wide FOV",                wide,  eyeShift, conv);

    TestTransposed("transposed storage, standard",     std60, eyeShift, conv);
    TestTransposed("transposed storage, reversed-Z",   revZ,  eyeShift, conv);

    TestLegacyEquivalence(std60, eyeShift);
    TestDegenerateGuard();

    TestRingAdmission(std60);
    TestMovingCamera();
    TestRealBF3Matrices();
    TestNoCrossMatch(std60);
    TestDetection(std60);
    TestDetection(revZ);
    TestUnprojectionPrefilter();

    // How far the shear-only inverse lands from the full model's point, for the record.
    {
        StereoMatrixState st;
        st.projXScale = std60[0];
        float inv[16];
        Invert(std60, inv);
        float legacy[16];
        memcpy(legacy, inv, sizeof legacy);
        st.PatchInverse(legacy, false, eyeShift, 0.f);

        const float pt[4] = { 8.f, -4.f, 60.f, 1.f };
        const float b = st.EyeShiftB(eyeShift), m41 = -b * conv;
        float Pr[16];
        memcpy(Pr, std60, sizeof Pr);
        ForwardShiftRightEye(Pr, b, m41);
        float clip[4];
        MulPoint(pt, Pr, clip);
        const float v[4] = { clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3], 1.f };
        float w[4];
        MulPoint(v, legacy, w);
        printf("  shear-only reconstruction error at z=60: %.4f world units"
               " (StereoBase = %.4f)\n",
               fabsf(w[0] / w[3] - pt[0]), eyeShift * conv);
    }

    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
