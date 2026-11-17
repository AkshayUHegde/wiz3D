/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The stereo patchers for fragment and compute SPIR-V (FsPatch.cpp), by what each corrects: pure functions of the
// module (false, `out` untouched, without the shape), each reading the stereo block at (stereoSet, binding 0).

#pragma once

#include <stdint.h>
#include <vector>

namespace wiz3d { namespace vk {

// --- A position rebuilt from depth, ray * depth, gets the right eye's offset ---

// With every rule off: eye*right added where the ray is rows.(ndc,1) from a buffer (rule 1) or a varying feeding a
// shadow compare (rule 2).
struct ReconstructionRules
{
    bool skewExact = false;    // rule 1 undoes the vertex shift exactly instead: cx * fsSep * (conv - max(depth, wMin))
    bool fragDepth = false;    // rule 1 also takes a ray scaled by the fragment's own depth, 1 / gl_FragCoord.w
    bool volumeRays = false;   // rule 3: v.xyz / v.w from volume geometry, where the position addresses a 3D texture
    bool volumeConv = false;   // ... whose ray is a full-screen pass's (w = 1): eye*right*(1 - depth/conv)
    bool volumeLights = false; // ... also where the position is subtracted from a buffer vector (a light's centre)
    bool screenSkew = false;   // rule 4: a ray*depth point projected back through a buffer matrix gets x += fsSep*w
    bool skewedRays = false;   // ... also without a rule 2 match: the caller knows the vertex shader hands skewed rays
};
struct ReconstructionCounts
{
    unsigned products = 0;     // positions patched, all rules
    unsigned volume = 0;       // ... of them, rule 3's
    unsigned screen = 0;       // rule 4's projections
};
bool PatchReconstructionFS(const uint32_t* code, size_t words, uint32_t stereoSet, const ReconstructionRules& rules,
                           std::vector<uint32_t>& out, ReconstructionCounts* counts = nullptr);
// (input rayLocation).xyz [/ w] * linear depth, a left-camera ray on a pinned full-screen pass:
// + normalize(ddx(ray)) * eye * (1 - max(z, wMin) / conv).
bool PatchScreenRayPositionFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t rayLocation, std::vector<uint32_t>& out, unsigned* products);

// --- A view ray is taken at the right eye ---

// A compute pass's view ray rows.(ndc, 1) read at ndc.x - sep (stereo set member 0, 0 in the left run): far clouds.
bool PatchSkyRaysCS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* rays);
// A compute pass rebuilding positions from such rays and projecting them through projBinding's members.
// counts: rays, positions, projections.
bool PatchEyeRaysCS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t projBinding, uint32_t projMembers,
                    std::vector<uint32_t>& out, unsigned* counts);
// normalize(ray varying / w) read at ndc.x - eye*P00/conv (P00 from projBinding's member): the right eye's view
// direction in a pinned lighting pass.
bool PatchViewRayFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t rayLocation, uint32_t uvLocation, uint32_t projBinding, uint32_t projMember,
                    std::vector<uint32_t>& out, unsigned* dirs);

// --- A projection through a left-camera matrix is given the right camera's ---

// A direction (w = 0) through projBinding's members, used as a screen offset.
bool PatchDirProjectionFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t projBinding, uint32_t projMembers,
                          std::vector<uint32_t>& out, unsigned* dirs);
// P * (position, 1) with P = block bufBinding's [0][0].member.field (a shared, left-camera view-projection):
// clip.x + fsSep*(w - conv).
bool PatchBufferProjectionFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t bufBinding, uint32_t member, uint32_t field,
                             std::vector<uint32_t>& out, unsigned* projections);

// --- Data built once, for the left eye, is read at the left eye's pixel ---

// A clip-position varying (input `location`) read as a screen lookup: x - fsSep*(w - conv), the left eye's pixel for
// that point.
bool PatchClipVaryingLeftFS(const uint32_t* code, size_t words, uint32_t stereoSet, uint32_t location, std::vector<uint32_t>& out, unsigned* loads);
// A 3D fog volume sampled at (screen uv, depth slice): x - fsSep/2 * (1 - conv/depth), the left eye's pixel for the
// surface point.
bool PatchFogVolumeLeftFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* lookups);
// A fade dither indexed by gl_FragCoord.xy (noise tile or 3x3 table), read at the left eye's pixel:
// x - round(fsSep*(1 - conv/w)*halfWidth).
bool PatchScreenDitherFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* sites);
// Per-tile lists built for the left eye: a FragCoord pixel divided into a tile index is read at the left's pixel,
// x - round(fsSep*(1 - conv*FragCoord.w)*halfWidth) on screen; vm*: VKViewModel's scales above clip z vmClipZ.
bool PatchTileListsLeftFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* sites,
                          float vmClipZ = 0.f, float vmSep = 1.f, float vmConv = 1.f);

// --- A point the left camera projects is taken to this eye's ---

// A screen-space march (contact shadows) from the pixel's uv toward a point projected through a uniform block's camera
// rows: the end's uv-clip x + fsSep/2*(max(w, wMin) - conv); vm* as PatchTileListsLeftFS's.
bool PatchScreenMarchFS(const uint32_t* code, size_t words, uint32_t stereoSet, std::vector<uint32_t>& out, unsigned* marches,
                        float vmClipZ = 0.f, float vmSep = 1.f, float vmConv = 1.f);

// --- Camera matrices a fragment shader loads as raw dwords take the right eye's shift (CameraWindows.cpp) ---

struct CameraWindowsRules
{
    bool counters = false;         // the right eye's fragments are counted per window (the camera counters)
    bool screenX = false;          // screen x goes to the left eye's before the matrices multiply it
    uint32_t screenXBase = ~0u;    // ... the camera inverse at this dword; ~0: none named
};
bool PatchCameraWindowsFS(const uint32_t* code, size_t words, uint32_t stereoSet, const CameraWindowsRules& rules,
                          std::vector<uint32_t>& out, unsigned* windows);

// --- Right-run camera for compute and screen passes (ComputeRemap.cpp): VKComputeRemap levels, family list rules ---

struct ComputeRemapRules
{
    bool wide = false;             // level 2: the ndc rule also takes FAdd(FMul(a, b), c) chains and colour-output FS
    bool uvRays = false;           // level 3: a screen-uv varying times the depth read at it, in shadow-sampling FS
    bool storeCamera = false;      // the shader stores the camera inverse's w row for later readers
    bool counters = false;         // the camera counters are on
    bool tiledLights = false;      // tiled lighting: integer pixel to world, depth through shared memory
    bool rayRows = false;          // screen passes rebuilding from depth by the per-view inverse
    bool softRead = false;         // depth read at a varying built before the shift (soft particles)
    bool viewRows = false;         // view-vector rows of fragment shaders with no ndc reconstruction
    bool flareX = false;           // glows around points from a table filled from the left camera
    bool uvAffine = false;         // level 4: positions rebuilt from varyings alone, no FragCoord
    bool froxelLookups = false;    // FS froxel-volume lookups read at the right eye's uv
    bool fragRays = false;         // the uv-ray rule with the pixel's own uv
    bool perspInputs = false;      // (1/FragCoord.w) * (v * FragCoord.w) varyings: twins for unit-w pipelines only
};
// What it rewrote: ndc reconstructions, matrix rows, froxel rays, 3D reprojections, view vectors.
struct ComputeRemapCounts { unsigned ndc = 0, rows = 0, rays = 0, reproj = 0, eye = 0; };
bool PatchComputeRemap(const uint32_t* code, size_t words, uint32_t stereoSet, const ComputeRemapRules& rules,
                       std::vector<uint32_t>& out, ComputeRemapCounts& counts);

}} // namespace wiz3d::vk
