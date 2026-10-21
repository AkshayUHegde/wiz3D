// -----------------------------------------------------------------------------
// wiz3D stereo-params texture for DX11 (0.4.1)
//
// A 1x1 R32G32B32A32_FLOAT texture bound to SRV slot t125 on every stage, with
// per-eye bytes updated at Context11Proxy::BindEye:
//     r = separation (effective eye-shift magnitude, from gInfo.Input)
//     g = convergence
//     b = eye_sign  (-1.0 on left, +1.0 on right)
//     a = reserved
//
// Replacement shaders dropped in via inline <ShaderOverride> can
// `sample`/`Load` the texture at UV 0.0 to apply their own eye-shift math.
// This matches HelixMod / 3DMigoto's stereo-params convention for DX11 games
// so authors coming from those ecosystems can port fixes with minimal rework.
// (Our DX9 shim at S3DWrapper9/ShaderOverrideLoader bound at s0 / s13 which
// is the DX9 variant of the same convention.)
//
// Lifetime: per-device. Device11Proxy calls Ensure() lazily on first eye
// switch, Release() in its destructor. The texture lives in a static map
// keyed by ID3D11Device* under a mutex. Immediate + deferred contexts share
// the same texture — DYNAMIC + MAP_DISCARD lets each eye's write rename
// the underlying storage so there's no cross-eye stomp.
//
// Zero cost when no override is active: EnsureForDevice is called every
// eye switch but after first-call it's a hashmap hit + no work. The SRV
// is bound even without an active override — one PSSetShaderResources
// call per stage per eye, negligible. If this ever matters we can gate on
// "any override exists for this game" at profile load.
// -----------------------------------------------------------------------------
#pragma once

#include <d3d11.h>

namespace wiz3d
{

// Reserved SRV slot. HelixMod / 3DMigoto DX11 use t125 for stereo params;
// matching that convention lets ported shaders work with minimal edits.
constexpr UINT kStereoParamsSlot = 125;

// Lazily create the per-device texture + SRV. Idempotent; a second call for
// the same device returns the cached pair. Returns false when creation fails
// (very unusual — a tiny DYNAMIC texture on a working device).
bool EnsureStereoParams(ID3D11Device* dev);

// Split into two operations so Context11Proxy can route the Map through the
// IMMEDIATE context (never the right-eye recorder) while letting the SRV
// bindings go through whichever context the current draw uses. Rolling the
// Map through the recorder would land on RightRecorder11::Map, which only
// accepts DYNAMIC BUFFERS — a texture would be rejected via `Unexpected`
// and permanently stop batching for the frame. See 0.4.1 release notes.

// Update the per-device texture's bytes. MUST be called on the immediate
// context — passing a deferred/recording context will fail. The write is
// idempotent for the same (eye, sep, conv) so repeated calls in a frame
// are cheap. Safe to call before EnsureStereoParams — no-op in that case.
void UpdateStereoParamsBytes(ID3D11DeviceContext* immediateCtx, ID3D11Device* dev,
                             bool right, float separation, float convergence);

// Bind the per-device SRV to t125 on every programmable stage. Safe on the
// immediate context AND on RightRecorder11 (SSetShaderResources is a
// first-class recorded op). No Map/Unmap here — the texture-byte update is
// a separate call. No-op if EnsureStereoParams hasn't run yet.
void BindStereoParamsSRV(ID3D11DeviceContext* ctx, ID3D11Device* dev);

// Called from ~Device11Proxy. Releases the cached texture + SRV. Safe to
// call for a device that was never Ensure()'d (no-op).
void ReleaseStereoParams(ID3D11Device* dev);

}  // namespace wiz3d
