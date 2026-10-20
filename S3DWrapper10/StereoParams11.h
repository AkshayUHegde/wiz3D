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

// Called from Context11Proxy::BindEye at every left/right switch. Maps the
// per-device texture via MAP_WRITE_DISCARD on the given context, writes the
// four floats (sep, conv, eye_sign, 0), unmaps, then binds the SRV to
// slot t125 on VS/HS/DS/GS/PS/CS. Safe to call before EnsureStereoParams —
// it's a no-op in that case.
void BindStereoParamsForEye(ID3D11DeviceContext* ctx, ID3D11Device* dev,
                            bool right, float separation, float convergence);

// Called from ~Device11Proxy. Releases the cached texture + SRV. Safe to
// call for a device that was never Ensure()'d (no-op).
void ReleaseStereoParams(ID3D11Device* dev);

}  // namespace wiz3d
