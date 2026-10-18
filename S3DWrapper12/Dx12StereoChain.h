/* wiz3D - DX12 stereo swap-chain wrapper
 *
 * Wraps IDXGISwapChain3 to provide the game with ArraySize=2 eye-slice back
 * buffers (left eye to slice 0, right eye to slice 1), while the real desktop
 * swap chain stays mono. On Present we composite the two eye slices into the
 * desktop back buffer using the configured wiz3D output mode (SBS first, then
 * TAB and Anaglyph later).
 *
 * The game's path:
 *   1. HD3D proxy (atidxx64 + IsWindowedStereoEnabled hook) tells the game
 *      stereo is available.
 *   2. Game creates IDXGISwapChain1 with DXGI_SWAP_CHAIN_DESC1::Stereo = TRUE.
 *   3. D3d12VendorProxy::HookedCreateSwapChainForHwnd intercepts and calls
 *      our Dx12Stereo::Create() instead of the native path.
 *   4. We create a mono desktop chain, wrap it in StereoChain, lie to the
 *      game about Stereo=TRUE via GetDesc1.
 *   5. Game renders both eyes natively (DX12 ArraySize=2 render target).
 *   6. On Present our wrapper runs CopyPair to composite SBS/TAB/Anaglyph
 *      into the desktop back buffer, then real->Present(0, ...).
 *
 * Adapted from farmerarmor/RiseTombRaiderVR src/Dx12Stereo.cpp (LGPL 2.1
 * compatible, our rev 981de7b vendored there). Rise's Publication/OpenXR
 * companion-process layer is replaced with the in-process SBS composite;
 * everything else about the IDXGISwapChain3 wrapper is retained with
 * monitor-stereo adaptations (no headset-dimension overrides, fullscreen
 * allowed, no per-game binary fingerprint).
 */

#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dxgi1_5.h>

namespace wiz3d {
namespace Dx12Stereo {

// Native CreateSwapChainForHwnd function pointer (from the real IDXGIFactory2).
// Passed in by the vendor-proxy hook so our wrapper can call through to it
// cleanly without needing a static import or further vtable chasing.
using CreateForHwndFn = HRESULT (WINAPI*)(
    IDXGIFactory2*,
    IUnknown*,                                   /* ID3D12CommandQueue for DX12 */
    HWND,
    const DXGI_SWAP_CHAIN_DESC1*,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC*,
    IDXGIOutput*,
    IDXGISwapChain1**);

// Called from D3d12VendorProxy::HookedCreateSwapChainForHwnd when the game
// requests Stereo=TRUE and wiz3D DX12 stereo is enabled. If the game didn't
// ask for stereo, or the opaque param isn't a DX12 command queue, this falls
// through to the native call and returns its HRESULT unchanged.
HRESULT Create(
    CreateForHwndFn native,
    IDXGIFactory2*  factory,
    IUnknown*       opaque,
    HWND            window,
    const DXGI_SWAP_CHAIN_DESC1* desc,
    const DXGI_SWAP_CHAIN_FULLSCREEN_DESC* fullscreen,
    IDXGIOutput*    output,
    IDXGISwapChain1** result);

// Runtime enable/disable (reads from Config.xml at wrapper init, settable
// later via the control API if we grow one).
bool IsEnabled();
void SetEnabled(bool enabled);

// Output mode selection for the compositor pass on Present.
// MonoMirror: slice 0 copied to desktop (left eye only; debug / fallback).
// Sbs:        slice 0 → left half, slice 1 → right half of a doubled-width
//             real swap chain. Window stretches 2W to W for standard half-SBS.
enum class OutputMode { MonoMirror = 0, Sbs = 1 };
OutputMode GetOutputMode();
void SetOutputMode(OutputMode mode);

} // namespace Dx12Stereo
} // namespace wiz3d
