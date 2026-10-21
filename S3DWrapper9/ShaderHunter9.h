// -----------------------------------------------------------------------------
// wiz3D ShaderHunter — DX9
//
// Mirror of S3DWrapper10/ShaderHunter.{h,cpp} with DX9-specific capture.
// Same hotkey model (Ctrl+F2 enter, F2 / Shift+F2 cycle, F3 capture) and
// identical fast-path semantics (atomic active flag + hashmap lookups).
//
// Capture format for DX9 is HelixMod-compatible: dumps the shader's SM3
// asm via D3DXDisassembleShader into
//     wiz3D_Logs/<Game>/Shaders/Hunter/<CRC>_<stage>_<ts>.txt
// A suggested skeleton points at either (a) the DX9 <Mono> block for
// shader-level stereo-off, (b) a HelixMod ShaderOverride .txt the author
// can drop into <game>/ShaderOverride/{VertexShaders,PixelShaders}/
// <CRC>.txt and tune. Both authoring paths already shipped in 0.3.0;
// Hunter is the ergonomic layer on top that identifies the right CRC.
//
// See docs/SHADER_HUNTER_DESIGN.md and docs/SHADER_FIXES.md.
// -----------------------------------------------------------------------------
#pragma once

#include <cstdint>

namespace wiz3d
{
namespace dx9
{

enum class HuntStage : uint8_t
{
    VS = 0,
    PS = 1,
    _Count
};

// Fast-path gate used by hot callers. Returns false until Ctrl+F2 enters
// hunt mode. One relaxed atomic load — no lock.
bool HunterIsActive();

// Record a newly created shader. Called from CBaseStereoRenderer::Create*
// after the ShaderOverride substitution. `bytecode` is retained by shared_ptr
// so the capture-report path can disassemble it later without the game
// re-sending the blob.
void HunterNoteCreated(uint32_t crc, HuntStage stage, const void* bytecode, uint32_t sizeBytes);

// Record that a draw used these two bound shaders. 0 == stage unbound.
void HunterNoteDraw(uint32_t vsCrc, uint32_t psCrc);

// Returns true if the currently hunted CRC matches either bound shader.
// Called at every Draw*; symmetric skip keeps both eyes drawing the same
// no-op (no left-only ghost of the hidden shader).
bool HunterShouldSkipDraw(uint32_t vsCrc, uint32_t psCrc);

// Called from CBaseStereoRenderer::Present (both pre- and post-replay are
// fine — rebuilds the stable prev/next cycle from drawn CRCs, resets
// per-frame counters).
void HunterFrameEnd();

// Hotkey handlers (KeyboardHook thread).
void HunterOnToggle();
void HunterOnCyclePrev();
void HunterOnCycleNext();
void HunterOnCapture();

// OSD read API — POD snapshot, no locks held across GDI.
struct HunterState
{
    bool     active;
    uint32_t hiddenCRC;
    HuntStage hiddenStage;
    uint32_t drawsForHidden;
    size_t   cycleIdx;
    size_t   cycleSize;
    // 0.4.1 save-toast: non-zero flashDeadline means render a "Shader
    // saved: <leaf>" toast bottom-right until the deadline. Mirrors DX11.
    uint64_t flashDeadline;
    char     flashLeaf[128];
};
HunterState HunterGetState();

// 0.4.1 runtime profile reload for DX9 — mirrors the DX11 helpers in
// S3DWrapperD3D10.cpp. RequestProfileReload is the KeyboardHook handler
// (sets a between-frames atomic flag); MaybeReloadProfileBetweenFrames
// runs at the Present boundary (BaseSwapChain::Present), consumes the
// flag, and re-reads BaseProfile.xml via ReadCurrentProfile(). CRC-keyed
// data on g_ProfileData is replaced; the Feature registry is static set
// at CreateDevice time so engine-tag changes still need a game restart.
void RequestProfileReload();
void MaybeReloadProfileBetweenFrames();

}  // namespace dx9
}  // namespace wiz3d
