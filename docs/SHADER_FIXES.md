# Shader fixes in wiz3D

wiz3D authors stereo correctness per game via two complementary layers:

1. **CB-math features** (DX10/11, declarative XML in `BaseProfile.xml`) — the engine edits constant-buffer bytes before each eye's draws so the game's untouched shaders naturally produce per-eye output. One profile tunes a whole engine family (UE3/UE4, CryEngine, RAGE, Dunia, etc.). This is the primary system; most games get full stereo this way with no shader-level work.
2. **Shader-level fixes** (DX9 today, DX11 next release) — surgical per-shader bytecode replacement keyed by CRC32. For the ~15-30% of effects that can't be expressed as a CB edit: shaders that bypass CBs for stereo-relevant math (HUDs, crosshairs), write custom depth into output registers, or construct their own transforms from loose scalars.

This doc covers the shader-level fix layer. For CB-math features, see the inline comments in `S3DWrapper10/Feature*.cpp` (37 features, each with its XML tag documented in the file header) and the Profile entries in `S3DDriverSetup/Content/BaseProfile.xml`.

---

## DX9 — HelixMod-compatible, shipping since 0.3.0

Three layers in `S3DWrapper9/`, all gated on the `ShaderOverride/` folder existing next to the game. Zero cost when inactive.

### 1. Stereo-params texture

`BaseStereoRenderer.cpp` allocates a 1×16 `D3DFMT_A32B32G32R32F` texture, bound to `D3DVERTEXTEXTURESAMPLER0` (VS `s0`) and PS sampler `s13`. On each eye's draws it's populated with:

```
r = separation
g = convergence
b = eye_sign  (-1 for left, +1 for right)
a = magic     (iZ3D's traditional sentinel value)
```

Shaders sample at UV `0.0625` (`c200` in SM3 convention: `def c200, 0.0625, 0, 0, 0`) to read the per-eye params. This matches NVIDIA's de-facto stereo-texture layout, so community Helix fixes work unmodified.

Hardware-agnostic — mimicked in user code, no NVAPI runtime dependency.

### 2. `texldl` detection + double-shift skip

`ScanForStereoTexldl` walks each shader's bytecode at creation and flags shaders that sample from the stereo-params sampler. `SetViewStages` then skips wiz3D's own matrix-based eye offset for flagged shaders — if a replaced shader is applying its own eye shift, wiz3D's shouldn't double up.

### 3. ShaderOverride loader

`S3DWrapper9/ShaderOverrideLoader.cpp` scans for replacement bytecode at:

```
<game_folder>/ShaderOverride/VertexShaders/<CRC32_HEX>.txt
<game_folder>/ShaderOverride/PixelShaders/<CRC32_HEX>.txt
```

Runs `D3DXAssembleShader` on the file contents, substitutes the result for the original shader. File format is standard HelixMod SM3 asm text. Supports `//` comment headers (the assembler strips them).

CRC32 is computed via `CommonUtils/System.cpp::CalculateCRC32` on the unassembled bytecode buffer. Dump a shader via `DumpVS="1"` / `DumpPS="1"` in `wiz3D_Config.xml` to see its CRC in the dump header.

**wiz3D ships no fix files.** Users bring their own; this is a consumer, not a distributor. The folder-exists gate means games without an override folder incur zero per-shader file I/O.

### Example DX9 fix

```asm
// CRC: 0xA3B5F218
// Game: BioShock Infinite
// Problem: Deferred shadow sampled at mono depth
// Fix: Apply eye shift at correct depth via stereo-params sampler
//
vs_3_0
dcl_position v0
dcl_texcoord v1
dcl_position o0

def c200, 0.0625, 0, 0, 0

; Original transform
mul r0, c0, v0.x
mad r0, c1, v0.y, r0
mad r0, c2, v0.z, r0
mad r0, c3, v0.w, r0

; Stereo params at s0: (separation, convergence, eye_sign, _)
texldl r10, c200, s0

; pos.x += eye_sign * separation * (pos.w - convergence)
add r11.x, r0.w, -r10.y
mul r11.x, r11.x, r10.x
mad r0.x, r11.x, r10.z, r0.x

mov o0, r0
```

Drop at `<game>/ShaderOverride/VertexShaders/0xA3B5F218.txt`, launch the game. Done.

## DX10 / DX11 — roadmap

Not shipped yet. Current state:

- **CB-math features** are shipped (Akshay's PR #31, 0.4.0). 37 declarative tags in `BaseProfile.xml` cover most engine families.
- **Shader-level fixes** are planned for 0.5.0 as inline `<ShaderOverride CRC="..."><![CDATA[...]]></ShaderOverride>` entries in `BaseProfile.xml`, with the same three-layer architecture ported from DX9:
  - Stereo-params SRV at a reserved slot (likely `t125`)
  - `texldl`-equivalent scan on `sample` instructions against that slot → `usesStereoParams` flag on `ShaderAnalysis11Result`
  - Loader keyed on CRC32, consuming HLSL text compiled via `D3DCompile`
- **ShaderHunter** OSD mode (0.5.0): hotkey cycles the currently-drawn shaders, hides each in turn, captures the identified CRC + mini-report (disasm + CB layout + suggested ShaderPatch skeleton) to `wiz3D_Logs/<Game>/Shaders/Hunter/<CRC>_<timestamp>.txt`.

Inline-only. No `wiz3D_Overrides/` folder, no sidecar `.txt` sprawl: every fix lands as a reviewed PR against `BaseProfile.xml`, inheriting LGPL-2.1 by construction.

See `S3DWrapper10/ShaderModifier11.cpp` (automatic VS eye-shift token rewrite) and `S3DWrapper10/ShaderPatch11.cpp` (automatic right-eye shader twinning) for the existing engine-internal shader-modification paths. Both run without user authoring; the planned ShaderOverride loader is additive.

## DX8 — planned (0.5.x)

DX8 shaders are SM1.0/1.1 assembly only. `S3DWrapper8/Direct3DShader8.cpp:662-670` already routes DX8 shader creation through `D3DXAssembleShader` from `d3dx9.h`, so the DX9 ShaderOverride loader ports mechanically. Deferred until after DX10/11 ships because the community fix corpus for DX8 games is small.

## DX7 — not applicable

Fixed-function pipeline, no programmable shaders. Stereo correctness is handled by projection-matrix manipulation in the fixed-function state — no shader-level fix layer is needed or possible.

## Authoring switches (`wiz3D_Config.xml`)

Under `<GlobalSettings>`:

- `CollectDebugInformation="1"` — DX10/11 only. Reports CB layouts and matrix candidates to `wiz3D_proxy.log`. Primary authoring aid for CB-math profiles.
- `DumpVS="1"` / `DumpPS="1"` — dump every shader created by the game. Lands in:
  - DX10/11: `wiz3D_Logs/<GameName>/Shaders/Dump/DX10/{VS,PS,GS,HS,DS,CS}.txt`
  - DX9: `wiz3D_Logs/<GameName>/Shaders/Dump/DX9/{vs,ps}.{client,converted,modified}.txt`
- `VerboseFrameTrace="N"` with `FrameTraceStartFrame="M"` — DX10/11 only. Per-frame draw-by-draw timeline into `wiz3D_proxy.log` for `N` frames starting at Present #`M`.

Expensive flags — enable only during authoring, revert to `0` for play.

## Licensing

All in-repo profiles and shader fixes contributed via PR fall under the project's LGPL-2.1 by default (new code). Original iZ3D files retain their MIT headers.

**Do not submit shader fixes that are ports or derivatives of GPL-licensed HelixMod / Geo-11 / 3DMigoto fixes.** Clean-room reimplementations only, with pattern attribution as a courtesy. The repo's shader-fix pipeline is a *consumer* of the standard fix format, not a redistributor of upstream fix corpora.

If a specific community fix would materially help a shipped profile, ask the original author to grant us a fresh version under LGPL-2.1 (or MIT, which composes cleanly). Many will say yes.
