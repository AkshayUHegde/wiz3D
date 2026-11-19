# S3DWrapperVK

The Vulkan stereo wrapper: an explicit Vulkan layer, `VK_LAYER_wiz3D_stereo`, that renders every frame for both eyes
and presents them side by side or through the Simulated Reality weaver.

## How it loads

The game folder holds `vulkan-1.dll` (the proxy, `wiz3D-proxy/vulkan-1`), `VK_LAYER_wiz3D_stereo.json`,
`S3DWrapperVK.dll`, `S3DAPI.dll`, `S3DUtils.dll`, `ZLOg.dll`, `BaseProfile.xml` and `wiz3D_Config.xml`.

The proxy forwards every loader export to the system `vulkan-1.dll`. Before the loader's first layer scan it puts this
folder first in `VK_ADD_LAYER_PATH` and the layer first in `VK_LOADER_LAYERS_ENABLE`, keeping any values already set.
The loader then chains `S3DWrapperVK.dll` like any other layer: no registry entries and no admin rights.

Hosts that load the layer themselves (emulators) set those two variables instead of using the proxy.

## Stereo model

- **Vertex stages** get `x += sep * (w - conv)` after their own body, from a uniform block the layer owns at descriptor
  set 8 (`kStereoSet`). Pipeline layouts are padded so that set is free (`SpirvPatch.cpp`, `Pipelines.cpp`).
- **Images** that look like scene targets get a right-eye sibling bound into a mirror of the game's memory at the same
  offset, with sibling views and framebuffers (`Resources.cpp`; the rules are `StereoHeuristic.h`'s).
- **Passes** into a sibling-backed framebuffer, or dynamic rendering into sibling views, are recorded and replayed for
  the right eye at their end, with the right stereo set and shadow descriptor sets naming the siblings
  (`CommandBuffer.cpp`, `Descriptors.cpp`). Dispatches touching doubled images are doubled in place; copies, clears and
  barriers are mirrored.
- **Present** blits both eyes side by side into the presentable image, or hands them to the SR weaver through a private
  D3D11 device and shared textures (`Composite.cpp`, `Weave.cpp`). These are the two output modes; any other
  `OutputMode` is presented side by side and named in the log. A game presenting from a queue family with no
  graphics gets each eye copied for the weave; side by side and the on-screen display need blits and draws and are
  left out there.
- **On-screen display and screenshot**: the DX9/DX11 status strip, FPS counter and hotkey list are drawn over the final
  image (`OSDRendererVK.cpp`), and Ctrl+F12 writes both eyes at full size, taken before the composite
  (`ScreenshotVK.cpp`).
- **Separation and convergence** follow DX11: `sep = StereoBase * One_div_ZPS` scaled by the camera's P11, and
  `conv = 1 / One_div_ZPS` at either sign (`StereoParams.cpp`). Changed inputs are saved five seconds after the last change.
- **DLSS-SR** gets a right-eye twin feature with its own history at the NGX exports (`NgxVK.cpp`).
- **DLSS-G** over Streamline gets both eyes squeezed into one image, depth and motion too (`FrameGenVK.cpp`).

### The constants path (`DetectInverseMatrix`)

Screen passes rebuild positions from camera matrices in uniform buffers, which the vertex patch never touches. At each
submit the layer reads the uniform ranges of the bound sets through the game's own mapping, learns camera blocks per
(set layout, binding), and gives each such buffer a sibling the right eye's sets bind (`Constants.cpp`):

- The right projection is `P' = P` with the shader shift built in: `P'[8] += P00*e/C` and `P'[12] -= P00*e`.
- With `K = P^-1 P'`, a world-to-clip matrix becomes `M*K` and a clip-to-world matrix `K^-1*M`; view-space matrices and
  the camera position stay the game's, so everything the layer cannot reach still agrees between the eyes.
- A draw positioned through a matrix *claims* the registers it multiplies by, and only claimed projections are
  patched, so a screen pass reading the same block keeps its place (DX11's per-draw scoping).

Fragment and compute shaders that rebuild positions, rays or lookups from the left camera are patched by shape
(`FsPatch.cpp`, `CameraWindows.cpp`, `ComputeRemap.cpp`): the eye offset added to a camera-relative position, a ray
taken at the right eye, a lookup read at the left eye's pixel.

## Profiles

Per-game behaviour comes from the game's `<Profile>` in `BaseProfile.xml`, read once at the first `vkCreateInstance`.

- **Switches** are `<VKName Value="..."/>` elements, listed with their defaults in `ProfileVK.h`. An element that is no
  switch is logged and ignored.
- **Families** are engine or host modules enabled by one element each (`FamilyVK.h`): `<VKRage>` (`RageVK.cpp`),
  `<VKRpcs3>` (`RsxVK.cpp`), `<VKShadPS4>` (`ShadPs4VK.cpp`), `<VKShadGT>` (`ShadGtVK.cpp`) and `<VKKyty>`
  (`KytyVK.cpp`). Their children list the shaders each rule applies to, by CRC, and the engine's block layout. The
  shared files reach a family only through `Traits()` and the `Families*` queries.
- A game with no profile runs the core defaults.

## Logs

`wiz3D_Logs\S3DWrapperVK.log` next to the DLL, or `%LOCALAPPDATA%\wiz3D` where the game folder is read-only. The first
row names the profile, its families and every switch that differs from its default. Setup rows always print; the
per-window statistics (`[frame]`, `[cam]`, `[perf]`, `[img]`, `[mem]`) print with `CollectDebugInformation=1`.

## Building

`S3DDriver.sln`, x64 only. Vulkan-Headers and SPIRV-Headers come from `ThirdPartyLibs`; the weave needs the SR SDK the
other wrappers use.

## Known limitations

- The shared right-eye copy of a camera block keeps whichever write came last when two learned keys cover the same bytes.
- `IsAffine`'s tolerance scales with the matrix's largest element.
- `ConstantsSubmit` is one long function with family branches, and has no offline test.
- A module can be parsed several times along one rule chain.
- The replay keeps per-bind bookkeeping for screen triangles on its hot path.
