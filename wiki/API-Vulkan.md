# Vulkan

✅ **Working on the titles below.** The Vulkan Stereo-Injector is an explicit Vulkan layer, `VK_LAYER_wiz3D_stereo` (`S3DWrapperVK.dll`). It renders every frame for both eyes and presents them through the SR weave or as half side-by-side. Per-game behaviour comes from `BaseProfile.xml`; a game with no profile runs the core defaults.

## How the wrapper works

- `vulkan-1.dll` next to the game's `.exe` is a proxy: it forwards every call to the system Vulkan loader and tells that loader to chain the layer from the same folder. No registry entries, no admin rights.
- The layer patches vertex shaders to shift the right eye, gives scene images a right-eye copy, replays each pass for the right eye, and patches camera constants and screen-space shaders so lighting, shadows and reflections agree between the eyes.
- DLSS gets a right-eye twin at the NGX boundary. DLSS Frame Generation over Streamline gets both eyes squeezed into one image.
- Details for developers: `S3DWrapperVK/README.md`.

## Install path

`Stereo-Injector/Vulkan/64-bit/` (64-bit only). Copy its **contents** next to the `.exe` **that creates the Vulkan device**:

| File | What it is |
|---|---|
| `vulkan-1.dll` | The proxy. Leave it out for hosts listed under "No proxy" below. |
| `VK_LAYER_wiz3D_stereo.json`, `S3DWrapperVK.dll` | The layer and its manifest. |
| `S3DAPI.dll`, `S3DUtils.dll`, `ZLOg.dll` | Shared wiz3D libraries. |
| `BaseProfile.xml`, `wiz3D_Config.xml` | Game profiles and your settings. |

Rules that apply to every title:

- **The right folder is the game process's, not the launcher's.** Ports started by a `Play <Game>.exe` launcher run the game from a `bin\` or `out\` subfolder; the files go there (see the table).
- **Do not run the game as administrator** and remove any "Run as administrator" compatibility flag. The Vulkan loader ignores layer settings in elevated processes, and the game then runs in plain 2D.
- **Pipeline binary caches must be off** where the host has one. A cached pipeline is built without the layer's shader changes, so fixes silently do nothing. The table names the setting per host.
- **Launchers that update or repair the install** can put their own `vulkan-1.dll` back. Copy the proxy again after an update.
- **Log:** `wiz3D_Logs\S3DWrapperVK.log` in the folder holding the layer, or `%LOCALAPPDATA%\wiz3D\` when that folder is read-only. Its first row names the profile it matched.
- **Your tuned separation and convergence** are saved to `UserProfile.xml` next to the game's `.exe` five seconds after the last change. In a read-only game folder they are not saved.

### When a `vulkan-1.dll` is already there, or is refused

| Situation | Hosts | What to do |
|---|---|---|
| The host ships its own `vulkan-1.dll` in the game folder | The Last Guardian, inFAMOUS Second Son (`out\`) | Rename theirs (keep it as a backup) and copy the wiz3D proxy in its place. The proxy uses the system loader, so the GPU driver's Vulkan runtime must be installed. |
| The host refuses a `vulkan-1.dll` outside the Windows folder | RPCS3 ("module vulkan-1.dll was incorrectly installed") | **No proxy.** Copy everything except `vulkan-1.dll` and start the host through a script that sets two variables: see [RPCS3](#rpcs3). |
| The host copies or hard-links the DLLs in its folder | KytyPS5 (its shader precompile) | **No proxy.** Put everything except `vulkan-1.dll` in a `wiz3D\` subfolder and name it in `launch.json`: see [Demon's Souls](#demons-souls-kytyps5). |

Any host can use the no-proxy route. The two variables are:

```
VK_ADD_LAYER_PATH=<folder holding VK_LAYER_wiz3D_stereo.json>
VK_LOADER_LAYERS_ENABLE=VK_LAYER_wiz3D_stereo
```

## Output, hotkeys and the on-screen display

- **Output modes:** `SRWeave` and `HalfSBS` (`OutputMode` in `wiz3D_Config.xml`). Any other mode is presented as half side-by-side, and the log says so.
- **On-screen display:** the same status strip, FPS counter and hotkey list as DX9 / DX11 (`ShowOSD`, `ShowFPS`).
- **Games that present from a compute queue** (Wolfenstein II from its 3D menu on): the SR weave works. Half side-by-side and the on-screen display need drawing commands that queue cannot run, so Half SBS shows only the left eye there and the display is not drawn.

| Hotkey | Action |
|---|---|
| Numpad `*` | Stereo on / off |
| Numpad `+` / `-` | Separation |
| Shift + Numpad `+` / `-` | Screen plane (convergence). It can cross zero, as on DX11. |
| Numpad `7` / `8` / `9` | Presets |
| Ctrl + F12 | Stereo screenshot: both eyes at full size, right eye in the left half, in `Pictures\wiz3D Screenshots\<profile>\` |
| Shift + F1 | Hotkey list |
| Ctrl + F1 | Status strip + FPS |
| Ctrl + F5 | Reset the preset values to the loaded ones |

Auto-focus, the laser sight, ShaderHunter and the Alt + Numpad convergence shift have no effect under Vulkan.

## Games

Versions are the ones each title was accepted on. "Folder" is where the release files go.

### Native PC

| Game | Tested version | Folder | Requirements | Status |
|---|---|---|---|---|
| Red Dead Redemption 2 | `RDR2.exe` 1.0.1491.50 | Next to `RDR2.exe` | Graphics API: Vulkan. The game folder is usually read-only, so the log goes to `%LOCALAPPDATA%\wiz3D\` and hotkey tuning is not saved unless `UserProfile.xml` there is writable. | ✅ Working |
| DOOM (2016) | Steam build of 2026-11 (version not recorded) | Next to `DOOMx64vk.exe` | Graphics API: Vulkan (the game then runs `DOOMx64vk.exe`). | ✅ Working |
| DOOM Eternal | Version not recorded | Next to `DOOMEternalx64vk.exe` | **Video settings: "Present From Compute" Off.** | ✅ Working |
| Wolfenstein II: The New Colossus | Xbox app package 1.5.0.0 | Next to `NewColossus_x64vk.exe` (the `Content` folder) | **`OutputMode` SRWeave.** The game presents from a compute queue (see below). | ✅ Working |

### Emulators

| Game | Emulator and tested version | Folder | Requirements | Status |
|---|---|---|---|---|
| The Legend of Zelda: Breath of the Wild | Cemu 1.22.7 | Next to `Cemu.exe` | Graphics API: Vulkan in Cemu's general settings (`settings.xml`: `<api>1</api>`). No "Run as administrator" flag on `Cemu.exe`. | ✅ Working |
| The Legend of Zelda: Tears of the Kingdom | Ryubing (Ryujinx) 1.3.3 | Next to `Ryujinx.exe` | None. | ✅ Working |
| Ocarina of Time Remake | Lichii build-316 | Next to `lichii.exe` (the proxy works; no variables needed) | Close the emulator before replacing the files: the layer stays loaded until it exits. | ✅ Working |
| Donkey Kong Bananza | Lichii build-316 | Next to `lichii.exe` | As above. | ⚠️ Working with issues: artefacts in the tutorial video windows |
| Mario Kart World | Lichii build-316 | Next to `lichii.exe` | As above. | ⚠️ Working with issues: the shadows under the vehicles are wrong |
| Gran Turismo 5, Gran Turismo 6, Killzone 2, Killzone 3, Killzone HD | RPCS3 0.0.43-20561 | RPCS3's folder, **without** `vulkan-1.dll` | Start RPCS3 through the script below. Per-game config: "3D Display" off. | ✅ Working (accepted on 20561; 20607 not yet re-run) |
| Demon's Souls (PPSA01341, 1.07) | KytyPS5 (chenxiao07 fork) v20261102 | `<KytyPS5>\wiz3D\`, **without** `vulkan-1.dll` | `launch.json` environment and the 3D shader precompile below. | ✅ Working |

### PS4 ports (shadPS4-based)

All of these start from a launcher; the files go next to the game process.

| Game | Port and tested version | Folder | Requirements | Status |
|---|---|---|---|---|
| Bloodborne (CUSA03173, 1.09) | bbport, Jgamrot fork (`bb-probe.exe` of 2026-10-17) | `bbport-windows\bin\` | None. | ✅ Working |
| Shadow of the Colossus | ShadowOfTheColossus-PC 0.75 | `SotC-PC\bin\` | **Launcher, Effects: atmospheric fog Off** (`"fx_fog": false` in `user\launcher_settings.json`). | ✅ Working |
| Uncharted: The Nathan Drake Collection | uncharted-windows (`bb-probe.exe` of 2026-11-16) | `uncharted-windows\bin\` | None. | ✅ Working |
| The Last Guardian (CUSA03627, 1.03) | tlg_pc v2026.11.14 | `tlg-windows-x86_64\out\`, replacing the port's own `vulkan-1.dll` | Keep a backup of the port's `vulkan-1.dll`. | ✅ Working |
| inFAMOUS Second Son | infamous port v2026.10.28 | `infamous-windows-x86_64\out\`, replacing the port's own `vulkan-1.dll` | Use FSR, not the DLSS upscaler (see below). | ✅ Working with FSR |
| God of War III (CUSA01715) | SpartanPort (`spartanport-probe.exe` of 2026-10-22) | `spartanport-windows\bin\` | `settings.json`: `"pipeline_binary_cache": false`. | ✅ Working |
| DriveClub | driveclub-recomp 0.2.5 | `driveclub-recomp\runtime\` | None. | ✅ Working |
| Gran Turismo Sport (1.69) | RyGT prerelease 2026-10-29 | Next to `GTSport.exe` | None. | ✅ Working |

### Upscalers and frame generation

Tested = run in stereo and accepted. Untested = never run with this wrapper; it may or may not work.

| Game | DLSS / DLAA | DLSS Frame Generation | FSR |
|---|---|---|---|
| Red Dead Redemption 2 | ✅ DLSS tested | Untested | Untested |
| DOOM Eternal | ✅ DLSS tested | Untested | n/a |
| Bloodborne | ✅ DLSS tested | Untested | Untested |
| Shadow of the Colossus | ✅ DLSS tested | ✅ Tested | Untested |
| DriveClub | ✅ DLAA tested | Untested | Untested |
| Demon's Souls | ✅ DLSS tested | ✅ Tested (2x) | Untested |
| Uncharted | Untested | Untested | ✅ Tested (FSR 4 anti-aliasing + FSR 1) |
| The Last Guardian | Untested | Untested | ✅ Tested |
| inFAMOUS Second Son | ❌ No stereo with the DLSS upscaler | Untested | ✅ Tested |
| Gran Turismo Sport | ⚠️ DLSS runs per eye; a right-eye shimmer was reported and its fix has not been re-run | Untested | Untested |
| RPCS3 titles | n/a | n/a | ✅ Tested (RPCS3's own FSR output) |
| DOOM (2016), Wolfenstein II, Cemu, Ryubing, Lichii titles | n/a | n/a | n/a |

Frame generation carries both eyes in one image, so each eye has half the horizontal resolution while it is on.

## RPCS3

RPCS3 checks its loaded modules and exits if `vulkan-1.dll` is not the one in the Windows folder, so the proxy cannot be used.

1. Copy the contents of `Stereo-Injector/Vulkan/64-bit/` **except `vulkan-1.dll`** into the folder holding `rpcs3.exe`.
2. Save this next to `rpcs3.exe` as `rpcs3_3d.cmd` and always start RPCS3 with it. `rpcs3.exe` on its own runs in plain 2D.

```bat
@echo off
cd /d "%~dp0"
set "VK_ADD_LAYER_PATH=%~dp0"
set "VK_LOADER_LAYERS_ENABLE=VK_LAYER_wiz3D_stereo"
start "" "%~dp0rpcs3.exe" %*
```

3. In each game's custom configuration, leave RPCS3's own "3D Display" off: with it on, both would shift the image.
4. Killzone HD and Gran Turismo 6 need "Read Color Buffers" on, as they do without wiz3D.
5. The layer stays loaded until RPCS3's main window closes. Close RPCS3 completely before replacing the files.

## Demon's Souls (KytyPS5)

KytyPS5's shader precompile links the DLLs in its own folder, so the layer lives in a subfolder and no proxy is used.

1. Copy the contents of `Stereo-Injector/Vulkan/64-bit/` **except `vulkan-1.dll`** into `<KytyPS5>\wiz3D\`.
2. In `<KytyPS5>\launch.json`, set the environment (use your own path, with doubled backslashes):

```json
"environment": {
  "VK_ADD_LAYER_PATH": "C:\\KytyPS5\\wiz3D",
  "VK_LOADER_LAYERS_ENABLE": "VK_LAYER_wiz3D_stereo",
  "KYTY_PIPELINE_BINARIES": "0",
  "KYTY_MODULE_STORE": "0"
}
```

   An emulator update resets `launch.json`: put the block back afterwards. Keep a copy of the file.

3. **Precompile the shaders for 3D.** The emulator's own precompile builds the 2D pipelines; in 3D the game asks for different ones, and stutters while it compiles each of them for the first time. Save this next to the emulator's `precompile.cmd` as `precompile-3d.cmd` and run it once:

```bat
@echo off
set VK_ADD_LAYER_PATH=%~dp0wiz3D
set VK_LOADER_LAYERS_ENABLE=VK_LAYER_wiz3D_stereo
set KYTY_PIPELINE_BINARIES=0
set KYTY_MODULE_STORE=0
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0precompile-windows.ps1" %*
pause
```

   - It takes the options of the emulator's `precompile-windows.ps1` (`-Quick` for only the shaders recorded play used, `-Jobs N`).
   - A full run takes hours; it resumes when interrupted, and the game can start once it prints "ready to play".
   - **Run it again after every KytyPS5 update and every new wiz3D release**: both change the pipelines.

4. Frame generation is not active on the title screen or in menus; judge it in gameplay.

## Known issues

- inFAMOUS Second Son with the DLSS upscaler has no stereo. Use FSR.
- Wolfenstein II: SR weave only (Half SBS shows the left eye), and no on-screen display, because it presents from a compute queue.
- Shadow of the Colossus needs its atmospheric fog effect turned off.
- Donkey Kong Bananza (Lichii): the tutorial video windows show artefacts.
- Mario Kart World (Lichii): the shadows under the vehicles are wrong.
- Gran Turismo Sport with DLSS Frame Generation has not been run.
- DOOM: The Dark Ages hangs with the layer and has no profile.
- Under frame generation each eye has half the horizontal resolution.

## Performance note

Akshay reports **200 FPS in Demon's Souls** with 4× MFG + DLSS Quality + stereo enabled on his test rig. Vulkan + the emulator's compute-based approach scales well with modern GPUs.

## Reporting

Please report other Vulkan titles, with `S3DWrapperVK.log`, via [GitHub Issues](https://github.com/Stereopticon-Org/wiz3D/issues).
