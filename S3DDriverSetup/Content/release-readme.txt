# wiz3D, Quick Start


## Release Layout

wiz3D ships four render-path families. Each targets different hardware or
driver patterns; pick the one that matches the game you're setting up.

  Stereo-Injector/             Our own software stereo injector family.
                               Wraps the game's render path and generates
                               stereo frames in-process. Works on any GPU.
                               Output modes 1-10 share a unified numbering
                               scheme across every API; SR Weave is the
                               default on LeiaSR panels.
    DX7/                       DirectX 7 games (32-bit only).
    DX8/                       DirectX 8 games (32-bit only).
    DX9/32-bit|64-bit/         DirectX 9 games.
    DX10-11/32-bit|64-bit/     DirectX 10 / 11 games. Full COM-wrap +
                               DXBC analyser; per-game profiles ship in
                               BaseProfile.xml for 15+ engines.
    Vulkan/64-bit/             Vulkan games, emulators and PS4 ports.
                               A Vulkan layer plus a vulkan-1.dll proxy;
                               output is SR Weave or Half SBS.

  HD3D/32-bit|64-bit/         AMD HD3D vendor path. For games that
                               ship with AMD's HD3D stereo support.
                               Routes per-eye output through wiz3D's
                               software compositor at Present time.

  3D-Vision-Direct/            NVIDIA 3D Vision Direct Mode intercept.
    DX9|DX10|DX11/             For games that use NvAPI_Stereo_SetActiveEye
      32-bit|64-bit/           and render each eye explicitly. NVAPI
                               stereo bridge wired to wiz3D's separation
                               and convergence sliders on DX9 and DX11.

  OpenGL-Quad-Buffer-Stereo/   Captures native OpenGL quad-buffer stereo
    32-bit|64-bit/             (PFD_STEREO / GL_BACK_LEFT/RIGHT) from
                               games that produce it (id Tech 3/4, CAD
                               tools, flight sims). Unified output
                               numbering with the rest of the tree.


## Stereo-Injector Setup (DX7 / DX8 / DX9 / DX10-11)

1. Check if your game is 32-bit or 64-bit.
2. Copy the ENTIRE contents of the matching subfolder into the game folder
   next to the game .exe:
     Stereo-Injector\DX7\                         (DX7 games, 32-bit)
     Stereo-Injector\DX8\                         (DX8 games, 32-bit)
     Stereo-Injector\DX9\32-bit\  or  64-bit\     (DX9 games)
     Stereo-Injector\DX10-11\32-bit\  or  64-bit\ (DX10/11 games)
3. (Optional) Open wiz3D_Config.xml and change settings. Mode names and
   numbers are documented in-line in the file; the OutputMode / OutputMethodDll
   scheme is unified across every wiz3D path. SR Weave is the default on
   LeiaSR panels.
4. Launch the game normally. The proxy loads automatically.


## Stereo-Injector Setup (Vulkan)

1. Copy the ENTIRE contents of Stereo-Injector\Vulkan\64-bit\ next to the
   .exe that runs the game. For ports started by a launcher that is the
   bin\ or out\ subfolder, not the launcher's folder.
2. Do NOT run the game as administrator: the Vulkan loader then ignores
   the wiz3D layer and the game runs in 2D.
3. If the folder already holds a vulkan-1.dll (The Last Guardian, inFAMOUS
   Second Son), rename it as a backup and use the wiz3D one.
4. Hosts that refuse or bypass a local vulkan-1.dll take every file EXCEPT
   vulkan-1.dll, and are started with two environment variables set:
     VK_ADD_LAYER_PATH=<folder holding VK_LAYER_wiz3D_stereo.json>
     VK_LOADER_LAYERS_ENABLE=VK_LAYER_wiz3D_stereo
   - RPCS3: files in RPCS3's folder, started through a small .cmd script.
   - KytyPS5 (Demon's Souls): files in <KytyPS5>\wiz3D\, variables in
     launch.json, and a 3D shader precompile run once.
   The scripts, the launch.json lines and the precompile steps are in
   wiki\API-Vulkan.md (https://github.com/Stereopticon-Org/wiz3D/wiki).
5. Turn pipeline binary caches off where the host has one (God of War III:
   "pipeline_binary_cache": false in settings.json), or shader fixes are
   ignored. DOOM Eternal: turn "Present From Compute" off. Shadow of the
   Colossus: turn the atmospheric fog effect off in the launcher.
6. (Optional) wiz3D_Config.xml: OutputMode SRWeave or HalfSBS. Other modes
   are shown as Half SBS under Vulkan.
7. Hotkeys are the DX9 / DX11 ones: Numpad * toggles stereo, Numpad +/-
   separation, Shift + Numpad +/- screen plane, Ctrl+F12 stereo screenshot,
   Shift+F1 the hotkey list.
8. The log is wiz3D_Logs\S3DWrapperVK.log in that folder, or
   %LOCALAPPDATA%\wiz3D\ when the game folder is read-only.


## HD3D Setup

1. Check if your game is 32-bit or 64-bit.
2. Copy the ENTIRE contents of the `HD3D\32-bit` or `HD3D\64-bit` folder
   into the game's folder next to the game .exe.
3. (Optional) Open HD3D_Config.xml and change settings.
4. Launch the game.
5. Enable 'Stereoscopic 3D' or 'HD3D' in the game settings.


## 3D Vision Direct Mode Setup

1. Check if your game is DirectX 9, 10, or 11, and 32-bit or 64-bit.
2. Copy the ENTIRE contents of the matching
   `3D-Vision-Direct\<API>\<arch>` folder into the game's folder next to
   the game .exe.
3. (Optional) Open 3DVision_Config.xml and change settings.
4. Launch the game.
5. The game's own 3D Vision toggle in-game should now work. Separation
   and convergence are driven by wiz3D's sliders via the NVAPI stereo
   bridge (DX9 + DX11).


## OpenGL Quad-Buffer Stereo Setup

1. Check if your game is 32-bit or 64-bit.
2. Copy the ENTIRE contents of the `OpenGL-Quad-Buffer-Stereo\32-bit` or
   `OpenGL-Quad-Buffer-Stereo\64-bit` folder into the game folder next
   to the game .exe.
3. (Optional) Open wiz3D_Config.xml and change settings.
4. Launch the game.
5. Enable the game's native stereo / quad-buffer option.

---------------------------------------------------------------------------------

## Removing wiz3D

Use the Uninstall_wiz3D.bat (or Uninstall_HD3D.bat / Uninstall_3DVisionDirect.bat)
in the folder you installed to. The game will load the real system DLLs again.


## Troubleshooting

- Game won't launch? Check you have d3dx9_43.dll (install DirectX runtime)
- No stereo effect? Make sure EnableStereo="1" in the config XML
- Wrong display mode? Change OutputMode or OutputMethodDll in the config XML
- Game crashes? Remove the wrapper DLL from the game folder to disable
