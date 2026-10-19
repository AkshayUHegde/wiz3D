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

  HD3D/32-bit|64-bit/          AMD HD3D vendor path. For games that
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
