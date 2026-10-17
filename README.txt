# wiz3D — Quick Start


## Release Layout

wiz3D ships four render-path families. Each targets different hardware or
driver patterns; pick the one that matches the game you're setting up.

  StereoInjector/              Our own software stereo injector family.
                               Wraps the game's render path and generates
                               stereo frames in-process. Works on any GPU.
    dx7/                       DirectX 7 games (32-bit only).
    dx8/                       DirectX 8 games (32-bit only).
    dx9/x86|x64/               DirectX 9 games.
    dx10-11/x86|x64/           DirectX 10 / 11 games (Akshay's SwapChain
                               wrap; inline composite + SR weave).

  hd3d/x86|x64|dx12 x64/       AMD HD3D vendor path. For games that
                               ship with AMD's HD3D stereo support.
                               Routes per-eye output through wiz3D's
                               software compositor at Present time.

  3d-vision-direct/            NVIDIA 3D Vision Direct Mode intercept.
    dx9|dx10|dx11/x86|x64/     For games that use NvAPI_Stereo_SetActiveEye
                               and render each eye explicitly.

  opengl-quad-buffer-stereo/   Captures native OpenGL quad-buffer stereo
    x86|x64/                   (PFD_STEREO / GL_BACK_LEFT/RIGHT) from
                               games that produce it (id Tech 3/4, CAD
                               tools, flight sims).


## StereoInjector Setup (DX7 / DX8 / DX9 / DX10-11)

1. Check if your game is 32-bit (x86) or 64-bit (x64).
2. Copy the ENTIRE contents of the matching subfolder into the game folder
   next to the game .exe:
     StereoInjector\dx7\                    (DX7 games, 32-bit)
     StereoInjector\dx8\                    (DX8 games, 32-bit)
     StereoInjector\dx9\x86\  or  x64\      (DX9 games)
     StereoInjector\dx10-11\x86\  or  x64\  (DX10/11 games)
3. (Optional) Open wiz3D_Config.xml and change settings. Mode names and
   numbers are documented in-line in the file; the OutputMode / OutputMethodDll
   scheme is unified across every wiz3D path.
4. Launch the game normally. The proxy loads automatically.


## HD3D Setup

1. Check if your game is 32-bit or 64-bit.
2. Copy the ENTIRE contents of the `hd3d\x86` or `hd3d\x64` folder into
   the game's folder next to the game .exe.
3. (Optional) Open HD3D_Config.xml and change settings.
4. Launch the game.
5. Enable 'Stereoscopic 3D' or 'HD3D' in the game settings.


## 3D Vision Direct Mode Setup

1. Check if your game is DirectX 9, 10, or 11, and 32-bit or 64-bit.
2. Copy the ENTIRE contents of the matching `3d-vision-direct\<api>\<arch>`
   folder into the game's folder next to the game .exe.
3. (Optional) Open 3DVision_Config.xml and change settings.
4. Launch the game.
5. The game's own 3D Vision toggle in-game should now work.


## OpenGL Quad-Buffer Stereo Setup

1. Check if your game is 32-bit or 64-bit.
2. Copy the ENTIRE contents of the `opengl-quad-buffer-stereo\x86` or
   `opengl-quad-buffer-stereo\x64` folder into the game folder next to
   the game .exe.
3. (Optional) Open wiz3D_Config.xml and change settings.
4. Launch the game.
5. Enable the game's native stereo / quad-buffer option.

---------------------------------------------------------------------------------

## Removing wiz3D

Use the Uninstall_wiz3D.bat (or Uninstall_HD3D.bat / Uninstall_3DVisionDirect.bat)
in the folder you installed to. The game will load the real system DLLs again.


## Troubleshooting

- Game won't launch? → Check you have d3dx9_43.dll (install DirectX runtime)
- No stereo effect? → Make sure EnableStereo="1" in the config XML
- Wrong display mode? → Change OutputMode or OutputMethodDll in the config XML
- Game crashes? → Remove the wrapper DLL from the game folder to disable
