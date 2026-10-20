@echo off
echo ============================================
echo  wiz3D NvDirectMode (3D Vision Direct) Uninstaller
echo ============================================
echo.
echo This will remove ONLY wiz3D NvDirectMode proxy files from the current
echo directory. No game files will be modified or deleted.
echo.
echo Current directory: %CD%
echo.
pause

echo.
echo Removing NvDirectMode proxy DLLs...
if exist "d3d9.dll"      del /f /q "d3d9.dll"
if exist "d3d10.dll"     del /f /q "d3d10.dll"
if exist "d3d11.dll"     del /f /q "d3d11.dll"
if exist "dxgi.dll"      del /f /q "dxgi.dll"
if exist "opengl32.dll"  del /f /q "opengl32.dll"

echo Removing NvApiProxy DLLs...
if exist "nvapi.dll"     del /f /q "nvapi.dll"
if exist "nvapi64.dll"   del /f /q "nvapi64.dll"

echo Removing NvDirectMode config files...
if exist "3DVision_Config.xml" del /f /q "3DVision_Config.xml"

echo Removing NvDirectMode runtime files...
if exist "nvdirectmode_crash.dmp" del /f /q "nvdirectmode_crash.dmp"
REM 0.5.0: all logs live under wiz3D_Logs\ (nvdirectmode_proxy.log,
REM NvApiProxy.log, shader dumps, hunter captures). One rmdir cleans
REM up every log/dump wiz3D produced for this game folder.
if exist "wiz3D_Logs" rmdir /s /q "wiz3D_Logs"
REM Legacy pre-0.5.0 paths — remove if a user is upgrading from an
REM older build that still wrote these next to the exe.
if exist "nvdirectmode_proxy.log" del /f /q "nvdirectmode_proxy.log"
if exist "NvApiProxy.log"         del /f /q "NvApiProxy.log"
if exist "Logs"                   rmdir /s /q "Logs"

REM --- EOSStub handling (only ships in the 3d-vision-direct/dx11/x86 leaf;
REM the 'if exist' guards make this a silent no-op in every other leaf) ---
REM
REM EOSStub is a do-nothing replacement for EOSSDK-Win32-Shipping.dll shipped
REM alongside NvDirectMode DX11 x86 to disable Epic Online Services and its
REM overlay, which crashes Simulated Reality weaving. The stub REPLACES the
REM game's real EOSSDK-Win32-Shipping.dll, so removing our copy now leaves
REM the game without any EOS DLL - either broken or in whatever state the
REM user's own backup (.bak) is in.
REM
REM Policy:
REM  - If EOSStub.log exists, the stub was ACTIVELY USED. Removing the stub
REM    could break the game (it's currently relying on the stub). Leave the
REM    DLL in place; just clean up the log.
REM  - If EOSStub.log doesn't exist, the stub was never activated. Safe to
REM    remove.
REM  - Either way, point the user at their .bak backup / Steam-verify to
REM    restore original EOS if they want it back.
set _EOSSTUB_USED=0
REM 0.5.0: EOSStub.log now lives under wiz3D_Logs\. Legacy path checked
REM too in case the folder was already cleaned above.
if exist "wiz3D_Logs\EOSStub.log" set _EOSSTUB_USED=1
if exist "EOSStub.log"            set _EOSSTUB_USED=1

if "%_EOSSTUB_USED%"=="1" (
    echo.
    echo EOSStub was previously in use ^(EOSStub.log present^).
    echo Leaving EOSSDK-Win32-Shipping.dll in place so the game keeps running.
    echo To fully revert EOS, restore your EOSSDK-Win32-Shipping.dll.bak
    echo backup, or Steam-verify / re-install the game.
) else (
    if exist "EOSSDK-Win32-Shipping.dll" (
        echo.
        echo EOSStub was never activated ^(no EOSStub.log^) - removing.
        echo If you need EOS back, restore your EOSSDK-Win32-Shipping.dll.bak
        echo backup, or Steam-verify / re-install the game.
        del /f /q "EOSSDK-Win32-Shipping.dll"
    )
)

REM Always clean up the log itself (safe either way). The wiz3D_Logs
REM path is already handled by the rmdir earlier; this cleans the
REM legacy next-to-exe copy for backwards-compatible uninstalls.
if exist "EOSStub.log" del /f /q "EOSStub.log"
set _EOSSTUB_USED=

echo.
echo NvDirectMode files removed successfully.
echo.

REM Delete this uninstaller last
del /f /q "%~f0"
