@echo off
echo ============================================
echo  Wiz3D Uninstaller - OpenGL (x64)
echo ============================================
echo.
echo This will remove ONLY wiz3D files from the current directory.
echo No game files will be modified or deleted.
echo.
echo Current directory: %CD%
echo.
pause

echo.
echo Removing wiz3D proxy and wrapper files...
if exist "opengl32.dll" del /f /q "opengl32.dll"
if exist "S3DWrapperOGL.dll" del /f /q "S3DWrapperOGL.dll"
if exist "ZLOg.dll" del /f /q "ZLOg.dll"

echo Removing legacy files from older wiz3D installs (OGL no longer ships these)...
if exist "S3DAPI.dll" del /f /q "S3DAPI.dll"
if exist "S3DUtils.dll" del /f /q "S3DUtils.dll"
if exist "nvapi64.dll" del /f /q "nvapi64.dll"
if exist "BaseProfile.xml" del /f /q "BaseProfile.xml"
if exist "CommunityProfile.xml" del /f /q "CommunityProfile.xml"
if exist "OutputMethods" rmdir /s /q "OutputMethods"

echo Removing wiz3D config files...
if exist "wiz3D_Config.xml" del /f /q "wiz3D_Config.xml"

echo Removing wiz3D runtime files...
if exist "Statistic.xml" del /f /q "Statistic.xml"
if exist "wiz3D_proxy.log" del /f /q "wiz3D_proxy.log"
if exist "OpenGL32Proxy.log" del /f /q "OpenGL32Proxy.log"
if exist "OpenGLQuadBufferStereo.log" del /f /q "OpenGLQuadBufferStereo.log"
if exist "wiz3D_opengl32_proxy.log" del /f /q "wiz3D_opengl32_proxy.log"
if exist "wiz3D_ogl_diag.log" del /f /q "wiz3D_ogl_diag.log"
if exist "wiz3D_ogl_crash.dmp" del /f /q "wiz3D_ogl_crash.dmp"
if exist "wiz3d.dll" del /f /q "wiz3d.dll"
if exist "NvApiProxy.log" del /f /q "NvApiProxy.log"
if exist "wiz3D_crash.dmp" del /f /q "wiz3D_crash.dmp"

echo.
echo Wiz3D files removed successfully.
echo.

REM Delete this uninstaller last
del /f /q "%~f0"
