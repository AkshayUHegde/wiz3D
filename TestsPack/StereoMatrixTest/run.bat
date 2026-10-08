@echo off
rem Project : wiz3D
rem Copyright (C) Akshay Hegde 2026

rem Builds and runs the StereoMatrix proof (header-only: cl on one .cpp). vcvars64.bat is found by a recursive dir, as
rem vswhere's "Program Files (x86)" parentheses would end an enclosing for/if block.
setlocal
rem VsDevCmd.bat runs vswhere.exe by bare name, which cmd will not resolve from the current directory while
rem NoDefaultCurrentDirectoryInExePath is set; clearing it (scoped by setlocal) also lets the exe launch by name.
set "NoDefaultCurrentDirectoryInExePath="
set "VCVARS="
for /f "delims=" %%i in ('dir /b /s "%ProgramFiles%\Microsoft Visual Studio\vcvars64.bat" 2^>nul') do set "VCVARS=%%i"
if not defined VCVARS (
    echo Could not find vcvars64.bat under "%ProgramFiles%\Microsoft Visual Studio".
    exit /b 1
)
call "%VCVARS%" >nul || exit /b 1
pushd "%~dp0"
cl /nologo /W4 /EHsc /Fe:StereoMatrixTest.exe StereoMatrixTest.cpp || (popd & exit /b 1)
".\StereoMatrixTest.exe"
set RC=%ERRORLEVEL%
popd
exit /b %RC%
