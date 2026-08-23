@echo off
rem Build and run the StereoMatrix inverse-correction proof.
rem Header-only target, no project dependencies: cl on one .cpp is enough.
rem
rem vswhere lives under "Program Files (x86)", whose parentheses terminate any
rem enclosing for/if block, so locate vcvars64.bat by recursive dir instead --
rem VS 2017 and newer all install under %ProgramFiles%.
setlocal
rem VsDevCmd.bat:180 pushd's into the vswhere directory and then runs
rem "vswhere.exe" by bare name, which cmd refuses to resolve from the current
rem directory when NoDefaultCurrentDirectoryInExePath is set -- printing a
rem spurious "'vswhere.exe' is not recognized" and (per its own comment on line
rem 170) carrying on regardless. Clearing the variable is scoped to this
rem setlocal, and also lets us launch the built exe by name below.
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
