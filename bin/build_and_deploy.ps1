# wiz3D - build_and_deploy.ps1
#
# One-shot: builds all three wiz3D solutions for one or both architectures,
# then runs deploy_to_releases.ps1 to copy every freshly-built DLL into the
# appropriate releases/wiz3D/ subfolder.
#
# Configurations match what deploy_to_releases.ps1 ships:
#   S3DDriver.sln    -> Final Release (the shipping config; deploy reads
#                       bin/Final Release/<arch>/)
#   wiz3D-proxy.sln  -> Release (no Final Release config on that solution)
#   NvDirectMode.sln -> Release
#
# Usage:
#   .\bin\build_and_deploy.ps1                       # all slns, both archs
#   .\bin\build_and_deploy.ps1 -Arch Win32           # x86 only
#   .\bin\build_and_deploy.ps1 -Arch x64             # x64 only
#   .\bin\build_and_deploy.ps1 -SkipBuild            # deploy only
#   .\bin\build_and_deploy.ps1 -SkipProxy            # only build S3DDriver.sln (+ NvDirectMode if not skipped)
#   .\bin\build_and_deploy.ps1 -SkipNvDirectMode     # skip the Direct Mode proxies

[CmdletBinding()]
param(
    [ValidateSet('Win32', 'x64', 'both')]
    [string]$Arch = 'both',
    [switch]$SkipBuild,
    [switch]$SkipProxy,
    [switch]$SkipNvDirectMode,
    # Final Release is the default shipping build. 0.4.0 kept zlog live in
    # that config by removing FINAL_RELEASE from FinalRelease.props, so
    # DEBUG_MESSAGE / DEBUG_TRACE sites still emit — users and CI both get
    # the optimised build with log traces. -Diagnostic drops to Release
    # (same codegen, different linker/pdb layout) for local repro work.
    [switch]$Diagnostic
)

$ErrorActionPreference = 'Stop'
$repoRoot         = Split-Path -Parent $PSScriptRoot
$mainSln          = Join-Path $repoRoot 'S3DDriver.sln'
$proxySln         = Join-Path $repoRoot 'wiz3D-proxy\wiz3D-proxy.sln'
$nvDirectModeSln  = Join-Path $repoRoot 'NvDirectMode\NvDirectMode.sln'
$msbuild          = 'C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe'

if (-not (Test-Path $msbuild)) {
    throw "MSBuild not found at $msbuild. Install VS2026 (v145) or edit the path in this script."
}
if (-not (Test-Path $mainSln)) {
    throw "Solution not found: $mainSln"
}

$archs = if ($Arch -eq 'both') { @('Win32', 'x64') } else { @($Arch) }

$slnsToBuild = @($mainSln)
if (-not $SkipProxy -and (Test-Path $proxySln)) {
    $slnsToBuild += $proxySln
}
if (-not $SkipNvDirectMode -and (Test-Path $nvDirectModeSln)) {
    $slnsToBuild += $nvDirectModeSln
}

if (-not $SkipBuild) {
    foreach ($s in $slnsToBuild) {
        $slnName = Split-Path -Leaf $s
        # Final Release is the shipped config (optimised + zlog live after
        # the 0.4.0 FinalRelease.props change). -Diagnostic switches to
        # Release for a less-optimised local repro build. The proxy and
        # NvDirectMode solutions only define Debug/Release, so they stay
        # on Release either way.
        $cfg = if ($s -eq $mainSln) {
            if ($Diagnostic) { 'Release' } else { 'Final Release' }
        } else {
            'Release'
        }
        foreach ($a in $archs) {
            Write-Host ""
            Write-Host "=== Building $slnName  $cfg|$a ===" -ForegroundColor Cyan
            & $msbuild $s "/p:Configuration=$cfg" /p:Platform=$a /m /nologo /verbosity:minimal
            # MSBuild returns nonzero when ANY project fails. Most failing
            # projects are pre-existing test EXEs (boost x64, DXUT D3DX9
            # stubs); the user-facing release DLLs build first and deploy
            # will pick them up. So we just warn rather than abort.
            if ($LASTEXITCODE -ne 0) {
                Write-Host "  (msbuild exit $LASTEXITCODE - continuing; deploy will skip any DLLs that didn't build)" -ForegroundColor Yellow
            }
        }
    }

    # S3DWrapperD3D12 isn't in any solution (lives next to S3DWrapper12/).
    # Side-build it in Release|x64 only — DX12 is 64-bit, and this project
    # has no Final Release config to pick up through S3DDriver.sln.
    $s3dWrapperD3D12 = Join-Path $repoRoot 'S3DWrapper12\S3DWrapperD3D12.vcxproj'
    if ($archs -contains 'x64' -and (Test-Path $s3dWrapperD3D12)) {
        Write-Host ""
        Write-Host "=== Building S3DWrapperD3D12.vcxproj  Release|x64 (side-build) ===" -ForegroundColor Cyan
        & $msbuild $s3dWrapperD3D12 /p:Configuration=Release /p:Platform=x64 /m /nologo /verbosity:minimal
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  (S3DWrapperD3D12 msbuild exit $LASTEXITCODE - DX12 stereo wrapper may be missing from the hd3d/dx12 deploy)" -ForegroundColor Yellow
        }
    }
}

Write-Host ""
Write-Host "=== Deploying ===" -ForegroundColor Cyan
if ($Diagnostic) {
    & (Join-Path $PSScriptRoot 'deploy_to_releases.ps1') -Arch $Arch -MainBinDir 'Release'
} else {
    & (Join-Path $PSScriptRoot 'deploy_to_releases.ps1') -Arch $Arch
}
