# build_cuda.ps1 — compile dlgo CUDA kernels (gpu/cuda/kernels.cu) to a cubin.
#
# Produces gpu/cuda/kernels.cubin, which the Go package embeds (go:embed) and hands
# to cuModuleLoadData at run time. A cubin is native SASS for one architecture;
# sm_89 = Ada Lovelace (RTX 4070 Ti SUPER / AD103) here. Switch -Arch to build for
# another card (e.g. sm_86 for Ampere, sm_120 for Blackwell).
#
# nvcc needs an MSVC host compiler on Windows; this sets up the VS environment
# via vcvars64.bat (or locates it through vswhere) before invoking nvcc.

param(
    [string]$Arch = "sm_89",
    [string]$Out  = ""
)

$ErrorActionPreference = 'Stop'
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $Out) { $Out = Join-Path $here 'kernels.cubin' }
$src = Join-Path $here 'kernels.cu'

function Find-VcVars {
    $candidates = @(
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles}\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    )
    foreach ($c in $candidates) { if (Test-Path $c) { return $c } }
    return $null
}

$vcvars = Find-VcVars
if (-not $vcvars) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vswhere) {
        $inst = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($inst) {
            $cand = Join-Path $inst 'VC\Auxiliary\Build\vcvars64.bat'
            if (Test-Path $cand) { $vcvars = $cand }
        }
    }
}
if (-not $vcvars) { throw "Could not find vcvars64.bat (Visual Studio C++ tools)." }
Write-Output "using $vcvars"

$cmd = "call `"$vcvars`" >nul 2>&1 && nvcc -cubin -O3 --use_fast_math -arch=$Arch -o `"$Out`" `"$src`""
$log = & cmd /c $cmd 2>&1
$log | Out-File -Encoding utf8 (Join-Path $here 'nvcc.log')

if ($LASTEXITCODE -ne 0) {
    Write-Output $log
    throw "nvcc failed (see nvcc.log)"
}

$size = (Get-Item $Out).Length
Write-Output "cubin OK -> $Out ($size bytes, arch=$Arch)"
