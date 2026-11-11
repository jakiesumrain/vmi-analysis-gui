[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$PythonEmbedZip,
    [Parameter(Mandatory)][string]$VCRedistX64,
    [string]$InnoCompiler,
    [switch]$PrepareOnly
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Program exited with code $LASTEXITCODE"
    }
}

$PythonEmbedZip = (Resolve-Path -LiteralPath $PythonEmbedZip).Path
$VCRedistX64 = (Resolve-Path -LiteralPath $VCRedistX64).Path
if (!$PrepareOnly) {
    if (!$InnoCompiler) { throw 'Specify the path to Inno Setup ISCC.exe.' }
    $InnoCompiler = (Resolve-Path -LiteralPath $InnoCompiler).Path
}
$pythonHash = '90b4e5b9898b72d744650524bff92377c367f44bd5fbd09e3148656c080ad907'
if ((Get-FileHash -LiteralPath $PythonEmbedZip -Algorithm SHA256).Hash -ne $pythonHash) {
    throw 'Use the official Python 3.13.14 Windows x64 embeddable ZIP.'
}

Push-Location (Join-Path $PSScriptRoot '../..')
try {
    $readVersion = 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])'
    $version = Invoke-Checked uv @('run', '--no-sync', 'python', '-c', $readVersion)
    if ($version -notmatch '^(\d+\.\d+\.\d+)(?:(a|b|rc)(\d+))?$') {
        throw "Unsupported installer version: $version"
    }
    $releaseVersion = $Matches[1]
    if ($Matches[2]) {
        $phase = @{a = 'alpha'; b = 'beta'; rc = 'rc'}[$Matches[2]]
        $releaseVersion += "-$phase.$($Matches[3])"
    }
    # Keep the staging path short for Inno Setup 6 and deeply nested wheel data.
    $buildId = [guid]::NewGuid().ToString('N').Substring(0, 8)
    $build = Join-Path $PWD ".cache/vmi-analysis/w-$buildId"
    $stage = Join-Path $build 'stage'
    $runtime = Join-Path $stage 'runtime'
    $site = Join-Path $runtime 'Lib/site-packages'
    $output = Join-Path $PWD 'dist/windows'
    New-Item -ItemType Directory -Path $runtime, $output -Force | Out-Null
    Write-Host "Staging installer in $build"

    Expand-Archive -LiteralPath $PythonEmbedZip -DestinationPath $runtime
    Set-Content -LiteralPath (Join-Path $runtime 'python313._pth') -Encoding ascii -Value @(
        'python313.zip', '.', 'Lib\site-packages', 'import site'
    )
    foreach ($component in ('CRT', 'OpenMP')) {
        $dlls = @(Get-ChildItem -LiteralPath (Join-Path $VCRedistX64 "Microsoft.VC143.$component") -Filter '*.dll' -File)
        if ($dlls.Count -eq 0) { throw "Missing Visual C++ $component DLLs" }
        $dlls | Copy-Item -Destination $runtime
    }
    $python = Join-Path $runtime 'python.exe'
    $requirements = Join-Path $build 'requirements.txt'
    $wheelDir = Join-Path $build 'wheel'
    # Picasso declares these for its own GUI, web interface, and video tooling.
    # VMI uses Picasso's fitting routines and finds FFmpeg at runtime.
    $excluded = @('pyqt6', 'pyqt6-qt6', 'pyqt6-sip', 'streamlit', 'pyarrow', 'imageio-ffmpeg')
    $exportArgs = @('export', '--locked', '--no-dev', '--no-emit-project', '--output-file', $requirements)
    foreach ($package in $excluded) {
        $exportArgs += @('--no-emit-package', $package)
    }
    Invoke-Checked uv $exportArgs | Out-Null
    Invoke-Checked uv @('build', '--wheel', '--out-dir', $wheelDir)
    Invoke-Checked uv @('pip', 'install', '--python', $python, '--target', $site,
        '--link-mode', 'copy', '--only-binary', ':all:', '--no-deps', '--require-hashes', '-r', $requirements)
    $wheel = Get-ChildItem -LiteralPath $wheelDir -Filter '*.whl' -File
    Invoke-Checked uv @('pip', 'install', '--python', $python, '--target', $site,
        '--link-mode', 'copy', '--no-deps', $wheel.FullName)

    $compile = 'import py_compile, sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)'
    Invoke-Checked $python @('-B', '-c', $compile, (Join-Path $PSScriptRoot 'sitecustomize.py'),
        (Join-Path $runtime 'sitecustomize.pyc'))
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'launch.py'), (Join-Path $PSScriptRoot 'vmi-analysis.ico') -Destination $stage
    Copy-Item -LiteralPath 'LICENSE', 'NOTICE', 'docs/manual.pdf' -Destination $stage
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'third-party.md') -Destination $stage

    $check = 'import struct, sys; assert struct.calcsize("P") == 8 and sys.dont_write_bytecode; import cv2, h5py, nicegui, numba, scipy, win32api; from picasso import localize, gaussmle, gausslq, io'
    Invoke-Checked $python @('-c', $check)
    if ($PrepareOnly) {
        Write-Host "Prepared payload: $stage"
        return
    }

    $baseName = "VMI-Analysis-$releaseVersion-windows-x64-setup"
    Invoke-Checked $InnoCompiler @('/Qp', "/O$output", "/F$baseName", "/DStage=$stage",
        "/DReleaseVersion=$releaseVersion", (Join-Path $PSScriptRoot 'installer.iss'))
    $installer = Join-Path $output "$baseName.exe"
    $hash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $baseName.exe" | Set-Content -LiteralPath "$installer.sha256" -Encoding ascii
    Write-Host "Installer: $installer"
} finally {
    Pop-Location
}
