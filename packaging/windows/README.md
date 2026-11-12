# Windows x64 installer

The installer contains a private Python runtime, the Python packages used by
VMI Analysis, the Visual C++ runtime DLLs, and the user manual. Run Setup on
64-bit Windows 10 or later. Start **VMI Analysis** from the Start menu.
The app opens in the default web browser at `http://localhost:13517`.
The Start menu launcher runs in the background and writes server diagnostics
to `vmi-analysis-cache\server.log` in the installation directory.

The default experiment workspace is `%USERPROFILE%\VMI Analysis`.
Set `VMI_WORKSPACE_DIR` before starting the app to use another workspace.
Supply the experiment data and calibration files separately. The optional
MATLAB backend requires a MATLAB installation and CPBASEX basis data.
MP4 export uses FFmpeg from the system PATH or the executable specified by
`VMI_FFMPEG_EXE`. When FFmpeg is unavailable, the Movie panel saves an animated GIF.

To uninstall, choose **Stop server** in VMI Analysis **Settings** and confirm.
In **Control Panel**, open **Programs and
Features**. Select **VMI Analysis** and click **Uninstall**. The uninstaller
removes the installed files, shortcuts, and the settings and cache in the
installation directory. It preserves the experiment workspace.

## Build

Use a Windows x64 build machine with `uv` and Inno Setup 6.7 or later.
Run `uv sync --locked` from the checkout. Obtain these build inputs.

- [Python 3.13.14 Windows embeddable package (64-bit)](https://www.python.org/downloads/release/python-31314/).
  The build checks its SHA-256 digest.
- The x64 `VC\Redist\MSVC\<version>\x64` directory from a Visual Studio
  installation that includes the VC143 redistributable files.

```powershell
.\packaging\windows\build.ps1 `
    -PythonEmbedZip 'C:\path\python-3.13.14-embed-amd64.zip' `
    -VCRedistX64 'C:\path\VC\Redist\MSVC\14.44.35112\x64' `
    -InnoCompiler 'C:\path\Inno Setup 6\ISCC.exe'
```

The build can download wheels. Set `UV_OFFLINE=1` when the lockfile inputs
are available in the local cache. The resulting Setup executable installs
from its embedded files. The executable and SHA-256 checksum appear in
`dist/windows/`. The version comes from `pyproject.toml`. Use
`-PrepareOnly` to stage and check the payload before compiling Setup.

Before a public release, test installation and uninstallation on a clean,
disconnected Windows x64 machine. Run a multi-worker job and export a GIF.
With system FFmpeg available, also check MP4 export. Check that uninstall
preserves the experiment workspace.
