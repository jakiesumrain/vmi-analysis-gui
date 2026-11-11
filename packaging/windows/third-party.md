# Third-party components in the Windows installer

- The private runtime includes CPython 3.13.14. See `runtime/LICENSE.txt`
  for the Python license. Source is available from
  [Python.org](https://www.python.org/downloads/release/python-31314/).
- The installer includes app-local Microsoft Visual C++ x64 runtime DLLs.
  The build uses the redistributable files supplied with Visual Studio.
- The package directories under `runtime/Lib/site-packages/` contain the
  licenses supplied with the Python dependencies.
- Movie export uses the FFmpeg executable supplied by the locked
  [imageio-ffmpeg wheel](https://github.com/imageio/imageio-ffmpeg).
  The bundled executable is a Gyan Doshi Windows essentials build.
  See `FFmpeg-build.txt` for its version and configuration.
  See `FFmpeg-GPLv3.txt` for its license.
  The [build release](https://github.com/GyanD/codexffmpeg/releases/tag/9.0.2)
  identifies the upstream build. The
  [FFmpeg source](https://ffmpeg.org/releases/ffmpeg-9.0.2.tar.xz)
  is available from the FFmpeg project.
