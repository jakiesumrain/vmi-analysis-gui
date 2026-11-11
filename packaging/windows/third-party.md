# Third-party components in the Windows installer

- The private runtime includes CPython 3.13.14. See `runtime/LICENSE.txt`
  for the Python license. Source is available from
  [Python.org](https://www.python.org/downloads/release/python-31314/).
- The installer includes app-local Microsoft Visual C++ x64 runtime DLLs.
  The build uses the redistributable files supplied with Visual Studio.
- The package directories under `runtime/Lib/site-packages/` contain the
  licenses supplied with the Python dependencies.
