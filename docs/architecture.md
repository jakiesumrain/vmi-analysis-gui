# Architecture

VMI Analysis GUI is a local NiceGUI application for velocity-map imaging data.
The [user manual](manual.pdf) describes panel controls, input formats, and outputs.

## Source and workspace

The source package is `src/vmi_analysis/`. Run `uv sync` from the checkout
to install it in editable mode. Start the app with `uv run vmi-analysis`.

`vmi_analysis.paths` resolves the workspace independently of the installed
package. A checkout uses the repository root. An installed copy uses the startup
directory. Set `VMI_WORKSPACE_DIR` to choose another workspace.

The workspace contains `centroiding/` and `circularization/` for working data.
The GUI stores preview artifacts in `.cache/vmi-analysis/`. NiceGUI stores form
values in `.nicegui/` unless `NICEGUI_STORAGE_PATH` specifies another path.
Keep experiment files and generated outputs outside version control.

## Panels and previews

`gui/app.py` defines tab order, startup, and global settings.
`gui/panels/` contains the forms and result views. Each panel starts with
a title and a short summary. Binning, PES, and Pipeline are immature and
need restructuring.

- Trig Calibration remaps a NumPy detector image and fits selected rings in
  the web process. It can export a Gascooke-compatible calibration.
- Circularize applies a calibration in a subprocess. The corrected image
  retains the input grid and the selected correction center.
- `gui/panels/polar_preview.py` shares display controls between the two panels.
  Each polar preview has separate percentile limits. Trig Calibration has
  crosshair guides. Circularize has a vertical radius guide.
- Circularize displays the polar comparison below **Run**. Its results below
  the console display the corrected Cartesian image in Plotly and the current
  reassignment map when applicable.
- Centroiding and Accumulate also use Plotly. Static image inspection uses
  `gui/assets/image_zoom.js`.

The [design reference](DESIGN.md) describes the visual tokens.
`gui/theme.py` loads the palette and `gui/assets/theme.css`.
The browser loads the packaged assets directly.

## Jobs and resources

Job panels start analysis programs with `sys.executable -m`. A job stores
its command arguments, working directory, and result paths when submitted.
Relative paths resolve from the subprocess working directory.

- `gui/jobs.py` schedules jobs and captures their logs. It starts up to two
  jobs by default. The user can change that limit in **Settings**.
- `gui/resources.py` declares the paths each job reads or writes. Jobs can
  share read access. A write waits if it conflicts with a read or another
  write. Unrelated jobs can use an available slot.
- Pipeline reserves the resources of all four stages for the entire job.
  Its stages run in order. The optional skip check tests whether the declared
  output files exist.
- The MATLAB CPBASEX backend uses an exclusive reservation for its shared
  working data. The backend uses `CPBASEX_DIR` when set. Otherwise it uses
  an existing legacy checkout or the bundled code. Supply the basis data separately.
- `JobView` displays job status and logs. One button starts, stops, or cancels
  the panel job. A browser reload can reattach to an active job by name.
- Cancel and shutdown stop subprocess trees. On Windows the manager uses
  `taskkill /T /F`. On POSIX it signals the process group.

**Keep running on disconnect** defaults to off. When enabled, it leaves the
server available after all browsers disconnect. When it is off, the server
waits two seconds by default after the last browser disconnects and checks
for a reconnect before shutting down. **Stop server** in Settings shuts down
the server and cancels jobs for all browsers.

## Settings

`centroiding/options.py` defines Centroiding defaults and CLI serialization.
`gui/settings.py` imports supported legacy TOML values once. It reads the
legacy file without changing it. GUI jobs pass explicit CLI arguments.

`gui/workflow.py` exports and restores panel settings in one JSON file.
A workflow stores paths and settings. Save a Trig Calibration session
separately to preserve the applied processing, selected band, masks, and fits.

## Verification

Run the Python suite from the checkout.

```sh
uv run python -m unittest discover -s tests -v
```

The suite covers unit and integration tests. Browser checks and fixtures
are in `tests/browser/`. See the comments at the top of each check for
the fixture and execution instructions.
