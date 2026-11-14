# AGENTS.md

Use the [user manual](docs/manual.pdf) for installation, dataset formats,
panel and CLI options, calibration, and troubleshooting. Its source is
[docs/manual.tex](docs/manual.tex). See [architecture](docs/architecture.md)
and [GUI design](docs/DESIGN.md) for implementation context.

## Development

- Keep application code under `src/vmi_analysis/`, with CSS and JavaScript
  under `gui/assets`. Keep CLI scripts standalone with neutral defaults.
  Launch from the checkout with `uv run vmi-analysis`.
- Run `uv sync` before commands and tests. Run the local suite with
  `uv run python -m unittest discover -s tests -v`.
- Use `vmi_analysis.paths` for workspace locations. Put disposable artifacts
  under `.cache/vmi-analysis/`. Preserve user data, existing basis files,
  saved `.nicegui` keys, legacy data roots, and CLI-relative path semantics.

## Analysis and GUI invariants

- Centroiding uses a supplied alpha to scale energy coordinates only. Keep
  intensity and the CSV intensity column unchanged. Xe fits use pixel² and
  report a suggested alpha. GMD weighting is a per-frame scalar.
- Keep the MATLAB CPBASEX path sequential because it shares
  `examples/image.txt`. Preserve `CPBASEX_DIR` and legacy checkout support.
- `centroiding/options.py` owns defaults and CLI serialization.
  `gui/settings.py` owns GUI persistence and read-only legacy migration.
  TOML is an explicit CLI option. The GUI uses saved fields and passes explicit
  arguments. Pipeline overrides dataset, map, and tuning mode only in its
  submitted snapshot. Read result paths from the job snapshot.
- Launch analysis scripts as subprocesses. Use one `JobView.run_button` per
  job panel and match its `job_name` to the submitted job name for reconnect.
  Preserve job snapshots and stop all process trees during server shutdown.
- Trig Calibration runs interactive fitting in the web process. Use the
  same detector origin in Trig Calibration and Circularize. Circularize
  preserves the input matrix shape and the selected origin.
- Keep `gui/resources.py` reservations in sync with CLI reads and writes.
  Reserve every Pipeline stage for the full run. Only an independent
  standalone GMD Map may write beside standalone Centroiding. Use
  `delete_outputs` for clear actions.
- Use the shared `image_zoom.js` overlay. Pan, zoom, and cropping affect the
  preview only. Reset restores the original image. Browser regression checks
  are in `tests/browser/`.
- Keep panel summaries short. Describe the task without naming the Python
  module. Binning, PES, and Pipeline are immature and need restructuring.

## Git workflow

Branch from `main` using `feature/<name>` or `fix/<name>`. The owner has
authorized agents to commit, push the branch, and open a PR. Do not merge the
PR or fast-forward local `main`; after opening a PR, wait for the owner to
review and merge. All changes reach `main` through PRs. Run CI only when
requested. Preserve unrelated edits and dataset settings.
If direct GitHub access fails, use `HTTPS_PROXY=http://127.0.0.1:7890`.
