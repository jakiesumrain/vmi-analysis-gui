# AGENTS.md

Use the [user manual](docs/manual.pdf) for installation, dataset formats,
panel and CLI options, calibration, and troubleshooting. Its source is
[docs/manual.tex](docs/manual.tex). Read the [Agent-native design](docs/agent-first-design.md)
and [architecture](docs/architecture.md) for the product and implementation.
Use the [panel playbook](docs/panel-development.md) and [GUI design](docs/DESIGN.md)
when developing panels.

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
  arguments. Read result paths from the job snapshot.
- Launch analysis scripts as subprocesses. Use one `JobView.run_button` per
  job panel and match its `job_name` to the submitted job name for reconnect.
  Preserve job snapshots and stop all process trees during server shutdown.
  An approved Agent run records `job.extra["panel_form"]`; the panel adopts it
  through `attach_builder` so its fields match the submitted run.
- Trig Calibration runs interactive fitting in the web process. Use the
  same detector origin in Trig Calibration and Circularize. Circularize
  preserves the input matrix shape and the selected origin.
- Keep `gui/resources.py` reservations in sync with CLI reads and writes.
  Reserve every stage of a multi-stage job for the full run. Only an independent
  standalone GMD Map may write beside standalone Centroiding. Use
  `delete_outputs` for clear actions.
- Register supported operations in the owning panel and `gui/panel_operations.py`.
  Return an unsubmitted job and review summary from preparation. Use the same
  validated builder from the manual form. Keep the fallback ion panels manual.
- Use the shared `image_zoom.js` overlay. Pan, zoom, and cropping affect the
  preview only. Reset restores the original image. Browser regression checks
  are in `tests/browser/`.
- Agent sessions are per account and use the container launcher in sandboxed
  shared mode. Keep workspace writes in the account's mount and give shared
  datasets and analysis records read-only mounts. Keep standard panel jobs in
  Bubblewrap and reviewed Python in a separate run container.
  Manual-only shared mode disables Agent entry points and uses the shared
  service account for panel jobs. See [deployment](docs/shared-server.md).
  Pass API keys only through the child environment; never save them in settings,
  workflows, model configuration, arguments, or logs. Clear key copies and remove
  temporary configuration on startup failure, process exit, stop, and shutdown.
  Agent is an interactive session panel, not a finite `JobView` job.
- Follow the workbench patterns in [GUI design](docs/DESIGN.md) and use the
  UI/UX Pro Max skill for UI design. Keep panel summaries short. Describe
  the task without naming the Python module. Binning is immature and
  needs restructuring.

## Git workflow

Branch from `main` using `feature/<name>` or `fix/<name>`. The owner has
authorized agents to commit, push the branch, and open a PR. Do not merge the
PR or fast-forward local `main`; after opening a PR, wait for the owner to
review and merge. All changes reach `main` through PRs. Run CI only when
requested. Preserve unrelated edits and dataset settings.
If direct GitHub access fails, use `HTTPS_PROXY=http://127.0.0.1:7890`.
