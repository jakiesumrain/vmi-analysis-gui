# Contributing to vmi-analysis-gui

This repository provides analysis tools for VMI photoelectron data and a
NiceGUI web interface.

**The owner reviews and approves changes before merge.** Agents follow the
owner authorization in [AGENTS.md](AGENTS.md). Use a pull request and a squash
merge for every change to `main`.

## Before you start

- Read [AGENTS.md](AGENTS.md) and [Architecture](docs/architecture.md).
- Open an issue or draft pull request to discuss substantial changes before
  implementation.
- Keep datasets, analysis outputs, `.nicegui/`, `.cache/`, and
  machine-specific paths outside version control.

## Set up

```bash
uv sync                 # creates .venv and installs everything (editable)
uv run vmi-analysis     # → http://localhost:3157
```

Install Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
Install ffmpeg for MP4 output. Install MATLAB if you use the CPBASEX backend.

## Branch and commits

Branch off `main`:

```bash
git fetch origin
git switch -c feature/<short-name> origin/main   # or fix/<short-name>
```

- Keep each branch focused on one topic.
- Keep each commit buildable. Use an imperative subject. Explain the reason
  for the change in the commit body.
- Never commit directly to `main`, and never force-push a branch under review.
- Leave unrelated local edits out of the commit — for example a regenerated
  `uv.lock` — and stage explicit paths.

## Local tests

There is no CI. Run the checks yourself and paste the evidence in the pull
request:

```bash
uv run python -m unittest discover -s tests -v
```

- For GUI changes, start the app and test the affected panel.
  Include a screenshot or GIF in the pull request.
- For image-overlay or picker changes, use the fixtures and browser checks
  in `tests/browser/`. Each check describes its setup in its opening comments.
- For numerical changes, describe the inputs. State whether the input was
  synthetic or user-supplied.

## Pull requests

- Fill in the PR template (`.github/pull_request_template.md`): why, what,
  test evidence, UI evidence, and backward compatibility.
- Keep the change focused, and call out behavior, CLI, or configuration
  changes.
- Request the owner's review, then wait. Do not merge your own PR.
- Address every review thread with follow-up commits, re-run the checks, and
  re-request review. Rebase on `main` when the branch falls behind.

## Review and merge

The owner reviews and approves the pull request. After approval, squash-merge
the pull request and fast-forward local `main`. If the owner asks you to merge,
use these commands.

```bash
gh pr merge --squash
git switch main && git fetch origin && git merge --ff-only origin/main
```

## House rules

The highlights that review most often catches (full detail in `AGENTS.md`):

- Keep application code under `src/vmi_analysis/`. Start the GUI with
  `uv run vmi-analysis`.
- Analysis scripts stay standalone-runnable (`uv run python -m
  vmi_analysis.<module> --help`), with neutral defaults and no
  machine-specific paths.
- Use one `JobView.run_button` for each job panel. Update `gui/resources.py`
  when the CLI changes the paths it reads or writes.
- Run jobs as subprocesses. Trig Calibration runs its interactive tools in
  the web process.
- Preserve legacy data roots and saved `.nicegui` keys. Resolve relative
  CLI paths from the subprocess working directory.
- MATLAB/CPBASEX work stays sequential because it shares
  `examples/image.txt`.
- Update `README.md`, `docs/`, or `AGENTS.md` when behavior or workflow
  changes.

## License

The project uses the [Apache License, Version 2.0](LICENSE).
Bundled CPBASEX uses a separate
[MIT license](vendor/pbasex-matlab/LICENSE.txt).
