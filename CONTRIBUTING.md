# Contributing to vmi-analysis-gui

Thanks for helping. This is a research tool: a VMI photoelectron analysis
pipeline plus a NiceGUI web interface. Changes here affect real experiment
data, so the review bar is explicit.

**The owner reviews and approves every change before merge.** Contributors
propose; the owner disposes. `main` is never written to directly, and squash is
the only enabled merge method.

## Before you start

- Read `AGENTS.md` (pipeline order, constraints, traps) and
  `docs/architecture.md` (GUI design decisions) first.
- For anything non-trivial, open an issue or a **draft pull request** so the
  approach can be agreed before code is written.
- No data is bundled with this repository. Never commit datasets, analysis
  outputs, `.nicegui/`, `.cache/`, or machine-specific paths.

## Set up

```bash
uv sync                 # creates .venv and installs everything (editable)
uv run vmi-analysis     # → http://localhost:3157
```

Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/) are required. Optional:
ffmpeg for MP4 movie output, MATLAB only for the CPBASEX inversion backend.

## Branch and commits

Branch off `main`:

```bash
git fetch origin
git switch -c feature/<short-name> origin/main   # or fix/<short-name>
```

- One topic per branch; keep diffs reviewable.
- Commit in buildable steps; use an imperative subject and explain *why* in the
  body.
- Never commit directly to `main`, and never force-push a branch under review.
- Leave unrelated local edits out of the commit — for example a regenerated
  `uv.lock` — and stage explicit paths.

## Verify locally

There is no CI. Run the checks yourself and paste the evidence in the pull
request:

```bash
uv run python -m unittest discover -s tests -v
```

- GUI changes: start the app and exercise the affected panel; include a
  screenshot or GIF in the PR.
- Image-overlay or picker changes: run the browser checks described in
  `README.md` under "Further reading".
- Numerical changes: describe the inputs used. No dataset is bundled, so state
  whether the input was synthetic or user-supplied.

## Pull requests

- Fill in the PR template (`.github/pull_request_template.md`): why, what,
  test evidence, UI evidence, and backward compatibility.
- Keep the change focused, and call out behavior, CLI, or configuration
  changes.
- Request the owner's review, then wait. Do not merge your own PR.
- Address every review thread with follow-up commits, re-run the checks, and
  re-request review. Rebase on `main` when the branch falls behind.

## Review and merge

The owner reviews and approves. After approval the PR is **squash-merged**, the
head branch is deleted, and local `main` is fast-forwarded. If the owner asks
you to merge after approving, use:

```bash
gh pr merge --squash
git switch main && git fetch origin && git merge --ff-only origin/main
```

## House rules

The highlights that review most often catches (full detail in `AGENTS.md`):

- Application code lives under `src/vmi_analysis/`; do not reintroduce
  root-level launcher shims.
- Analysis scripts stay standalone-runnable (`uv run python -m
  vmi_analysis.<module> --help`), with neutral defaults and no
  machine-specific paths.
- GUI: every panel uses one `JobView.run_button` Run/Stop control, and
  `gui/resources.py` file-access declarations stay in sync with the CLI.
  Jobs run as subprocesses, never imported into the web process.
- Preserve legacy data roots and saved `.nicegui` keys; relative CLI paths
  resolve from the subprocess working directory.
- MATLAB/CPBASEX work stays sequential because it shares
  `examples/image.txt`.
- Update `README.md`, `docs/`, or `AGENTS.md` when behavior or workflow
  changes.

## License

Contributions to the project are licensed under the
[Apache License, Version 2.0](LICENSE). Bundled CPBASEX has a separate
[MIT license](vendor/pbasex-matlab/LICENSE.txt).
