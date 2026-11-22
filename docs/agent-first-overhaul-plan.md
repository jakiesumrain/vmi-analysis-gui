# Agent-first overhaul plan

**Starting point** — `feature/agent-panel` at `17cd20a`. The embedded Agent
can already explain, propose, and track approved TPX3 Dump and Ion TOF jobs,
including GMD2 gating and peak fitting. Keep that workflow working while
extending it to the rest of the workbench. The
[Agent-first design guide](agent-first-design.md) defines the product goals.

## Scope

Users should be able to direct and monitor the standard analysis panels in
natural language, hand work back to a panel, attach a screenshot for interpretation,
run an analysis outside the built-in panels, and turn a repeatable analysis
into a temporary or saved panel. The standalone ion fallback panels are
GUI-only and outside the Agent rollout. Prioritize the embedded Agent. External
agent access through CLI and MCP is deferred. Existing scientific CLI
commands remain part of the app's runtime and test surface.

## Architecture

| Area | Design |
| --- | --- |
| Panel operations | Give operations stable names and typed inputs, units, defaults, and result shapes. Reuse existing job builders and scientific functions. Share validation, workspace checks, and resource reservations across GUI and Agent requests. |
| View context | Capture the invoking browser tab's panel, run, trace, selection, and relevant settings. Distinguish unsaved form values from submitted settings and Agent drafts. Tie each draft to its source run and view revision. |
| Runs and workflows | Keep approved parameters, dependencies, progress, and outputs in server-owned records. Reuse job IDs and scheduling. Persist workflow checkpoints and run metadata by workspace. Record interruption after a server restart and resume from a verified checkpoint. |
| Results and plots | Provide bounded summaries, measurements, trace metadata, and links to files or exact values. Include axis units, visible bounds, selection, and whether the displayed values were binned or downsampled. |
| Execution and approvals | Show the resolved inputs, parameters, outputs, and steps in an editable proposal. One approval can cover the stated workflow. Pause for unresolved scientific choices. Use the existing workspace permissions and job lifecycle. Route user-authored analyses through the isolated execution environment. |

## Delivery phases

### 0. Reconcile the branch and establish a baseline

Compare `17cd20a` with current `main` before making the first code change.
The locally tracked `main` includes Live TPX3 and the standalone ion
fallback panels, which are absent from the Agent branch. Bring those features
into the overhaul baseline, reconcile overlapping ion and sidebar code, and
record the resulting baseline commit. Verify the fallback panels through their
manual GUI workflows.

**Acceptance** — Existing GUI workflows and the Agent's ion workflow pass
on the integrated branch. Verify the shared-server image build and a
non-privileged Python and HDF5 run on Linux. Capture representative results
for subsequent GUI and Agent parity tests.

### 1. Build the shared operation interface and prove a contextual fit

Extract the reusable interface from the current TPX3 Dump and Ion TOF
integration. First connect the active Ion TOF selection to **Ask Agent**.
Support reading the spectrum, placing a draft fit window on the plot, editing
that draft from the plot or form, and submitting it through the scheduler.

**Acceptance** — Select a peak, ask the Agent to fit it, drag a proposed
bound, approve, and inspect the result in Ion TOF. Run the same resolved
inputs manually and compare the scientific output. A proposal from one tab
stays attached to its original run when the user changes tabs or results.

### 2. Add workflow execution and roll out across panels

Move cross-panel sequencing into server-owned workflows. Each step receives
explicit input references. Report progress, actionable errors, and result
links in both the workbench and chat. A waiting workflow presents the plot
or settings the user needs to make a decision.

| Rollout group | Initial coverage |
| --- | --- |
| Ion workflows | TPX3 Dump, Ion TOF, and Ion Pipeline. Cover dump handoff, GMD2 alignment and gating, and fit selection. |
| Electron batch work | GMD Map, Dark Frame, Accumulate, Centroiding, Binning, and Circularize. Expose setup, submission, monitoring, and artifacts. |
| Interactive electron work | Trig Calibration and PES. Expose selected bands, fit windows, overlays, and results. |
| Previews and streaming | Movie and Live TPX3. Expose preview controls and useful bounded summaries. Keep camera acquisition controls outside this scope. |
| Electron Pipeline | Inventory its current behavior and agree on its workflow contract before exposing orchestration controls. |

**Acceptance** — Approve a TPX3 Dump → Ion TOF workflow in one review. Verify
output handoff, pause and resume at an ambiguous GMD shift, and cancel from
either interface. Follow the workflow after reloading the browser. For each
Agent-enabled panel, document supported actions and compare equivalent GUI and Agent runs.
Verify that another workspace cannot read or alter the run.

### 3. Add screenshot interpretation

Accept a screenshot attached by the user and allow an optional link to an
authorized run. Provide panel captures for display inspection. Use a
vision-capable model for images and the app's data for exact measurements.
Explain when an interpretation is based on visible features alone.

**Acceptance** — With a screenshot of an Ion TOF result, identify a visible
feature and verify a claimed position or width against the linked data.
Verify the upload flow with and without a run link, and show a clear
capability message when the configured model cannot accept images.

### 4. Add ad hoc analysis and temporary panels

Run user-directed Python analyses with the app's locked dependencies in the
user's isolated workspace. Capture the script, inputs, parameters, version,
and generated files. Show those artifacts in the workbench with run status
and logs.

Define temporary panels with a versioned description of fields, validation,
script entry point, and result views. Have the app render the description
using its existing components. Run the script in an isolated worker through
the job system. The same operation interface exposes the resulting panel to
the Agent. Let users discard a draft, save a reusable workspace version, or
propose a built-in version through the normal project review process.

**Acceptance** — Ask the Agent to analyze peak drift across several Ion TOF
runs, inspect the script and outputs, generate a temporary panel, and use it
on a second dataset. Reopen the saved version and reproduce the result.
Verify workspace ownership and the cancellation and failed-script paths.

## Release criteria

- The GUI remains usable for manual setup, execution, and inspection.
- Equivalent settings give equivalent scientific results across entry points.
- Approved runs record the acting user, resolved settings, input and output
  references, and relevant app or script versions.
- Progress, failures, and pending decisions stay visible. Browser reload,
  interruption, cancellation, and repeated approval have defined outcomes.
- Local Windows and shared Linux deployments pass their supported workflow
  checks. Test the shared worker and model capabilities in the actual
  deployment environment before enabling user-created panels there.
