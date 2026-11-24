# Agent-first overhaul plan

**Current baseline** — Phase 2C has reached the planned panel coverage.
PR #85 added screenshot paste and file drop to the Agent composer.

**Next milestone** — Verify Phase 2C end to end, then link screenshots to
run data in Phase 3. The
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
| Panel registration | Each participating panel registers stable operation IDs, descriptions, typed inputs with units and defaults, and result references. It supplies handlers that reuse its manual validation, job builders, and analysis code. Registration is explicit in app setup. The Agent discovers and describes operations from this registry on demand. |
| Agent core | The core owns discovery, proposals, approval, execution, status, cancellation, and access checks. It uses the shared job scheduler and resolves handlers from the registry. Panel-specific prompts, validation, and routing belong with the registered operation or view. |
| View context | Panels can supply optional hooks for the invoking browser tab's run, trace, selection, and relevant settings, as well as for opening drafts and results. Keep unsaved inputs, submitted settings, and Agent drafts distinct. Tie drafts to their source run and view revision. Apply submitted settings once per run and preserve later user edits. |
| Runs and workflows | Keep approved parameters, dependencies, progress, and outputs in server-owned records. Reuse job IDs and scheduling. Persist workflow checkpoints and run metadata by workspace. Record interruption after a server restart and resume from a verified checkpoint. |
| Results and plots | Provide bounded summaries, measurements, trace metadata, and links to files or exact values. Include axis units, visible bounds, selection, and whether the displayed values were binned or downsampled. |
| Execution and approvals | Show the resolved inputs, parameters, outputs, and steps in an editable proposal. One approval can cover the stated workflow. Pause for unresolved scientific choices. Use the existing workspace permissions and job lifecycle. Route user-authored analyses through the isolated execution environment. |

## Delivery phases

### 0–1. Completed baseline and prototypes

The integrated branch includes Live TPX3 and the GUI-only fallback panels.
The Linux shared-server image passed a non-privileged Python and HDF5 smoke
test. The ion prototypes established contextual fit drafts, one-approval
Dump → TOF, activity reporting, and form adoption. Use these flows as
regression checks when moving their registration and view hooks.

### 2A. Completed panel-owned ion registrations

TPX3 Dump and Ion TOF now own their operation definitions and use shared
validation and job builders for manual and Agent runs. Ion TOF owns its
plot context and fit draft review. Dump → TOF composes the registered
operations and runs with one approval.

**Regression checks** — The Agent discovers the operations at runtime.
Manual and Agent runs with the same resolved settings agree. A proposed
fit opens on the correct plot and stays tied to the original tab and run. Inputs and results appear in the normal panel after approval, with
later manual edits preserved across reload. Invalid inputs and access to
another workspace are rejected on the server.

### 2B. Completed extension contract check with Dark Frame

Dark Frame registers its operation and shares validation, execution, and
output handling with its manual run. Manual and Agent runs produced the
same mean and median dark-frame arrays and statistics.

**Acceptance** — Adding Dark Frame requires no edits to the Agent core,
chat UI, or Pi tool definitions. Test discovery, validation, approval,
progress, cancellation, and result handoff. Compare the output with a
manual run using the same resolved settings. The migrated Ion TOF fit is
the corresponding test for an interactive panel.

### 2C. Finish the ion workflow and roll out across panels

Dump → TOF passes the exact output collection and matching metadata to
Ion TOF. An optional GMD2 gate accepts a specified shift or checks
alignment first. Automatic selection requires each count source's
Pearson and Spearman maxima to agree on the same shift, reach 0.50, and
exceed their runner-up by at least 0.05. These thresholds control when the
workflow asks for confirmation. When the criteria are unmet, the Agent
and Ion TOF panels show the correlations and ask the user to confirm
the shift. A paused run keeps its file reservations and releases its
worker slot. It survives browser reload within the same server process.

Ion Pipeline shares a job builder between the panel and the Agent. The
approval shows the resolved GMD2 gate, shift, fit source, peak windows,
and LUNA settings. The Agent can follow output by completed delay
position. Results include the yield CSV, yield plot, and a ratio CSV
for the first configured peak divided by the second. The Ion Pipeline
panel adopts the approved inputs and shows the same results as a
manual run.

The rollout groups below summarize Phase 2C coverage.

| Rollout group | Initial coverage |
| --- | --- |
| Ion workflows | TPX3 Dump, Ion TOF, and Ion Pipeline. Cover dump handoff, GMD2 alignment and gating, and fit selection. |
| Electron batch work | Dark Frame, GMD Map, Accumulate, Centroiding, Binning, and Circularize share their manual job builders with the Agent. |
| Interactive electron work | Trig Calibration passes the loaded polar image context and selected ring band to the Agent. The panel reviews the draft, runs the same fit as the manual button, and hands accepted calibrations to Circularize. PES also shares its spectrum job builder with the Agent. The Agent reviews fit drafts against the current displayed spectra and runs the same peak fit as the manual button. |
| Previews and streaming | Movie uses its manual preview and export jobs for Agent runs. Live TPX3 exposes read-only summaries from the active browser tab. Camera acquisition controls stay outside this scope. |

**Acceptance** — Verify output handoff, pause and resume at an ambiguous
GMD shift, and cancel from either interface. Follow the workflow after
reloading the browser. For each Agent-enabled panel, document supported
actions and compare equivalent GUI and Agent runs. Check workspace isolation.
Before claiming server-restart recovery, verify that the workflow can
resume from a persisted checkpoint without repeating completed work.

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
the job system. Expose the resulting panel through a trusted operation
adapter that validates its declarative definition. Let users discard a draft,
save a reusable workspace version, or
propose a built-in version through the normal project review process.

**Acceptance** — Ask the Agent to analyze peak drift across several Ion TOF
runs, inspect the script and outputs, generate a temporary panel, and use it
on a second dataset. Reopen the saved version and reproduce the result.
Verify workspace ownership and the cancellation and failed-script paths.

## Release criteria

- The GUI remains usable for manual setup, execution, and inspection.
- A standard panel can join by registering its operations and any view hooks
  through normal app setup. The shared Agent core stays stable.
- Equivalent settings give equivalent scientific results across entry points.
- Approved runs record the acting user, resolved settings, input and output
  references, and relevant app or script versions.
- Progress, failures, and pending decisions stay visible. Browser reload,
  interruption, cancellation, and repeated approval have defined outcomes.
- Local Windows and shared Linux deployments pass their supported workflow
  checks. Test the shared worker and model capabilities in the actual
  deployment environment before enabling user-created panels there.
