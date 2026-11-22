# Agent-first overhaul plan

**Starting point** — `feature/agent-panel` at `534c219`. The embedded Agent
can propose and track TPX3 Dump and Ion TOF jobs, draft an Ion TOF fit from a
plot selection, and run Dump → TOF with one approval. The panels can adopt
the submitted settings and show the results of Agent runs. The
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

### 2A. Move existing ion support behind panel-owned registrations

Move TPX3 Dump and Ion TOF operation definitions and dispatch out of the
central Agent module. Move Ion TOF-specific prompt context, draft review,
and result navigation out of the Agent chat UI and into the Ion TOF
integration. Register Dump → TOF as a workflow composed of those
operations. Keep the existing GUI and Agent flows on the same validation
and execution paths.

**Acceptance** — The Agent discovers the operations at runtime. A manual
run and an Agent run with the same resolved settings agree. A proposed fit
still opens on the correct plot and remains tied to the original tab and
run. Inputs and results appear in the normal panel after approval, with
later manual edits preserved across reload. Invalid inputs and access to
another workspace are rejected on the server.

### 2B. Prove the extension contract with Dark Frame

Add Dark Frame through its operation registration and normal app setup.
Give the Agent enough structured information to explain the settings,
prepare a run, monitor it, and open its output.

**Acceptance** — Adding Dark Frame requires no edits to the Agent core,
chat UI, or Pi tool definitions. Test discovery, validation, approval,
progress, cancellation, and result handoff. Compare the output with a
manual run using the same resolved settings. The migrated Ion TOF fit is
the corresponding test for an interactive panel.

### 2C. Finish the ion workflow and roll out across panels

Extend the existing Dump → TOF workflow to pause for a GMD2 alignment
decision and resume from that step. Pass explicit input and output
references between operations. Report progress, errors, and results in
the workbench and chat. Roll out in the groups below after the extension
contract passes.

| Rollout group | Initial coverage |
| --- | --- |
| Ion workflows | TPX3 Dump, Ion TOF, and Ion Pipeline. Cover dump handoff, GMD2 alignment and gating, and fit selection. |
| Electron batch work | After Dark Frame, add GMD Map, Accumulate, Centroiding, Binning, and Circularize. Expose setup, submission, monitoring, and artifacts. |
| Interactive electron work | Trig Calibration and PES. Expose selected bands, fit windows, overlays, and results. |
| Previews and streaming | Movie and Live TPX3. Expose preview controls and useful bounded summaries. Keep camera acquisition controls outside this scope. |
| Electron Pipeline | Inventory its current behavior and agree on its workflow contract before exposing orchestration controls. |

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
