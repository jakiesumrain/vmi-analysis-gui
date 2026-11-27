# Add or extend a panel

Read the [capability matrix](agent-first-design.md#agent-capabilities) first. Decide what the user needs to run, what data the result should expose, and which actions require a current view.

## Build one scientific path

1. **Define the science and the CLI.** Put the analysis in a standalone module under `src/vmi_analysis/`. Set units, defaults, and input validation there or in its shared options. Keep CLI-relative paths relative to its working directory.
2. **Build one job in the panel module.** Accept plain parameters, resolve workspace paths, validate values, and return an unsubmitted `Job`. Store resolved output references and the relevant form values in `job.extra`. Use subprocesses for batch work. Interactive fitting uses the panel's existing in-process path where appropriate.
3. **Declare file access.** Map every actual input, output, and exclusive resource in `gui/resources.py` or on the job's explicit access fields. Include generated sidecars and multi-stage intermediates. For output deletion use `delete_outputs` with the panel's declared files.
4. **Connect the manual form.** Add a `JobView` with the submitted job name, one Run/Stop control, console, and result builder. Use the shared form components and read the job snapshot when opening results. Register the navigation entry in `gui/app.py`.
5. **Register the Agent operation.** Add `agent_operations()` to the panel and wire the provider in `gui/panel_operations.py`. Give `PanelOperation` a stable ID, description, parameters with units, the preparation callback, and the job names and outputs it exposes. The callback maps Agent inputs to the shared job builder and returns the job plus a review summary. Keep panel-specific defaults and validation out of the Agent session code.
6. **Connect approved runs back to the panel.** Set the form snapshot at submission and restore it through `attach_builder` and `adopt_agent_form` where that panel adopts Agent runs. Keep later manual edits. Add an artifact resolver for dynamic outputs or a measurement callback for exact values drawn from saved results.

## Choose the smallest interaction contract

| Task | Contract | Example |
| --- | --- | --- |
| Prepare a batch run from named inputs | `PanelOperation.prepare` | Dark Frame, ion histogram generation |
| Review an action tied to current panel state | `PanelView` with a binding and review hook | Trig Calibration and PES fitting, Ion momenta gates and reconstruction, Live TPX3 receiver control |
| Read current state from a running panel | `PanelInspection` | Summarize the current Live TPX3 preview |
| Measure a completed result | `PanelOperation.measure_result` with job ownership checks | Ion TOF peak or bin measurements |
| Combine jobs with a known handoff | `Job.stages`, `Stage.result`, optional `Checkpoint` | TPX3 Dump → Ion TOF → optional GMD2 gate |

## Small example: Dark Frame

In [`dark.py`](../src/vmi_analysis/gui/panels/dark.py), `dark_frame_job` validates the directory, HDF5 filename pattern, dataset, method, and frame count. Both Run and `dark_frame` call it. It records `dark_npy`, `info`, and the form snapshot. `_prepare_agent` translates scoped paths and formats the approval summary, leaving validation and command construction to the builder. The result view reopens those recorded files, and the attach hook adopts the approved form.

When changing this capability, test a synthetic dark HDF5 fixture through both entry points. Compare the resolved parameters and output arrays. Check that a wrong path or a conflicting write fails or waits at the intended boundary.

## Workflow example: TPX3 Dump → Ion TOF

[`panel_operations.py`](../src/vmi_analysis/gui/panel_operations.py) prepares the dump and passes its planned HDF5 or acquisition collection to Ion TOF. The optional GMD2 gate uses the matching metadata. If automatic alignment is ambiguous, the job waits for the user to pick a shift. Resolve the checkpoint on the existing job. Keep the workflow's reservations and attach each stage result to the regular panel so the user can inspect it there.

## Save and load Workbench settings

Use `persist()` for editable panel settings that should travel in a workflow
file. It registers the control when the Workbench builds the panel. For
settings whose controls appear only after a result opens, register the stored
values in `workflow.result_fields()` and read them when building the result
view. Check that loading them restores the controls without starting a run or
live preview.

## Review checklist

- Run the operation's normal GUI and CLI entry points. Compare equivalent manual and approved Agent results when both are exposed.
- Review parameters before submission and verify decline, invalid inputs, and duplicate approval paths.
- Exercise ownership, cancellation, and competing reads or writes when the change touches a shared workflow.
- Reload the browser during a run and verify status, form adoption, and result paths.
- Update the [capability matrix](agent-first-design.md#agent-capabilities), [manual](manual.tex), and [deployment guide](shared-server.md) when their contracts change.
