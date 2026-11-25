# Agent-first design for VMI Analysis

The embedded Agent is part of the scientific workbench. A user can describe a
task in natural language, inspect and adjust the Agent's proposal in the
relevant panel, and follow the run from either the panel or the conversation.
The user can take over at any point. Build this experience inside the app
first. Defer external-agent integrations until the embedded workflows mature.

## Give supported panels a shared set of operations

Enable Agent control and monitoring for the standard analysis panels. The
standalone ion fallback panels are GUI-only. Each supported panel declares
its operations and describes their inputs, units,
defaults, validation, and outputs. The GUI and the Agent use the same
application operations, scientific code, job scheduler, and workspace
permissions.

- **Batch analysis.** Prepare, validate, submit, monitor, and cancel jobs.
- **Interactive analysis.** Read selections and stage proposed changes on
  the plot and in the form. Let the user adjust them before running the analysis.
- **Live previews.** Inspect the state of the preview, use its available
  controls, and monitor summaries or user-defined conditions such as gaps
  in the frame stream.
- **Results.** Open a specific run, read warnings and measurements, and
  compare its outputs with another run.

Use the context of the view that invoked the Agent. Keep the active run,
selected trace, visible range, and unsaved form values distinct. Attach a
proposal to the specific run and inputs it was prepared for, so navigating
to another result cannot silently change what it applies to.

## Keep execution and progress visible

Show resolved settings, inputs, outputs, and decisions that require the
user's judgment before submission. Let users review and approve a complete
sequence of steps together. If a scientific decision remains unresolved,
pause the workflow and open the relevant diagnostic so the user can decide
and resume.

Keep run state in the app so the panel and the conversation show the same
progress, warnings, and results. An approved run should remain inspectable
after a browser reload. Record the input and output references, resolved
parameters, and app or analysis version so the user can reproduce the result.

## Let the Agent inspect data and images

Expose panel state and structured plot data directly. Include trace labels,
axis units and scales, visible bounds, and the current selection. Link the
displayed trace to its source result when it has been binned or downsampled.

Support screenshots that users capture and attach to the conversation.
A vision-capable model can interpret visible features, annotations, and
warnings. When a screenshot is linked to a run, use its data and settings to
verify quantitative claims. Panel captures can also help diagnose rendering
issues such as clipped labels or overlapping traces.

## Support analyses beyond the existing panels

Let the Agent write and run analysis scripts in the user's workspace. Use the
app's locked Python dependencies for the Agent execution environment. Retain
the script and its version, inputs, resolved parameters, and outputs alongside
the run. Make plots and tables available in the workbench for inspection.

When an analysis becomes repeatable, let the Agent propose a temporary panel
in the current workspace. The proposal defines the form fields, units,
defaults, validation, and result views, and references the analysis script.
Render it with the app's existing components. Execute the script in the user's
analysis environment through the job system. Use the same interface for
inputs, execution, progress, and results as built-in panels, so the Agent can
operate on either kind.

The user can discard the draft, save it as a versioned reusable workspace
panel, or propose it for inclusion as a built-in panel through the project's
normal review and release process.

## Verify the full workflow on Ion TOF

Build on the existing TPX3 Dump and Ion TOF integration. Use equivalent
inputs to verify that a GUI run and an Agent run produce the same scientific
result. Then take an Ion TOF result through the complete path.

1. Select a peak on the spectrum and ask the Agent to fit it. Adjust its
   proposed window on the plot, approve the fit, and inspect the result.
2. Attach a screenshot of the result and, when linked to the run, verify
   the interpretation against the underlying data.
3. Ask the Agent to write a peak-drift analysis across several runs.
4. Turn the script into a temporary panel and run it on a second dataset.
   Verify that the user can review, rerun, and save the panel for reuse.

Apply the same principles as additional panels gain Agent support.
