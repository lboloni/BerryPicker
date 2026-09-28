# Experiment-flow design

## Objective

A flow turns a set of expruns into a repeatable research procedure. It
prepares an isolated workspace, generates the expruns it needs, runs the
existing stage notebooks in an explicit order, stops on failure, and leaves
behind both the scientific products and the executed notebooks needed to
understand what happened.

The goal is not to build a general workflow engine. A BerryPicker flow is a
small, readable notebook for a known procedure, such as training every
sensor processor and regressor of a visual proprioception study. It should
make the complete procedure easy to start, inspect, rerun, and audit without
hiding research decisions behind dynamic scheduling machinery.

## General principles

- **The procedure is explicit.** Queue order and membership come from short
  generator and queue-building code in the flow notebook.
- **Stage notebooks remain independently runnable.** A flow supplies the
  same parameters a user would supply directly; it does not create a second
  implementation of an experiment.
- **Each execution is isolated.** Generated expruns, results, and executed
  notebooks share one flow workspace outside the source tree.
- **Inspection has no side effects.** Queue construction resolves expruns
  without creating their result directories.
- **Failures remain failures.** A failed stage stops the queue and is
  re-raised after the final report has recorded partial progress.
- **Completion is evidence-based.** A stage is complete only when its result
  provenance contains `time_done`.
- **The design stays deliberately small.** There is no DAG database,
  component registry, automatic implementation selection, or hidden retry.

The configuration and result conventions used below are defined in
[DESIGN-ExpRun.md](DESIGN-ExpRun.md). The helpers are in `src/flow.py`.

## Isolated workspace

`setup_flow(flow_name, families)` creates the following workspace below the
machine-specific `flows_path`:

```text
<flows_path>/<flow_name>/
  expruns/              copied families and generated expruns
  results/              experimental data and models
  executed-notebooks/   the stage notebooks as executed by Papermill
```

It copies each listed family from the currently active exprun path, not from
a hard-coded built-in location, and then points `Config` to the workspace
with `set_exprun_path()` and `set_results_path()`. It returns
`(expruns_path, results_path, notebooks_path)`. Copying a family also copies
its default and thus its `input-to-notebook` value.

Each flow lists the families it needs explicitly: the families it generates
runs into, and the families its stages load as components (for example
`robot_al5d` and `demonstration`). Most flows then import a demonstration
pack with `import_demopack()`, which copies the demonstrations into
`results/demonstration/<demopack>` and returns their split into training and
evaluation groups.

Generated data and executed notebooks never belong in the source repository.

## Building the execution queue

Unlike a flow over a checked-in collection, a BerryPicker flow **generates**
most of its expruns, because their training data depends on the imported
demopack. Each generator function writes one exprun YAML into the workspace
and returns the flow entry for it:

```python
def generate_sp_conv_vae(params, exp_name, run_name):
    val = {...}
    path = pathlib.Path(Config().get_exprun_path(), exp_name, run_name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(val, f)
    return flow_entry("Train_SP_Conv-VAE", exp_name, run_name, 0, creation_style)
```

`flow_entry(name, experiment, run, index, creation_style)` resolves the
exprun with `create_data_dir=False` and takes the notebook from
`exp["input-to-notebook"][index]`. The generated YAML is therefore the only
place where the notebook is chosen: a run in a homogeneous family inherits
it from the copied default, while a run in a mixed family such as
`visual_proprioception` has it written by the generator. The entry contains
only a display name, the experiment, the run, the notebook, and the creation
style; a generator may add further keys for its own bookkeeping.

The usual order is:

1. train each sensor processor;
2. train each regressor or controller on top of it;
3. verify, where the flow includes verification; and
4. run the comparisons.

A flow may also have several phases. `Flow_FilteredVsUnfiltered` trains its
base regressor, tunes the filter parameters in the flow notebook on that
regressor's predictions, and only then generates and queues the filtered
runs, their verification, and the comparison.

## Creation styles and data ownership

Every stage receives one of the three creation styles of
[DESIGN-ExpRun.md](DESIGN-ExpRun.md). The flow's `creation_style` parameter
applies to producer stages: trainings, the verification of a separate
verification run (as in behavior cloning), and comparisons.

The second entry point of the same exprun, such as `Verify_Conv_VAE` after
`Train_Conv_VAE` on one run, always receives `exist-ok`. It reopens the
directory of its producer; `discard-old` or `version` would remove or move
the model it is supposed to verify.

Trained models are authoritative inputs; verification figures and
comparisons are reproducible derived products. A comparison exprun owns its
own result directory and must not modify the runs it reads.

## Execution and failure behavior

`run_flow(entries, expruns_path, results_path, notebooks_path)` executes the
queue in order through `run_notebook()`. Each stage receives the same
Papermill parameters:

- `experiment`
- `run`
- `creation_style`
- `expruns_path`
- `results_path`

The notebook runs with the `berrypicker` kernel and its own directory as the
working directory. Its executed copy is written to
`executed-notebooks/<notebook>_<experiment>_<run>.ipynb`. One overall
progress bar shows the queue total, the current stage, and the number of
stages remaining; Papermill's cell-level progress remains local to the
current stage.

Exceptions are not suppressed. The top-level flow catches a stage exception
only long enough to execute its final report cell:

```python
flow_error = None
try:
    run_flow(entries, expruns_path, results_path, notebooks_path)
except Exception as error:
    flow_error = error
```

The report records partial progress, after which the original exception is
re-raised. Papermill therefore marks the top-level flow as failed, and the
executed notebook of the failed stage remains available as the diagnostic
artifact.

## Completion and final report

Every top-level flow notebook ends with a report cell:

```python
final_results_path = pathlib.Path(results_path, <comparison experiment>, <comparison run>)
report = display_flow_report(
    entries, results_path, final_results_path, flow_error=flow_error)
if flow_error is not None:
    raise flow_error
if not report["all-results-present"]:
    raise Exception("Flow finished without all expected stage results.")
```

`get_flow_report()` checks every queued stage at
`results/<experiment>/<run>/exprun.yaml`. A stage is complete when that file
contains `time_done`. The report distinguishes complete, partial, and
no-result executions and lists each incomplete stage with its expected
result directory. If execution returns without an exception but a marker is
missing, the final cell raises; incomplete results cannot be presented as
success.

`display_flow_report()` shows clickable absolute paths for the flow
workspace, the complete results tree, and the final-results directory
(normally the directory of the main comparison). It lists every PDF below
the final-results directory as a link, and it can embed selected previews
passed as `preview_files`. The paths remain visible and copyable even if a
notebook frontend refuses to open a local `file:` link.

The checked-in flow notebook contains no saved outputs; the rendered report
belongs to its executed copy.

## Current flows

All flows are in `src/<area>/`. Their `flow_name` parameter selects the
workspace; reusing a name with `exist-ok` reuses previously trained models.

`visual_proprioception/Flow_VisualProprioception.ipynb` trains the selected
single-view sensor processors (Conv-VAE, VGG19, ResNet50, ViT) for latent
sizes 128 and 256, one regressor on top of each, and the comparisons of all
regressors and per latent size. Final results:
`visual_proprioception_collections/vp_comp_flow_all`.

`visual_proprioception/Flow_VisualProprioception_multi.ipynb` extends this to
the multi-view sensor processors (ViT, CNN, and conv encoder backbones with
the five fusion heads), with additional single-view-only and
multi-view-only comparisons. Final results:
`visual_proprioception_collections/flow_vp_comp_all`.

`visual_proprioception/Flow_PtunVsRandProj_128.ipynb` trains two
proprioception-tuned CNN variants (created with `create_exprun_variant`),
regressors on them and on two training-free random-projection encoders, and
compares the four on the held-out `vp_testing` group. Final results:
`visual_proprioception_collections/comp_ptun_vs_randproj_128`.

`visual_proprioception/Flow_FilteredVsUnfiltered.ipynb` trains one regressor
on a training-free encoder, tunes EMA and Kalman filters on its training
predictions, verifies the unfiltered and both filtered runs, and compares
them. It is the cheapest flow to run end to end. Final results:
`visual_proprioception_collections/comp_filter_<encoder>`.

`behavior_cloning/Flow_BehaviorCloning.ipynb` trains and verifies a Conv-VAE
sensor processor, trains and verifies MLP, LSTM, residual LSTM, and LSTM-MDN
controllers on it, and compares them. It also writes a `runbc_<run>` exprun
per controller for `Run_BehaviorCloning.ipynb`, which drives the robot and is
not part of the queue. Final results: `behavior_cloning/_flow_bc_compare`.

`visual_proprioception/MultiFlow_VisualProprioception.ipynb` is not a flow
over stages but a sweep over flows: it runs
`Flow_VisualProprioception.ipynb` once per simulated camera, with a separate
workspace for each. The executed flow notebooks go to
`<flows_path>/multiflow_visualproprioception/executed-notebooks`. A failing
flow stops the sweep.
