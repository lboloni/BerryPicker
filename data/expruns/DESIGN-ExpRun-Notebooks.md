# Exp/run notebook entry points

## Purpose

Some exp/runs describe work that is performed by a notebook. For example, a
sensor-processing exp/run can be used first by a training notebook and later by
a verification notebook. The `input-to-notebook` field records these notebook
entry points as part of the exp/run itself.

This makes the relationship available both to a person inspecting an exp/run
and to a flow that generates and executes exp/runs. It also keeps the notebook
choice next to the configuration that the notebook accepts.

`input-to-notebook` is descriptive metadata. Loading an exp/run does not
automatically execute a notebook.

## Field contract

Every resolved exp/run has this field:

```yaml
input-to-notebook:
  - sensorprocessing/Train_Conv_VAE.ipynb
  - sensorprocessing/Verify_Conv_VAE.ipynb
```

The value is a list, including when it is empty. Every item is the POSIX path of
an existing notebook relative to `src`. Therefore the example above identifies
`src/sensorprocessing/Train_Conv_VAE.ipynb` and
`src/sensorprocessing/Verify_Conv_VAE.ipynb`.

A notebook belongs in the list when the exp/run is one of its primary inputs:
the notebook is intended to load that experiment and run and produce, verify,
compare, or display its result. A notebook that only loads the exp/run as a
transitive component of another experiment is not listed. Flow notebooks and
the output notebooks produced by Papermill are not entry points either.

The list order follows the normal workflow. A training or data-production
notebook precedes a verification notebook. Current flows use this order when
selecting the train and verify steps.

An empty list means that the exp/run is a supporting configuration and has no
independent notebook entry point:

```yaml
input-to-notebook: []
```

## Defaults and run overrides

`Config().get_experiment(experiment, run)` merges the experiment-family default
with the run configuration. Run fields override default fields. A system-
dependent configuration, when present, is merged afterward. Consequently,
`input-to-notebook` uses the same inheritance mechanism as every other exp/run
field and is accessed as:

```python
exp["input-to-notebook"]
```

Put the field in `_defaults_<experiment>.yaml` when every run in the family has
the same notebook entry points. Sensor-processing families are the common
example:

```yaml
# data/expruns/sensorprocessing_conv_vae/_defaults_sensorprocessing_conv_vae.yaml
input-to-notebook:
  - sensorprocessing/Train_Conv_VAE.ipynb
  - sensorprocessing/Verify_Conv_VAE.ipynb
```

Families containing different kinds of runs have an empty default and override
it in each runnable configuration. Behavior cloning has separate training,
verification, robot-running, and comparison runs. Visual proprioception has
single-view and multiview runs. Its comparison collections similarly select a
single-view or multiview comparison notebook. For example:

```yaml
# data/expruns/behavior_cloning/bc_lstm_00.yaml
input-to-notebook:
  - behavior_cloning/Train_BehaviorCloning.ipynb
```

Keeping the common case in the family default avoids repeating metadata, while
run overrides make mixed families unambiguous.

## Using the field in a flow

A flow is a notebook that generates exp/runs and executes their notebooks in
an explicit order. The shared helpers are in `src/flow.py`.

`setup_flow(flow_name, families)` creates an external workspace below the
machine-specific `flows_path`:

```text
<flows_path>/<flow_name>/
  expruns/              copied families and generated run configurations
  results/              experimental data and models
  executed-notebooks/   the notebooks executed by Papermill
```

It copies the listed exp/run families from the currently active exp/run path
and points `Config` at the workspace with `set_exprun_path()` and
`set_results_path()`. Copying a family also copies its default and thus its
`input-to-notebook` value.

For a homogeneous family, a generated run inherits the field from the copied
default. For a mixed family, the generator writes the appropriate override
into the generated run. The flow entry is then created from the exp/run with
`flow_entry()`, which resolves the exp/run with `create_data_dir=False` and
takes the notebook from `input-to-notebook`:

```python
values = copy.copy(params)
values["input-to-notebook"] = [
    "visual_proprioception/Train_VisualProprioception.ipynb",
    "visual_proprioception/Verify_VisualProprioception.ipynb",
]
_write_exprun(experiment, run, values)

train_entry = flow_entry(f"Train_{run}", experiment, run, 0, creation_style)
verify_entry = flow_entry(f"Verify_{run}", experiment, run, 1, "exist-ok")
```

A producer entry receives the flow's `creation_style`. The second entry point
of the same exp/run (a verification of a trained model) receives `exist-ok`,
because it reopens the directory of the producer rather than replacing it. A
generated comparison run usually contains only its comparison notebook and
uses index zero.

`run_flow(entries, expruns_path, results_path, notebooks_path)` executes the
entries in order with one overall progress bar. Every notebook receives the
same Papermill parameters:

```python
parameters = {
    "experiment": entry["experiment"],
    "run": entry["run"],
    "creation_style": entry["creation_style"],
    "expruns_path": expruns_path,
    "results_path": results_path,
}
```

This is also the interface for running a stage notebook directly: its cell
tagged `parameters` defines these five names with useful defaults.

The notebook loads the resolved configuration with
`Config().get_experiment(experiment, run, creation_style=creation_style)`.
`Config` derives and creates the result directory as
`<experiment_data>/<experiment>/<run>` (and adds a subrun component when
requested), exposing it as `exp["data_dir"]`. The notebook writes its
experimental artifacts there, and its last cell calls `exp.done()`, which
records `time_done` in the `exprun.yaml` of the result directory.

A failing notebook stops the flow. The flow catches the exception only to run
its final report cell, `display_flow_report()`, which lists the completed and
incomplete stages (a stage is complete when its `exprun.yaml` contains
`time_done`) and links the workspace and the final results. The report cell
then re-raises the exception, so Papermill also marks the flow as failed, and
the executed notebook of the failed stage remains in `executed-notebooks`.

Using the exp/run field as the source of the execution entry prevents the
generated configuration and the flow's notebook choice from drifting apart.

## Consistency checks

`src/test/test_exprun_notebooks.py` checks that:

- every default and every resolved run has a list-valued
  `input-to-notebook` field;
- every listed path exactly matches an existing notebook below `src`;
- notebook references used by flow generators are declared by an exp/run;
- every flow uses the helpers of `src/flow.py` and ends with the report;
- every declared notebook has the five standard parameters and ends with
  `exp.done()` (except `Verify_Demonstration`, which redirects `data_dir` to an
  external import directory); and
- stage and flow notebooks contain no saved outputs.

`src/test/test_flows.py` tests the helpers of `src/flow.py` and the creation
styles of `Config.get_experiment`.

Run the checks from the repository root with:

```shell
PYTHONPATH=src python -m unittest src.test.test_exprun_notebooks src.test.test_flows -v
```
