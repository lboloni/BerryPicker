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

A flow normally creates an external setup with two sibling directories:

```text
<setup>/
  expruns/    copied defaults and generated run configurations
  results/    experimental data, models, and executed notebooks
```

The flow points `Config` at these directories with `set_exprun_path()` and
`set_results_path()`, then copies each required experiment family. Copying the
family also copies its default and thus its `input-to-notebook` value.

For a homogeneous family, a generated run can inherit the field from the copied
default. For a mixed family, the generator writes the appropriate override into
the generated run. The same value should then be used for the execution entry:

```python
values = copy.copy(params)
values["input-to-notebook"] = [
    "visual_proprioception/Train_VisualProprioception.ipynb",
    "visual_proprioception/Verify_VisualProprioception.ipynb",
]

_write_exprun(experiment, run, values)

train_entry = {
    "name": f"Train_{run}",
    "notebook": values["input-to-notebook"][0],
    "experiment": experiment,
    "run": run,
    "expruns_path": expruns_path.as_posix(),
    "results_path": results_path.as_posix(),
}
```

The verification step for this exp/run uses
`values["input-to-notebook"][1]`. A generated comparison run usually contains
only its applicable comparison notebook and uses index zero.

The execution portion of the flow passes at least the experiment and run to the
selected notebook. External flows also pass the external exprun and results
paths:

```python
parameters = {
    "experiment": entry["experiment"],
    "run": entry["run"],
    "expruns_path": entry["expruns_path"],
    "results_path": entry["results_path"],
}
```

The notebook loads the resolved configuration with
`Config().get_experiment(experiment, run, ...)`. `Config` derives and creates the
result directory as `<experiment_data>/<experiment>/<run>` (and adds a subrun
component when requested), exposing it as `exp["data_dir"]`. The notebook then
writes its experimental artifacts there. Papermill's executed copy of the
notebook can also be written under the flow's `results` directory.

Using the exp/run field as the source of the execution entry prevents the
generated configuration and the flow's notebook choice from drifting apart.

## Consistency checks

`src/test/test_exprun_notebooks.py` checks that:

- every default and every resolved run has a list-valued
  `input-to-notebook` field;
- every listed path exactly matches an existing notebook below `src`;
- notebook paths in automation configurations exist; and
- notebook references used by flow generators are declared by an exp/run.

Run the checks from the repository root with:

```shell
PYTHONPATH=src python -m unittest src.test.test_exprun_notebooks -v
```
