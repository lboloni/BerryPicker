# Experiment/run design

## Objective

An experiment/run, abbreviated **exp/run** or **exprun**, is the complete,
human-readable configuration for one reproducible unit of research work, such
as training a sensor processor, verifying it, or comparing a set of
regressors. It answers three questions without requiring someone to read
Python code:

1. What configuration should be used?
2. Which notebook performs the work?
3. Where are the resulting data and provenance stored?

An exprun is configuration, not an implementation. Models and training code
remain in Python, orchestration belongs to flows, and generated data belongs
in the result tree. Keeping these responsibilities separate makes experiments
easy to inspect, rerun, archive, and move between machines.

## General principles

- **Configuration is data.** Checked-in YAML describes a run; it does not
  contain Python code.
- **Resolved expruns are complete.** Code accesses fields as `exp["field"]`
  and assumes required values are present and correctly formatted. Invalid
  research configuration should fail rather than be silently repaired.
- **Templates and results are different trees.** Editing a template must not
  modify old results, and generating results must not dirty the source tree.
- **Machine differences stay out of the templates.** Paths and hardware
  details live in the machine-specific settings, not in checked-in runs.
- **Notebook paths are metadata.** An exprun declares its entry points, while
  loading the exprun has no execution side effect.
- **Provenance travels with results.** The resolved configuration saved into
  a result directory records what produced that directory.

## Settings

The exp/run framework is implemented by `src/exp_run_config.py`. Every entry
point sets the project name before the first use of `Config`:

```python
from exp_run_config import Config
Config.PROJECTNAME = "BerryPicker"
```

`Config()` is a singleton. It reads `~/.config/BerryPicker/mainsettings.yaml`,
whose only field, `configpath`, points to a machine-specific settings file.
These files live in the `Lotzi-BerryPicker-Settings` repository, one per
machine. The settings used by the exp/run framework are:

- `experiment_data`: the root of the result tree;
- `experiment_system_dependent_dir`: the root of the system-dependent
  overlays (see below);
- `flows_path`: where flow workspaces are created;
- `demopacks_path`: where demonstration packs are stored.

`Config().runtime` holds values that are computed at startup and never saved,
most importantly `Config().runtime["device"]`, the torch device.

## Template organization

Built-in templates live in the BerryPicker repository under:

```text
data/expruns/
  <family>/
    _defaults_<family>.yaml
    <run>.yaml
```

The defaults file contains values shared by a family. A run file contains the
values specific to one named run. An optional system-dependent overlay is
read from:

```text
<experiment_system_dependent_dir>/<family>/<run>_sysdep.yaml
```

`Config().get_experiment(experiment, run)` merges these in order: the family
default, then the run, then the overlay. Later values override earlier ones.
The overlay is appropriate for paths or machine characteristics (for example
a camera or USB port), not for changing the scientific meaning of a run.

Two helpers create templates in an external exp/run tree. They refuse to
work while the built-in tree is active:

- `copy_experiment(family, run=None)` copies a run, or the whole family, from
  the built-in tree.
- `create_exprun_variant(family, run, changes, new_run_name)` resolves a
  built-in run, applies `changes`, and writes the result as a new run.

Flows do not use `copy_experiment`; `setup_flow()` copies families from the
currently active tree (see [DESIGN-Flows.md](DESIGN-Flows.md)).

## Configuration and result paths

`Config` exposes separate APIs for the two trees:

- `get_exprun_path()` and `set_exprun_path()` select the configuration
  templates. By default this is the built-in `data/expruns`.
- `get_results_path()` and `set_results_path()` select the result tree. By
  default this is the machine's `experiment_data`.

The old names `set_experiment_path`, `get_experiment_path`, and
`set_experiment_data` raise an exception; callers must use the new names.

## Creating the result directory

A run result lives below:

```text
<results_path>/<experiment>/<run>/             (or .../<run>/<subrun>/)
```

`get_experiment(experiment, run, subrun=None, creation_style="exist-ok",
create_data_dir=True)` resolves the exprun, sets `exp["data_dir"]`, and
prepares the directory according to `creation_style`:

- `exist-ok` reuses the directory, creating it only when absent;
- `version` moves an existing directory to a timestamped backup and starts
  fresh;
- `discard-old` deletes an existing directory and starts fresh.

Any other value raises an exception. `exist-ok` means reuse the directory; it
does not mean that the computation is skipped. A notebook decides for itself
whether an existing result (for example a trained model) can be reused.

Resolving an exprun for inspection uses `create_data_dir=False`. This returns
the exprun, including its `data_dir`, without creating or changing anything.
Flows use it to read the notebook entry points of the runs they queue.

## Notebook entry points

Every resolved exprun has an `input-to-notebook` list. Each entry is the POSIX
path of a notebook relative to `src`:

```yaml
input-to-notebook:
  - sensorprocessing/Train_Conv_VAE.ipynb
  - sensorprocessing/Verify_Conv_VAE.ipynb
```

A notebook belongs in the list when the exprun is one of its primary inputs:
the notebook loads that experiment and run and produces, verifies, compares,
or displays its result. A notebook that only loads the exprun as a component
of another experiment is not listed. Flow notebooks are not entry points.

The list order follows the normal workflow: a training or data-production
notebook precedes a verification notebook. An empty list means that the
exprun is a supporting configuration without its own notebook.

The field is inherited like any other field. Put it in the family default
when every run has the same entry points, as in the sensor-processing
families. Families containing different kinds of runs, such as
`behavior_cloning` and `visual_proprioception`, have an empty default and
override it in each run. Flows select notebooks only through this field.

## Stage notebook contract

A notebook listed in some `input-to-notebook` is a **stage notebook**. It can
be run directly or by a flow, with the same interface. It has exactly one
cell tagged `parameters`, defining at least:

```python
creation_style = "exist-ok"
expruns_path = None   # if not None, an external exp/run tree
results_path = None   # if not None, an external result tree
experiment = "sensorprocessing_conv_vae"
run = "sp_vae_128"
```

Alternative runs are listed as commented assignments. The next cell points
`Config` at `expruns_path` and `results_path` when they are given, and loads
the primary exprun with
`Config().get_experiment(experiment, run, creation_style=creation_style)`.
Other expruns loaded as components (the sensor processor of a regressor, the
robot, a demonstration) use the default `exist-ok`.

The last cell of the notebook calls `exp.done()` on the primary exprun (see
below). The one exception is `demonstration/Verify_Demonstration.ipynb`,
which redirects `data_dir` to an external import directory, where `done()`
would write a file.

## Result provenance and completion

The result directory contains an `exprun.yaml` copy of the resolved
configuration alongside the generated data. It is written when the directory
is created, and `time_started` is recorded at that point. Timers
(`exp.start_timer(name)` and `exp.end_timer(name)`) add their start, end, and
duration to it.

Calling `exp.done()` adds `time_done` and saves. It is called only after the
notebook has produced every result it declares. The marker therefore means
that the stage completed, not merely that its directory exists.
`Train_RCCO` also has it set by the training recipe after a successful
export.

Known limitations:

- When two entry points share an exprun (a training and its verification),
  they share one `exprun.yaml`, so `time_done` does not tell them apart.
- Under `exist-ok`, a `time_done` from an earlier successful run remains. This
  is intended, since the result is reused.
- `done()` on an existing directory rewrites `exprun.yaml` from the current
  configuration, which drops the earlier `time_started` and timer fields.

## Relationship to flows

An exprun describes one unit of work. A flow generates or selects several
such units, isolates their configuration and result trees, executes their
declared notebooks, and reports on the results. That orchestration contract
is defined in [DESIGN-Flows.md](DESIGN-Flows.md).

## Consistency checks

`src/test/test_exprun_notebooks.py` checks that:

- every default and every resolved run has a list-valued
  `input-to-notebook` field whose entries are existing notebooks below `src`;
- every stage notebook has the standard parameters and ends with
  `exp.done()`, except the notebooks listed in `NOT_DONE`;
- notebook references in flow notebooks are declared by a checked-in exprun;
- every flow uses the helpers of `src/flow.py` and ends with the report; and
- stage and flow notebooks contain no saved outputs or execution counts.

`src/test/test_flows.py` tests the creation styles and the flow helpers. Run
the checks from the repository root with:

```shell
PYTHONPATH=src python -m unittest src.test.test_exprun_notebooks src.test.test_flows -v
```
