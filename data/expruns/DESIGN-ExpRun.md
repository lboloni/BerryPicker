# Experiment/run design in BerryPicker

The exp/run framework is implemented by the ExpRunFlow library
(`exprunflow.exp_run_config`), whose `docs/DESIGN-ExpRun.md` is the
authoritative contract: templates and their merge order, the exprun and
result paths, the creation styles, the `input-to-notebook` field, the stage
notebook contract, `exp.done()`, and `configure_deterministic_run`. This
document describes only how BerryPicker uses it.

## Settings

`src/exp_run_config.py` re-exports the library and sets the BerryPicker
values:

```python
Config.PROJECTNAME = "BerryPicker"
Config.SRC_ROOT = pathlib.Path(__file__).resolve().parent
Config.KERNEL_NAME = "berrypicker"
```

Code imports `from exp_run_config import Config`. Most entry points still
set `Config.PROJECTNAME = "BerryPicker"` themselves; this is redundant but
harmless.

`Config()` reads `~/.config/BerryPicker/mainsettings.yaml`, whose `configpath`
points to a machine-specific settings file. These files live in the
`Lotzi-BerryPicker-Settings` repository, one per machine. Besides the
framework settings (`experiment_data`, `experiment_system_dependent_dir`,
`flows_path`), BerryPicker uses `demopacks_path`, where demonstration packs
are stored.

`Config().runtime["device"]` is the torch device used by all training and
inference code.

## Templates

The built-in templates are in `data/expruns/<family>/`. The overlay is used
for machine characteristics such as a camera or the USB port of a robot.

## Notebook entry points

`input-to-notebook` entries are relative to `src`:

```yaml
input-to-notebook:
  - sensorprocessing/Train_Conv_VAE.ipynb
  - sensorprocessing/Verify_Conv_VAE.ipynb
```

The sensor-processing families have the same entry points for every run, so
the field is in their family default. Families containing different kinds of
runs, such as `behavior_cloning` and `visual_proprioception`, have an empty
default and override it in each run.

## Exceptions to the stage notebook contract

- `demonstration/Verify_Demonstration.ipynb` does not call `exp.done()`: it
  redirects `data_dir` to an external import directory, where `done()` would
  write a file.
- `Train_RCCO` also has `time_done` set by the training recipe after a
  successful export.

## Consistency checks

`src/test/test_exprun_notebooks.py` checks that:

- every default and every resolved run has a list-valued
  `input-to-notebook` field whose entries are existing notebooks below `src`;
- every stage notebook has the standard parameters and ends with
  `exp.done()`, except the notebooks listed in `NOT_DONE`;
- notebook references in flow notebooks are declared by a checked-in exprun;
- every flow uses the helpers of `src/flow.py` and ends with the report; and
- stage and flow notebooks contain no saved outputs or execution counts.

`src/test/test_flows.py` tests the creation styles and the flow report with
the real BerryPicker configuration. The framework itself is tested in
ExpRunFlow. Run the checks from the repository root with:

```shell
PYTHONPATH=src python -m unittest src.test.test_exprun_notebooks src.test.test_flows -v
```
