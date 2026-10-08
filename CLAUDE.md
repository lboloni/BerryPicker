# CLAUDE.md

## Code style (research code, not production)
- exp/run objects come from `Config().get_experiment(...)`; fields are accessed as `exp["field"]`.
- Assume all exp/run fields are present and correctly formatted. Do not add validation or error handling for missing/malformed config, or for values that would be invalid at runtime.
- This is research code: on error, the right behavior is to let an exception propagate and abort — do not silently catch or work around errors.
- Do not add input validation for cases that would crash anyway.
- Keep code short and human-understandable, matching the style already present in the file you're editing.
- When asked to design or propose a fix, produce a design only — do not change existing code unless explicitly told to implement it. When implementing, make the smallest possible change and follow the surrounding code's existing style.

## Config / exp_run_config bootstrap
The codebase is driven by the `Config` singleton of the ExpRunFlow library (`exprunflow.exp_run_config`, installed editable from the `ExpRunFlow` checkout next to BerryPicker). `src/exp_run_config.py` and `src/flow.py` are thin shims that re-export the library and set the BerryPicker values `Config.PROJECTNAME`, `Config.SRC_ROOT` and `Config.KERNEL_NAME`; code keeps importing `from exp_run_config import Config` and `from flow import ...`. The torch device (`Config().runtime["device"]`) and `exprunflow.reproducibility` also come from the library.
- Every entry point sets `Config.PROJECTNAME = "BerryPicker"` before first use (now redundant with the shim, but harmless).
- `Config()` reads `~/.config/BerryPicker/mainsettings.yaml`, whose only field, `configpath`, points at a machine-specific settings YAML that lives outside this repo and is not committed.
- Experiment definitions live in `data/expruns/<family>/`: one `_defaults_<family>.yaml` plus one YAML per run. `Config().get_experiment(family, run)` merges defaults -> run config -> an optional system-dependent override, then creates `exp["data_dir"]` under the machine's configured `experiment_data` root.
- Without a working `mainsettings.yaml`/`configpath` on the current machine, most modules will fail on import or on first `Config()` call — that is expected, not something to work around defensively.

## Where the design lives
Design docs are the source of truth for what's actually implemented vs. proposed-but-not-built; don't infer status from code alone.
- `src/sensorprocessing/DESIGN-*.md` — composite SP engine, VAE-GAN, multiview VAE, proprioception-tuned CNN, random-projection baseline, sensor-processing memory/temporal context. Several of these (e.g. composite ops, sensor-processing memory) are proposals, not yet implemented.
- `src/visual_proprioception/DESIGN-TemporalVisualProprioception.md` — recursive estimators (EMA, Kalman, LSTM) over the observation stream, with the measured per-frame motion and error figures that motivate them. A proposal; no temporal component exists.
- `src/robot_controller/DESIGN-RobotController.md` — the RCCO graph-controller framework.
- `src/robot/wxai/PLAN-WXAI.md` — the Trossen WidowX AI support (real arm on `trossen_arm`, kinematic fake driver, MuJoCo simulation with cameras, leader arm): the design, what is implemented, and the open problems that need Linux or the arm. `INSTALL-WXAI.md` next to it.
- `data/expruns/robot_controller/DESIGN_Training_Recipe.md` — the staged training recipe and controller-bundle export.
- `ExpRunFlow/docs/DESIGN-ExpRun.md` and `ExpRunFlow/docs/DESIGN-Flows.md` — the generic exp/run and flow contracts, shared with WaterberryFarms.
- `data/expruns/DESIGN-ExpRun.md` — the exp/run framework as used in BerryPicker: templates, paths, creation styles, the `input-to-notebook` field, the stage notebook contract and `exp.done()`.
- `data/expruns/DESIGN-Flows.md` — flows: the workspace, queue building with `flow_entry`, fail-fast execution, the final report, and the current BerryPicker flows.

## Tests
Tests are plain `unittest`/`pytest` under `src/test/`, mirroring the `src/` package layout. Each test file inserts `src/` onto `sys.path` itself, so tests run from the repo root:
```
pytest src/test
```

## Notebooks
Training/verification entry points are Jupyter notebooks (run via Papermill), not scripts. Each exp/run declares which notebook(s) are canonical via the `input-to-notebook` field (see `DESIGN-ExpRun.md`). Flows run them in order (see `DESIGN-Flows.md`).
