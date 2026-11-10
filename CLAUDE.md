# CLAUDE.md

## Code style (research code, not production)
- exp/run objects come from `Config().get_experiment(...)`; fields are accessed as `exp["field"]`.
- Assume all exp/run fields are present and correctly formatted. Do not add validation or error handling for missing/malformed config, or for values that would be invalid at runtime.
- This is research code: on error, the right behavior is to let an exception propagate and abort — do not silently catch or work around errors.
- Do not add input validation for cases that would crash anyway.
- Keep code short and human-understandable, matching the style already present in the file you're editing.
- Line endings are LF everywhere (`.gitattributes`: `* text=auto eol=lf`). The files that had CRLF or mixed endings were converted on 2026-11-03. If a file still has CRLF (e.g. a new file written on Windows), convert it to LF instead of preserving the CRLF.
- When asked to design or propose a fix, produce a design only — do not change existing code unless explicitly told to implement it. When implementing, make the smallest possible change and follow the surrounding code's existing style.

## Config / exp_run_config bootstrap
The codebase is driven by the `Config` singleton of the ExpRunFlow library (`exprunflow.exp_run_config`, installed editable from the `ExpRunFlow` checkout next to BerryPicker). `src/exp_run_config.py` and `src/flow.py` are thin shims that re-export the library and set the BerryPicker values `Config.PROJECTNAME`, `Config.SRC_ROOT` and `Config.KERNEL_NAME`; code keeps importing `from exp_run_config import Config` and `from flow import ...`. The torch device (`Config().runtime["device"]`) and `exprunflow.reproducibility` also come from the library.
- Every entry point sets `Config.PROJECTNAME = "BerryPicker"` before first use (now redundant with the shim, but harmless).
- `Config()` reads `~/.config/BerryPicker/mainsettings.yaml`, whose only field, `configpath`, points at a machine-specific settings YAML that lives outside this repo and is not committed.
- Experiment definitions live in `data/expruns/<family>/`: one `_defaults_<family>.yaml` plus one YAML per run. `Config().get_experiment(family, run)` merges defaults -> run config -> an optional system-dependent override, then creates `exp["data_dir"]` under the machine's configured `experiment_data` root.
- Without a working `mainsettings.yaml`/`configpath` on the current machine, most modules will fail on import or on first `Config()` call — that is expected, not something to work around defensively.

## Where the design lives
Design docs are the source of truth for what's actually implemented vs. proposed-but-not-built; don't infer status from code alone.
- `src/sensorprocessing/DESIGN-*.md` — composite SP engine, VAE-GAN, multiview VAE, proprioception-tuned CNN, random-projection baseline, sensor-processing memory/temporal context. The composite engine and the sensor-processing memory framework (context, `reset_context()`, `dt`) are implemented; the concrete composite operations and temporal operations are proposals, not yet implemented. The other documents describe implemented code.
- `src/visual_proprioception/DESIGN-TemporalVisualProprioception.md` — recursive estimators (EMA, Kalman, LSTM) over the observation stream, with the measured per-frame motion and error figures that motivate them. The EMA and Kalman filters on the predicted positions are implemented (`visproprio_filters.py`, the `position_filter` field, `Flow_FilteredVsUnfiltered.ipynb`); the LSTM and every estimator inside the sensor processor are proposals.
- `src/robot_controller/DESIGN-RobotController.md` — the RCCO graph-controller framework.
- `src/robot_controller/DESIGN-BehaviorCloningFlow.md` — the behavior cloning flows on the RCCO framework (verification, comparison, LSTM and MDN controllers), in three phases, each with its implementation record. Implemented.
- `src/visual_proprioception/DESIGN-Import-Competitors.md` — external robot pose estimators (RoboPEPP, HPE) run from BerryPicker as `external_proprioception` exp/runs. The BerryPicker side is implemented and the ground-truth reference has run; the methods themselves have not been run.
- `src/visual_proprioception/DESIGN-ProprioceptionByClaude.md` — the record of a one-off measurement (robot positions guessed from video); not a component.
- `src/demonstration/DESIGN-Aloha-WidowX-AI-SimpleImport.md` — the partial importer of Aloha WidowX-AI demonstrations (`import_aloha.py`). Implemented.
- `src/visual_proprioception/PLAN-MultiRobot.md` — visual proprioception on other robots (Franka Panda, Unitree G1 arm, Inspire hand): parts 1–3 implemented, the full reproduction and the clean protocol remain; one package and exp/run family per robot, the joint-vector `JointPosition` base class, `robot/position_factory.py`, the imported packs, and the cross-robot flow reproducing Robi's 2025-10 results.
- `src/visual_proprioception/transfer/DESIGN-Transfer.md` — the transfer of visual proprioception across robots (RAS 2027 paper): the baselines and five designs (several sources, keypoint latent, masking, AdaBN, meta-learning), the full and smoke flows, and how to find the datasets on a machine. Implemented, not yet run.
- `src/robot/wxai/PLAN-WXAI.md` — the Trossen WidowX AI support (real arm on `trossen_arm`, kinematic fake driver, MuJoCo simulation with cameras, leader arm): the design, what is implemented, and the open problems that need Linux or the arm. `INSTALL-WXAI.md` next to it.
- `data/expruns/robot_controller/DESIGN_Training_Recipe.md` — the staged training recipe and controller-bundle export.
- `ExpRunFlow/docs/DESIGN-ExpRun.md` and `ExpRunFlow/docs/DESIGN-Flows.md` — the generic exp/run and flow contracts, shared with WaterberryFarms.
- `data/expruns/DESIGN-ExpRun.md` — the exp/run framework as used in BerryPicker: templates, paths, creation styles, the `input-to-notebook` field, the stage notebook contract and `exp.done()`.
- `data/expruns/DESIGN-Flows.md` — flows: the workspace, queue building with `flow_entry`, fail-fast execution, the final report, and the current BerryPicker flows.

## Tests
Tests are plain `unittest`/`pytest` under `src/test/`, mirroring the `src/` package layout. The test files insert `src/` onto `sys.path` themselves (except `test_flows.py`, which expects `src` to be on the path already, e.g. `PYTHONPATH=src` when run alone), so tests run from the repo root, with the Python of the machine's venv (there is no venv in the repository; see `src/install/INSTALL.md`):
```
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 <venv>/bin/python -m pytest src/test
```
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` is needed on machines that source ROS: ROS puts its own pytest plugins on `PYTHONPATH`, and they fail inside the venv. Run the whole suite in one call; it must not depend on which folders are run. Tests import project modules through their packages (`demonstration.import_aloha`), because a folder on `sys.path` such as `src/demonstration` turns `demonstration` into the module `demonstration.py` for all later tests. Tests for a robot stack are meant to be skipped unless the robot is listed in `robots` of the `machine/current` exp/run (see `data/expruns/machine/README.md`); at present only the `wxai` tests are gated this way, so e.g. `src/test/robot/widowx/test_interbotix_gazebo_bridge.py` stops the collection on a machine without ROS (`rclpy`).

## Notebooks
Training/verification entry points are Jupyter notebooks (run via Papermill), not scripts. Each exp/run declares which notebook(s) are canonical via the `input-to-notebook` field (see `DESIGN-ExpRun.md`). Flows run them in order (see `DESIGN-Flows.md`).
