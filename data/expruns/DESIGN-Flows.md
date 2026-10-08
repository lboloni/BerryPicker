# Experiment flows in BerryPicker

The flow helpers (`setup_flow`, `flow_entry`, `run_flow`, `run_notebook`,
`get_flow_report`, `display_flow_report`) are implemented by the ExpRunFlow
library (`exprunflow.flow`), whose `docs/DESIGN-Flows.md` is the
authoritative contract: the isolated workspace, flow entries, creation
styles, fail-fast execution and the final report. `src/flow.py` re-exports
them; flow notebooks import `from flow import ...`. This document describes
only how BerryPicker uses them. The exp/run conventions are in
[DESIGN-ExpRun.md](DESIGN-ExpRun.md).

## Workspace and demonstrations

Each flow lists the families it needs explicitly: the families it generates
runs into, and the families its stages load as components (for example
`robot_al5d` and `demonstration`). Most flows then import a demonstration
pack with `import_demopack()`, which copies the demonstrations into
`results/demonstration/<demopack>` and returns their split into training and
evaluation groups.

## Generated expruns

Unlike a flow over a checked-in collection, a BerryPicker flow **generates**
most of its expruns, because their training data depends on the imported
demopack. Each generator function writes one exprun YAML into the workspace
and returns its `flow_entry`. A run in a homogeneous family inherits its
notebook from the copied default, while a run in a mixed family such as
`visual_proprioception` has it written by the generator.

The usual order is:

1. train each sensor processor;
2. train each regressor or controller on top of it;
3. verify, where the flow includes verification; and
4. run the comparisons.

`Flow_FilteredVsUnfiltered` has several phases: it trains its base
regressor, tunes the filter parameters in the flow notebook on that
regressor's predictions, and only then generates and queues the filtered
runs, their verification, and the comparison.

## Creation styles

The flow's `creation_style` applies to trainings, the verification of a
separate verification run (as in behavior cloning), and comparisons. The
second entry point of the same exprun, such as `Verify_Conv_VAE` after
`Train_Conv_VAE` on one run, always receives `exist-ok`.

## Execution

The stage notebooks run with the `berrypicker` kernel
(`Config.KERNEL_NAME`, set in `src/exp_run_config.py`), which
`src/install/berry_install.sh` registers.

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

`robot_controller/Flow_RCCO_BehaviorCloning.ipynb` trains one RCCO
controller (AL5D): it trains the sensor processing chosen by `sp_type`
(ResNet-50, VGG19, Conv-VAE-Neo, or VAE-GAN), trains the controller chosen by
`controller_type` (an MLP, or a plain or residual LSTM with an MLP or MDN
head) starting from its encoder, and verifies the exported controller bundle
with teacher forcing on the held-out `bc_testing` group. Final results:
`robot_controller_verify/_flow_verify_<sp_type>_<controller_type>`.

`robot_controller/Flow_RCCO_Compare.ipynb` does the same for every
`(sp_type, controller_type)` pair in `controllers`, on the same demonstration
split, training each sensor processing once, and compares the verified
controllers. Final results: `robot_controller_compare/_flow_compare`. Both
flows use the generators in `src/robot_controller/rcco_flow.py`; see
`src/robot_controller/DESIGN-BehaviorCloningFlow.md`.

`visual_proprioception/MultiFlow_VisualProprioception.ipynb` is not a flow
over stages but a sweep over flows: it runs
`Flow_VisualProprioception.ipynb` once per simulated camera, with a separate
workspace for each. The executed flow notebooks go to
`<flows_path>/multiflow_visualproprioception/executed-notebooks`. A failing
flow stops the sweep.
