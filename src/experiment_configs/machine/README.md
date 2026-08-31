# Machine bindings

`current.yaml` is the safe repository baseline. Each computer supplies its own
`machine/current_sysdep.yaml` under `experiment_system_dependent_dir`. That file
selects the physical devices and must not be committed.

Use `current_sysdep.template.yaml` as the starting point. A binding must name a
supported participant `factory` and whether it is available. Hardware bindings
also name their local `exp`/`run`; behavior-specific runs such as AutoMove can
be supplied by the collection recipe instead.
`resources` prevent a collection recipe from acquiring the same physical device
twice.

The component templates `controllers/fixed_cameras_sysdep.template.yaml` and
`robot_al5d/pulse_controller_00_sysdep.template.yaml` belong in the same local
system-dependent directory. They hold device indices and serial ports.
