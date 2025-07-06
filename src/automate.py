from exp_run_config import Config, Experiment
Config.PROJECTNAME = "BerryPicker"

import pathlib
import papermill as pm

def automate_exprun(notebook, name, params):
   """Automates the execution of a notebook. It is assumed that in the notebook there is cell tagged "parameters", and in general that the notebook is idempotent."""

   ext_path = pathlib.Path(Config()["experiment_external"])
   params["external_path"] = str(pathlib.Path(ext_path, "_automate").resolve())
   notebook_path = pathlib.Path(notebook)
   output_path = pathlib.Path(ext_path, "_automation_output")
   output_filename = f"{notebook_path.stem}_{name}_output{ notebook_path.suffix}"
   output = pathlib.Path(output_path, notebook_path.parent, output_filename)
   output.parent.mkdir(exist_ok=True)
   print(output)

   try:
      pm.execute_notebook(
         notebook,
         output.absolute(),
         cwd=notebook_path.parent,
         parameters=params
      )
   except Exception as e:
      print(f"There was an exception {e}")

experiment = "automate"
run = "automate_00"
exp = Config().get_experiment(experiment, run)

for item in exp["exps_to_run"]:
    print(f"***Automating {item['name']}")
    #notebook = params["notebook"]
    automate_exprun(item["notebook"], item["name"], item["params"])