import pathlib
import papermill as pm

ext_path = pathlib.Path(Config()["experiment_external"])

params = {}
params["run"] = "sp_vae_256"
params["external_path"] = str(pathlib.Path(ext_path, "_automate").resolve())
params["epochs"] = 10

print(params["external_path"])

output_path = pathlib.Path(ext_path, "_automation_output")
output = pathlib.Path(output_path, "Train-Conv-VAE-output.ipynb")

try:
   pm.execute_notebook(
      'sensorprocessing/Train-Conv-VAE.ipynb',
      output.absolute(),
      cwd="sensorprocessing",
      parameters=params
   )
except Exception as e:
   print(f"There was an exception {e}")
