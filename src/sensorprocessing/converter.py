import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open('/home/sa641631/WORK/BerryPicker-Flows/VisualProprioception_flow_07/result/Train_VisualProprioception_visual_proprioception_vp_vit_base_256_0001_output.ipynb') as f:
    notebook = nbformat.read(f, as_version=4)

# Convert to Python
python_exporter = PythonExporter()
python_code, _ = python_exporter.from_notebook_node(notebook)

# Write to a Python file
with open('/home/sa641631/WORK/BerryPicker-Flows/VisualProprioception_flow_07/result/Train_VisualProprioception_visual_proprioception_vp_vit_base_256_0001_output.py', 'w') as f:
    f.write(python_code)