import nbformat
from nbconvert import PythonExporter

# Load the notebook
with open('/home/ssheikholeslami/BerryPicker/src/visual_proprioception/Compare_VisualProprioception_multiview.ipynb') as f:
    notebook = nbformat.read(f, as_version=4)

# Convert to Python
python_exporter = PythonExporter()
python_code, _ = python_exporter.from_notebook_node(notebook)

# Write to a Python file
with open('Train_VisualProprioception_multiview.py', 'w') as f:
    f.write(python_code)