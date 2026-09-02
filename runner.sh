source $HOME/WORK/BerryPicker/vm/berrypickervenv/bin/activate
which pip                      # must be .../berrypickervenv/bin/pip, not anaconda's

pip install --upgrade pip
pip install --no-cache-dir ipykernel
pip install --no-cache-dir pyyaml papermill numpy pyserial opencv-python
pip install --no-cache-dir approxeng.input pillow matplotlib pandas
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu132