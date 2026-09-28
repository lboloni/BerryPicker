# How to install BerryPicker on a new Linux environment?

## Things to configure before running the install script

### Install python, and venv

```
sudo apt install python3.10-venv
sudo apt install python-is-python3
sudo apt install ffmpeg
```

### Make the user sudo capable:

```
sudo usermod -aG sudo <username>
```

or add username into  /etc/sudoers 

## Install BerryPicker

```
berry_install.sh
```

## Activate BerryPicker

This is needed before running vscode from the same terminal, or before running the flow notebooks from the command line. 

```
berry_activate.sh
```

This activates the virtual environment. 

## Uninstall BerryPicker

```
berry_uninstall.sh
```