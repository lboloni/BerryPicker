#!/bin/bash
set -x # turn on echo
~/WORK/BerryPicker/vm/berrypickervenv/bin/jupyter kernelspec remove -f berrypicker
rm -rf ~/.config/BerryPicker
rm -rf ~/WORK/BerryPicker
echo Uninstalled BerryPicker
