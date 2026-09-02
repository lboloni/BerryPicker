# git reset origin/main                 # HEAD back to Lotzi's real commit; keeps every file as is
# git status --short                    # should show only  M and ?? lines (your changes), no D
# git add src
# git commit -m "Port multi view sensor processing onto the refactored code base. Shared fusion heads for ViT, CNN and conv encoder, training on the harness, fixed multi view flow and configs, tests"


# git checkout --theirs src/sensorprocessing/Train_ProprioTuned_CNN_multiview.ipynb \
#                       src/visual_proprioception/Flow_VisualProprioception_multi.ipynb \
#                       src/visual_proprioception/Train_VisualProprioception_multiview.ipynb
# git checkout --ours   src/visual_proprioception/Compare_VisualProprioception_multiview_and_singleview.ipynb
# git add src/sensorprocessing/Train_ProprioTuned_CNN_multiview.ipynb \
#         src/visual_proprioception/Flow_VisualProprioception_multi.ipynb \
#         src/visual_proprioception/Train_VisualProprioception_multiview.ipynb \
#         src/visual_proprioception/Compare_VisualProprioception_multiview_and_singleview.ipynb
# GIT_EDITOR=true git rebase --continue
# git log --oneline -3      # expect your commit on top of feeff1a


# python -c "import demonstration; print(demonstration.__file__)"
# find . -name 'demonstration.py' -not -path './demonstration/*'
# echo "PYTHONPATH=$PYTHONPATH"


# cd /home/sa641631/WORK/BerryPicker/src/BerryPicker/src
# source /home/sa641631/WORK/BerryPicker/vm/berrypickervenv/bin/activate
# python -m pytest test/sensorprocessing test/training_harness test/visual_proprioception -q
# grep -rn "sys.modules" test/ robot/ demonstration/ 2>/dev/null | grep -v "\.pyc"


import socket, torch
print(socket.gethostname(), torch.cuda.get_device_name(0))