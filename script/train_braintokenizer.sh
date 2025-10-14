#!/bin/bash
# signal_type=['eeg','meg','both']: when set as eeg, the dataset will only load eeg data

export PYTHONPATH=./
deepspeed  --num_gpus=8 \
   braintokenizer/launcher.py \
   --launcher=pdsh \
   --signal_type=both \
   --epoch=16 \
   --codebook_size=512 \
   --codebook_dim=256 \
   --num_quantizers=4 