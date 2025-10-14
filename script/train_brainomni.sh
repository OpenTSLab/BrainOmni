#!/bin/bash
# signal_type=['eeg','meg','both']: when set as eeg, the dataset will only load eeg data
# tokenizer_path: should point to a folder, which contain BrainTokenizer.pt and model_cfg.json

export PYTHONPATH=./
deepspeed  --num_gpus=8 \
    brainomni/launcher.py \
   --launcher=pdsh \
   --signal_type=both \
   --model_size=tiny \
   --tokenizer_path=ckpt_collection/braintokenizer \
   --num_quantizers_used=4 \
   --epoch=32