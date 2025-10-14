#!/bin/bash
# time        -> how long is each sample
# stride      -> the stride when sampling the original emeg data
# max_workers -> number of multiprocessing
# we train tokenizer using time=10 stride=10 data   <--- you can change the setting in braintokenizer/config.py
# we train omni      using time=30 stride=30 data   <--- you can change the setting in brainomni/config.py
export PYTHONPATH=./
python factory/process.py \
    --time 10 \
    --stride 10 \
    --max_workers 32