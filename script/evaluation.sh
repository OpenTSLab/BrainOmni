#!/bin/bash
MASTER_PORT=$((RANDOM % 101 + 20000))
export MASTER_PORT=$MASTER_PORT
export PYTHONPATH=./

n_fold=5
split_list="0 1 2 3 4"
seed_list="42 3407"
dataset_list="ad65 asd74 FACED MDD MEG-MMI pd31 PhysioNet-MI Somato_EMEG TUAB TUEV WBCIC_SHU"
lr_list="3e-6 1e-5 3e-5"

for dataset in $dataset_list
do
   for lr in $lr_list
   do
      for split_fold in $split_list
      do
         for seed in $seed_list
         do
            lr2=$(awk "BEGIN { printf \"%.6f\", $lr * 10 }")
            deepspeed  --num_gpus=8 --master_port $MASTER_PORT \
               downstream/launcher.py \
               --launcher=pdsh \
               --seed ${seed} \
               --dataset_name=${dataset} \
               --ckpt_path=ckpt_collection/tiny \
               --pretrained \
               --n_fold=${n_fold} \
               --fold_index=${split_fold} \
               --head_lr=${lr2} \
               --backbone_lr=${lr}
         done
      done
   done
done

# freeze backbone
dataset_list="ad65 TUAB TUEV"
for dataset in $dataset_list
do
   for lr in $lr_list
   do
      for split_fold in $split_list
      do
         for seed in $seed_list
         do
            lr2=$(awk "BEGIN { printf \"%.6f\", $lr * 10 }")
            deepspeed  --num_gpus=8 --master_port $MASTER_PORT \
               downstream/launcher.py \
               --launcher=pdsh \
               --seed ${seed} \
               --dataset_name=${dataset} \
               --ckpt_path=ckpt_collection/tiny \
               --pretrained \
               --frozen \
               --n_fold=${n_fold} \
               --fold_index=${split_fold} \
               --head_lr=${lr2} \
               --backbone_lr=${lr}
         done
      done
   done
done

# this will compute the averaged result of 10 fold for each experiment
python dowstream/metrics_stat.py