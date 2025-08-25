#!/bin/bash

dataset="CivilCommentsFine"
# dataset="MultiNLI"
arch="bert-base-uncased"
# resume = "output/_attrNo/MultiNLI_ERM_hparams0_seed0/model.pkl"

lora=false
mask=false
mixout=false
lr=1e-5

if [ "$lora" = true ]; then
  lr=1e-4
  echo "Using LoRA with learning rate $lr"
fi
if [ "$mask" = true ]; then
  lr=1e-5
  echo "Using Masking with learning rate $lr"
fi
if [ "$mixout" = true ]; then
  lr=1e-5
  echo "Using Mixout with learning rate $lr"
fi


python -m subpopbench.train \
       --algorithm ERM \
       --text_arch $arch \
       --dataset $dataset \
       --train_attr no \
       --data_dir "/export/livia/home/vision/masih/local/masih/dataset" \
       --output_folder_name "" \
       --hparams "{\"batch_size\": 64, \"lr\": ${lr}, \"lora\": ${lora}, \"lora_rank\": 64, \"mask\": ${mask}, \"mask_prob\": 0.9, \"mask_seed\": 42, \"mixout\": ${mixout}, \"mixout_refresh\": 50, \"mixout_ema\": 0.3}" \
       --seed 1 \
       --checkpoint_freq 100 \
       # --resume $resume