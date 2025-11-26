#!/bin/bash

dataset="CivilCommentsFine"
# dataset="MultiNLI"
arch="bert-base-uncased"
# resume = "output/_attrNo/MultiNLI_ERM_hparams0_seed0/model.pkl"

lora=false
mask=false
mixout=false
lr=1e-5
batch_size=196
lora_rank=8
lora_alpha=16
mask_prob=0.98
mixout_refresh=50
mixout_ema=0.3
store_postfix="full"

if [ "$lora" = true ]; then
  lr=1e-4
  store_postfix="lora"
  echo "Using LoRA with learning rate $lr"
fi
if [ "$mask" = true ]; then
  lr=1e-5
  store_postfix="mask"
  echo "Using Masking with learning rate $lr"
fi
if [ "$mixout" = true ]; then
  lr=1e-5
  store_postfix="mixout"
  echo "Using Mixout with learning rate $lr"
fi


python -m subpopbench.train \
      --algorithm ERM \
      --text_arch $arch \
      --dataset $dataset \
      --train_attr no \
      --data_dir "/export/livia/home/vision/masih/local/masih/dataset" \
      --output_folder_name "" \
      --store_postfix $store_postfix \
      --hparams "{\"batch_size\": ${batch_size}, \"lr\": ${lr}, \"lora\": ${lora}, \"lora_rank\": ${lora_rank}, \"lora_alpha\": ${lora_alpha}, \"mask\": ${mask}, \"mask_prob\": ${mask_prob}, \"mask_seed\": 42, \"mixout\": ${mixout}, \"mixout_refresh\": ${mixout_refresh}, \"mixout_ema\": ${mixout_ema}}" \
      --seed 1 \
      --checkpoint_freq 100 \
      # --resume $resume