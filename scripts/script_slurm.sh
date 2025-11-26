#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=7:0:0    
#SBATCH --output=./slurm_logs/R-%x.%j.out
#SBATCH --mail-user=<masseeh.amin@gmail.com>
#SBATCH --mail-type=ALL

# abort on error
set -e

dataset="CivilCommentsFine"
arch="bert-base-uncased"

ft="full"  # full/lp/lora/dora/mask/gmixout/mixout
mixout=false
lr=1e-5
batch_size=196
lora_rank=8
lora_alpha=16
mask_sparsity=0.02
mixout_refresh=50
mixout_ema=0.3
store_postfix="full"
seed=1
algorithm="ERM"
dense=true
mask_momentum=false


if [ -z "$SLURM_JOB_ID" ]; then
    echo "Running on a local machine"
    outdir=./output
    data_dir="/export/livia/home/vision/masih/local/masih/dataset"
else
    echo "Running on a cluster"
    outdir=$project/workspace/SubpopBench/output
    data_dir=$SLURM_TMPDIR
fi

function parse_args
{

    # positional args
    args=()

    # named args
    while [ "$1" != "" ]; do
        case "$1" in
            --ft )                        ft="$2";                      shift 2;;
            --seed )                      seed="$2";                    shift 2;;
            * )                           args+=("$1");                 shift;;         # if no match, add it to the positional args
        esac
    done

    if [[ "${ft}" != "full" \
            &&  "${ft}" != "lp" \
            &&  "${ft}" != "lora" \
            &&  "${ft}" != "dora" \
            && "${ft}" != "mask" \
            && "${ft}" != "gmixout" \
            && "${ft}" != "mixout" ]]; then
        echo "ft must be one of: lp/full/lora/dora/mask/gmixout/mixout"
        usage
        exit;
    fi
}

parse_args "$@"

if [ $ft == "lp" ]; then
  lr=1e-4
  algorithm="LP"
  store_postfix="lr${lr}"
  echo "Using LP with learning rate $lr"
fi
if [ $ft == "lora" ]; then
  lr=1e-4
  algorithm="LoRAERM"
  store_postfix="rank${lora_rank}_alpha${lora_alpha}_lr${lr}"
  echo "Using LoRA with learning rate $lr"
fi
if [ $ft == "mask" ]; then
  lr=1e-5
  algorithm="MaskERM"
  store_postfix="sparsity${mask_sparsity}_lr${lr}"
  echo "Using Masking with learning rate $lr"
fi
if [ $ft == "mixout" ]; then
  lr=1e-5
  mixout=true
  mixout_refresh=1
  mixout_ema=1.0
  mask_momentum=false
  algorithm="GMixoutERM"
  store_postfix="sparsity${mask_sparsity}_lr${lr}"
  echo "Using Mixout with learning rate $lr"
fi
if [ $ft == "gmixout" ]; then
  lr=1e-5
  algorithm="GMixoutERM"
  mask_momentum=true
  store_postfix="sparsity${mask_sparsity}_refresh${mixout_refresh}_ema${mixout_ema}_lr${lr}"
  echo "Using GMixout with learning rate $lr"
fi

if [ -z "$SLURM_JOB_ID" ]; then
    echo "skip"
else
    bash scripts/prepare_dataset.sh
    echo "syncing SubpopBench repo in $SLURM_TMPDIR"
    rsync -av ../SubpopBench $SLURM_TMPDIR --exclude output --exclude .venv --exclude .git --exclude slurm_logs
    cd $SLURM_TMPDIR/SubpopBench

    bash scripts/create_venv_slurm.sh
    echo "activating virtual environment"
    source .venv/bin/activate
fi

echo "running experiment"

python -m subpopbench.train \
      --algorithm $algorithm \
      --text_arch $arch \
      --dataset $dataset \
      --use_es \
      --es_patience 100 \
      --train_attr no \
      --data_dir $data_dir \
      --output_dir $outdir \
      --output_folder_name "" \
      --store_postfix $store_postfix \
      --hparams "{\"batch_size\": ${batch_size}, \"lr\": ${lr}, \"lora_rank\": ${lora_rank}, \"lora_alpha\": ${lora_alpha}, \"mask_sparsity\": ${mask_sparsity}, \"mask_seed\": 42, \"mask_momentum\": ${mask_momentum}, \"mixout_refresh\": ${mixout_refresh}, \"mixout\": ${mixout},  \"mixout_ema\": ${mixout_ema}, \"dense\": ${dense}}" \
      --seed $seed \
      --checkpoint_freq 100 \
      # --resume $resume