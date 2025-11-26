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

outdir=$project/workspace/SubpopBench/output
data_dir=$SLURM_TMPDIR

function parse_args
{

    # positional args
    args=()

    # named args
    while [ "$1" != "" ]; do
        case "$1" in
            --ft )                        ft="$2";                      shift 2;;
            * )                           args+=("$1");                 shift;;         # if no match, add it to the positional args
        esac
    done

    if [[ "${ft}" != "full" \
            &&  "${ft}" != "lora" \
            &&  "${ft}" != "dora" \
            && "${ft}" != "mask" \
            && "${ft}" != "mixout" ]]; then
        echo "ft must be one of: lp/full/lora/dora/mask/mixout"
        usage
        exit;
    fi
}

parse_args "$@"

if [ $ft == "lora" ]; then
  lora=true
elif [ $ft == "mask" ]; then
  mask=true
elif [ $ft == "mixout" ]; then
  mixout=true
fi

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

bash scripts/prepare_dataset.sh

# skip rsync if the current directory is already $SLURM_TMPDIR/SubpopBench
if [ "$PWD" == "$SLURM_TMPDIR/SubpopBench" ]; then
    echo "already in $SLURM_TMPDIR/SubpopBench"
    echo "skip rsync"
else
    echo "syncing SubpopBench repo in $SLURM_TMPDIR"
    rsync -av ../SubpopBench $SLURM_TMPDIR --exclude output --exclude .venv --exclude .git --exclude slurm_logs
    cd $SLURM_TMPDIR/SubpopBench
fi

bash scripts/create_venv_slurm.sh
echo "activating virtual environment"
source .venv/bin/activate
echo "running experiment"

python -m subpopbench.train \
      --algorithm ERM \
      --text_arch $arch \
      --dataset $dataset \
      --train_attr no \
      --data_dir $data_dir \
      --output_dir $outdir \
      --output_folder_name "" \
      --store_postfix $store_postfix \
      --hparams "{\"batch_size\": ${batch_size}, \"lr\": ${lr}, \"lora\": ${lora}, \"lora_rank\": ${lora_rank}, \"lora_alpha\": ${lora_alpha}, \"mask\": ${mask}, \"mask_prob\": ${mask_prob}, \"mask_seed\": 42, \"mixout\": ${mixout}, \"mixout_refresh\": ${mixout_refresh}, \"mixout_ema\": ${mixout_ema}}" \
      --seed 1 \
      --checkpoint_freq 100 \
      # --resume $resume