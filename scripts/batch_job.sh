#!/bin/bash
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:v100l:4
#SBATCH --time=7:0:0    
#SBATCH --output=./slurm_logs/R-%x.%j.out
#SBATCH --mail-user=<masseeh.amin@gmail.com>
#SBATCH --mail-type=ALL

set -e

bash scripts/script_slurm.sh --ft lora --lr 5e-5
bash scripts/script_slurm.sh --ft dora --lr 5e-5
bash scripts/script_slurm.sh --ft mask --lr 1e-4
bash scripts/script_slurm.sh --ft mask --lr 5e-4
bash scripts/script_slurm.sh --ft mixout --lr 5e-5
bash scripts/script_slurm.sh --ft mixout --lr 3e-5
bash scripts/script_slurm.sh --ft gmixout --lr 5e-5 --mixout_refresh 10 --mixout_ema 0.3
bash scripts/script_slurm.sh --ft gmixout --lr 3e-5 --mixout_refresh 10 --mixout_ema 0.3