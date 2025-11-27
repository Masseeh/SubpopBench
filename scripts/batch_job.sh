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

for ft in lora dora mask mixout
do
    for lr in 1e-5 3e-5 5e-5 1e-4
    do
        bash scripts/script_slurm.sh --ft $ft --lr $lr
    done
done

for lr in 1e-5 3e-5 5e-5 1e-4
do
    for mixout_refresh in 5 10 20
    do
        for mixout_ema in 0.1 0.3 0.5
        do
            bash scripts/script_slurm.sh --ft gmixout --lr $lr --mixout_refresh $mixout_refresh --mixout_ema $mixout_ema
        done
    done
done