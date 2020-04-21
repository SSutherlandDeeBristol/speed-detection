#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0-08:00:00
#SBATCH --mem 64GB
#SBATCH --gres gpu:1

module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python3 train.py --lr=0.001 --bs=32
