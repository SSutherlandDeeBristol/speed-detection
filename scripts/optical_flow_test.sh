#!/usr/bin/env bash
#SBATCH --partition gpu_veryshort
#SBATCH --time 0-00:01:00
#SBATCH --mem 64GB
#SBATCH --gres gpu:1
#SBATCH --mail-type=FAIL

module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python3 optical_flow_test.py --mode=forward