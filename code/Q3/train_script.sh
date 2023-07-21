#!/bin/bash

#SBATCH -n 4
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=t5_training
#SBATCH --output=/cluster/home/rdanis/%x.out                                                                         
#SBATCH --error=/cluster/home/rdanis/%x.err
#SBATCH --gpus=1
#SBATCH --mail-type=NONE

module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1

python3 /cluster/home/rdanis/LLM-Assignment-2/code/Q3/train.py
