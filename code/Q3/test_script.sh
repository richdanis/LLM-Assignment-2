#!/bin/bash

#SBATCH -n 4
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --job-name=testing
#SBATCH --output=/cluster/home/rdanis/%x.out                                                                         
#SBATCH --error=/cluster/home/rdanis/%x.err
#SBATCH --gpus=1
#SBATCH --mail-type=NONE

module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1

NAME=$1

if [ -z "$2" ]
  then
    python3 /cluster/home/rdanis/LLM-Assignment-2/code/Q3/test.py --model_name $NAME
  else
    python3 /cluster/home/rdanis/LLM-Assignment-2/code/Q3/test.py --model_name $NAME --pre_trained
fi
