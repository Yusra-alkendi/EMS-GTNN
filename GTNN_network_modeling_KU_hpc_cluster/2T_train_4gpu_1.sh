#!/bin/bash


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBATCH --gres=gpu:4

#SBATCH --job-name=python_MoSeg
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --account=kuin0013
#SBATCH --output=1T_train_4gpu.%j.out
#SBATCH --error=1T_train_4gpu.%j.err

module purge
module load miniconda/3

conda activate MoSegGPU
python 2T_train_4gpu_3LGTNNwGF.py $1
