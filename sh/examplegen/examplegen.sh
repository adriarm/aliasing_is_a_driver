#!/bin/bash

# Slurm logging options
#SBATCH -o "slurm_logs/examplegen_%j.log"

# Slurm resources options
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8

# Debug info
set | grep SLURM | while read line; do echo "# $line"; done
echo "# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""
nvidia-smi

# Loading the required module
source /etc/profile
module load anaconda/2021b
export PYTHONPATH=$PYTHONPATH:/home/gridsan/amunoz/projects/AntialiasedNetworks/src/robust_layers/antialiased/stylegan3

# Save datetime as variable
hydra_now=$(date +"%Y-%m-%d_%H-%M-%S")
export HYDRA_NOW=$hydra_now
echo "# HYDRA_NOW="$HYDRA_NOW
echo ""

# Message
msg='examplegen'

# Example generations
srun python3 launch_examplegen.py msg=$msg dataset=cifar10 +model_path='""'