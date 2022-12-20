#!/bin/bash

# Slurm logging options
#SBATCH -o "slurm_logs/cifar10_vgg11_%j.log"

# Slurm resources options
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8

# Debug info
set | grep SLURM | while read line; do echo "# $line"; done
echo "# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo ""
nvidia-smi

# Parallelism
export MASTER_PORT=12340
echo "# MASTER_PORT="$MASTER_PORT

export WORLD_SIZE=$(($SLURM_NTASKS_PER_NODE * $SLURM_NNODES))
echo "# WORLD_SIZE="$WORLD_SIZE

echo "# NODELIST="$SLURM_NODELIST

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "# MASTER_ADDR="$MASTER_ADDR
echo ""

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
msg='cifar10_vgg11'

# Experiments
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_antialiased trainer=cifar10_vgg11
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_vanilla trainer=cifar10_vgg11
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_vanilla trainer=cifar10_vgg11 +trainer.adversarial_training.attack=PGD
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_antialiased_partial2 trainer=cifar10_vgg11 +trainer.adversarial_training.attack=PGD
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_initialblur trainer=cifar10_vgg11