#!/bin/bash

# Slurm logging options
#SBATCH -o "slurm_logs/typewise_%j.log"
#SBATCH --mail-user=adrianrm@mit.edu
#SBATCH --mail-type=ALL

# Slurm resources options
#SBATCH --nodes=4
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
export PYTHONPATH=$PYTHONPATH:$PWD/src/robust_layers/antialiased/stylegan3

# Save datetime as variable
hydra_now=$(date +"%Y-%m-%d_%H-%M-%S")
export HYDRA_NOW=$hydra_now
echo "# HYDRA_NOW="$HYDRA_NOW
echo ""

# Message
msg='typewise'

# vgg11+cifar10
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_blur trainer=cifar10_vgg11
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_quantile trainer=cifar10_vgg11
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_blur trainer=cifar10_vgg11 +trainer.adversarial_training.attack=PGD
srun python3 launch_train.py msg=$msg dataset=cifar10 model=vgg11_quantile trainer=cifar10_vgg11 +trainer.adversarial_training.attack=PGD

# resnet50+cifar10
srun python3 launch_train.py msg=$msg dataset=cifar10 model=resnet50_blur trainer=cifar10_resnet50 trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=cifar10 model=resnet50_quantile trainer=cifar10_resnet50 trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=cifar10 model=resnet50_blur trainer=cifar10_resnet50 +trainer.adversarial_training.attack=PGD trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=cifar10 model=resnet50_quantile trainer=cifar10_resnet50 +trainer.adversarial_training.attack=PGD trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64

# vgg11+tinyimagenet
srun python3 launch_train.py msg=$msg dataset=tinyimagenet model=vgg11_blur trainer=tinyimagenet_vgg11 trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=tinyimagenet model=vgg11_quantile trainer=tinyimagenet_vgg11 trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=tinyimagenet model=vgg11_blur trainer=tinyimagenet_vgg11 +trainer.adversarial_training.attack=PGD trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=tinyimagenet model=vgg11_quantile trainer=tinyimagenet_vgg11 +trainer.adversarial_training.attack=PGD trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64

# resnet50+tinyimagenet
srun python3 launch_train.py msg=$msg dataset=tinyimagenet model=resnet50_blur trainer=tinyimagenet_resnet50 trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=tinyimagenet model=resnet50_quantile trainer=tinyimagenet_resnet50 trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=tinyimagenet model=resnet50_blur trainer=tinyimagenet_resnet50 +trainer.adversarial_training.attack=PGD trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
srun python3 launch_train.py msg=$msg dataset=tinyimagenet model=resnet50_quantile trainer=tinyimagenet_resnet50 +trainer.adversarial_training.attack=PGD trainer.train_loader.batch_size=64 trainer.valid_loader.batch_size=64
