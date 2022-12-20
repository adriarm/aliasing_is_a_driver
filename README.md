# Aliasing is a Driver of Adversarial Attacks

This is the official implementation of Aliasing is a Driver of Adversarial Attacks.

<p align="center">
  <img width="100%" src="https://adriarm.github.io/_pages/aliasing_is_a_driver/files/toy_example_2.svg">
</p>

In this work, we investigate the hypothesis that the existence of adversarial perturbations is due in part to aliasing in neural networks. We made heavy use of the fast up-sampling, down-sampling, and anti-aliased ReLu implementations of Stylegan3 [[Karras et al](https://github.com/NVlabs/stylegan3)].

[[Webpage](https://adriarm.github.io/_pages/aliasing_is_a_driver/)] 
[[Paper](https://adriarm.github.io/_pages/aliasing_is_a_driver/files/paper.pdf)]
[arXiv]

# Requirements

To use our code run the following:
```
git clone https://github.com/adriarm/aliasing_is_a_driver
cd aliasing_is_a_driver
pip install -r requirements.txt
```

# Experiments

To re-run our experiments, you will need a distributed cluster running Slurm. The current slurm configuration requests 8 nodes with 1 GPU each (we used Nvidia Volta V100).
```
# Download Tiny-Imagenet
mkdir datasets
wget -P datasets/ http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip datasets/tiny-imagenet-200.zip

# Logs
mkdir slurm_logs # folder to deposit the logs

# Submit scripts to Slurm
sbatch ./sh/train/cifar10_vgg11.sh
sbatch ./sh/train/cifar10_resnet50.sh
sbatch ./sh/train/tinyimagenet_vgg11.sh
sbatch ./sh/train/tinyimagenet_resnet50.sh
sbatch ./sh/train/typewise.sh
```
If you want to use a different node/gpu configuration, you will need to change the resource request slurm commands in the .sh files, as well as the batch size used, accordingly. The default per-gpu batch size for 8 GPUs is found in the corresponding config file in `conf/train/trainer`.
```
# Slurm resources options
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
.
.
.
srun python3 launch_train.py ... trainer.train_loader.batch_size=X trainer.train_loader.valid_size=X
```

# Citation
