import logging
import os
import random
import sys

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torchattacks
import torchvision
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from src.datasets import datasets
from src.models.models import get_model
from src.train.trainer import Trainer


def set_seed(seed=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)


def save_config(file, cfg):
  with open(os.path.join(os.getcwd(), '.hydra', f'{file}.yaml'), 'w') as f:
    print(OmegaConf.to_yaml(cfg), file=f)

@hydra.main(config_path='conf/train', config_name='config')
def main(cfg: DictConfig) -> None:
  OmegaConf.resolve(cfg)
  print(OmegaConf.to_yaml(cfg))

  # Seed
  set_seed(cfg.seed)

  # Slurm setup
  cfg.world_size = int(os.environ["WORLD_SIZE"])
  cfg.ngpus_per_node = torch.cuda.device_count()
  cfg.rank = int(os.environ['SLURM_PROCID'])
  cfg.gpu = cfg.rank % torch.cuda.device_count()
  
  dist.init_process_group(backend=cfg.dist_backend, init_method=cfg.dist_url,
                                world_size=cfg.world_size, rank=cfg.rank)
  torch.cuda.set_device(cfg.gpu)
  torch.distributed.barrier()

  # Save config
  if cfg.rank == 0:
    save_config('config', cfg)

  # Luanch train
  launch_train(cfg)

def launch_train(cfg):
  log = logging.getLogger('train')

  log.info(f'Getting dataset {cfg.dataset.name}')
  dataset = datasets.get_dataset(cfg.dataset)
  log.info(f'Train Dataset length={len(dataset.train)}')
  log.info(f'Valid Dataset length={len(dataset.valid)}')
  log.info(f'Test Dataset length={len(dataset.test)}')
  log.info(f'in_size={dataset.in_size}, num_classes={dataset.num_classes}')

  log.info(f'Getting model')
  model = get_model(cfg=cfg.model, num_classes=dataset.num_classes, in_size=dataset.in_size, normalizer=dataset.normalizer)

  log.info(f'Constructing trainer')
  trainer = Trainer(cfg=cfg.trainer)

  log.info(f'Starting fit')
  test_results = trainer.fit(cfg, cfg.rank, cfg.gpu, model, dataset, save_best=cfg.save_best, save_last=cfg.save_last)

  if cfg.rank == 0:
    test_results.to_hdf(os.path.join(os.path.join(os.getcwd(), 'train.h5')), key='test')

if __name__ == '__main__':
  if int(os.environ['SLURM_PROCID']) != 0:
    sys.argv.append('hydra.output_subdir=null')
    sys.argv.append('hydra/job_logging=disabled')
  main()
