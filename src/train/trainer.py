import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional, Union

import contexttimer
import hydra
import numpy as np
import pandas as pd
import tabulate
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchattacks
from numpy.random import default_rng
from omegaconf import MISSING, DictConfig, OmegaConf, SCMode, open_dict
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from src.utils.distributed import all_reduce, is_dist_avail_and_initialized


# Loader configs
@dataclass
class LoaderConfig:
  batch_size: int = MISSING
  num_workers: int = 0

# Optimizer config
@dataclass
class SGDConfig:
  lr: float = 0.01
  momentum: float = 0.9
  weight_decay: float = 0.005


# Scheduler config:
@dataclass
class OneCycleLRConfig:
  pct_start: float = 0.3
  anneal_strategy: str = 'cos'
  max_lr: float = 0.02
  div_factor: int = 10
  final_div_factor: int = 10
  three_phase: bool = False


# AdversarialTraining config
@dataclass
class AdversarialTrainingConfig:
  attack: str = 'None'
  adversarial_batch_size: float = 0.5


# Restarting config
@dataclass
class RestartConfig:
  state_dict_path: Optional[str] = None


# Checkpoint config
@dataclass
class CheckpointConfig:
  last_state_dict_path: str = MISSING #os.path.join(os.getcwd(), 'checkpoints', 'last_state_dict.pt')
  best_state_dict_path: str = MISSING #os.path.join(os.getcwd(), 'checkpoints', 'best_state_dict.pt')


# Train config
@dataclass
class TrainConfig:
  max_epochs: int = MISSING
  grad_acum: int = 1

  train_loader: LoaderConfig = LoaderConfig()
  valid_loader: LoaderConfig = LoaderConfig()
  optim: SGDConfig = SGDConfig()
  scheduler: OneCycleLRConfig = OneCycleLRConfig()
  adversarial_training: AdversarialTrainingConfig = AdversarialTrainingConfig()
  restart: RestartConfig = RestartConfig()
  checkpoint: CheckpointConfig = MISSING #CheckpointConfig()


def build_default_config():
  checkpoint = CheckpointConfig(
    last_state_dict_path=os.path.join(os.getcwd(), 'checkpoints', 'last_state_dict.pt'),
    best_state_dict_path=os.path.join(os.getcwd(), 'checkpoints', 'best_state_dict.pt'),
  )
  _instance = TrainConfig(checkpoint=checkpoint)
  return _instance


def update_config(_instance: TrainConfig, cfg: DictConfig):
  default_cfg = OmegaConf.structured(_instance)
  if cfg is not None:
    default_cfg = OmegaConf.merge(default_cfg, cfg)
  cfg = default_cfg

  return cfg


class Trainer:
  def __init__(self, cfg: DictConfig):
    _instance = build_default_config()
    cfg = update_config(_instance, cfg)

    self.cfg = cfg
    self.best_results = None

  def train_epoch(self, model, train_loader, criterion, optimizer, scheduler, adv_train_attack):
    device = next(model.parameters()).device

    model.train()
    optimizer.zero_grad()
    for batch_num, (inputs, targets) in enumerate(train_loader, start=1):
      inputs, targets = inputs.to(device), targets.to(device)
      if adv_train_attack is not None:
        model.eval()
        k = int(self.cfg.train_loader.batch_size *
                self.cfg.adversarial_training.adversarial_batch_size)
        inputs[:k] = adv_train_attack(inputs[:k], targets[:k])
        model.train()
      outputs = model(inputs)

      loss = criterion(outputs, targets)
      loss.backward()

      if batch_num % self.cfg.grad_acum == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

  def validate(self, model, valid_loader):

    device = next(model.parameters()).device
    correct = 0
    total = 0

    model.eval()
    for _, (inputs, targets) in enumerate(valid_loader):
      inputs, targets = inputs.to(device), targets.to(device)

      pred = model(inputs).argmax(-1)
      correct += (pred == targets).sum().item()
      total += targets.size(0)

    results = torch.tensor([correct, total], dtype=torch.float64, device=device)
    correct, total = all_reduce(results).cpu().numpy()
    return 100 * correct / total

  def test(self, model, test_loader,  evaluation_attacks={('', 'clean'):None}):

    device = next(model.parameters()).device
    epoch_results = pd.DataFrame(
        columns=evaluation_attacks.keys(), index=['acc'], data=0.0)
    epoch_results.style.set_table_styles([dict(selector='th', props=[('text-align', 'left')] )])
    total = 0

    model.eval()
    for _, (inputs, targets) in enumerate(test_loader):
      inputs, targets = inputs.to(device), targets.to(device)

      for key, attack in evaluation_attacks.items():
        # new variable inputs_ to not destroy clean inputs
        inputs_ = inputs
        if attack is not None:
          inputs_ = attack(inputs, targets)

        with torch.no_grad():
          pred = model(inputs_).argmax(-1)
          correct = (pred == targets).sum().item()
          epoch_results[key] += correct

      total += targets.size(0)

    # Reduce results
    results_data = torch.tensor(epoch_results.to_numpy(), dtype=torch.float64, device=device)
    results_data = all_reduce(results_data).cpu().numpy()
    epoch_results = pd.DataFrame(data=results_data, index=epoch_results.index, columns=epoch_results.columns)
    
    total =  torch.tensor([total], dtype=torch.float64, device=device)
    total, = all_reduce(total).cpu().numpy()
    
    epoch_results = 100 * (epoch_results / total)

    # Format results
    columns = ['params', 'field', 'value']
    rows = []
    for key in epoch_results.columns:
      params, attack_name = key
      params = f'{params},attack={attack_name}'
      acc = epoch_results.loc['acc', key]
      row = (params, 'acc', acc)
      
      rows.append(row)
    epoch_results = pd.DataFrame(data=rows, columns=columns)

    return epoch_results

  def fit(self, cfg, rank, gpu, model, dataset, save_best=True, save_last=False):
      log = logging.getLogger(f'train.fit')

      # Wrap model
      model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
      model = model.to(f'cuda:{gpu}')
      model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[gpu],
      )

      # Initialize datasets
      train_sampler = torch.utils.data.distributed.DistributedSampler(
          dataset.train,
          shuffle=True,
          seed=42,
      )
      train_loader = torch.utils.data.DataLoader(
        dataset.train, **self.cfg.train_loader, pin_memory=True, sampler=train_sampler, drop_last=False)

      valid_sampler = torch.utils.data.distributed.DistributedSampler(
          dataset.valid,
          shuffle=False,
      )
      valid_loader = torch.utils.data.DataLoader(
        dataset.valid, **self.cfg.valid_loader, pin_memory=True, sampler=valid_sampler, drop_last=False)

      test_sampler = torch.utils.data.distributed.DistributedSampler(
          Subset(dataset.test, indices=default_rng(42).choice(len(dataset.test), int(cfg.test_fraction*len(dataset.test)), replace=False)),
          shuffle=False,
      )
      test_loader = torch.utils.data.DataLoader(
          dataset.test, **self.cfg.valid_loader, pin_memory=True, sampler=test_sampler, drop_last=False)

      # Initializing loop objects
      start_epoch = 0
      optimizer = optim.SGD(model.parameters(), **self.cfg.optim)
      scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        epochs=self.cfg.max_epochs, steps_per_epoch=len(train_loader)//self.cfg.grad_acum,
        **self.cfg.scheduler
      )
      criterion = nn.CrossEntropyLoss()

      # Restarting train
      if self.cfg.restart.state_dict_path is not None:
        restart_state_dict = torch.load(self.cfg.restart.state_dict_path)

        start_epoch = int(restart_state_dict['epoch'])
        model.load_state_dict(restart_state_dict['model_state_dict'])
        optimizer.load_state_dict(restart_state_dict['optimizer_state_dict'])
        scheduler.load_state_dict(restart_state_dict['scheduler_state_dict'])

        log.info(
              f'Restarted train at epoch {start_epoch} from {self.cfg.restart.state_dict_path}')

      self.best_state_dict = {
                'epoch': start_epoch,
                'model_cntr_args': model.module.cntr_args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
      }

      # Checkpointing last
      if save_last and rank == 0:
        os.makedirs(os.path.dirname(
            self.cfg.checkpoint.last_state_dict_path), exist_ok=True)
        log.info(
            f'Saving last_state_dict to {self.cfg.checkpoint.last_state_dict_path}')

      # Checkpointing best
      if save_best and rank == 0:
        os.makedirs(os.path.dirname(
            self.cfg.checkpoint.best_state_dict_path), exist_ok=True)
        log.info(
            f'Saving best_state_dict to {self.cfg.checkpoint.best_state_dict_path}')

      # Adversarial training
      if self.cfg.adversarial_training.attack == 'None':
        adv_train_attack = None
      elif self.cfg.adversarial_training.attack == 'FGSM':
        adv_train_attack = torchattacks.FGSM(model, eps=8/255.0)
        adv_train_attack.set_mode_targeted_least_likely()
      elif self.cfg.adversarial_training.attack == 'PGD':
        adv_train_attack = torchattacks.PGD(
            model, eps=8/255.0, alpha=2/255.0, steps=7, random_start=True)
      log.info(
            f'Using adversarial training {self.cfg.adversarial_training.attack}')

      self.best_validation_acc = 0

      log.info('Starting train')
      with contexttimer.Timer() as train_timer:
        for epoch in range(start_epoch, self.cfg.max_epochs):
          train_sampler.set_epoch(epoch)

          # Training epoch
          self.train_epoch(model, train_loader, criterion, optimizer, scheduler, adv_train_attack)

          log.info(f'Finished train epoch {epoch}')

          # Evaluation loop
          validation_acc = self.validate(model, valid_loader)

          log.info(f'Finished eval epoch {epoch}, acc={validation_acc:.2f}')

          # Checkpointing last
          self.last_state_dict = {
                'epoch': epoch + 1,
                'model_cntr_args': model.module.cntr_args,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
          }
          if rank == 0 and save_last:
            torch.save(self.last_state_dict, self.cfg.checkpoint.last_state_dict_path)

          # Checkpoint best
          if validation_acc > self.best_validation_acc:
            self.best_validation_acc = validation_acc

            self.best_state_dict = {
                  'epoch': epoch + 1,
                  'model_cntr_args': model.module.cntr_args,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(),
            }
            if rank == 0 and save_best:
              torch.save(self.best_state_dict, self.cfg.checkpoint.best_state_dict_path)
      log.info('Finished training')
      log.info(f'Best validation accuracy: {self.best_validation_acc}')
      model.load_state_dict(self.best_state_dict['model_state_dict'])
      model.eval()
      model = model.module
      model.eval()

      log.info(f'Starting test')
      evaluation_attacks = {
            'eps=0': {
                'clean': None
            },
            'eps=1': {
                'FGSM': torchattacks.FGSM(model, eps=1/255.0),
                'FGSM-L2': torchattacks.PGDL2(model, eps=32.0*1/255.0, alpha=32.0*1/255.0, steps=1, random_start=False),
                'PGD': torchattacks.PGD(model, eps=1/255.0, steps=20),
                'PGD-L2': torchattacks.PGDL2(model, eps=32.0*1/255.0, steps=20),
            },
            'eps=2': {
                'FGSM': torchattacks.FGSM(model, eps=2/255.0),
                'FGSM-L2': torchattacks.PGDL2(model, eps=32.0*2/255.0, alpha=32.0*2/255.0, steps=1, random_start=False),
                'PGD': torchattacks.PGD(model, eps=2/255.0, steps=20),
                'PGD-L2': torchattacks.PGDL2(model, eps=32.0*2/255.0, steps=20),
            },
            'eps=3': {
                'FGSM': torchattacks.FGSM(model, eps=3/255.0),
                'FGSM-L2': torchattacks.PGDL2(model, eps=32.0*3/255.0, alpha=32.0*3/255.0, steps=1, random_start=False),
                'PGD': torchattacks.PGD(model, eps=3/255.0, steps=20),
                'PGD-L2': torchattacks.PGDL2(model, eps=32.0*3/255.0, steps=20),
            },
            'eps=4': {
                'FGSM': torchattacks.FGSM(model, eps=4/255.0),
                'FGSM-L2': torchattacks.PGDL2(model, eps=32.0*4/255.0, alpha=32.0*4/255.0, steps=1, random_start=False),
                'PGD': torchattacks.PGD(model, eps=4/255.0, steps=20),
                'PGD-L2': torchattacks.PGDL2(model, eps=32.0*4/255.0, steps=20),
            },
            'eps=5': {
                'FGSM': torchattacks.FGSM(model, eps=5/255.0),
                'FGSM-L2': torchattacks.PGDL2(model, eps=32.0*5/255.0, alpha=32.0*5/255.0, steps=1, random_start=False),
                'PGD': torchattacks.PGD(model, eps=5/255.0, steps=20),
                'PGD-L2': torchattacks.PGDL2(model, eps=32.0*5/255.0, steps=20),
            },
            'eps=6': {
                'FGSM': torchattacks.FGSM(model, eps=6/255.0),
                'FGSM-L2': torchattacks.PGDL2(model, eps=32.0*6/255.0, alpha=32.0*6/255.0, steps=1, random_start=False),
                'PGD': torchattacks.PGD(model, eps=6/255.0, steps=20),
                'PGD-L2': torchattacks.PGDL2(model, eps=32.0*6/255.0, steps=20),
            },
            'eps=7': {
                'FGSM': torchattacks.FGSM(model, eps=7/255.0),
                'FGSM-L2': torchattacks.PGDL2(model, eps=32.0*7/255.0, alpha=32.0*7/255.0, steps=1, random_start=False),
                'PGD': torchattacks.PGD(model, eps=7/255.0, steps=20),
                'PGD-L2': torchattacks.PGDL2(model, eps=32.0*7/255.0, steps=20),
            },
            'eps=8': {
                'FGSM': torchattacks.FGSM(model, eps=8/255.0),
                'FGSM-L2': torchattacks.PGDL2(model, eps=32.0*8/255.0, alpha=32.0*8/255.0, steps=1, random_start=False),
                'PGD': torchattacks.PGD(model, eps=8/255.0, steps=20),
                'PGD-L2': torchattacks.PGDL2(model, eps=32.0*8/255.0, steps=20),
            },
        }
      evaluation_attacks = {
        (params, attack_name): attack for params in evaluation_attacks.keys()
                                    for attack_name, attack in evaluation_attacks[params].items()}
      with contexttimer.Timer() as test_timer:
        test_results = self.test(model, test_loader, evaluation_attacks=evaluation_attacks)
      log.info(f'Finished test')
      test_results = test_results.append({'params':'train', 'field':'time', 'value':train_timer.elapsed}, ignore_index=True)
      test_results = test_results.append({'params':'test', 'field':'time', 'value':test_timer.elapsed}, ignore_index=True)
      log.info('\n' + test_results.to_string(float_format='{:,.2f}'.format))

      return test_results
