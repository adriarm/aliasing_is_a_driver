import logging
import os
import random
import sys

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data.dataset import Subset

from src.datasets import datasets
from src.models.models import get_checkpoint, get_model
from src.train.trainer import Trainer
from src.utils.distributed import all_reduce, is_dist_avail_and_initialized, all_gather


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

@hydra.main(config_path='conf/examplegen', config_name='config')
def main(cfg: DictConfig) -> None:
  OmegaConf.resolve(cfg)
  print(OmegaConf.to_yaml(cfg))

  # Seed
  set_seed(cfg.seed)

  # Save config
  save_config('config', cfg)

  # Launch
  launch_examplegen(cfg)

def launch_examplegen(cfg):
  log = logging.getLogger('examplegen')

  # Load dataset
  log.info(f'Getting dataset {cfg.dataset}')
  dataset = datasets.get_dataset(cfg.dataset)
  dataset = Subset(dataset.test, indices=np.random.choice(len(dataset.test), cfg.num_examples, replace=False))

  test_loader = torch.utils.data.DataLoader(
    dataset, **cfg.evaluator.test_loader, pin_memory=True, drop_last=False)
  
  # Load model
  log.info(f'Loading model {cfg.model_path}')
  hc = HydraConfig.instance().get()
  model, checkpoint = get_checkpoint(os.path.join(hc.runtime.cwd, cfg.model_path, 'checkpoints', 'best_state_dict.pt'))
  state_dict = checkpoint['model_state_dict']
  # state_dict = {k.removeprefix('module.'):v for k,v in state_dict.items()}
  state_dict = {k[len('module.'):]:v for k,v in state_dict.items()}

  model.load_state_dict(state_dict)
  model.eval()
  model = model.to(f'cuda')

  # compute and save metric
  log.info(f'Generating examples')

  ## FGSM
  images = []
  predictions = []
  for epsilon in range(0, 9):
    attack = torchattacks.FGSM(model, eps=epsilon/255.0)

    images_batch, predictions_batch = generate_examples(model, attack, test_loader)

    images.append(images_batch)
    predictions.append(predictions_batch)
  images = np.stack(images) # len(epsilon), num_examples, H, W, C
  predictions = np.stack(predictions) # len(epsilon), num_examples, 3

  np.savez(os.path.join(os.getcwd(), 'FGSM.npz'), images=images, predictions=predictions)
  log.info('Saved FGSM')

  ## FGSM-L2
  images = []
  predictions = []
  for epsilon in range(0, 9):
    attack = torchattacks.PGDL2(model, eps=32.0*epsilon/255.0, steps=1, random_start=False)

    images_batch, predictions_batch = generate_examples(model, attack, test_loader)

    images.append(images_batch)
    predictions.append(predictions_batch)
  images = np.stack(images) # len(epsilon), num_examples, H, W, C
  predictions = np.stack(predictions) # len(epsilon), num_examples, 3

  np.savez(os.path.join(os.getcwd(), 'FGSM_L2.npz'), images=images, predictions=predictions)
  log.info('Saved FGSM-L2')

  ## PGD
  images = []
  predictions = []
  epsilon_list = list(range(0, 8+1))
  steps_list = [20] #list(range(1, 20+1))
  for epsilon in epsilon_list:
    for steps in steps_list:
      attack = torchattacks.PGD(model, eps=epsilon/255.0, steps=steps)

      images_batch, predictions_batch = generate_examples(model, attack, test_loader)

      images.append(images_batch)
      predictions.append(predictions_batch)
  images = np.stack(images) # len(epsilon)*len(steps), num_examples, H, W, C
  predictions = np.stack(predictions) # len(epsilon)*len(steps), num_examples, 3

  images = np.reshape(images, (len(epsilon_list), len(steps_list), *images.shape[1:]))
  predictions = np.reshape(predictions, (len(epsilon_list), len(steps_list), *predictions.shape[1:]))

  np.savez(os.path.join(os.getcwd(), 'PGD.npz'), images=images, predictions=predictions)
  log.info('Saved PGD')

  ## PGD-L2
  images = []
  predictions = []
  epsilon_list = list(range(0, 8+1))
  steps_list = [20] #list(range(1, 20+1))
  for epsilon in epsilon_list:
    for steps in steps_list:
      attack = torchattacks.PGDL2(model, eps=32.0*epsilon/255.0, steps=steps)

      images_batch, predictions_batch = generate_examples(model, attack, test_loader)

      images.append(images_batch)
      predictions.append(predictions_batch)
  images = np.stack(images) # len(epsilon)*len(steps), num_examples, H, W, C
  predictions = np.stack(predictions) # len(epsilon)*len(steps), num_examples, 3

  images = np.reshape(images, (len(epsilon_list), len(steps_list), *images.shape[1:]))
  predictions = np.reshape(predictions, (len(epsilon_list), len(steps_list), *predictions.shape[1:]))

  np.savez(os.path.join(os.getcwd(), 'PGD_L2.npz'), images=images, predictions=predictions)
  log.info('Saved PGD-L2')
  
def generate_examples(model, attack, loader):  

  images = [] #np.empty((N, 32, 32, 3), dtype=np.float32) 
  predictions = [] #np.empty((N, 3), dtype=np.float32) # target, confidence

  for inputs, labels in loader:
    inputs, labels = inputs.to('cuda'), labels.to('cuda')
    attacked_inputs = attack(inputs, labels)
    with torch.no_grad():
      outputs = model(attacked_inputs)
      outputs = F.softmax(outputs, dim=-1)
      conf, pred = torch.max(outputs.data, 1)
    
    # save images
    if torch.allclose(inputs, attacked_inputs):
      image_batch = attacked_inputs
      # print('0 epsilon detected')
    else:
      image_batch = attacked_inputs - inputs
    images.append(torch.permute(image_batch, (0, 2, 3, 1)).detach().cpu().numpy())

    # alignment
    flat_image_batch = image_batch.flatten(start_dim=1)
    flat_inputs = inputs.flatten(start_dim=1)
    cossim = torch.nn.functional.cosine_similarity(flat_image_batch, flat_inputs, dim=1, eps=1e-08)

    # save prediction
    conf_batch = conf.detach().cpu().numpy()
    pred_batch = pred.detach().cpu().numpy()
    labels_batch = labels.detach().cpu().numpy()
    cossim_batch = cossim.detach().cpu().numpy()
    # print(f'conf_batch:{conf_batch.shape}', f'cossim_batch:{cossim_batch.shape}')
    predictions_batch = np.stack((conf_batch, pred_batch, labels_batch, cossim_batch), axis=-1)
    predictions.append(predictions_batch)

  images = np.concatenate(images)
  predictions = np.concatenate(predictions)

  return images, predictions

if __name__ == '__main__':
  main()