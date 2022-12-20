import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torchvision
from omegaconf import MISSING, DictConfig, OmegaConf
from src.antialiased_layers.conv import AAConv2d, AAConv2dConfig
from src.antialiased_layers.filters import InitialBlur2d
from src.antialiased_layers.maxpool import (AAMaxPool2d,
                                                   AAMaxPool2dConfig)
from src.antialiased_layers.relu import AAReLU, AAReLUConfig
from torch import Tensor
from torchvision.models.vgg import VGG

log = logging.getLogger(__name__)  


# Resnet wrapper
class AAVGG(VGG):
  
   def forward(self, inputs: Tensor):
    outputs = inputs
    # Normalizer
    outputs = self.normalizer(outputs)
    # Initial Blur
    outputs = self.initial_blur(outputs)
    # VGG
    outputs = self.features(outputs)
    outputs = self.avgpool(outputs)
    outputs = torch.flatten(outputs, 1)
    outputs = self.classifier(outputs)
    return outputs


# Data classes
@dataclass
class VGGBlockConfig:
  conv: AAConv2dConfig = AAConv2dConfig()
  relu: AAReLUConfig = AAReLUConfig()
  maxpool: AAMaxPool2dConfig = AAMaxPool2dConfig()


@dataclass
class VGGConfig:
  # Arch
  arch: str = 'vgg'
  # Initial Blur
  initial_blur: int = 0
  # VGGBlocks
  block0: VGGBlockConfig = VGGBlockConfig()
  block1: VGGBlockConfig = VGGBlockConfig()
  block2: VGGBlockConfig = VGGBlockConfig()
  block3: VGGBlockConfig = VGGBlockConfig()
  block4: VGGBlockConfig = VGGBlockConfig()


def expand_cfg(cfg):
  default_cfg = VGGConfig()
  cfg = OmegaConf.merge(default_cfg, cfg)
  return cfg


def antialiased_vgg(model_cntr, num_classes: int, cfg: DictConfig = None, in_size: int = None, normalizer: nn.Module = None):
  # Expand cfg
  cfg = expand_cfg(cfg)

  # Construct vgg11
  vggnet = model_cntr(num_classes=num_classes)

  # Change type to wrapper
  vggnet.__class__ = AAVGG

  # Save constructor parameters
  vggnet.cntr_args = {'num_classes':num_classes, 'cfg':cfg, 'in_size':in_size, 'normalizer':normalizer}
  vggnet.num_classes = num_classes
  vggnet.cfg = cfg

  # Normalize
  vggnet.normalizer = normalizer if normalizer is not None else nn.Identity()

  # Anti-aliased modifications
  ## Initial Blur
  vggnet.initial_blur = InitialBlur2d(order=cfg.initial_blur)
  ## VGGBlocks
  block_idx = 0
  for module_idx, m in enumerate(vggnet.features):
    block_cfg = getattr(cfg, f'block{block_idx}')
    if isinstance(m, nn.Conv2d):
      vggnet.features[module_idx] = AAConv2d(m, **block_cfg.conv)
    if isinstance(m, nn.ReLU):
      vggnet.features[module_idx] = AAReLU(**block_cfg.relu)
    if isinstance(m, nn.MaxPool2d):
      vggnet.features[module_idx] = AAMaxPool2d(m, **block_cfg.maxpool)   
      block_idx = block_idx+1        

  return vggnet
