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
from torchvision.models.resnet import ResNet

log = logging.getLogger(__name__)  


# Resnet wrapper
class AAResNet(ResNet):
  
   def forward(self, inputs: Tensor):
    outputs = inputs
    outputs = self.normalizer(outputs)
    outputs = self.initial_blur(outputs)
    outputs = self._forward_impl(outputs)
    return outputs


# Data classes
@dataclass
class ResLayerConfig:
  conv: AAConv2dConfig = AAConv2dConfig()
  relu: AAReLUConfig = AAReLUConfig()


@dataclass
class ResnetConfig:
  # Arch
  arch: str = 'resnet'
  # Initial Blur
  initial_blur: int = 0
  # Layer0
  conv1: AAConv2dConfig = AAConv2dConfig()
  relu: AAReLUConfig = AAReLUConfig()
  maxpool: AAMaxPool2dConfig = AAMaxPool2dConfig()  
  # ResLayers
  layer1: ResLayerConfig = ResLayerConfig()
  layer2: ResLayerConfig = ResLayerConfig()
  layer3: ResLayerConfig = ResLayerConfig()
  layer4: ResLayerConfig = ResLayerConfig()


def expand_cfg(cfg):
  default_cfg = ResnetConfig()
  cfg = OmegaConf.merge(default_cfg, cfg)
  return cfg


def antialiased_resnet(model_cntr, num_classes: int, cfg: DictConfig = None, in_size: int = None, normalizer: nn.Module = None):
  # Expand cfg
  cfg = expand_cfg(cfg)

  # Construct resnet50
  resnet = model_cntr(num_classes=num_classes)

  # Change type to wrapper
  resnet.__class__ = AAResNet

  # Save constructor parameters
  resnet.cntr_args = {'num_classes':num_classes, 'cfg':cfg, 'in_size':in_size, 'normalizer':normalizer}
  resnet.num_classes = num_classes
  resnet.cfg = cfg

  # Normalize
  resnet.normalizer = normalizer if normalizer is not None else nn.Identity()

  # Anti-aliased modifications
  ## Initial Blur
  resnet.initial_blur = InitialBlur2d(order=cfg.initial_blur)
  ## Layer0
  resnet.conv1 = AAConv2d(resnet.conv1, **cfg.conv1)
  resnet.relu = AAReLU(**cfg.relu)
  resnet.maxpool = AAMaxPool2d(resnet.maxpool, **cfg.maxpool)
  ## ResLayers
  for reslayer_id in [1, 2, 3, 4]:
    name = f'layer{reslayer_id}'
    layer_cfg = getattr(cfg, name)
    resnet.add_module(name, AAConv2d.convert_antialiased(getattr(resnet, name), layer_cfg.conv))
    resnet.add_module(name, AAReLU.convert_antialiased(getattr(resnet, name), layer_cfg.relu))

  # Size modifications
  if in_size is not None and in_size <= 64:
    resnet.conv1 = AAConv2d(nn.Conv2d(3, resnet.conv1.conv.out_channels, kernel_size=3, stride=1, padding=1, bias=False), **cfg.conv1)
    resnet.maxpool = nn.Identity()

  return resnet
