# Using filtered_lrelu from Stylegan3

from dataclasses import dataclass
from typing import Optional

import numpy as np
import src.antialiased_layers.filters as filters
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.antialiased_layers.stylegan3.torch_utils.ops.filtered_lrelu import \
    filtered_lrelu
from torch import Tensor


@dataclass
class AAReLUConfig:
  L: int = 1
  U: int = 1
  q: float = -1


class AAReLU(nn.Module):
  def __init__(self, L: int = 1, U: int = 1, q: float = -1) -> None:
    super().__init__()
    self.L = L
    self.U = U
    self.q = q

    # Build low-pass filters
    for numtaps in [3, 5, 7]:
      fc = 1 / (2 * self.L * self.U)
      lpf = filters.make_kaiser_filter(fc, numtaps=numtaps)
      self.register_buffer(f'lpf_{numtaps}', lpf, persistent=False)


  def get_lpf(self, inputs: Tensor):
    H = inputs.shape[-1]
    numtaps = np.clip(2*((H+1)//8) + 1, a_min=3, a_max=7)
    lpf = getattr(self, f'lpf_{numtaps}')
    return lpf


  def forward(self, inputs: Tensor) -> Tensor:
    outputs = inputs
    # Quantile Adjust
    outputs = self.quantile_adjust(outputs)
    # Up Apply Down
    lpf = self.get_lpf(outputs)
    padding = (lpf.shape[0] - 1)
    outputs = filtered_lrelu(outputs, fu=lpf, fd=lpf, up=self.U, down=self.U, padding=padding, gain=1, slope=0, impl='cuda')
    return outputs


  def quantile_adjust(self, inputs: Tensor):
    if self.q > 0:
      batch_size, num_channels, _, _ = inputs.shape
      q_inputs = torch.quantile(inputs.view(
          batch_size, num_channels, -1), q=self.q, dim=-1)
      inputs = inputs + F.relu(-q_inputs[:, :, None, None])
    return inputs


  @classmethod
  def convert_antialiased(cls, module: nn.Module, cfg: AAReLUConfig):
    module_output = module
    if isinstance(module, nn.ReLU):
      module_output = AAReLU(**cfg)
    for name, child in module.named_children():
      module_output.add_module(
          name, cls.convert_antialiased(child, cfg)
      )
    del module
    return module_output
