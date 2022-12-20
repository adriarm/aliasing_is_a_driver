from dataclasses import dataclass

import numpy as np
import src.antialiased_layers.filters as filters
import src.antialiased_layers.stylegan3.torch_utils.ops.upfirdn2d as upfirdn2d
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.conv import Conv2d


@dataclass
class AAConv2dConfig:
  antialiasing: bool = False


class AAConv2d(nn.Module):

  def __init__(
      self,
      conv: Conv2d,
      antialiasing: bool = False
  ) -> None:
    super().__init__()
    self.conv = conv
    self.antialiasing = antialiasing

    # Build low-pass filters
    for numtaps in [3, 5, 7]:
      fc = 1 / (2 * self.conv.stride[0]) if self.antialiasing else 0.5
      lpf = filters.make_kaiser_filter(fc, numtaps=numtaps)
      self.register_buffer(f'lpf_{numtaps}', lpf, persistent=False)
  

  def get_lpf(self, inputs: Tensor):
    H = inputs.shape[-1]
    numtaps = np.clip(2*((H+1)//8) + 1, a_min=3, a_max=7)
    lpf = getattr(self, f'lpf_{numtaps}')
    return lpf


  def forward(self, inputs: Tensor) -> Tensor:
    outputs = inputs
    # Blur
    lpf = self.get_lpf(outputs)
    padding = (lpf.shape[0] - 1) // 2
    outputs = upfirdn2d.upfirdn2d(outputs, lpf, padding=padding)
    # Conv
    outputs = self.conv(outputs)
    return outputs


  @classmethod
  def convert_antialiased(cls, module: nn.Module, cfg: AAConv2dConfig):
    module_output = module
    if isinstance(module, nn.Conv2d):
      module_output = AAConv2d(module, **cfg)
    for name, child in module.named_children():
      module_output.add_module(
          name, cls.convert_antialiased(child, cfg)
      )
    del module
    return module_output


