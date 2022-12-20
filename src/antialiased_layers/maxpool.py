# Using upfirdn2d from Stylegan3

from dataclasses import dataclass
from typing import Optional

import numpy as np
import src.antialiased_layers.filters as filters
import src.antialiased_layers.stylegan3.torch_utils.ops.upfirdn2d as upfirdn2d
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.pooling import MaxPool2d


@dataclass
class AAMaxPool2dConfig:
  pre_blur: bool = False
  post_blur: bool = False


class AAMaxPool2d(nn.Module):

  def __init__(
      self,
      maxpool: MaxPool2d,
      pre_blur: bool = False,
      post_blur: bool = False
  ) -> None:
    super().__init__()
    self.U = 2
    self.maxpool = maxpool
    self.pre_blur = pre_blur
    self.post_blur = post_blur

    # Build pre-blur low-pass filters
    for numtaps in [3, 5, 7]:
      K = (self.maxpool.kernel_size * self.U - self.U + 1) if self.pre_blur else 1
      fc = 1 / (2 * K)
      lpf = filters.make_kaiser_filter(fc, numtaps=numtaps)
      self.register_buffer(f'pre_lpf_{numtaps}', lpf, persistent=False)

    # Build post-blur low-pass filters
    for numtaps in [3, 5, 7]:
      fc = 1 / (2 * self.maxpool.stride * self.U) if self.post_blur else 0.5
      lpf = filters.make_kaiser_filter(fc, numtaps=numtaps)
      self.register_buffer(f'post_lpf_{numtaps}', lpf, persistent=False)


  def get_pre_lpf(self, inputs: Tensor):
    H = inputs.shape[-1]
    numtaps = np.clip(2*((H+1)//8) + 1, a_min=3, a_max=7)
    lpf = getattr(self, f'pre_lpf_{numtaps}')
    return lpf


  def get_post_lpf(self, inputs: Tensor):
    H = inputs.shape[-1]
    numtaps = np.clip(2*((H+1)//8) + 1, a_min=3, a_max=7)
    lpf = getattr(self, f'post_lpf_{numtaps}')
    return lpf


  def forward(self, inputs: Tensor):
    outputs = inputs
    # Oversampling
    U = self.U if self.pre_blur else 1
    # Pre-blur
    lpf = self.get_pre_lpf(outputs)
    padding = (lpf.shape[0] - 1) // 2
    outputs = upfirdn2d.upfirdn2d(outputs, lpf, up=U, padding=padding)
    # Pool
    kernel_size = self.maxpool.kernel_size * U - U + 1
    stride = self.maxpool.stride * U
    padding = (kernel_size - stride + 1) // 2
    outputs = F.max_pool2d(outputs, kernel_size, stride=1, padding=padding)
    offset = int(padding == kernel_size // 2)
    # Post-blur
    lpf = self.get_post_lpf(outputs) # Uses up-sampled size
    padding = (lpf.shape[0] - 1) // 2 + offset
    outputs = upfirdn2d.upfirdn2d(outputs, lpf, down=stride, padding=padding)
    outputs = outputs[..., offset:, offset:]
    return outputs


  @classmethod
  def convert_antialiased(cls, module: nn.Module, cfg: AAMaxPool2dConfig):
    module_output = module
    if isinstance(module, nn.MaxPool2d):
      module_output = AAMaxPool2d(module, **cfg)
    for name, child in module.named_children():
      module_output.add_module(
          name, cls.convert_anti_aliased_conv(child, cfg)
      )
    del module
    return module_output
