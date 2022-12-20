import math

import numpy as np
import scipy.special
import src.antialiased_layers.stylegan3.torch_utils.ops.upfirdn2d as upfirdn2d
import torch
import torch.nn as nn
from scipy.signal import firwin
from torch import Tensor


# Initial Blur module
class InitialBlur2d(nn.Module):
  def __init__(
      self,
      order: int = 1
  ) -> None:
    super().__init__()
    self.order = order

    self.register_buffer('lpf', make_binomial_filter(order=order))

  def forward(self, inputs: Tensor):
    lpf = self.lpf
    padding = (lpf.shape[0] - 1) // 2
    outputs = upfirdn2d.upfirdn2d(inputs, lpf, padding=padding)
    return outputs

# Gaussian filter
def make_gaussian_filter(stddev):
  order = math.ceil(3*stddev)
  n = np.arange(-order, order+1)
  h = math.exp(-stddev) * scipy.special.iv(n, stddev)
  h = np.array(h, dtype=np.float32)
  h = h / np.sum(h)
  h = torch.from_numpy(h)
  return h


# Binomial filter
def make_binomial_filter(order):
  h = np.array([1, 2, 1])
  return exponentiate_filter(h, order)

def exponentiate_filter(h, pow):
  h = (np.poly1d(h)**pow).coeffs
  h = np.array(h, dtype=np.float32)
  h = h / np.sum(h)
  h = torch.from_numpy(h)
  return h


# Kaiser filter
def make_kaiser_filter(fc, fh=None, numtaps=5):
  if fc == 0.5:
    h = np.array([1.0], dtype=np.float32)
  else:
    fh = (1/2) - fc if fh is None else fh
    h = firwin(numtaps, cutoff=fc, width=2*fh, fs=1)
    h = h / np.sum(h)
    h = h.astype(np.float32)
  h = torch.from_numpy(h)
  return h
