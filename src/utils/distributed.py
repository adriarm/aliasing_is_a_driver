import torch
import torch.distributed as dist
from typing import List


## Distributed utils
def is_dist_avail_and_initialized():
  if not dist.is_available():
      return False
  if not dist.is_initialized():
      return False
  return True

def all_reduce(t: torch.Tensor) -> torch.Tensor:
  if is_dist_avail_and_initialized():
    dist.barrier()
    dist.all_reduce(t)
  return t

def all_gather(tensor_list: List[torch.Tensor], tensor: torch.Tensor):
  if is_dist_avail_and_initialized():
    dist.barrier()
    dist.all_gather(tensor_list, tensor)
  return tensor_list