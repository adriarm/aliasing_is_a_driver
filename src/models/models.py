# Inspiration on registering scheme
""" @inproceedings{Karras2021,
  author = {Tero Karras and Miika Aittala and Samuli Laine and Erik Harkonen and Janne Hellsten and Jaakko Lehtinen and Timo Aila},
  title = {Alias-Free Generative Adversarial Networks},
  booktitle = {Proc. NeurIPS},
  year = {2021}
} """

#----------------------------------------------------------------------------

from functools import partial

import torch
import torchvision
from src.models.resnet import antialiased_resnet
from src.models.vgg import antialiased_vgg
# from src.models.vision_transformer import antialiased_vision_transformer

_model_dict = {}

def register_model(fn):
    assert callable(fn)
    _model_dict[fn.__name__] = fn
    return fn

def is_valid_model(model):
    return model in _model_dict

def list_valid_models():
    return list(_model_dict.keys())

#----------------------------------------------------------------------------

def get_model(cfg, **kwargs):
  return _model_dict[cfg.arch](cfg=cfg, **kwargs)

def get_checkpoint(path_to_checkpoint):
  checkpoint = torch.load(path_to_checkpoint, map_location='cpu')
  cntr_args = checkpoint['model_cntr_args']
  model = get_model(**cntr_args)
  model.eval()
  return model, checkpoint

#----------------------------------------------------------------------------

@register_model
def resnet50(cfg, **kwargs):
  return antialiased_resnet(torchvision.models.resnet50, cfg=cfg, **kwargs)

@register_model
def vgg11(cfg, **kwargs):
  return antialiased_vgg(torchvision.models.vgg11_bn, cfg=cfg, **kwargs)

# @register_model
# def vit(cfg, **kwargs):
#   assert 'in_size' in kwargs, '"in_size" must be passed in.'
#   model_cntr = partial(
#         torchvision.models.vision_transformer._vision_transformer,
#         patch_size=(kwargs['in_size']//16),
#         num_layers=12,
#         num_heads=12,
#         hidden_dim=768,
#         mlp_dim=3072,
#         weights=None,
#         progress=False
#   )
#   return antialiased_vision_transformer(model_cntr, cfg=cfg, **kwargs)
