#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess

# from src.modules.classes import *
import src.modules.loss_funcs as lf
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.constants import *
from src.modules.classes import *

# def variable_hook(grad):
#     print('len_x')
#     print('grad', grad)
#     return grad*.1

# def clip_grad(grad):
#     print('x')
#     print('grad', grad)
#     return torch.clamp(grad, min=-10, max=10)

x = torch.tensor([0.001, 0.1, 0.0], requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0], requires_grad=True) 
loss_fn = lf.get_loss_func('angle_squared_loss_with_L2')
loss = loss_fn(x, y)
loss.backward()
print(x.grad)
############
# batch_size = x.shape[0]
# zeros, ones = torch.zeros(batch_size), torch.ones(batch_size)

# # * Check for zero-length vectors --> neutralize them by adding the target. Makes gradient 0 (which is what matters) and make other entries in batch count more. 
# len_x_checker = torch.sqrt(torch.sum(x*x, dim=-1))
# zero_vectors = torch.where(len_x_checker <= eps, ones, zeros)
# n_zeros = torch.sum(zero_vectors)
# mult_factor = batch_size/(batch_size-n_zeros)
# x_altered = x #+ zero_vectors.unsqueeze(1)*y

# # * Add eps to avoid infinite gradient.
# len_x = torch.sqrt(eps**2+torch.sum(x_altered*x_altered, dim=-1))# + zero_vectors
# len_y = torch.sqrt(torch.sum(y*y, dim=-1))
# dot_prods = torch.sum(x*y, dim=-1)
# cos = dot_prods/(len_x*len_y*(1+eps))
# loss = torch.acos(cos)**2
# loss_mean = torch.mean(loss)


# # len_x_checker.register_hook(variable_hook)
# x.register_hook(clip_grad)

# loss_mean.backward()