#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess
from multiprocessing import Pool, cpu_count

# from src.modules.classes import *
import src.modules.loss_funcs as lf
import src.modules.helper_functions as hf
from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.constants import *
from src.modules.classes import *
import src.modules.preprocessing as pp
from subprocess import call


# tot = []

# for entry in d['dom_charge_over_vertex']:
#     tot.extend(entry)

# tot = sorted(tot)
# tot = np.array(tot)
# print('we done here too')

# tot = (tot - np.median(tot))/calc_iqr(tot)
# path = get_project_root() + '/plots/dom_d_mink_to_prev.png'
# title = r'$d_{Minkowski}$ from $DOM_{t-1}$ to $DOM_{t} $'
# pd = {'data': [tot[:167000]], 'log': [False]}#, 'title': title, 'savefig': path}
# f = rpt.make_plot(pd)
# print(tot.shape)
# tot = []
# for entry in d['dom_t']:
#     tot.extend(entry)

# tot = sorted(tot)
# pd = {'data': [tot], 'log': [False]}#, 'title': title, 'savefig': path}
# f = rpt.make_plot(pd)
