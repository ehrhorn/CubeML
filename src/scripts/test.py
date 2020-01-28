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

# * First, make all directories.
n_dirs = 1131
path = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2/'
for i in range(n_dirs):
    if not Path(path+str(i)).exists():
        Path(path+str(i)).mkdir()

# * Now, prep a list of from-paths
# hep_dir = '/groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/'
# gpu_dir = '/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2'
# dir_names = [path + str(i) for i ]

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
