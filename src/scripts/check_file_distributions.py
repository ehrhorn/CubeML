import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess
from random import shuffle as shuffler

# from src.modules.classes import *
import src.modules.loss_funcs as lf
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
import src.modules.reporting as rpt

# %% ENERGY DISTRIBUTIONS
particle = 'muon_neutrino'
dataset = get_project_root()+get_path_from_root('/CubeML/data/oscnext-genie-level5-v01-01-pass2')

train, val, test = split_files_in_dataset(dataset, particle=particle)

# * Get random files
rand_train = np.arange(len(train))
shuffler(rand_train)
energy_train = []
n_in_file_train = []

rand_val = np.arange(len(val))
shuffler(rand_val)
energy_val = []
n_in_file_val = []

rand_test = np.arange(len(test))
shuffler(rand_test)
energy_test = []
n_in_file_test = []

n_wanted = 50000
key = 'raw/true_primary_time'

n_read = 0
for index in rand_train:
    file = get_project_root() + train[index]
    if n_read >= n_wanted:
        break
     
    with h5.File(file, 'r') as f:
        # energy = f[key]
        n_in_file_train.append(f['meta/events'][()])
        # n_read += len(energy)
        # energy_train.extend(energy)

n_read = 0
for index in rand_val:
    file = get_project_root() + train[index]
    if n_read >= n_wanted:
        break
     
    with h5.File(file, 'r') as f:
        # energy = f[key]
        n_in_file_val.append(f['meta/events'][()])
        # n_read += len(energy)
        # energy_val.extend(energy)

# d = {'data': [energy_train, energy_val]}
# fig = rpt.make_plot(d)
title = 'Distribution of number of events in files'
path_save = get_project_root() + '/plots/n_events_in_files.png'
d = {'data': [n_in_file_train, n_in_file_val], 'density': [True, True], 'title': title, 'label': ['Train', 'Val'], 'savefig': path_save}
fig2 = rpt.make_plot(d)

# d = {'x': [np.arange(len(energy_train))], 'y': [energy_train]}
# train_fig = rpt.make_plot(d_train)
# tot_n_doms = [entry for entry in tot_n_doms if entry<100]
