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

def get_tot_charge(dataset, d):
    # * SEEMS TO WORK
    charge = 'raw/dom_charge'
    d['dom_tot_charge'] = np.array([0.0]*f['meta/events'][()])
    for i, event in enumerate(dataset[charge]):
        d['dom_tot_charge'][i] = np.sum(event)
    return d

def get_tot_charge_frac(dataset, d):
     # * SEEMS TO WORK
    charge = 'raw/dom_charge'
    key = 'dom_tot_charge_frac'
    if not 'dom_tot_charge' in d:
        d = get_tot_charge(dataset, d)
    
    d[key] = [[]]*f['meta/events'][()]
    for i_event, event in enumerate(dataset[charge]):
        # * charge_frac.shape = (n_doms,)
        d[key][i_event] = event/d['dom_tot_charge'][i_event]
    
    return d

def get_frac_of_n_doms(dataset, d):
    # * SEEMS TO WORK
    charge = 'raw/dom_charge'
    key = 'dom_frac_of_n_doms'
    d[key] = [[]]*f['meta/events'][()]
    for i_event, event in enumerate(dataset[charge]):
        # * charge_frac.shape = (n_doms,)
        n_doms = event.shape[0]
        d[key][i_event] = np.arange(1, n_doms+1)/n_doms

    return d
data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2'
particle_code = '140000'

file_list = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5' and confirm_particle_type(particle_code, file)])

N_FILES = len(file_list)
    
for file in file_list:
    # print(Path(file).name)
    # * open file and prep new file
    d = {}
    with h5.File(file, 'r') as f:
        d = get_tot_charge(f, d)
        d = get_tot_charge_frac(f, d)
        d = get_frac_of_n_doms(f, d)
    break
    # * calculate relevant stuff

    # * put in new file and save it
tot_charge_fracs = []
for entry in d['dom_frac_of_n_doms']:
    tot_charge_fracs.extend(entry)
pd = {'data': [tot_charge_fracs]}
f = rpt.make_plot(pd)