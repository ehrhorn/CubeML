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

def calc_dist():
    pass

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

def get_d_to_prev(dataset, d):
    x, y, z = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z'
    key = 'dom_d_to_prev'
    n_events = f['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):
        n_doms = event.shape[0]
        x_diff = dataset[x][i_event][1:] - dataset[x][i_event][:-1]
        y_diff = dataset[y][i_event][1:] - dataset[y][i_event][:-1]
        z_diff = dataset[z][i_event][1:] - dataset[z][i_event][:-1]
        dists = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        dists = np.append([0.0], dists)
        d[key][i_event] = dists
    return d

def get_v_from_prev(dataset, d):
    t = 'raw/dom_time'
    key = 'dom_v_from_prev'
    if 'dom_d_to_prev' not in d:
        d = get_d_to_prev(dataset, d)

    n_events = f['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):
        t_diff = dataset[t][i_event][1:] - dataset[t][i_event][:-1]

        # * Time has discrete values due to the clock on the electronics + the pulse extraction algorithm bins the pulses in time --> more discreteness.
        t_diff = np.where(t_diff==0, 1.0, t_diff)
        t_diff = np.append([np.inf], t_diff)
        
        v = d['dom_d_to_prev'][i_event]/t_diff
        d[key][i_event] = v
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
        # d = get_tot_charge(f, d)
        # d = get_tot_charge_frac(f, d)
        # d = get_frac_of_n_doms(f, d)
        d = get_d_to_prev(f, d)
        d = get_v_from_prev(f, d)

    break
    # * calculate relevant stuff

    # * put in new file and save it

#%%

tot = []
for entry in d['dom_v_from_prev']:
    tot.extend(entry)

tot = sorted(tot)
pd = {'data': [tot], 'log': [False]}
f = rpt.make_plot(pd)