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

particle = 'muon_neutrino'
dataset = get_project_root()+get_path_from_root('/CubeML/data/oscnext-genie-level5-v01-01-pass2')

tot_energy = []
tot_tot_charge = []
events_wanted = np.inf
events_loaded = 0
for file in Path(dataset).iterdir():
    if events_loaded >= events_wanted:
        break
    if not (file.suffix == '.h5' and confirm_particle_type(get_particle_code(particle), file)):
        continue
     
    with h5.File(file, 'r') as f:
        energy = f['raw/true_primary_energy']
        charge = f['raw/dom_charge']
        n = f['meta/events'][()]

        tot_energy.extend(energy)
        tot_charge = [0]*n
        for i, event in enumerate(charge):
            tot_charge[i] = np.sum(event)
        tot_tot_charge.extend(tot_charge)

        events_loaded += len(energy)

# tot_n_doms = [entry for entry in tot_n_doms if entry<100]
# %% 
# * SORTED WRT ENERGY
energy_sorted, tot_charge_sorted = sort_pairs(tot_energy, tot_tot_charge)

FRAC = 0.1
from_, to_ = 0.0, 0.1
end = int(FRAC*len(tot_energy))
from_i, to_i = int(from_*len(tot_energy)), int(to_*len(tot_energy))

from2_, to2_ = 0.9, 1.0
end = int(FRAC*len(tot_energy))
from2_i, to2_i = int(from2_*len(tot_energy)), int(to2_*len(tot_energy))

path1 = get_project_root() + '/plots/transformed_E_dist.png'
title1 = 'Transformed energy distribution'
d1 = {'data': [tot_charge_sorted[from_i:to_i], tot_charge_sorted[from2_i:to2_i]], 'density': [True, True]}#, 'title': title1, 'savefig': path1}
a1 = rpt.make_plot(d1)
x = np.arange(len(energy_sorted))
d = {'x': [x], 'y': [energy_sorted]}
f = rpt.make_plot(d)

# * MAKING A CUT IN TOT CHARGE
tot_charge_sorted, energy_sorted = sort_pairs(tot_tot_charge, tot_energy)
tot_charge_sorted = np.array(tot_charge_sorted)
energy_sorted = np.array(energy_sorted)
charge_cut = 80.0
indices = tot_charge_sorted < charge_cut

energy_cutted = energy_sorted[indices]
d1 = {'data': [energy_sorted[indices], energy_sorted[~indices]], 'density': [False, False]}#, 'title': title1, 'savefig': path1}

f = rpt.make_plot(d1)