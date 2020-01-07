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
dataset = '/home/bjoern/Thesis/CubeML/data/oscnext-genie-level5-v01-01-pass2'

tot_n_doms = []
tot_energy = []
for file in Path(dataset).iterdir():
    if not (file.suffix == '.h5' and confirm_particle_type(get_particle_code(particle), file)):
        continue
     
    with h5.File(file, 'r') as f:
        print(f['raw'].keys)
        n_doms = [x.shape[0] for x in f['raw/dom_charge']]
        energy = f['raw/true_primary_energy']

        tot_n_doms.extend(n_doms)
        tot_energy.extend(energy)
doms, energy = sort_pairs(tot_n_doms, tot_energy)

# tot_n_doms = [entry for entry in tot_n_doms if entry<100]
# %%
d1 = {'data': [energy[0:3000], energy[-3000:-50]]}
a1 = rpt.make_plot(d1)

d2 = {'data': [doms[0:3000], doms[-3000:-50]]}
a2 = rpt.make_plot(d2)

# %%
morethan200 = len([entry for entry in doms if entry > 200])
print(morethan200/len(doms))