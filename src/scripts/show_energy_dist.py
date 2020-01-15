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
events_wanted = np.inf
events_loaded = 0
for file in Path(dataset).iterdir():
    if events_loaded >= events_wanted:
        break
    if not (file.suffix == '.h5' and confirm_particle_type(get_particle_code(particle), file)):
        continue
     
    with h5.File(file, 'r') as f:
        energy = f['raw/true_primary_energy']
        events_loaded += len(energy)
        tot_energy.extend(energy)

# tot_n_doms = [entry for entry in tot_n_doms if entry<100]
# %%

path1 = get_project_root() + '/plots/transformed_E_dist.png'
title1 = 'Transformed energy distribution'
d1 = {'data': [tot_energy]}#, 'title': title1, 'savefig': path1}
a1 = rpt.make_plot(d1)
