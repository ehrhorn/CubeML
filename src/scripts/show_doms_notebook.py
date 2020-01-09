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

tot_n_doms = []
tot_energy = []
doms_wanted = 100000
doms_total = 0
for file in Path(dataset).iterdir():
    if doms_total >= doms_wanted:
        break
    if not (file.suffix == '.h5' and confirm_particle_type(get_particle_code(particle), file)):
        continue
     
    with h5.File(file, 'r') as f:
        n_doms = [x.shape[0] for x in f['raw/dom_charge']]
        energy = f['raw/true_primary_energy']
        doms_total += len(n_doms)
        tot_n_doms.extend(n_doms)
        tot_energy.extend(energy)
doms, energy = sort_pairs(tot_n_doms, tot_energy)

# tot_n_doms = [entry for entry in tot_n_doms if entry<100]
# %%
FRAC = 0.1
from_, to_ = 0.0, 0.9
end = int(FRAC*len(doms))
from_i, to_i = int(from_*len(doms)), int(to_*len(doms))

path1 = get_project_root() + '/plots/%s_energy_vs_seqlen.png'%(particle)
title1 = '%s: Bottom and upper %.0f %% seq. length log(e) dist'%(particle, FRAC*100)
d1 = {'data': [energy[0:end], energy[-end:-50]], 'title': title1, 'savefig': path1}
a1 = rpt.make_plot(d1)

path2 = get_project_root() + '/plots/%s_seqlen.png'%(particle)
title2 = '%s: Seq. length dist (entries: %.2e)'%(particle, len(doms[from_i:to_i]))
d2 = {'data': [doms[from_i:to_i]], 'title': title2, 'savefig': path2}
a2 = rpt.make_plot(d2)

# %%
morethan200 = len([entry for entry in doms if entry > 200])
print(morethan200/len(doms))
