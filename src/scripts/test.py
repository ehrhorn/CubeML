import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess
from multiprocessing import Pool, cpu_count

from src.modules.classes import *
import src.modules.loss_funcs as lf
import src.modules.helper_functions as hf
from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.constants import *
from src.modules.classes import *
import src.modules.preprocessing as pp

def calc_permutation_importance(save_dir, wandb_ID=None):
    
    # * Load the best model
    hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(save_dir)
    model = load_best_model(save_dir)

    # * Setup dataloader and generator - num_workers choice based on gut feeling - has to be high enough to not be a bottleneck
    # * SET MODE TO PERMUTE IN COLLATE_FN
    n_predictions_wanted = data_pars.get('n_predictions_wanted', np.inf)
    LOG_EVERY = int(meta_pars.get('log_every', 200000)/4) 
    VAL_BATCH_SIZE = data_pars.get('val_batch_size', 256) # ! Predefined size !
    device = get_device()
    dataloader_params_eval = get_dataloader_params(VAL_BATCH_SIZE, num_workers=8, shuffle=False, dataloader=data_pars['dataloader'])
    val_set = load_data(hyper_pars, data_pars, arch_pars, meta_pars, 'predict')
    collate_fn = get_collate_fn(data_pars, mode='permute', permute_features=seq_features)
    val_generator = data.DataLoader(val_set, **dataloader_params_eval, collate_fn=collate_fn)
    N_VAL = get_set_length(val_set)

    # * Get keys to calculate permutation importance on
    seq_features = data_pars['seq_feat']
    scalar_features = data_pars['scalar_feat']

    # * Permute 
    # # * Read the predictions. Each group in the h5-file corresponds to a raw data-file. Each group has same datasets.
    # file_address = save_dir+'/data/predictions.h5'
    # with h5.File(file_address, 'r') as f:
    #     for key in f:
    #         print(key)
    # print(seq_features)
    # print(scalar_features)


# weights = '/groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/weights/inverse_performance_muon_energy.pickle'
# weights = pickle.load(open(weights, 'rb'))
# interpolator = weights['interpolator']
# x = np.linspace(0.0, 3.0, 200)
# y = interpolator(x)

# d = {'x': [x], 'y': [y]}
# d['savefig'] = get_project_root()+'/WEIGHT_TEST.png'
# fig = rpt.make_plot(d)
intervals = np.linspace(0.0, 3.0, 19)
lowers = intervals[:-1]
uppers = intervals[1:]
centers = (uppers+lowers)/2
means = np.linspace(8.0, 10.0, 18)
sigmas = np.linspace(3.0, 1.0, 18)
n_events = np.arange(40000, 40018)

medians, upper_percs, lower_percs = [], [], []
all_energy, all_errors = [], []

for lower, upper, mean, sigma, events in zip(lowers, uppers, means, sigmas, n_events):
    energy = np.random.uniform(low=lower, high=upper, size=(events,))
    errors = np.random.normal(loc=mean, scale=sigma, size=(events,))

    medians.append(np.percentile(errors, 50))
    upper_percs.append(np.percentile(errors, 84))
    lower_percs.append(np.percentile(errors, 16))

    all_energy.extend(energy)
    all_errors.extend(errors)
# %%

# fig = rpt.make_plot(d)

d2 = {}
d2['hist2d'] = [all_energy, all_errors]
d2['zorder'] = 0
f2 = rpt.make_plot(d2)
#%%
d = {}
d['x'] = [centers, centers, centers]
d['y'] = [upper_percs, medians, lower_percs]
d['drawstyle'] = ['steps-mid', 'steps-mid', 'steps-mid']
d['color'] = ['red','red', 'red']
d['zorder'] = [1, 1, 1]
d['savefig'] = get_project_root() + '/LOLOMG.png'
f3 = rpt.make_plot(d, h_figure=f2)