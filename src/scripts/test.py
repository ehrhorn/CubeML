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
n = 1000
n_exp = 1000
x = np.random.normal(size=(n,n_exp))
y = np.random.normal(size=(n,n_exp))

res = np.sum(x*y, axis=0)
mean = np.mean(res)
std = np.std(res)
print('Sums: %d, experiments: %d'%(n, n_exp))
print('Mean: %.2f, std: %.2f'%(mean, std))
print('Divided by sqrt(n): Mean: %.2f, std: %.2f'%(mean/np.sqrt(n), std/np.sqrt(n)))