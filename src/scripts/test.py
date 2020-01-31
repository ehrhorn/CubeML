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

    # * Get keys to calculate permutation importance on
    seq_features = data_pars['seq_feat']
    scalar_features = data_pars['scalar_feat']

    
path = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2/pickles'
names = [int(dir_.name) for dir_ in Path(path).iterdir()]
ints = np.arange(1131)
difference = set(names).symmetric_difference(set(ints))
print(difference)