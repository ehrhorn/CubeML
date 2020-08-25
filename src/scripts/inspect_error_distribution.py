# import torch
# from matplotlib import pyplot as plt
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
# from time import time
# from scipy.stats import norm
# import subprocess
# from multiprocessing import Pool, cpu_count

# from src.modules.classes import *
# from pathlib import Path
# from src.modules.reporting import *
# from src.modules.constants import *
# import src.modules.loss_funcs as lf
from src.modules.helper_functions import *
# from src.modules.eval_funcs import *
from src.modules.reporting import *
from src.modules.classes import *
# from src.modules.preprocessing import *
# # import src.modules.preprocessing as pp
# from src.modules.main_funcs import *
# import shelve
# import sys
# from time import sleep
import numpy as np
import pickle

from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
from src.modules.thesis_plotting import *
import os

p = '/home/bjoernhm/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/full_reg/2020-04-16-11.34.16/data/predictions.h5'

with h5.File(p, 'r') as f:
    en = f['true_primary_energy'][:]
    ids = np.array([str(i) for i in f['index'][:]])

db = SqliteFetcher(PATH_VAL_DB)
db_ids = db.ids
overlap_ids = np.isin(ids, db_ids)
f_i = ids[overlap_ids]
true_e = db.fetch_features(
    all_events=f_i,
    scalar_features=['true_primary_energy']
)
e_t = np.array([d['true_primary_energy'] for i, d in true_e.items()])
e_p = en[overlap_ids]
error = e_p-e_t 

e_p, error = sort_pairs(e_p, error)
bins = np.linspace(min(e_p), max(e_p), num=20)
# e_p_bins, error_bins = bin_data(e_p, error, bins)
e_p_bins, error_bins = bin_data(e_p, error, bins)

for e_p_bin, error_bin in zip(e_p_bins, error_bins):
    mean = np.mean(error_bin)
    std = np.std(error_bin)
    center = np.median(e_p_bin)
    print(center, mean, std)


d = {
    'data': [e_t, e_p, error1],
    'savefig': get_project_root() + '/LOL.png'
}
_ = make_plot(d)
    # for key in f:
        # print(key)