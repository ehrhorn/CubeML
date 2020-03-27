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
from src.modules.main_funcs import *
import shelve
import sys
from time import sleep

ids = [10, 3, 5]
sequential = ['dom_charge', 'dom_x']
scalar = ['dom_timelength_fwhm', 'linefit_point_on_line_z']
target = ['true_primary_energy', 'true_primary_direction_z']
mask = ['SRTInIcePulses']
db = SqliteFetcher(PATH_TRAIN_DB)
a = db.make_batch(all_events=ids, seq_features=sequential, scalar_features=scalar, target_features=target, mask=mask)

print(a[0])
# a = np.array([0, 1, 2, 3])
# mask = np.array([0, 1, 0, 1], dtype=bool)
# print(a[mask])


# path = mask_dir + '/muon_neutrino_test.pickle'
# mask = pickle.load(open(path, 'rb'))
# print(len(mask))
# x = np.arange(len(ids))
# d = {'x': [x], 'y': [ids]}
# d['savefig'] = get_project_root()+'/WTF.png'
# _=make_plot(d)
# path = '/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/weights/pickle_weights/inverse_performance_muon_energy.pickle'
# weights = pickle.load(open(path, 'rb'))
# interpolator = weights['interpolator']
# x = np.linspace(0.0, 4.0)
# y = interpolator(x)
# d = {'x': [x], 'y': [y]}
# d['savefig'] = get_project_root()+'/WEIGHT_TEST_OLD.png'
# d['yscale'] = 'log'
# _ = make_plot(d)
