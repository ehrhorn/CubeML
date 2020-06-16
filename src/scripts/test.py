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
import src.modules.helper_functions as hf
# from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.classes import *
# from src.modules.preprocessing import *
# # import src.modules.preprocessing as pp
# from src.modules.main_funcs import *
# import shelve
# import sys
# from time import sleep
import numpy as np
import pickle

def tanh(x):
    y = 10.0*np.tanh(x/10.0)
    return y

path = get_project_root() + '/models/oscnext-genie-level5-v01-01-pass2/regression/nue_numu/test_2020.06.15-21.38.20/data/predictions.h5'

with h5.File(path, 'r') as f:
    for key in f:
        print(key)

# x = np.linspace(-50, 50, 200)
# y = tanh(x)
# d = {'x': [x], 'y': [y], 'savefig': get_project_root()+'/lol.png'}
# _ = rpt.make_plot(d)