import torch
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
from src.modules.classes import *
from src.modules.preprocessing import *
# import src.modules.preprocessing as pp
from src.modules.main_funcs import *
import shelve
import sys
from time import sleep


batch_size = 256
n_experiments = 20
path_train_db = '/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/train_transformed.db'

def _normal(x, mean, sigma):
    const = 1.0/(sigma*np.sqrt(2*3.14159))
    exponent = -0.5*((x-mean)/sigma)*((x-mean)/sigma)

    return const*np.exp(exponent)
x = np.linspace(-5.0, 5.0)
mean = 0
sigma = 1.0

y = _normal(x, mean, sigma)
d = {'x': [x], 'y': [y]}
d['savefig'] = get_project_root()+'/LOL.png'
_ = make_plot(d)