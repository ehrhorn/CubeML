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

data1 = np.array([0, 1, 2])
data2 = np.array([np.nan, 3, 3])
n_nans, data1, data2 = hf.strip_nans(data1, data2)
print(n_nans)
print(data1)
print(data2)
# print('WARNING: %d NAN(S) FOUND IN I3 PERFORMANCE PLOT!'%(n_nans)) if n_nans>0 else None