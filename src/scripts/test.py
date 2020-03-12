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

n = 1000
percentile = 68
sigma = np.sqrt((percentile/100)*n*(1-(percentile/100)))
mean = n*percentile/100
mean = int(mean)
upper = int(mean+sigma+1)
lower = int(mean-sigma)
x = np.random.uniform(low=0.0, high=1.0, size=n)
sorted(x)
print(np.nanpercentile(x, 68.2), sorted(x)[mean], sorted(x)[upper], sorted(x)[lower])
