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

x = np.arange(10)
y = np.arange(10)
path = get_project_root() + '/LOL.png'
pathpickle = get_project_root() + '/pickle_LOL.pickle'

d = {'x': [x], 'y': [y], 'savefig': path}
fig = make_plot(d)

h_fig = pickle.load(open(pathpickle, 'rb'))
ax = h_fig.gca()
ax.set_title('WHAT IS UP')
h_fig.savefig(path)
print(type(h_fig))