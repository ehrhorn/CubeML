# import torch
# from matplotlib import pyplot as plt
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# import h5py as h5
# from time import time
# from scipy.stats import norm
# import subprocess
# from multiprocessing import Pool, cpu_count

from src.modules.classes import *
from pathlib import Path
from src.modules.reporting import *
from srsc.modules.constants import *
# import src.modules.loss_funcs as lf
# import src.modules.helper_functions as hf
# from src.modules.eval_funcs import *
# import src.modules.reporting as rpt
# from src.modules.classes import *
# from src.modules.preprocessing import *
# # import src.modules.preprocessing as pp
# from src.modules.main_funcs import *
# import shelve
# import sys
# from time import sleep

phi = np.random.uniform(low=0.0, high=np.pi, size=(1000))
x = np.cos(phi)
_ = make_plot(
    {'data': [x], 'savefig': get_projet_root()+'/LOL.png'}
)