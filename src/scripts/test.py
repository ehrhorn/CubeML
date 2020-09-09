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



x = np.linspace(10, 20)
y = np.linspace(10, 20)

d = {
    'x': [x],
    'y': [y],
    'label': ['lol']
}

f = make_plot(d)
axes = f.axes
labels = []
lines = []
for ax in axes:
    new_lines, new_labels = ax.get_legend_handles_labels()
    lines = lines + new_lines
    labels = labels + new_labels
    print(labels)
print(axes)


