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

path = PATH_DATA_OSCNEXT + '/weights/energy_balanced_alpha100.pickle'
x = np.linspace(0.0, 3.0, num=200)
spline = pickle.load(open(path, 'rb'))

def inverse_dec(x, alpha=1.0, gamma=0.05, x0=.5, x1=3.0):
    if x < x0:
        p = alpha
    else:
        beta = gamma*(x1-x0)/(1-gamma)
        p = alpha*beta/(beta + x-x0)
    return p


def normalizer(x, y):
    dx = x[1:] - x[:-1]
    meany = (y[1:] + y[:-1])/2
    c = 1/np.sum(dx*meany)
    return c

# y = spline(x)
w = np.array([inverse_dec(e) for e in x])
w2 = np.flip(
    [inverse_dec(e, x0=.5) for e in x]
)
pdf = normalizer(x, 1/spline(x))/spline(x)
w_normed = w*normalizer(x, pdf*w)
w2_normed = w2*normalizer(x, pdf*w2)
weighted_dist = pdf*w_normed
weighted2_dist = pdf*w2_normed

from scipy import interpolate
interpolator1 = interpolate.interp1d(x, w_normed, fill_value="extrapolate", kind='quadratic')
interpolator2 = interpolate.interp1d(x, w2_normed, fill_value="extrapolate", kind='quadratic')
y1 = interpolator1(x)
y2 = interpolator2(x)
path1 = PATH_DATA_OSCNEXT + '/weights/inverse_low_E.pickle'
with open(path1, 'wb') as f:
    pickle.dump(interpolator1, f)
path2 = PATH_DATA_OSCNEXT + '/weights/inverse_high_E.pickle'
with open(path2, 'wb') as f:
    pickle.dump(interpolator2, f)
# w = post*spline(x)
d = {
    'x': [x, x],
    # 'x': [x, x, x, x, x],
    # 'y': [weighted_dist, pdf, weighted2_dist],
    'y': [y1, y2],
    # 'yscale': 'log',
    'savefig': get_project_root() + '/LOL.png'
}
_ = make_plot(d)