from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
import healpy as hp
import numpy as np
import argparse
import os

def LeakyReLU(x):
    y = np.array(
        [np.max([0.01*entry, entry]) for entry in x]
    )
    return y

def Mish(x):
    y = x * np.tanh(
        np.log(1+np.exp(x))
    ) 
    return y

def deriv(x, y):
    dx = x[1]-x[0]
    deriv = (y[1:]-y[:-1])/dx
    return deriv

setup_pgf_plotting()

# make_data()
x = np.linspace(-4, 3, 300)
leakyrelu = LeakyReLU(x)
mish = Mish(x)
legend = [r'LeakyReLU', r'$\frac{d}{dx}$ LeakyReLU', r'Mish', r'$\frac{d}{dx}$ Mish']
d = {
    'x': [x, x[:-1], x, x[:-1]],
    'y': [leakyrelu, deriv(x, leakyrelu), mish, deriv(x, mish)],
    'linestyle': ['solid', 'dotted', 'solid', 'dotted'],
    'color': ['b', 'b', 'orange', 'orange'],
    'title': 'Nonlinear Activation Functions',
    'yrange': {'top': 1.5, 'bottom': None},
    'label': legend
    }

f = make_plot(d, for_thesis=True)
# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

# single_fig, 2subfigs
FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(width, height)

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
