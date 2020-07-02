from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
from matplotlib import pyplot as plt
import os

setup_pgf_plotting()

def logcosh(x):
    y = np.log(
        np.cosh(x)
    )
    return y

def LE(x):
    y = np.log(x)
    return y

def RE(x):
    y = x - 1
    return y
# f, _ = plt.subplots(1, 2)

# ============================================================================
# PLOT 1  
# ============================================================================
frac = np.linspace(0.1, 3.0)
le = logcosh(LE(frac))
re = logcosh(RE(frac))

d1 = {
    'x': [frac, frac],
    'y': [le, re],
    'xlabel': r'$\frac{E_{\mathrm{reco}}}{E_{\mathrm{true}}}$',
    'ylabel': r'Loss',
    'title': r'Relative (RE) vs logarithmic (LE) error',
    'label': [r'logcosh(LE)', r'logcosh(RE)'],
    'yrange': [0.0, 1.0],
    'grid': True
}
f = make_plot(d1, for_thesis=True)

FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
