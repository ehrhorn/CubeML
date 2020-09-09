from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
from src.modules.thesis_plotting import *
from src.modules.reporting import *
import os

setup_pgf_plotting()

# ============================================================================
# IMPORT/MAKE DATA     
# ============================================================================

def make_data():
    
    models = [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-28-16.25.11?workspace=user-bjoernmoelvig', # logcosh
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-02-13.20.30?workspace=user-bjoernmoelvig', # L1
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-02-19.57.38?workspace=user-bjoernmoelvig', # L1 + Penalty
    ]

    perfs = []
    errs = []
    edges = []

    from_root = '/home/bjoernhm/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/direction_reg/'
    model_names = ['logcosh', 'L1', 'L1 + pen']
    # perf_key = 'directional_error_68th'
    # perf_key_err = 'directional_error_err68th'

    perf_key = 'polar_error_sigma'
    perf_key_err = perf_key + 'err'
    # Load the predictions
    for i_model, model_full in enumerate(models):
        model = model_full.split('/')[-1].split('?')[0]
        perf_path = from_root + model + '/data/Performance.pickle'
        perf = pickle.load(open(perf_path, 'rb'))
        perfs.append(
            getattr(perf, perf_key)
        )
        errs.append(
            getattr(perf, perf_key_err)
        )
        edges.append(
            getattr(perf, 'bin_edges')
        )

    return perfs, errs, model_names, edges

perfs, errs, model_names, edges = make_data()
title = r'$RNN_{Direction}(3, 256, 2)$'
d = {
    'edges': edges, 
    'y': perfs, 
    'yerr': errs, 
    'xlabel': r'log(E) [E/GeV]', 
    'ylabel': r'$W(\Delta \theta)$ [deg]', 
    'grid': True, 
    'label': model_names, 
    'yrange': {'bottom': 0.001}, 
    'title': title,
    'markersize': 0.5,
    'linewidth': 1.5,
}
f = make_plot(d, for_thesis=True)

# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# broad_figure: FOTW = 2.
# single_fig, 2subfigs

FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)