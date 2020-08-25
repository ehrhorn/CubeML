from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
from src.modules.thesis_plotting import *
import os

setup_pgf_plotting()

# ============================================================================
# IMPORT/MAKE DATA     
# ============================================================================
names = [
    'energy_balanced_alpha70.pickle',
    'inverse_low_E.pickle',
    'inverse_high_E.pickle',
    'inverse_performance_muon_energy.pickle'
]
labels = [
    r'$w_{balanced}^{\alpha=0.7}$',
    r'$w_{low}$',
    r'$w_{high}$',
    r'$w_{blinding}$'
]
x = np.linspace(0.0, 3.0, 200)
d = {
    'x': [],
    'y': [],
    'xlabel': r'Energy [GeV]',
    'ylabel': r'Weight',
    'xscale': 'log',
    'yscale': 'log',
    'label': [],
    'title': 'Regression Weights'
}
for name, label in zip(names, labels):
    path = PATH_DATA_OSCNEXT + '/weights/' + name
    interpolator = pickle.load(
        open(path, 'rb')
    )
    y = interpolator(x)
    d['x'].append(10**x)
    d['y'].append(y)
    d['label'].append(label)
# all_energy, all_weights = make_data()
# indices = all_energy<3.0
# energy = all_energy[indices]
# weights = all_weights[indices]
# energy, weights = sort_pairs(energy, weights)
# bin_edges = np.linspace(0.0, 3.0, num=150)
# energy_binned, weights_binned = bin_data(energy, weights, bin_edges)
# energy_bins = [len(e) for e in energy_binned]
# energy_weighted = [
#     energy_bins[i]*weights_binned[i][len(weights_binned[i])//2] for i in range(len(weights_binned))
# ]
# bin_edges = 10**np.linspace(0.0, 3.0, num=150)
# d = {
#     'data': [bin_edges[:-1], bin_edges[:-1]], 
#     'bins': [bin_edges, bin_edges],
#     'histtype': ['step', 'step'],
#     'alpha': [1.0, 1.0],
#     'weights': [energy_bins, energy_weighted],
#     # 'density': [True, True],
#     'xscale': 'log',
#     'xlabel': r'Energy [GeV]',
#     'ylabel': r'Count',
#     'label': [r'Raw', r'Weighted'],
#     'title': r'Weighted energy distribution $(N_{tot}=2\cdot 10^6)$'
#     }



f = make_plot(d, for_thesis=True)
ax = f.gca()

# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
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
