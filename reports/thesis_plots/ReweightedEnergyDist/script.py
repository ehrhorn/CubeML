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
def make_data(energy_key, weight_keys):
    dbs = [PATH_TRAIN_DB]
    data_d = {weight_key: [] for weight_key in weight_keys}
    data_d[energy_key] = []
    transformer = joblib.load(
        open(PATH_DATA_OSCNEXT + '/sqlite_transformers.pickle', 'rb')
    )[energy_key]
    for db_path in dbs:
        scalars = [energy_key] + weight_keys
        # Load seq lengths
        db = SqliteFetcher(db_path)
        data_dicts = db.fetch_features(
            all_events=db.ids[:2000000], 
            scalar_features=scalars
            )
        for weight_key in weight_keys:
            data_d[weight_key].extend(
                [d[weight_key] for i, d in data_dicts.items()]
                )
        data_d[energy_key].extend(
            [d[energy_key] for i, d in data_dicts.items()]
            )
    
    data_d[energy_key] = np.squeeze(
                transformer.inverse_transform(
                    np.array(
                        data_d[energy_key]
                        ).reshape(-1, 1)
                    )
                ) 

    return data_d

energy_key = 'true_primary_energy'
weight_keys = [
    'energy_balanced_alpha70',
    'inverse_low_E',
    'inverse_high_E',
    'inverse_performance_muon_energy'
]
labels = [
    r'$w_{balanced}^{\alpha=0.7}$',
    r'$w_{low}$',
    r'$w_{high}$',
    r'$w_{blinding}$'
]

data_d = make_data(energy_key, weight_keys)
all_energy = data_d[energy_key]
indices = all_energy<3.0

energy = all_energy[indices]
bin_edges = np.linspace(0.0, 3.0, num=150)
bin_edges_log = 10**np.linspace(0.0, 3.0, num=150)

d = {
    'data': [bin_edges_log[:-1]], 
    'bins': [bin_edges_log],
    'histtype': ['step'],
    'alpha': [1.0],
    'weights': [],
    # 'density': [True, True],
    'xscale': 'log',
    'xlabel': r'Energy [GeV]',
    'ylabel': r'Count',
    'label': [],
    'title': r'Weighted energy distributions $(N_{tot}=2\cdot 10^6)$'
    }
for weight_key, label in zip(weight_keys, labels):
    weights = np.array(data_d[weight_key])[indices]
    energy_sorted, weights_sorted = sort_pairs(energy, weights)
    energy_binned, weights_binned = bin_data(
        energy_sorted, weights_sorted, bin_edges
        )
    energy_bins = [len(e) for e in energy_binned]
    energy_weighted = []
    for i in range(len(weights_binned)):
        running = 0
        while True:
            if weights_binned[i][running] != None:
                break
            else:
                running += 1
        energy_weighted.append(
            energy_bins[i]*weights_binned[i][running] 
        )
    d['data'].append(bin_edges_log[:-1])
    d['bins'].append(bin_edges_log)
    d['histtype'].append('step')
    d['alpha'].append(1.0)
    d['weights'].append(energy_weighted)
    d['label'].append(label)
d['weights'].append(energy_bins)
d['label'].append('Raw')


f = make_plot(d, for_thesis=True)
ax = f.gca()
update_ylabels(ax)

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
