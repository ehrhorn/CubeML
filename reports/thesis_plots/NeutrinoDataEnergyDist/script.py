from src.modules.thesis_plotting import *
from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
import os

setup_pgf_plotting()

# ============================================================================
# IMPORT/MAKE DATA     
# ============================================================================
def make_data():
    dbs = [PATH_TRAIN_DB, PATH_VAL_DB, PATH_TEST_DB]
    particles = ['electron_neutrino', 'muon_neutrino', 'tau_neutrino']
    suffixes = ['_train.pickle', '_val.pickle', '_test.pickle']
    key = 'true_primary_energy'
    transformer = joblib.load(
        open(PATH_DATA_OSCNEXT + '/sqlite_transformers.pickle', 'rb')
    )[key]
    data_d = {} 
    i = 0
    for particle in particles:
        all_energies = np.array([])
        for db_path, suffix in zip(dbs, suffixes):

            # Load mask
            path = PATH_DATA_OSCNEXT + '/masks/' + particle + suffix
            mask = [str(e) for e in pickle.load(open(path, 'rb'))]

            # Load energy
            db = SqliteFetcher(db_path)
            data_dicts = db.fetch_features(
                all_events=mask, scalar_features=[key]
                )
            energies_trans = np.array(
                [d[key] for e, d in data_dicts.items()]
            )

            # Inverse transform them
            energies = np.squeeze(
                transformer.inverse_transform(energies_trans.reshape(-1, 1))
                ) 

            # Add to all
            all_energies = np.append(all_energies, energies)
            print(i)
            i+=1
        data_d[particle] = all_energies

    # Decide on bin size
    iqr = np.percentile(data_d['muon_neutrino'], 75)-np.percentile(data_d['muon_neutrino'], 25)
    n_data = data_d['muon_neutrino'].shape[0]
    bin_width = 2*iqr/ (n_data**0.3333)
    n_bins = int(4/bin_width)

    hist_vals = {}
    for particle, data in data_d.items():
        hist_vals[particle], edges = np.histogram(data, bins=n_bins, range=(0.0, 4.0))
    hist_vals['edges'] = edges
    path = Path(os.path.realpath(__file__))

    # Save data
    with open(str(path.parent) + '/data.pickle', 'wb') as f:
        pickle.dump(hist_vals, f)

path = Path(os.path.realpath(__file__))

# Save data
with open(str(path.parent) + '/data.pickle', 'rb') as f:
    data = pickle.load(f)

edges = 10**data['edges']
weights = [data['muon_neutrino'], data['electron_neutrino'], data['tau_neutrino']]
n_e = np.sum(weights[1])
n_mu = np.sum(weights[0])
n_tau = np.sum(weights[2])
labels = [r'$\nu_{\mu}$, N = %.2f $\cdot 10^6$'%(n_mu/1e6), 
    r'$\nu_{e}$, N = %.2f $\cdot 10^6$'%(n_e/1e6),
    r'$\nu_{\tau}$, N = %.2f $\cdot 10^6$'%(n_tau/1e6)
    ]

d = {
    'data': [[edges[:-1], edges[:-1], edges[:-1]]], 
    'bins': [edges], 
    'weights': [weights],
    'stacked': [True],
    'label': [labels],
    'xlabel': r'Energy [GeV]',
    # 'ylabel': r'Entries',
    'title': r'OscNext Lvl5',
    'xscale': 'log',
    'grid': False,
    }

f = make_plot(d, for_thesis=True)
def update_ylabels(ax):
    ylabels = [str(int(label/1000))+'k' for label in ax.get_yticks()]
    ax.set_yticklabels(ylabels)
ax = f.gca()
update_ylabels(ax)

FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
