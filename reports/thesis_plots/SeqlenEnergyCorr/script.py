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
def make_data():
    dbs = [PATH_TRAIN_DB, PATH_VAL_DB, PATH_TEST_DB]
    cleaned_key = 'srt_in_ice_pulses_event_length'
    energy_key = 'true_primary_energy'
    data_d = {cleaned_key: [], energy_key: []} 
    transformer = joblib.load(
        open(PATH_DATA_OSCNEXT + '/sqlite_transformers.pickle', 'rb')
    )[energy_key]
    for db_path in dbs:

        # Load seq lengths
        db = SqliteFetcher(db_path)
        data_dicts = db.fetch_features(
            all_events=db.ids, 
            meta_features=[cleaned_key], 
            scalar_features=[energy_key]
            )
        data_d[cleaned_key].extend(
            [d[cleaned_key] for i, d in data_dicts.items()]
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

    return data_d[energy_key], data_d[cleaned_key]
energy, seqlen = make_data()
corrcoeff = np.corrcoef(energy, seqlen)[1, 0]

d = {
    'hist2d': [np.clip(energy, 0.0, 3.0), np.clip(seqlen, 0, 83)], 
    'n_bins': 21,
    'range': [[0.0, 3.0], [0, 80]],
    'xlabel': r'log$_{10}$ E [E/GeV]',
    'ylabel': r'Sequence length',
    'title': r'OscNext lvl5'
    }

f = make_plot(d, for_thesis=True)
# Get the current axis 
ax = f.gca()   

# Assume colorbar was plotted last one plotted last
cb = ax.collections[-1].colorbar
# cb = im[-1].colorbar   

# Do any actions on the colorbar object (e.g. remove it)
def update_ylabels(cb):
    ylabels = [str(int(label/1000))+'k' for label in cb.ax.get_yticks()]
    cb.ax.set_yticklabels(ylabels)
update_ylabels(cb)
# cb.remove()
# ============================================================================
# ADD TEXT
# ============================================================================
text = r'Pearson Corr.: %.2f'%(corrcoeff)
xpos = 0.14
ypos = 0.8
f.text(xpos, ypos, text)

# ============================================================================
# FINAL EDITS     
# ============================================================================
# ax.set_xticks([], [])
# ax.set_yticks([], [])

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
