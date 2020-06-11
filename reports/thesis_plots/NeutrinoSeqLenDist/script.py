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
    cleaned_key = 'srt_in_ice_pulses_event_length'
    uncleaned_key = 'split_in_ice_pulses_event_length'
    data_d = {cleaned_key: [], uncleaned_key: []} 

    for db_path in dbs:

        # Load seq lengths
        db = SqliteFetcher(db_path)
        data_dicts = db.fetch_features(
            all_events=db.ids, meta_features=[cleaned_key, uncleaned_key]
            )
        data_d[cleaned_key].extend(
            [d[cleaned_key] for i, d in data_dicts.items()]
            )
        data_d[uncleaned_key].extend(
            [d[uncleaned_key] for i, d in data_dicts.items()]
            )
    
    # Decide on bin size
    maxlen = 200
    minlen = 0
    bins = maxlen-minlen + 1

    hist_vals = {}
    for key, data in data_d.items():
        data_clipped = np.clip(data, 0, maxlen)
        hist_vals[key], edges = np.histogram(data_clipped, bins=bins, range=(minlen-0.5, maxlen+0.5))
    hist_vals['edges'] = edges

    path = Path(os.path.realpath(__file__))

    # Save data
    with open(str(path.parent) + '/data.pickle', 'wb') as f:
        pickle.dump(hist_vals, f)

# make_data()

path = Path(os.path.realpath(__file__))

# Save data
with open(str(path.parent) + '/data.pickle', 'rb') as f:
    data = pickle.load(f)

uncleaned = data['srt_in_ice_pulses_event_length'] 
cleaned = data['split_in_ice_pulses_event_length']
edges = data['edges']
labels = [
    r'SRT-cleaned events', 
    r'Uncleaned events'
    ] 
frac_above = cleaned[-1]/np.sum(cleaned[:-1])
# print('Fraction dropped: %.2f percent'%(frac_above*100))
d = {
    'data': [edges[:-1], edges[:-1]], 
    'bins': [edges, edges], 
    'weights': [uncleaned, cleaned],
    'label': labels,
    'xlabel': r'Event Sequence Length',
    # 'ylabel': r'Entries',
    'title': r'OscNext lvl5'
,
    'grid': False,
    }

f = make_plot(d, for_thesis=True)


# ============================================================================
# FINAL EDITS     
# ============================================================================
def update_ylabels(ax):
    ylabels = [str(int(label/1000))+'k' for label in ax.get_yticks()]
    ax.set_yticklabels(ylabels)
def update_xlabels(ax):
    xlabels = [str(int(label)) for label in ax.get_xticks()]
    xlabels[-1] = '>'+xlabels[-1]
    ax.set_xticklabels(xlabels)
ax = f.gca()
update_ylabels(ax)
# ============================================================================
# ADD TEXT
# ============================================================================
text = r'Overflow $\sim 1$\%'
xpos = 0.65
ypos = 0.25
f.text(xpos, ypos, text)

# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right

FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
