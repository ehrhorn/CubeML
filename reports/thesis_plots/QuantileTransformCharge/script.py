from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
from matplotlib import pyplot as plt
import os

def make_data():
    db_path = PATH_TRAIN_DB
    key = 'dom_charge'
    transformer = joblib.load(
        open(PATH_DATA_OSCNEXT + '/sqlite_transformers.pickle', 'rb')
    )[key]
    db = SqliteFetcher(db_path)
    # Lets go with 1M ~ approximtely 1M/50 = 20k events
    ids = [str(e) for e in range(20000)]

    all_data = db.fetch_features(all_events=ids, seq_features=[key])
    data_lists = [data[key] for event_id, data in all_data.items()]
    data_transformed = np.array(flatten_list_of_lists(data_lists))
    data = np.squeeze(
        transformer.inverse_transform(data_transformed.reshape(-1, 1))
        ) 

    return data, data_transformed
data, data_transformed = make_data()

setup_pgf_plotting()

# ============================================================================
# SET LABELS      
# ============================================================================

# x and y
f, _ = plt.subplots(1, 2)
entries = data.shape[0]
entries_text = r'N = %.1f M$'%(entries/1000000)


# Part 1
mean1, std1 = np.mean(data), np.std(data)
d1 = {
    'data': [data],
    'xlabel': r'Charge',
    'title': r'Before transformation',
    'grid': False,
    'bins': [100],
    'range': [0.0, 5.0]
}
f = make_plot(d1, h_figure=f, axes_index=0, for_thesis=True)
ax = f.get_axes()[0]
update_ylabels(ax)
mean1t = r'$\hat{\mu} = %.2f$'%(mean1)
std1t = r'$\hat{\sigma} = %.2f$'%(std1)

# Part 2
mean2, std2 = np.mean(data_transformed), np.std(data_transformed)
d2 = {
    'data': [data_transformed],
    'xlabel': r'Charge',
    'title': r'After transformation',
    'grid': False,
    'bins': [100],
    'range': [-3.5, 4.5]
}
f = make_plot(d2, h_figure=f, axes_index=1, for_thesis=True)
ax = f.get_axes()[1]
update_ylabels(ax)
mean2t = r'$\hat{\mu} = %.2f$'%(mean2)
std2t = r'$\hat{\sigma} = %.2f$'%(std2)

xpos = 0.29
ypos = 0.6
x_disp = 0.47
y_disp = -0.06

f.text(xpos, ypos, mean1t)
f.text(xpos, ypos+y_disp, std1t)
f.text(xpos, ypos-y_disp, entries_text)

f.text(xpos+x_disp, ypos, mean2t)
f.text(xpos+x_disp, ypos+y_disp, std2t)
f.text(xpos+x_disp, ypos-y_disp, entries_text)

# ============================================================================
# ADD TEXT
# ============================================================================
# text = r'Optimal complexity'
# xpos = 0.42
# ypos = 0.6
# f.text(xpos, ypos, text)

# ============================================================================
# FINAL EDITS     
# ============================================================================

# ax = plt.gca()
# ax.set_xticks([], [])
# ax.set_yticks([], [])

# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# single_fig, 2subfigs
FOTW = get_frac_of_textwidth(keyword='subplots', rows=1, cols=2)
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width, rows=1, cols=2)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
