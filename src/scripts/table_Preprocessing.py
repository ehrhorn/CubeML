from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
from matplotlib import pyplot as plt
import sklearn
import os
import pandas as pd

def make_data():
    seq_keys = [
        'dom_charge', 
        'dom_x', 
        'dom_y', 
        'dom_z', 
        'dom_time', 
        'dom_atwd',
        'dom_pulse_width'
    ]
    target_keys = [
        'true_primary_energy', 
        'true_primary_position_x', 
        'true_primary_position_y', 
        'true_primary_position_z', 
        'true_primary_time', 
        'true_primary_direction_x', 
        'true_primary_direction_y', 
        'true_primary_direction_z'
    ]
    db_path = PATH_TRAIN_DB
    key = 'dom_charge'
    transformers = joblib.load(
        open(PATH_DATA_OSCNEXT + '/sqlite_transformers.pickle', 'rb')
    )
    db = SqliteFetcher(db_path)
    # Lets go with 1M ~ approximtely 1M/50 = 20k events
    ids = [str(e) for e in range(1000)]

    all_data = db.fetch_features(
        all_events=ids, 
        seq_features=seq_keys, 
        scalar_features=target_keys
        )
    data_d = {key: [] for key in all_data['0']}
    for key in target_keys:
        data_d[key] = [data[key] for event_id, data in all_data.items()]
    for key in seq_keys:
        data_d[key].extend(
            flatten_list_of_lists(
                [data[key] for event_id, data in all_data.items()]
                )
            )
    # Calculate means and std's before and after transformation
    dicts = {}
    table = np.empty((5, len(seq_keys)+len(target_keys)), dtype=object)
    for i_key, key in enumerate(data_d):
        data = data_d[key]
        d = {}
        if key in transformers:
            if type(transformers[key]) == sklearn.preprocessing._data.QuantileTransformer:
                name = 'ToNormal'
            elif sklearn.preprocessing._data.RobustScaler:
                if key == 'true_primary_energy':
                    name = 'LogRobust'
                else:
                    name = 'Robust'
            table[0, i_key] = name
            table[3, i_key] = r'%.2f'%(np.mean(data))
            table[4, i_key] = r'%.2f'%(np.std(data))
            data_pre = np.squeeze(
                transformers[key].inverse_transform(
                    np.array(data).reshape(-1, 1)
                )
            )
            if key == 'true_primary_energy':
                table[1, i_key] = r'%.2e'%(np.mean(10**data_pre))
                table[2, i_key] = r'%.2e'%(np.std(10**data_pre))
            else:
                table[1, i_key] = r'%.2e'%(np.mean(data_pre))
                table[2, i_key] = r'%.2e'%(np.std(data_pre))
        else:
            table[0, i_key] = 'None'
            table[1, i_key] = r'%.2f'%(np.mean(data))
            table[2, i_key] = r'%.2f'%(np.std(data))
            table[3, i_key] = r'-'
            table[4, i_key] = r'-'

    index = [r'Transformation', r'$\mu$, before', r'$\sigma$, before', r'$\mu$, after', r'$\sigma$, after']
    columns = []
    for col in [key for key in data_d]:
        split = col.split('_')
        new_col = r'\_'.join(split)
        columns.append(new_col)
    table_pd = pd.DataFrame(
        np.transpose(table),                       # values
        index=columns,    # 1st column as index
        columns=index)                # 1st row as the column names

    return table_pd
table_pd = make_data()
print(table_pd.to_latex(escape=False))
a+=1

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

# # Print as a LaTeX-table
# means_arr, medians_arr = np.array(means), np.array(medians)
# table = np.vstack((means, medians))
# Cols = Confs + ['Rest']
# table_pd = pd.DataFrame(table,                       # values
#                         index=['Mean', 'Median'],    # 1st column as index
#                         columns=Cols)                # 1st row as the column names
# if PrintTable:
#     print(table_pd)
#     print('')
#     print(table_pd.to_latex(float_format=lambda x: '%.2f' % x))