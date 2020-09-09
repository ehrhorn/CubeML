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
    from_root = '/home/bjoernhm/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/direction_reg'
    model_names = ['logcosh', 'L1', 'L1 + pen']
    keys = [
        'true_primary_direction_x', 
        'true_primary_direction_y', 
        'true_primary_direction_z'
    ]

    hists = {
        key: [] for key in keys
    }
    truths = {
        key: [] for key in keys
    }
    
    length_dists = []
    length_edges = []
    n_bins = 110
    
    # Load the predictions
    for i_model, model_full in enumerate(models):
        model = model_full.split('/')[-1].split('?')[0]
        perf = Performance('/'.join([from_root, model]), run_perf_eval=False)
        raw_pred_dict = read_pickle_predicted_h5_data_v2(
            perf._pred_path, perf._target_keys
        )
        
        # Calculate histograms and length distributions
        lengths = np.zeros(len(raw_pred_dict['indices']))
        for key in keys:
            hist, bin_edges = np.histogram(
                np.clip(raw_pred_dict[key], -1.0, 1.0),
                bins=n_bins,
                density=True
            )
            hists[key].append(hist)
            
            lengths = lengths + raw_pred_dict[key]*raw_pred_dict[key]
        lengths = np.sqrt(lengths)
        hist, bin_edges = np.histogram(
                np.clip(lengths, 0.0, 1.2),
                bins=n_bins,
                density=True
            )
        length_dists.append(hist)
        length_edges.append(bin_edges)

    
    # Load truths
    event_ids = [str(idx) for idx in raw_pred_dict['indices']]
    del raw_pred_dict['indices']

    fetched = perf.db.fetch_features(
        all_events=event_ids, 
        scalar_features=[key for key in raw_pred_dict],
    )
    true_dict = {
        key: np.zeros(len(event_ids)) for key in perf._true_keys
    }
    for i_event, idx in enumerate(event_ids):
        for key in perf._true_keys:
            true_dict[key][i_event] = fetched[idx][key]
    
    # Make truth histograms
    for key, data in true_dict.items():
        hist, bin_edges = np.histogram(
            data,
            bins=n_bins,
            density=True
        )
        truths[key].append(hist)
    
    
    # Dump data
    datadump = {
        'predictions': hists,
        'truths': truths,
        'models': model_names,
        'lengths': length_dists,
        'length_edges': length_edges
    }
    path = Path(os.path.realpath(__file__))

    # Save data
    with open(str(path.parent) + '/data.pickle', 'wb') as f:
        pickle.dump(datadump, f)

def make_first_plot(data, f):
    preds = data['predictions']
    truths = data['truths']
    names = data['models']

    key = 'true_primary_direction_y'
    xlabel = r'$x$-component'
    title = r'Predicted direction, $x$-component'
    hists = []
    placeholder = []
    bin_edges = []
    labels = []
    linewidths = []
    for pred, name in zip(preds[key], names):
        n_bins = pred.shape[0]
        hists.append(pred)
        placeholder.append(np.linspace(-1, 1, num=n_bins))
        bin_edges.append(np.linspace(-1, 1, num=n_bins+1))
        labels.append(name)
        linewidths.append(1.2)

    hists.append(truths[key])
    placeholder.append(np.linspace(-1, 1, num=n_bins))
    bin_edges.append(np.linspace(-1, 1, num=n_bins+1))
    labels.append('Truths')
    d = {
        'data': placeholder, 
        'bins': bin_edges, 
        'weights': hists, 
        'histtype': ['step']*len(hists), 
        'label': labels,
        'linewidth': 1.4,
        'alpha': 1.0,
        'grid': False, 
        'ylabel': 'Density',
        'xlabel': xlabel,
        'title': title
    }


    f = make_plot(d, h_figure=f, axes_index=0, for_thesis=True)
    return f
    

def make_second_plot(data, f):
    lengths = data['lengths']
    names = data['models']
    lengths_edges = data['length_edges']

    xlabel = r'Vector length'
    title = r'Predicted direction vector length'
    hists = []
    placeholder = []
    bin_edges = []
    labels = []
    linewidths = []
    for length_hist, edges, name in zip(lengths, lengths_edges, names):
        n_bins = length_hist.shape[0]
        hists.append(length_hist)
        placeholder.append(edges[:-1])
        bin_edges.append(edges)
        labels.append(name)
        linewidths.append(1.2)

    d = {
        'data': placeholder, 
        'bins': bin_edges, 
        'weights': hists, 
        'histtype': ['step']*len(hists), 
        'label': labels,
        'linewidth': 1.4,
        'alpha': 1.0,
        'grid': False, 
        # 'ylabel': 'Density',
        'xlabel': xlabel,
        'title': title,
        'legend_loc': 'upper left'
    }


    f = make_plot(d, h_figure=f, axes_index=1, for_thesis=True)

    return f

f, axs = plt.subplots(1, 2)

# make_data()
path = Path(os.path.realpath(__file__))
data = pickle.load(open(str(path.parent) + '/data.pickle', 'rb'))
f = make_first_plot(data, f)
f = make_second_plot(data, f)
# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# broad_figure: FOTW = 2.
# single_fig, 2subfigs
FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(8.4, 3.4)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True, png_name='fig2')