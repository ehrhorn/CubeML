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

def make_data(
    models, 
    perf_key, 
    legend_loc=None, 
    ylabel=None, 
    f=None, 
    i_plot=0,
    legend=False,
    rel_imp_label=True,
    right_ylabel=True
):

    perfs = []
    errs = []
    retro_perfs = []
    retro_errs = []
    edges = []
    rel_imp = []
    rel_imp_err = []
    
    # Load the predictions
    for i_model, model_full in enumerate(models):
        model = webpage_to_modelname(model_full)
        path = locate_model(model) + '/data/Performance.pickle'
        perf = pickle.load(open(path, 'rb'))

        perfs.append(getattr(perf, perf_key + '_sigma'))
        errs.append(getattr(perf, perf_key + '_sigmaerr'))
        retro_perfs.append(getattr(perf, 'retro_' + perf_key + '_sigma'))
        retro_errs.append(getattr(perf, 'retro_' + perf_key + '_sigmaerr'))
        edges.append(getattr(perf, 'bin_edges'))
        rel_imp.append(getattr(perf, perf_key + '_RI'))
        rel_imp_err.append(getattr(perf, perf_key + '_RIerr'))

        energy_counts = getattr(perf, 'counts')
        
    d = {
        'edges': [edges[0]]*3, 
        'y': [perfs[0], retro_perfs[0], perfs[1]], 
        'yerr': [errs[0], retro_errs[0], errs[1]], 
        'color': ['tab:blue', 'tab:orange', 'tab:cyan'],
        'xlabel': r'log(E) [E/GeV]', 
        'ylabel': ylabel,
        'grid': False, 
        'label': ['Ensemble', 'Retro', 'Best submodel'], 
        'yrange': {'bottom': 0.001}, 
        'markersize': 0.8,
        'linewidth': 1.2,
    }
    if not legend:
        del d['label']
    f = make_plot(
        d,
        position=[0.125, 0.26, 0.475, 0.42],
        for_thesis=True,
        h_figure=f
    )

     # ENERGY PLOT
    d = {
        'data': [edges[0][:-1]], 
        'bins': [edges[0]],
        'weights': [energy_counts*8], 
        'histtype': ['step'], 
        'log': [True], 
        'color': ['gray'], 
        'twinx': True, 
        'grid': False, 
        'label': ['Training Events'],
        'combine_labels': True,
        'ylabel': 'Training Events',
        'legend_loc': legend_loc,
        'frameon': False
    }
    if not legend:
        del d['label']
    if not right_ylabel:
        del d['ylabel']
    f = make_plot(d, h_figure=f, axes_index=0)


    # RELATIVE IMPROVEMENT PLOT
    d = {
        'edges': [edges[0], edges[1]], 
        'y': [rel_imp[0], rel_imp[1]], 
        'yerr': [rel_imp_err[0], rel_imp_err[1]],  
        'color': ['tab:blue', 'tab:cyan'],
        'xlabel': r'$\log_{10}(E)$ [$E$/GeV]', 
        'ylabel': 'Rel. Imp.', 
        'grid': True, 
        'y_minor_ticks_multiple': 0.2
    }

    if not rel_imp_label:
        del d['ylabel']
        
    yrange_d = {}
    if max(-0.5, min(rel_imp[0])) == -0.5:
        yrange_d['bottom'] = -0.5
        yrange_d['top'] = 0.5
        d['yrange'] = yrange_d
    
    d['subplot'] = True
    d['axhline'] = [0.0]
    # f = make_plot(d, h_figure=f, position=[0.625, 0.11, 0.475, 0.15])
    f = make_plot(d, h_figure=f, position=[0.125, 0.11, 0.475, 0.15])


    return f

energy_models = [
    'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-08-19.05.11?workspace=user-bjoernmoelvig', # ER ens
    'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-04-05.14.55?workspace=user-bjoernmoelvig', # Fullreg
]

direction_models = [
    'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-08-13.01.59?workspace=user-bjoernmoelvig', # DR ens
    'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-02-00.33.27?workspace=user-bjoernmoelvig', # DR best single
]
f1  = make_data(
    energy_models, 
    'log_frac_E_error', 
    ylabel=r'$W\left(\log_{10} \left[ \frac{E_{pred}}{E_{true}} \right]\right)$',
    legend_loc=(0.15, 0.05),
    legend=True,
    right_ylabel=False
)
f2  = make_data(
    direction_models, 
    'polar_error', 
    ylabel=r'$W(\Delta \theta)$ [deg]',
    legend_loc=(0.15, 0.05),
    rel_imp_label=False
)


# SAVE LOGFRAC 2D ERROR DISTRIBUTION ASWELL
model = webpage_to_modelname(energy_models[0])
model_path = locate_model(model)
wanted_plot = 'log_frac_E_error_2DPerformance' 
data_path = model_path + '/figures/pickle_'

plotname = 'log_frac_E_2D'

save_path = get_project_root() + '/reports/thesis_plots/all_pgf/ensemble_performance_'
path = data_path + wanted_plot + '.pickle'
with open(path, 'rb') as f:
    fig = pickle.load(f)
    savefig_path = save_path + plotname + '.png'
    fig.savefig(savefig_path, bbox_inches='tight')

# Move copies of z-score distributions 
import os
newname = 'ensemble_performance_zscore_energy.pgf'
pgf_path = model_path + '/figures/distributions/z_score_true_primary_energy.pgf'
destination = ' /home/bjoernhm/CubeML/reports/thesis_plots/all_pgf/'
os.system('cp ' + pgf_path + destination + newname)

model = webpage_to_modelname(direction_models[0])
model_path = locate_model(model)
newname = 'ensemble_performance_zscore_z_dir.pgf'
pgf_path = model_path + '/figures/distributions/z_score_true_primary_direction_z.pgf'
os.system('cp ' + pgf_path + destination + newname)

# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# broad_figure: FOTW = 2.
# single_fig, 2subfigs
for f, name in zip([f1, f2], ['energy', 'polar']):
    FOTW = get_frac_of_textwidth(keyword='single_fig')
    width = get_figure_width(frac_of_textwidth=FOTW)
    height = get_figure_height(width=width)
    f.set_size_inches(1.25*width, 1.25*height)

    path = Path(os.path.realpath(__file__))
    save_thesis_pgf(path, f, save_pgf=True, png_name=name, pgf_name=name)