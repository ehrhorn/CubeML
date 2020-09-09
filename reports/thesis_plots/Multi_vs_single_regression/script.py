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
    compare_to = 'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-21-15.26.17?workspace=user-bjoernmoelvig'
    model = compare_to.split('/')[-1].split('?')[0]
    model_path = locate_model(model) + '/data/Performance.pickle'
    compare_perf = pickle.load(
        open(
            model_path, 'rb'
        )
    )
    
    # Energy models
    models = [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-27-11.43.44?workspace=user-bjoernmoelvig', #Energy model
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-28-16.25.11?workspace=user-bjoernmoelvig', # Direction model
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-29-23.17.17?workspace=user-bjoernmoelvig', # Vertex model
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-29-23.17.17?workspace=user-bjoernmoelvig', # Vertex model
    ]
    wanted_feats = [
        'log_frac_E_error_sigma',
        'polar_error_sigma',
        'len_error_68th',
        'vertex_t_error_sigma'
        ]
    wanted_errors = [
        'log_frac_E_error_sigmaerr',
        'polar_error_sigmaerr',
        'len_error_err68th',
        'vertex_t_error_sigmaerr'
        ]
    
    plot_names = [
        'Energy Reco. (logarithmic error)',
        'Polar Angle Reco.',
        'Interaction Vertex Reco.',
        'Interaction Time Reco.',
    ]



    perfs = []

    for i_model, model_full in enumerate(models):
        model = model_full.split('/')[-1].split('?')[0]
        model_path = locate_model(model) + '/data/Performance.pickle'
        perfs.append(
            pickle.load(
                open(
                    model_path, 'rb'
                )
            )
        )
        
    return compare_perf, perfs, wanted_feats, wanted_errors, plot_names

baseline, perfs, wanted_feats, wanted_errors, plot_names = make_data()

logE = getattr(perfs[0], 'bin_centers')
logE_edges = getattr(perfs[0], 'bin_edges')
logE_errs = [(logE_edges[1]-logE_edges[0])/2 for i in range(len(logE))]
i_plots = [
    (0, 0), 
    (0, 1), 
    (1, 0), 
    (1, 1), 
    (2, 0), 
    (2, 1), 
]
f, axs = plt.subplots(2, 2, sharex=True)#, sharey=True)

for name, errname, perf, i_plot, plot_name in zip(
    wanted_feats, wanted_errors, perfs, i_plots, plot_names
):
    bl = getattr(baseline, name)
    bl_err = getattr(baseline, errname)
    mod = getattr(perf, name)
    moderr = getattr(perf, errname)

    rel_imp, sigma_rel_imp = calc_relative_error(bl, mod, e1=bl_err, e2=moderr)

    axs[i_plot].axhline(y=0.0, color='k', ls='--', linewidth=1.0)
   
    axs[i_plot].errorbar(
        logE, 
        -rel_imp, 
        yerr=sigma_rel_imp, 
        xerr=logE_errs, 
        fmt='.',
        markersize=0.5,
        linewidth=1.2,
        label='Additional feats.',
        # color='#ff7f0e'
        color='black'
    )

    axs[i_plot].grid(alpha=0.7)
    axs[i_plot].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[i_plot].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[i_plot].grid(True, which='minor', alpha=0.5, linestyle=':')
    axs[i_plot].set_title(plot_name)
    
    if i_plot[0] == 1:
        axs[i_plot].set_xlabel(r'$\log_{10}(E)$ [$E$/GeV]')
    if i_plot[1] == 0:
        axs[i_plot].set_ylabel(
            r'Relative Improvement'
        )

f.tight_layout(h_pad=1.3)
# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# broad_figure: FOTW = 2.
# single_fig, 2subfigs
FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(7.4, 4.0)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
