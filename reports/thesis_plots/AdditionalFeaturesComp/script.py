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
    
    models = [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-03-05-21.09.41?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-04-20-08.19.33?workspace=user-bjoernmoelvig',
        # 'https://app.wandb.ai/cubeml/cubeml/runs/2020-03-20-11.41.40?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-04-16-11.34.16?workspace=user-bjoernmoelvig'
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

        
    return perfs

perfs = make_data()

wanted_feats = [
    'directional_error_68th',
    'polar_error_sigma',
    'log_frac_E_error_sigma',
    'relative_E_error_sigma',
    'len_error_68th',
    'vertex_t_error_sigma',   
]

wanted_errors = [
    'directional_error_err68th',
    'polar_error_sigmaerr',
    'log_frac_E_error_sigmaerr',
    'relative_E_error_sigmaerr',
    'len_error_err68th',
    'vertex_t_error_sigmaerr',   
]

plot_names = [
    'Direction Reco.',
    'Polar Angle Reco.',
    'Energy Reco. (logarithmic error)',
    'Energy Reco. (relative error)',
    'Interaction Vertex Reco.',
    'Interaction Time Reco.',
]

rel_imps1 = []
rel_imp_sigmas1 = []

rel_imps2 = []
rel_imp_sigmas2 = []

logE = getattr(perfs[0], 'bin_centers')
logE_edges = getattr(perfs[0], 'bin_edges')
logE_errs = [(logE_edges[1]-logE_edges[0])/2 for i in range(len(logE))]

for name, errname in zip(wanted_feats, wanted_errors):
    one = getattr(perfs[0], name)
    oneerr = getattr(perfs[0], errname)
    two = getattr(perfs[1], name)
    twoerr = getattr(perfs[1], errname)
    three = getattr(perfs[2], name)
    threeerr = getattr(perfs[2], errname)

    rel_imp, sigma_rel_imp = calc_relative_error(one, two, e1=oneerr, e2=twoerr)
    rel_imps1.append(-rel_imp)
    rel_imp_sigmas1.append(sigma_rel_imp)

    rel_imp, sigma_rel_imp = calc_relative_error(one, three, e1=oneerr, e2=threeerr)
    rel_imps2.append(-rel_imp)
    rel_imp_sigmas2.append(sigma_rel_imp)

f, axs = plt.subplots(3, 2, sharex=True, sharey=True)
i_plot = [
    (0, 0), 
    (0, 1), 
    (1, 0), 
    (1, 1), 
    (2, 0), 
    (2, 1), 
]
n_plot = 0
for plot_name, perf_imp, perf_err, i_plot, perf_imp2, perf_err2 in zip(
    plot_names, rel_imps1, rel_imp_sigmas1, i_plot, rel_imps2, rel_imp_sigmas2
):
    axs[i_plot].axhline(y=0.0, color='k', ls='--', linewidth=1.0)
   
    axs[i_plot].errorbar(
        logE, 
        perf_imp, 
        yerr=perf_err, 
        xerr=logE_errs, 
        fmt='.',
        markersize=0.5,
        linewidth=1.2,
        label='Additional feats.'
        # color='#ff7f0e'
        # color='black'
    )
    axs[i_plot].errorbar(
        logE, 
        perf_imp2, 
        yerr=perf_err2, 
        xerr=logE_errs, 
        fmt='.',
        markersize=0.5,
        linewidth=1.2,
        label='Retrained Id.'
        # color='#ff7f0e'
        # color='black'
    )

    if i_plot == (0, 0):
        axs[i_plot].legend()

    axs[i_plot].set_ylim(bottom=-0.12, top=0.12)

    axs[i_plot].grid(alpha=0.7)
    axs[i_plot].yaxis.set_minor_locator(MultipleLocator(0.05))
    axs[i_plot].xaxis.set_minor_locator(MultipleLocator(0.25))
    axs[i_plot].grid(True, which='minor', alpha=0.5, linestyle=':')
    

    
    axs[i_plot].set_title(plot_name)
    
    if i_plot[0] == 2:
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
f.set_size_inches(7.4, 6.0)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
