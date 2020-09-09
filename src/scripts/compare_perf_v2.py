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
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-27-13.33.29?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-27-11.43.44?workspace=user-bjoernmoelvig'
    ]
    wanted_feats = ['log_frac_E_error_sigma']
    wanted_errors = ['log_frac_E_error_sigmaerr']

    # # Direction models
    # models = [
    #     'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-28-16.25.11?workspace=user-bjoernmoelvig',
    #     'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-28-22.42.34?workspace=user-bjoernmoelvig'
    # ]
    # wanted_feats = ['directional_error_68th','polar_error_sigma']
    # wanted_errors = ['directional_error_err68th','polar_error_sigmaerr']

    # # Vertex models
    # models = [
    #     'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-29-23.17.17?workspace=user-bjoernmoelvig',
    #     'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-29-06.15.51?workspace=user-bjoernmoelvig'
    # ]
    # wanted_feats = ['len_error_68th','vertex_t_error_sigma']
    # wanted_errors = ['len_error_err68th','vertex_t_error_sigmaerr'] 


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
        
    return compare_perf, perfs, wanted_feats, wanted_errors

baseline, perfs, wanted_feats, wanted_errors = make_data()

rel_imps = []
rel_imp_sigmas = []

logE = getattr(baseline, 'bin_centers')
logE_edges = getattr(baseline, 'bin_edges')
logE_width = [(logE_edges[1]-logE_edges[0])/2 for i in range(len(logE))]


for name, errname in zip(wanted_feats, wanted_errors):
    bl = getattr(baseline, name)
    bl_err = getattr(baseline, errname)
    f, axs = plt.subplots(1, 1)
    axs.grid(alpha=0.7)
    axs.yaxis.set_minor_locator(MultipleLocator(0.05))
    axs.xaxis.set_minor_locator(MultipleLocator(0.25))
    axs.grid(True, which='minor', alpha=0.5, linestyle=':')
    axs.set_xlabel(r'$\log_{10}(E)$ [$E$/GeV]')
    axs.set_ylabel(
        r'Relative Improvement'
    )
    for perf in perfs:
        
        mod = getattr(perf, name)
        moderr = getattr(perf, errname)

        rel_imp, sigma_rel_imp = calc_relative_error(bl, mod, e1=bl_err, e2=moderr)
        plot_name = perf.model_dir.split('/')[-1]

        axs.axhline(y=0.0, color='k', ls='--', linewidth=1.0)

        n_stds = np.sum(-rel_imp)/np.sqrt(np.sum(sigma_rel_imp*sigma_rel_imp))
        print(plot_name, 'Standard deviations from 0 (%s): %.2f'%(name, n_stds))
        axs.errorbar(
            logE, 
            -rel_imp, 
            yerr=sigma_rel_imp, 
            xerr=logE_width, 
            fmt='.',
            markersize=0.5,
            linewidth=1.2,
            label=plot_name
        )
    
    axs.legend()
    f.savefig(get_project_root() + '/reports/plots/comparison_' + name + '.png')


    

# for perf_imp, perf_err, i_plot, perf_imp2, perf_err2 in zip(
#     rel_imps1, rel_imp_sigmas1, i_plot, rel_imps2, rel_imp_sigmas2
# ):
    
#     axs.errorbar(
#         logE, 
#         perf_imp2, 
#         yerr=perf_err2, 
#         xerr=logE_width, 
#         fmt='.',
#         markersize=0.5,
#         linewidth=1.2,
#         label='Retrained Id.'
#         # color='#ff7f0e'
#         # color='black'
#     )

#     if i_plot == (0, 0):

#     # axs.set_ylim(bottom=-0.12, top=0.12)

    

# f.tight_layout(h_pad=1.3)
# # Standard ratio of width to height it 6.4/4.8
# # Standard figure: FOTW = 1.0
# # Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# # broad_figure: FOTW = 2.
# # single_fig, 2subfigs
# FOTW = get_frac_of_textwidth(keyword='single_fig')
# width = get_figure_width(frac_of_textwidth=FOTW)
# height = get_figure_height(width=width)
# f.set_size_inches(7.4, 6.0)

# # ============================================================================
# # SAVE PGF AND PNG FOR VIEWING    
# # ============================================================================

# path = Path(os.path.realpath(__file__))
# # save_thesis_pgf(path, f, save_pgf=True)
