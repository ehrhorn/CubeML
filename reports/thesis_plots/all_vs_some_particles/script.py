from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
from src.modules.thesis_plotting import *
from src.modules.reporting import *
import os
import json

setup_pgf_plotting()

# ============================================================================
# IMPORT/MAKE DATA     
# ============================================================================

def make_data(
    models, masknames
    ):

    # Load the predictions
    for model_full, maskname in zip(models, masknames):
        print('')
        model = webpage_to_modelname(model_full)
        path = locate_model(model)
        perf = Performance(path, run_perf_eval=False, mask=maskname)
        # perf = pickle.load(open(path, 'rb'))
        energy_dict, pred_dict, crs_dict, true_dict, n_doms, _, _, _ = perf._get_data_dicts(mask=maskname)

        energy_transformed = inverse_transform(
            energy_dict, get_project_root()+perf.model_dir
        )
        # We want energy as array
        energy_arr = np.array(
            convert_to_proper_list(energy_transformed[perf._energy_key[0]])
        )
        perf._calculate_performance(
            energy_arr, 
            pred_dict, 
            crs_dict, 
            true_dict, 
            n_doms, 
            save_2D_perf_plot=False
        )
        
        path = Path(os.path.realpath(__file__))
        savepath = '/'.join(str(path).split('/')[:-1]) + '/' + maskname + '_all_subset.pickle'

        with open(savepath, 'wb') as f:
            pickle.dump(perf, f)
    


    return None

def get_title(mask):
    if mask == 'electron_neutrino':
        title = r'$e$-neutrino'
    elif mask == 'muon_neutrino':
        title = r'$\mu$-neutrino'
    else:
        title = r'$\tau$-neutrino'
    return title

def make_plots(models):

    perf_keys = [
        'polar_error',
        'log_frac_E_error',
        'directional_error',
        'len_error'
    ]
    ylabels = [
        r'$W(\Delta \theta)$ [deg]',
        r'$W\left(\log_{10} \left[ \frac{E_{pred}}{E_{true}} \right]\right)$',
        r'$U\left(\Delta\Psi\right)$ [deg]',
        r'$U(|\vec{x}_{reco}-\vec{x}_{true}|)$ [m]'
    ]
    i_plot = 0

    for model_full in models:
        print('')
        model = webpage_to_modelname(model_full)
        model_path = locate_model(model)
        mask_path = model_path + '/data_pars.json'
        with open(mask_path) as f:
            data_pars = json.load(f)
        maskname = data_pars['masks'][-1]
        path = Path(os.path.realpath(__file__))
        savepath = '/'.join(str(path).split('/')[:-1]) + '/' + maskname + '_all_subset.pickle'

        with open(savepath, 'rb') as f:
            perf_cls_all = pickle.load(f)
        
        perf_path = model_path + '/data/Performance.pickle'
        with open(perf_path, 'rb') as f:
            perf_cls_single = pickle.load(f)
        
        for perf_key, ylabel in zip(perf_keys, ylabels):
            
            try:
                perf_all = getattr(perf_cls_all, perf_key + '_sigma')
            except AttributeError:
                perf_all = getattr(perf_cls_all, perf_key + '_68th') 
            
            try:
                err_all = getattr(perf_cls_all, perf_key + '_sigmaerr')
            except AttributeError:
                err_all = getattr(perf_cls_all, perf_key + '_err68th')
            
            try:
                perf_single = getattr(perf_cls_single, perf_key + '_sigma')
                perf_retro = getattr(perf_cls_single, 'retro_' + perf_key + '_sigma')
            except AttributeError:
                perf_single = getattr(perf_cls_single, perf_key + '_68th') 
                perf_retro = getattr(perf_cls_single, 'retro_' + perf_key + '_68th')
            
            try:
                err_single = getattr(perf_cls_single, perf_key + '_sigmaerr')
                err_retro = getattr(perf_cls_single, 'retro_' + perf_key + '_sigmaerr')
            except AttributeError:
                err_single = getattr(perf_cls_single, perf_key + '_err68th')
                err_retro = getattr(perf_cls_single, 'retro_' + perf_key + '_err68th')

            rel_imp_all = getattr(perf_cls_all, perf_key + '_RI')
            rel_imp_single = getattr(perf_cls_single, perf_key + '_RI')
            rel_imp_err_all = getattr(perf_cls_all, perf_key + '_RIerr')
            rel_imp_err_single = getattr(perf_cls_single, perf_key + '_RIerr')
            
            edges = getattr(perf_cls_single, 'bin_edges')
            energy_counts = getattr(perf_cls_single, 'counts')

            # rel_imp = getattr(perf, perf_key + '_RI'))
            # rel_imp_err = getattr(perf, perf_key + '_RIerr'))

            d = {
                'edges': [edges]*3, 
                'y': [perf_all, perf_single, perf_retro], 
                'yerr': [err_all, err_single, err_retro], 
                'xlabel': r'log(E) [E/GeV]', 
                'ylabel': ylabel,
                'grid': False, 
                'label': ['All', 'Single', 'Retro'], 
                'yrange': {'bottom': 0.001}, 
                'markersize': 0.8,
                'linewidth': 1.2,
                'title': get_title(maskname)
            }
            if i_plot != 0:
                del d['label']
            f = make_plot(
                d,
                for_thesis=True,
                position=[0.125, 0.26, 0.475, 0.42],
            )

            # ENERGY PLOT
            d = {
                'data': [edges[:-1]], 
                'bins': [edges],
                'weights': [energy_counts*8], 
                'histtype': ['step'], 
                'log': [True], 
                'color': ['gray'], 
                'twinx': True, 
                'grid': False, 
                'label': ['Training Events'],
                'combine_labels': True,
                'ylabel': 'Training Events',
                'legend_loc': (0.15, 0.05),
                'frameon': False
            }
            if i_plot != 0:
                del d['label']
            if perf_key != 'polar_error':
                del d['ylabel']
            f = make_plot(d, h_figure=f, axes_index=0)
            i_plot += 1

            # RELATIVE IMPROVEMENT PLOT
            d = {
                'edges': [edges]*2, 
                'y': [rel_imp_all, rel_imp_single], 
                'yerr': [rel_imp_err_all, rel_imp_err_single],  
                'xlabel': r'$\log_{10}(E)$ [$E$/GeV]', 
                'ylabel': 'Rel. Imp.', 
                'grid': True, 
                'y_minor_ticks_multiple': 0.2
            }

            if perf_key == 'polar_error':
                del d['ylabel']
                
            yrange_d = {}
            if max(-0.5, min(rel_imp_all+rel_imp_single)) == -0.5:
                yrange_d['bottom'] = -0.5
                yrange_d['top'] = 0.5
                d['yrange'] = yrange_d
            
            d['subplot'] = True
            d['axhline'] = [0.0]
            # f = make_plot(d, h_figure=f, position=[0.625, 0.11, 0.475, 0.15])
            f = make_plot(d, h_figure=f, position=[0.125, 0.11, 0.475, 0.15])

            # Standard ratio of width to height it 6.4/4.8
            # Standard figure: FOTW = 1.0
            # Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
            # broad_figure: FOTW = 2.
            # single_fig, 2subfigs
            # for f, name in zip([f1, f2], ['energy', 'polar']):
            FOTW = get_frac_of_textwidth(keyword='single_fig')
            width = get_figure_width(frac_of_textwidth=FOTW)
            height = get_figure_height(width=width)
            f.set_size_inches(1.25*width, 1.25*height)
            name = maskname + '_' + perf_key
            path = Path(os.path.realpath(__file__))
            save_thesis_pgf(path, f, save_pgf=True, png_name=name, pgf_name=name)

    return f

masks = ['muon_neutrino', 'electron_neutrino', 'tau_neutrino']

all_model = [
    'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-04-15.05.38?workspace=user-bjoernmoelvig'
]*len(masks)

specific_models = [
    'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-04-14.28.14?workspace=user-bjoernmoelvig',
    'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-04-11.11.57?workspace=user-bjoernmoelvig',
    'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-31-04.03.42?workspace=user-bjoernmoelvig'
]

make_plots(specific_models)

# perf = make_data(all_model, masks)


# f1  = make_data(
#     energy_models, 
#     'log_frac_E_error', 
#     ylabel=r'$W\left(\log_{10} \left[ \frac{E_{pred}}{E_{true}} \right]\right)$',
#     legend_loc=(0.15, 0.05),
#     legend=True,
#     right_ylabel=False
# )
# f2  = make_data(
#     direction_models, 
#     'polar_error', 
#     ylabel=r'$W(\Delta \theta)$ [deg]',
#     legend_loc=(0.15, 0.05),
#     rel_imp_label=False
# )


# # SAVE LOGFRAC 2D ERROR DISTRIBUTION ASWELL
# model = webpage_to_modelname(energy_models[0])
# model_path = locate_model(model)
# wanted_plot = 'log_frac_E_error_2DPerformance' 
# data_path = model_path + '/figures/pickle_'

# plotname = 'log_frac_E_2D'

# save_path = get_project_root() + '/reports/thesis_plots/all_pgf/ensemble_performance_'
# path = data_path + wanted_plot + '.pickle'
# with open(path, 'rb') as f:
#     fig = pickle.load(f)
#     savefig_path = save_path + plotname + '.png'
#     fig.savefig(savefig_path, bbox_inches='tight')

# # Move copies of z-score distributions 
# import os
# newname = 'ensemble_performance_zscore_energy.pgf'
# pgf_path = model_path + '/figures/distributions/z_score_true_primary_energy.pgf'
# destination = ' /home/bjoernhm/CubeML/reports/thesis_plots/all_pgf/'
# os.system('cp ' + pgf_path + destination + newname)

# model = webpage_to_modelname(direction_models[0])
# model_path = locate_model(model)
# newname = 'ensemble_performance_zscore_z_dir.pgf'
# pgf_path = model_path + '/figures/distributions/z_score_true_primary_direction_z.pgf'
# os.system('cp ' + pgf_path + destination + newname)

# # Standard ratio of width to height it 6.4/4.8
# # Standard figure: FOTW = 1.0
# # Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# # broad_figure: FOTW = 2.
# # single_fig, 2subfigs
# for f, name in zip([f1, f2], ['energy', 'polar']):
#     FOTW = get_frac_of_textwidth(keyword='single_fig')
#     width = get_figure_width(frac_of_textwidth=FOTW)
#     height = get_figure_height(width=width)
#     f.set_size_inches(1.25*width, 1.25*height)

#     path = Path(os.path.realpath(__file__))
#     save_thesis_pgf(path, f, save_pgf=True, png_name=name, pgf_name=name)