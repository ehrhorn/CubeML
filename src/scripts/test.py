#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm

# from src.modules.classes import *
# from src.modules.loss_funcs import *
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
# from src.modules.main_funcs import *

class AziPolarPerformance:
    '''A class to create azimuthal and polar error plots - one 2D-histogram and two performance plots.
    '''

    def __init__(self, model_dir, wandb_ID=None):
        _, data_pars, _, meta_pars = load_model_pars(model_dir)
        prefix = 'transform'+str(data_pars['file_keys']['transform'])
        from_frac = data_pars['train_frac']
        to_frac = data_pars['train_frac'] + data_pars['val_frac']

        self.model_dir = get_path_from_root(model_dir)
        self.data_dir = data_pars['data_dir']
        self.meta_pars = meta_pars
        self.prefix = prefix
        self.e_key = self._get_energy_key()

        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        self.data_dict = self._get_data_dict()
        self.calculate()

    def _get_data_dict(self):
        full_pred_address = self._get_pred_path()
        keys = self._get_keys()
        data_dict = read_predicted_h5_data(full_pred_address, keys)
        return data_dict

    def _get_energy_key(self):
        dataset_name = get_dataset_name(self.data_dir)

        if dataset_name == 'MuonGun_Level2_139008':
            energy_key = ['true_muon_energy']
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            energy_key = ['true_neutrino_energy']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return energy_key

    def _get_pred_path(self):
        path_to_data = get_project_root() + self.model_dir + '/data'
        for file in Path(path_to_data).iterdir():
            if file.suffix == '.h5':
                path = str(file)
        return path
    
    def _get_keys(self):
        funcs = get_eval_functions(self.meta_pars)
        keys = []

        for func in funcs:
            keys.append(func.__name__)
        return keys


    def calculate(self):
        # Read data
        energy = read_h5_directory(self.data_dir, self.e_key, self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)

        # Transform back and extract values into list
        energy = inverse_transform(energy, get_project_root() + self.model_dir)
        energy = [y for _, y in energy.items()]
        energy = [x[0] for x in energy[0]]
        self.counts, self.bin_edges = np.histogram(energy, bins=12)
        
        # Calculate performance as a fn of energy for polar and azi errors
        polar_error = self.data_dict['polar_error']
        print('\nCalculating polar performance...')
        self.polar_sigmas, self.polar_errors = calc_perf2_as_fn_of_energy(energy, polar_error, self.bin_edges)
        print('Calculation finished!')

        azi_error = self.data_dict['azi_error']
        print('\nCalculating azimuthal performance...')
        self.azi_sigmas, self.azi_errors = calc_perf2_as_fn_of_energy(energy, azi_error, self.bin_edges)
        print('Calculation finished!')

    def get_polar_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.polar_sigmas], 'yerr': [self.polar_errors], 'xlabel': r'log(E) [GeV]', 'ylabel': 'Error [Deg]', 'grid': False}

    def get_azi_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.azi_sigmas], 'yerr': [self.azi_errors], 'xlabel': r'log(E) [GeV]', 'ylabel': 'Error [Deg]', 'grid': False}

    def get_energy_dict(self):
        return {'data': [self.bin_edges[:-1]], 'bins': [self.bin_edges], 'weights': [self.counts], 'histtype': ['step'], 'log': [True], 'color': ['lightgray'], 'twinx': True, 'grid': False, 'ylabel': 'Events'}

    def save(self):

        # Save Azi first
        perf_savepath = get_project_root()+self.model_dir+'/data/AziErrorPerformance.pickle'
        img_address = get_project_root()+self.model_dir+'/figures/AziErrorPerformance.png'
        d = self.get_azi_dict()
        h_fig = make_plot(d)
        d_energy = self.get_energy_dict()
        d_energy['savefig'] = img_address
        _ = make_plot(d_energy, h_figure=h_fig)


        # Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'AziErrorPerformance': wandb.Image(im, caption='AziErrorPerformance')}, commit = False)

        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)
        
        # Save polar next
        perf_savepath = get_project_root()+self.model_dir+'/data/PolarErrorPerformance.pickle'
        img_address = get_project_root()+self.model_dir+'/figures/PolarErrorPerformance.png'
        d = self.get_polar_dict()
        h_fig = make_plot(d)
        d_energy = self.get_energy_dict()
        d_energy['savefig'] = img_address
        _ = make_plot(d_energy, h_figure=h_fig)

        # Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'PolarErrorPerformance': wandb.Image(im, caption='PolarErrorPerformance')}, commit = False)

        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)


model_dir = '/models/oscnext-genie-level5-v01-01-pass2/regression/direction_reg/test_2019.12.13-16.29.08'
azipolar = AziPolarPerformance(model_dir)
#%%
def make_plot(plot_dict, h_figure=None, axes_index=None, position=[0.125, 0.11, 0.775, 0.77]):
    '''A custom plot function using PyPlot. If 'x' AND 'y' are in plot_dict, a xy-graph is returned, if 'data' is given, a histogram is returned. 

    Example dictionary: 
    plot_dict = {'data': [set1, set2], 'xlabel': '<LABEL_NAME>', 'ylabel': '<LABEL_NAME>', 'label':['<PLOT1_NAME>', '<PLOT2_NAME>']}

    Input: Figure dictionary
    Output: Figure handle
    '''
    plt.style.use('default')
    alpha = 0.25
    if 'grid' in plot_dict:
        grid_on = plot_dict['grid']
    else:
        grid_on = True

    if h_figure == None:

        h_figure = plt.figure()
        h_axis = h_figure.add_axes(position)

    if 'twinx' in plot_dict and h_figure != None:
        if plot_dict['twinx']:
            if axes_index == None:
                h_axis = h_figure.axes[0].twinx()
            else:
                h_axis = h_figure.axes[axes_index].twinx()
    
    if 'subplot' in plot_dict:
        h_axis = h_figure.add_axes(position)
    
    # Make a xy-plot
    if 'x' in plot_dict and 'y' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])
        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])
        
        for i_set, dataset in enumerate(plot_dict['y']):
            plot_keys = ['label']
            # Set baseline
            d = {'linewidth': 1.5}
            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            plt.plot(plot_dict['x'][i_set], dataset, **d)
            
            if 'xscale' in plot_dict: h_axis.set_xscale(plot_dict['xscale'])
            if 'yscale' in plot_dict: h_axis.set_yscale(plot_dict['yscale'])
        
        # Plot vertical lines if wanted
        if 'axvline' in plot_dict:
            for vline in plot_dict['axvline']:
                h_axis.axvline(x=vline, color = 'k', ls = ':')
            
        if 'label' in plot_dict: h_axis.legend()
        
    elif 'data' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])
        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])

        for i_set, data in enumerate(plot_dict['data']):
            
            plot_keys = ['label', 'alpha', 'density', 'bins', 'weights', 'histtype', 'log', 'color']
            
            # Set baseline
            if len(plot_dict['data']) > 1:
                d = {'alpha': 0.6, 'density': False, 'bins': 'fd'}
            else:
                d = {'alpha': 1.0, 'density': False, 'bins': 'fd'}

            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            
            plt.hist(data, **d)

            if 'label' in plot_dict: h_axis.legend()
        
        if 'yscale' in plot_dict: h_axis.set_yscale(plot_dict['yscale'])
        if 'xscale' in plot_dict: h_axis.set_xscale(plot_dict['xscale'])

    elif 'hist2d' in plot_dict:

        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])

        set1 = plot_dict['hist2d'][0]
        set2 = plot_dict['hist2d'][1]

        if type(set1) == torch.Tensor: set1 = set1.cpu().numpy()
        if type(set2) == torch.Tensor: set2 = set2.cpu().numpy()

        # Get bin-widths
        _, widths1 = np.histogram(set1, bins='fd')
        _, widths2 = np.histogram(set2, bins='fd')

        # Rescale 
        widths1 = np.linspace(min(widths1), max(widths1), int(0.5 + widths1.shape[0]/4.0))
        widths2 = np.linspace(min(widths2), max(widths2), int(0.5 + widths2.shape[0]/4.0))
        plt.hist2d(set1, set2, bins = [widths1, widths2])
        plt.colorbar()

    elif 'hexbin' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])

        set1 = plot_dict['hexbin'][0]
        set2 = plot_dict['hexbin'][1]

        if type(set1) == torch.Tensor: set1 = set1.cpu().numpy()
        if type(set2) == torch.Tensor: set2 = set2.cpu().numpy()

        # Get bin-widths - my attempt to modularize generation of 2d-histograms
        _, widths1 = np.histogram(set1, bins='fd')
        _, widths2 = np.histogram(set2, bins='fd')

        if 'range' in plot_dict:
            xmin = plot_dict['range'][0][0]
            xmax = plot_dict['range'][0][1]
            ymin = plot_dict['range'][1][0]
            ymax = plot_dict['range'][1][1]
        else:
            try:
                if set1.shape[0] == 0:
                    print('ERROR: Empty list was given to make_plot (hexbin). No plot created.')
                    return -1
                xmin = set1.min()
                xmax = set1.max()
                ymin = set2.min()
                ymax = set2.max()
            except AttributeError:
                if len(set1) == 0:
                    print('ERROR: Empty list was given to make_plot (hexbin). No plot created.')
                    return -1
                xmin = min(set1)
                xmax = max(set1)
                ymin = min(set2)
                ymax = max(set2)
            
        # Rescale. Factor of 4 comes form trial and error
        n1 = int(0.5 + widths1.shape[0]/4.0)
        n2 = int(0.5 + widths2.shape[0]/4.0)
        plt.hexbin(set1, set2, gridsize = (n1, n2))
        plt.axis([xmin, xmax, ymin, ymax])
        plt.colorbar()
        
    elif 'edges' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])

        # Calculate bin centers and 'x-error'
        centers = []
        xerrs = []
        for edges in plot_dict['edges']:
            centers.append(calc_bin_centers(edges))
            xerrs.append(calc_dists_to_binedges(edges))
        
        for i_set in range(len(centers)):
            x = centers[i_set]
            xerr = xerrs[i_set]
            y = plot_dict['y'][i_set]
            yerr = plot_dict['yerr'][i_set]
            edges = plot_dict['edges'][i_set]

            plot_keys = ['label']
            # Set baseline
            d = {'linewidth': 1.5}
            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            
            plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='.', **d)
            
        if 'xscale' in plot_dict: h_axis.set_xscale(plot_dict['xscale'])
        if 'yscale' in plot_dict: h_axis.set_yscale(plot_dict['yscale'])

        # Plot vertical lines if wanted
        if 'axvline' in plot_dict:
            for vline in plot_dict['axvline']:
                h_axis.axvline(x=vline, color = 'k', ls = ':')
            
        if 'label' in plot_dict: h_axis.legend()

    else:
        raise ValueError('Unknown plot wanted!')
    
    
    if grid_on:
        h_axis.grid(alpha=alpha)

    if 'text' in plot_dict:
        plt.text(*plot_dict['text'], transform=h_axis.transAxes)
    
    if 'savefig' in plot_dict: 
            h_figure.savefig(plot_dict['savefig'])
            print('\nFigure saved at:')
            print(plot_dict['savefig'])

    return h_figure

# azipolar.save()
# d = azipolar.get_azi_dict()
# d.save()
# d['grid'] = False
# h_fig = make_plot(d)
# d2 = azipolar.get_energy_dict()
# d2['twinx'] = True
# # d2['grid'] = False
# h_fig = make_plot(d2, h_figure=h_fig)

d = azipolar.get_azi_dict()
h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
d['subplot'] = True
h_fig = make_plot(d, h_figure=h_fig, position=[0.125, 0.11, 0.775, 0.15])

#%%
_, data_pars, _, meta_pars = load_model_pars(model_dir)
data_dir = data_pars['data_dir']
azi_crs = read_h5_directory(data_dir, ['retro_crs_prefit_azi', 'retro_crs_prefit_zen'], 'transform0', from_frac=0.8, to_frac=1.0)
true = read_h5_directory(data_dir, ['true_neutrino_direction_x', 'true_neutrino_direction_y',  'true_neutrino_direction_z'], 'transform0', from_frac=0.8, to_frac=1.0)

# azi_crs = inverse_transform(azi_crs, get_project_root() + model_dir)
true = inverse_transform(true, get_project_root() + model_dir)
azi_crs = convert_keys(azi_crs, [key for key in azi_crs], ['azi', 'zen'])
true = convert_keys(true, [key for key in true], ['x', 'y', 'z'])
def get_retro_crs_prefit_azi_error(retro_dict, true_dict, units='degrees'):
    # use atan2 to calculate angle 
    # - see https://pytorch.org/docs/stable/torch.html#torch.atan
    pi = 3.14159265359

    xy_truth = torch.tensor([true['x'], true['y']])
    azi_truth_signed = torch.atan2(xy_truth[1, :], xy_truth[0, :])

    # Convert retro_crs to signed angle
    pred_signed = [entry if entry < pi else entry - 2*pi for entry in retro_dict['azi']]

    # add 180 degrees - retro_crs appears to predict direction neutrino came from and not neutrino direction..
    pred_signed = torch.tensor([entry-pi if entry > 0 else entry + pi for entry in pred_signed], dtype=azi_truth_signed.dtype)
    diff = pred_signed-azi_truth_signed
    true_diff = torch.where(abs(diff)>pi, -2*torch.sign(diff)*pi+diff, diff)

    if units == 'radians':
        return true_diff 

    elif units == 'degrees':
        return true_diff*(180/pi)

def get_retro_crs_prefit_polar_error(retro_dict, true_dict, units='degrees'):
    pi = 3.14159265359

    x_true, y_true, z_true = true_dict['x'], true_dict['y'], true_dict['z']
    dir_truth = torch.tensor([x_true, y_true, z_true])
    length_truth = torch.sum(dir_truth*dir_truth, dim=0)**0.5
    polar_truth = torch.acos(dir_truth[2, :]/length_truth)

    # retro_crs seems to predit the direction the neutrino came from and not the neutrinos direction - therefore do a parity.
    polar_preds = pi-torch.tensor(retro_dict['zen'], dtype=polar_truth.dtype)
    if units == 'radians':
        diff = polar_preds-polar_truth
    elif units == 'degrees':
        diff = (180/pi)*(polar_preds-polar_truth)
    
    return diff   
azi_crs_error = get_retro_crs_prefit_azi_error(azi_crs, true)
polar_crs_error = get_retro_crs_prefit_polar_error(azi_crs, true)
_ = make_plot({'data': [polar_crs_error]})
# _ = make_plot({'data': [azi_crs_error]})

        # prefix = 'transform'+str(data_pars['file_keys']['transform'])
        # from_frac = data_pars['train_frac']
        # to_frac = data_pars['train_frac'] + data_pars['val_frac']

        # self.model_dir = get_path_from_root(model_dir)
# [0.125, 0.11, 0.775, 0.77]
# azipolar.save()

# for key, item in azi.data_dict.items():
#     print(key, item)
# #%%
# n = 1000
# n_bootstraps = 1000
# dist_sorted = np.random.normal(size=n)
# dist_sorted.sort()
# indices = np.arange(0, n)


# p = 0.75
# sigma = np.sqrt(p*n*(1-p))
# mean = int(n*p)
# plussigma = int(mean+sigma+1)
# minussigma = int(mean-sigma-1)
# print(mean, plussigma, minussigma)
# print('True: %.3f [%.3f, %.3f]'%(norm.ppf(p), norm.ppf(p+np.sqrt(p*(1-p)/n)), norm.ppf(p-np.sqrt(p*(1-p)/n))))


# # bootstrap
# bootstrap_indices = np.random.choice(indices, size=(n, n_bootstraps))
# bootstrap_indices.sort(axis=0)
# # print(bootstrap_indices)
# bootstrap_samples = dist_sorted[bootstrap_indices]

# bootstrap_mean = bootstrap_samples[mean, :]
# bootstrap_plussigma = bootstrap_samples[plussigma, :]
# bootstrap_minussigma = bootstrap_samples[minussigma , :]
# fig = make_plot({'data': [bootstrap_mean, bootstrap_plussigma, bootstrap_minussigma]})
# bootstrap_estimate_mean = np.mean(bootstrap_mean)
# bootstrap_estimate_plussigma = np.mean(bootstrap_plussigma)
# bootstrap_estimate_minussigma = np.mean(bootstrap_minussigma)
# print('Estimate: %.3f [%.3f, %.3f]'%(bootstrap_estimate_mean, bootstrap_estimate_plussigma, bootstrap_estimate_minussigma))





# # print(bootstrap_samples)
# #%%
# fig = make_plot({'data': [dist_sorted, bootstrap_samples]})
# #%%
# dist = np.random.normal(size=(n, n))
# dist.sort(axis=0)
# percentile = dist[int(n*p), :]
# fig = make_plot({'data': [percentile]})
# mean = np.mean(percentile)
# std = np.std(percentile)

# print(dot_prods.sum()/batch_size, test.sum()/batch_size)
# def read_one_at_a_time(path, index):
#     key = 'toi_point_on_line_y'
#     with h5.File(path, 'r') as f:
#         data = f['raw/'+key][index]
#     return data

# def read_batch(path, indices):
#     key = 'toi_point_on_line_y'
#     with h5.File(path, 'r') as f:
#         data = f['raw/'+key][indices]
#     return data

# def read_npy(path):
#     data = np.load(path)
#     return data

# # save a numpy file
# pwd = '/home/bjoernhm/CubeML/src/scripts/'
# name = 'numpy_file.npy'
# # a = np.array([123.23])
# # np.save(pwd+name, a)

# path = '/home/bjoernhm/CubeML/data/MuonGun_Level2_139008/000001.h5'
# batch_size = 64
# tests = 100
# data = read_one_at_a_time(path, 0)
# data = read_batch(path, list(range(batch_size)))

# t_oaat_start = time()
# for i in range(tests):
#     for j in range(batch_size):
#         data = read_one_at_a_time(path, 0)
# t_oaat_end = time()

# t_rb_start = time()
# for i in range(tests):
#     a = sorted(list(range(batch_size)))
#     data = read_batch(path, list(range(batch_size)))
# t_rb_end = time()

# t_np_start = time()
# for i in range(tests):
#     for j in range(batch_size):
#         data = read_npy(pwd+name)
# t_np_end = time()

# print('One at a time %d tests: %.3f seconds'%(tests, t_oaat_end-t_oaat_start) )
# print('One batch %d tests: %.3f seconds'%(tests, t_rb_end-t_rb_start) )
# print('One .npy at a time %d tests: %.3f seconds'%(tests, t_np_end-t_np_start) )



# log_performance_plots(path)

# print(torch.cuda.device_count())
# # TODO setup function that returns gpu-id so we can utilize both GPU's when they are available.
# for i in range(torch.cuda.device_count()):
#     print(torch.cuda.get_device_name(i))




# ### CALCULATE ICECUBE PERFORMANCE

# data_dir = '/data/MuonGun_Level2_139008'
# model_dir = '/models/MuonGun_Level2_139008/regression/direction_reg/2019-11-24-00.43.36'
# predictor_keys = ['toi_direction_x', 'toi_direction_y', 'toi_direction_z']
# true_keys = ['true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z']
# energy_key = ['true_muon_energy']
# prefix = 'transform0'
# from_frac = 0.8
# to_frac = 1.0

# true_vals = read_h5_directory(data_dir, true_keys, prefix, from_frac=from_frac, to_frac=to_frac)
# predictor_vals = read_h5_directory(data_dir, predictor_keys, prefix, from_frac=from_frac, to_frac=to_frac)
# energy = read_h5_directory(data_dir, energy_key, prefix, from_frac=from_frac, to_frac=to_frac)

# true_vals = inverse_transform(true_vals, get_project_root() + model_dir)
# predictor_vals = inverse_transform(predictor_vals, get_project_root() + model_dir)
# energy = inverse_transform(energy, get_project_root() + model_dir)

# expected_keys = ['x', 'y', 'z']
# predictor_keys = [key for key in predictor_vals]
# true_keys = [key for key in true_vals]
# energy_key = [key for key in energy]

# true_vals = convert_keys(true_vals, true_keys, expected_keys)
# predictor_vals = convert_keys(predictor_vals, predictor_keys, expected_keys)

# directional_error = directional_error_from_cartesian(predictor_vals, true_vals)
# energy = [x[0] for x in energy['true_muon_energy']]

# toi_edges, toi_maes, toi_errors = calc_perf_as_fn_of_energy(energy, directional_error)

# path = get_project_root()+'/models/MuonGun_Level2_139008/regression/direction_reg/2019-11-24-00.43.36/data/predict_model_72_Loss=0.08142805.h5'
# dir_error_best = read_predicted_h5_data(path, ['directional_error'])
# # energy = read_h5_directory(data_dir, energy_key, prefix=prefix,from_frac=from_frac, to_frac=to_frac)
# #%%
# best_edges, best_maes, best_errors = calc_perf_as_fn_of_energy(energy, dir_error_best['directional_error'])
# # best_edges = [x+0.1 for x in best_edges]
# # d = {'edges': [best_edges], 'y': [best_maes], 'yerr': [best_errors], 'xlabel': r'log(E)', 'ylabel': 'Error (degrees)'}
# toi_edges = [x+0.02 for x in toi_edges]
# d = {'edges': [toi_edges, best_edges], 'y': [toi_maes, best_maes], 'yerr': [toi_errors, best_errors], 'label': ['ToI', 'Model'], 'xlabel': r'log(E)', 'ylabel': 'Error (degrees)'}

# fig = make_plot(d)
