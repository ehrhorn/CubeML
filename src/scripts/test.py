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
# from src.modules.reporting import *

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
        try:
            energy = read_h5_directory(self.data_dir, ['true_neutrino_energy'], self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)
        except KeyError:
            energy = read_h5_directory(self.data_dir, ['true_muon_energy'], self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)

        # Transform back and extract values into list
        energy = inverse_transform(energy, get_project_root() + self.model_dir)
        energy = [y for _, y in energy.items()]
        energy = [x[0] for x in energy[0]]
        self.bin_edges = np.linspace(min(energy), max(energy), 13)

        # Calculate performance as a fn of energy for polar and azi errors
        polar_error = self.data_dict['polar_error']
        print(len(energy), len(polar_error), self.bin_edges)
        print('\nCalculating polar performance...')
        self.polar_sigmas, self.polar_errors = calc_perf2_as_fn_of_energy(energy, polar_error, self.bin_edges)
        print(self.polar_sigmas, self.polar_errors)
        print('Calculation finished!')

        azi_error = self.data_dict['azi_error']
        print('\nCalculating azimuthal performance...')
        self.azi_sigmas, self.azi_errors = calc_perf2_as_fn_of_energy(energy, azi_error, self.bin_edges)
        print('Calculation finished!')

    def get_polar_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.polar_sigmas], 'yerr': [self.polar_errors], 'xlabel': r'log(E) [GeV]', 'ylabel': 'Error [Deg]'}

    def get_azi_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.azi_sigmas], 'yerr': [self.azi_errors], 'xlabel': r'log(E) [GeV]', 'ylabel': 'Error [Deg]'}

    def save(self):

        # Save Azi first
        perf_savepath = get_project_root()+self.model_dir+'/data/AziErrorPerformance.pickle'
        img_address = get_project_root()+self.model_dir+'/figures/AziErrorPerformance.png'
        d = self.get_azi_dict()
        d['savefig'] = img_address
        _ = make_plot(d)

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
        d['savefig'] = img_address
        _ = make_plot(d)

        # Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'PolarErrorPerformance': wandb.Image(im, caption='PolarErrorPerformance')}, commit = False)

        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)


model_dir = '/media/data/bjoernhm/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/direction_reg/test_2019.12.12-12.41.10'
azipolar = AziPolarPerformance(model_dir)
# #%%
# d = azi.get_azi_dict()
# _ = make_plot(d)
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
