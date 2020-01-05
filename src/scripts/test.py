#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess

# from src.modules.classes import *
import src.modules.loss_funcs as lf
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
import src.modules.reporting as rpt

class VertexPerformance:
    """A class to create and save performance plots for interaction vertex predictions. If available, the relative improvement compared to Icecubes reconstruction is plotted aswell. A one-number performance summary is saved as the median of the total vertex distance error.     
    
    Raises:
        KeyError: If an unknown dataset is encountered.
    
    Returns:
        [type] -- Instance of class.
    """    

    def __init__(self, model_dir, wandb_ID=None):
        _, data_pars, _, meta_pars = load_model_pars(model_dir)
        prefix = 'transform'+str(data_pars['file_keys']['transform'])
        from_frac = data_pars['train_frac']
        to_frac = data_pars['train_frac'] + data_pars['val_frac']

        self.model_dir = get_path_from_root(model_dir)
        self.data_pars = data_pars
        self.meta_pars = meta_pars
        self.prefix = prefix
        self.energy_key = self._get_energy_key()
        self._reco_keys = self._get_reco_keys()
        self._true_xyzt_keys = get_target_keys(data_pars, meta_pars)

        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        data_dict = self._get_data_dict()
        self._create_performance_plots(data_dict)
        self._calculate_onenum_performance(data_dict)

    def _get_data_dict(self):
        full_pred_address = self._get_pred_path()
        keys = self._get_keys()

        data_dict = read_predicted_h5_data(full_pred_address, keys)
        return data_dict

    def _get_energy_key(self):
        dataset_name = get_dataset_name(self.data_pars['data_dir'])

        if dataset_name == 'MuonGun_Level2_139008':
            energy_key = ['true_muon_energy']
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            energy_key = ['true_primary_energy']
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

    def _get_reco_keys(self):
        dataset_name = get_dataset_name(self.data_pars['data_dir'])

        if dataset_name == 'MuonGun_Level2_139008':
            reco_keys = None
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            reco_keys = ['retro_crs_prefit_x', 'retro_crs_prefit_y', 'retro_crs_prefit_z', 'retro_crs_prefit_time']
            # self._true_xyz = ['true_primary_position_x', 'true_primary_position_y',  'true_primary_position_z']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return reco_keys
    
    def _create_performance_plots(self, data_dict):

        # * Transform back and extract values into list
        
        x_error = data_dict['vertex_x_error']
        print(type(x_error), x_error.shape)
        a += 1
        print('\nCalculating x performance...')
        self.x_sigmas, self.x_errors = calc_perf2_as_fn_of_energy(energy, x_error, self.bin_edges)
        print('Calculation finished!')

        y_error = data_dict['vertex_y_error']
        print('\nCalculating y performance...')
        self.y_sigmas, self.y_errors = calc_perf2_as_fn_of_energy(energy, y_error, self.bin_edges)
        print('Calculation finished!')

        z_error = data_dict['vertex_z_error']
        print('\nCalculating z performance...')
        self.z_sigmas, self.z_errors = calc_perf2_as_fn_of_energy(energy, z_error, self.bin_edges)
        print('Calculation finished!')

        t_error = data_dict['vertex_t_error']
        print('\nCalculating time performance...')
        self.t_sigmas, self.t_errors = calc_perf2_as_fn_of_energy(energy, t_error, self.bin_edges)
        print('Calculation finished!')

        # * If an I3-reconstruction exists, get it
        if self._reco_keys:
            pred_crs = read_h5_directory(self.data_pars['data_dir'], self._reco_keys, prefix=self.prefix, from_frac=self.from_frac, to_frac=self.to_frac, n_wanted=self.data_pars.get('n_predictions_wanted', np.inf), particle=self.data_pars['particle'])
            true = read_h5_directory(self.data_pars['data_dir'], self._true_xyzt_keys, prefix=self.prefix, from_frac=self.from_frac, to_frac=self.to_frac, n_wanted=self.data_pars.get('n_predictions_wanted', np.inf), particle=self.data_pars['particle'])

            # * Ensure keys are proper so the angle calculations work
            # pred_crs = inverse_transform(pred_crs, get_project_root() + self.model_dir)
            true = inverse_transform(true, get_project_root() + self.model_dir)

            pred_crs = convert_keys(pred_crs, [key for key in pred_crs], ['x', 'y', 'z', 't'])
            true = convert_keys(true, [key for key in true], ['x', 'y', 'z', 't'])
            true = { key: convert_to_proper_list(item) for key, item in true.items() }
            x_crs_error = vertex_x_error(pred_crs, true)
            y_crs_error = vertex_y_error(pred_crs, true)
            z_crs_error = vertex_z_error(pred_crs, true)
            t_crs_error = vertex_t_error(pred_crs, true)

            print('\nCalculating crs x performance...')
            self.x_crs_sigmas, self.x_crs_errors = calc_perf2_as_fn_of_energy(energy, x_crs_error, self.bin_edges)
            print('Calculation finished!')

            print('\nCalculating crs y performance...')
            self.y_crs_sigmas, self.y_crs_errors = calc_perf2_as_fn_of_energy(energy, y_crs_error, self.bin_edges)
            print('Calculation finished!')

            print('\nCalculating crs z performance...')
            self.z_crs_sigmas, self.z_crs_errors = calc_perf2_as_fn_of_energy(energy, z_crs_error, self.bin_edges)
            print('Calculation finished!')

            print('\nCalculating crs time performance...')
            self.t_crs_sigmas, self.t_crs_errors = calc_perf2_as_fn_of_energy(energy, t_crs_error, self.bin_edges)
            print('Calculation finished!')

            # * Calculate the relative improvement - e_diff/I3_error. Report decrease in error as a positive result
            a, b = calc_relative_error(self.x_crs_sigmas, self.x_sigmas, self.x_crs_errors, self.x_errors)
            self.x_relative_improvements, self.x_sigma_improvements = -a, b

            a, b = calc_relative_error(self.y_crs_sigmas, self.y_sigmas, self.y_crs_errors, self.y_errors)
            self.y_relative_improvements, self.y_sigma_improvements = -a, b

            a, b = calc_relative_error(self.z_crs_sigmas, self.z_sigmas, self.z_crs_errors, self.z_errors)
            self.z_relative_improvements, self.z_sigma_improvements = -a, b

            a, b = calc_relative_error(self.t_crs_sigmas, self.t_sigmas, self.t_crs_errors, self.t_errors)
            self.t_relative_improvements, self.t_sigma_improvements = -a, b
        
        else:
            self.x_relative_improvements = None
            self.x_sigma_improvements = None
            self.y_relative_improvements = None
            self.y_sigma_improvements = None
            self.z_relative_improvements = None
            self.z_sigma_improvements = None
            self.t_relative_improvements = None
            self.t_sigma_improvements = None
    

    def get_energy_dict(self):
        return {'data': [self.bin_edges[:-1]], 'bins': [self.bin_edges], 'weights': [self.counts], 'histtype': ['step'], 'log': [True], 'color': ['lightgray'], 'twinx': True, 'grid': False, 'ylabel': 'Events'}

    def get_x_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.x_sigmas], 'yerr': [self.x_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [m]', 'grid': False}
    def get_y_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.y_sigmas], 'yerr': [self.y_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [m]', 'grid': False}
    def get_z_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.z_sigmas], 'yerr': [self.z_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [m]', 'grid': False}
    def get_t_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.t_sigmas], 'yerr': [self.t_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [m]', 'grid': False}

    def get_rel_x_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.x_relative_improvements], 'yerr': [self.x_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}
    def get_rel_y_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.y_relative_improvements], 'yerr': [self.y_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}
    def get_rel_z_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.z_relative_improvements], 'yerr': [self.z_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}
    def get_rel_t_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.t_relative_improvements], 'yerr': [self.t_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}

# from src.modules.main_funcs import *
# model_path = '/home/bjoern/Thesis/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/vertex_reg/2020-01-04-02.15.02'
# a = VertexPerformance(model_path)
import time
def check_dir(path):
    n_exps = len([str(exp) for exp in Path(path).glob('*.json')])
    while n_exps>0:
        print(n_exps)
        n_exps = len([str(exp) for exp in Path(path).glob('*.json')])
        time.sleep(5)


path = get_project_root()+'/src/scripts/testlol'
check_dir(path)
print('outside')
# a = rpt.IceCubePerformance('oscnext-genie-level5-v01-01-pass2')
# d = a.get_y_dict()
# _ = rpt.make_plot(d)
#* #* print(bootstrap_samples)
#* #%%
#* fig = make_plot({'data': [dist_sorted, bootstrap_samples]})
#* #%%
#* dist = np.random.normal(size=(n, n))
#* dist.sort(axis=0)
#* percentile = dist[int(n*p), :]
#* fig = make_plot({'data': [percentile]})
#* mean = np.mean(percentile)
#* std = np.std(percentile)

#* print(dot_prods.sum()/batch_size, test.sum()/batch_size)
#* def read_one_at_a_time(path, index):
#*    key = 'toi_point_on_line_y'
#*    with h5.File(path, 'r') as f:
#*        data = f['raw/'+key][index]
#*    return data

#* def read_batch(path, indices):
#*    key = 'toi_point_on_line_y'
#*    with h5.File(path, 'r') as f:
#*        data = f['raw/'+key][indices]
#*    return data

#* def read_npy(path):
#*    data = np.load(path)
#*    return data

#* #* save a numpy file
#* pwd = '/home/bjoernhm/CubeML/src/scripts/'
#* name = 'numpy_file.npy'
#* #* a = np.array([123.23])
#* #* np.save(pwd+name, a)

#* path = '/home/bjoernhm/CubeML/data/MuonGun_Level2_139008/000001.h5'
#* batch_size = 64
#* tests = 100
#* data = read_one_at_a_time(path, 0)
#* data = read_batch(path, list(range(batch_size)))

#* t_oaat_start = time()
#* for i in range(tests):
#*    for j in range(batch_size):
#*        data = read_one_at_a_time(path, 0)
#* t_oaat_end = time()

#* t_rb_start = time()
#* for i in range(tests):
#*    a = sorted(list(range(batch_size)))
#*    data = read_batch(path, list(range(batch_size)))
#* t_rb_end = time()

#* t_np_start = time()
#* for i in range(tests):
#*    for j in range(batch_size):
#*        data = read_npy(pwd+name)
#* t_np_end = time()

#* print('One at a time %d tests: %.3f seconds'%(tests, t_oaat_end-t_oaat_start) )
#* print('One batch %d tests: %.3f seconds'%(tests, t_rb_end-t_rb_start) )
#* print('One .npy at a time %d tests: %.3f seconds'%(tests, t_np_end-t_np_start) )



#* log_performance_plots(path)

#* print(torch.cuda.device_count())
#* #* TODO setup function that returns gpu-id so we can utilize both GPU's when they are available.
#* for i in range(torch.cuda.device_count()):
#*    print(torch.cuda.get_device_name(i))




#* ###* CALCULATE ICECUBE PERFORMANCE

#* data_dir = '/data/MuonGun_Level2_139008'
#* model_dir = '/models/MuonGun_Level2_139008/regression/direction_reg/2019-11-24-00.43.36'
#* predictor_keys = ['toi_direction_x', 'toi_direction_y', 'toi_direction_z']
#* true_keys = ['true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z']
#* energy_key = ['true_muon_energy']
#* prefix = 'transform0'
#* from_frac = 0.8
#* to_frac = 1.0

#* true_vals = read_h5_directory(data_dir, true_keys, prefix, from_frac=from_frac, to_frac=to_frac)
#* predictor_vals = read_h5_directory(data_dir, predictor_keys, prefix, from_frac=from_frac, to_frac=to_frac)
#* energy = read_h5_directory(data_dir, energy_key, prefix, from_frac=from_frac, to_frac=to_frac)

#* true_vals = inverse_transform(true_vals, get_project_root() + model_dir)
#* predictor_vals = inverse_transform(predictor_vals, get_project_root() + model_dir)
#* energy = inverse_transform(energy, get_project_root() + model_dir)

# expected_keys = ['x', 'y', 'z']
#! predictor_keys = [key for key in predictor_vals]
#? true_keys = [key for key in true_vals]
#* energy_key = [key for key in energy]

#* true_vals = convert_keys(true_vals, true_keys, expected_keys)
#* predictor_vals = convert_keys(predictor_vals, predictor_keys, expected_keys)

#* directional_error = directional_error_from_cartesian(predictor_vals, true_vals)
#* energy = [x[0] for x in energy['true_muon_energy']]

#* toi_edges, toi_maes, toi_errors = calc_perf_as_fn_of_energy(energy, directional_error)

#* path = get_project_root()+'/models/MuonGun_Level2_139008/regression/direction_reg/2019-11-24-00.43.36/data/predict_model_72_Loss=0.08142805.h5'
#* dir_error_best = read_predicted_h5_data(path, ['directional_error'])
#* #* energy = read_h5_directory(data_dir, energy_key, prefix=prefix,from_frac=from_frac, to_frac=to_frac)
#* #%%
#* best_edges, best_maes, best_errors = calc_perf_as_fn_of_energy(energy, dir_error_best['directional_error'])
#* #* best_edges = [x+0.1 for x in best_edges]
#* #* d = {'edges': [best_edges], 'y': [best_maes], 'yerr': [best_errors], 'xlabel': r'log(E)', 'ylabel': 'Error (degrees)'}
#* toi_edges = [x+0.02 for x in toi_edges]
#* d = {'edges': [toi_edges, best_edges], 'y': [toi_maes, best_maes], 'yerr': [toi_errors, best_errors], 'label': ['ToI', 'Model'], 'xlabel': r'log(E)', 'ylabel': 'Error (degrees)'}

#* fig = make_plot(d)
