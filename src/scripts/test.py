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
from src.modules.reporting import make_plot
# from src.modules.main_funcs import *

class angle_loss(torch.nn.Module):
    '''takes two tensors with shapes (B, 3) as input and calculates the angular error. Adds 1e-8 to denominator to avoid division with zero.
    '''
    def __init__(self):
        super(angle_loss,self).__init__()
        
    def forward(self, x, y):
        dot_prods = torch.sum(x*y, dim=1)
        len_x = torch.sqrt(torch.sum(x*x, dim=1))
        len_y = torch.sqrt(torch.sum(y*y, dim=1))
        err = dot_prods/(len_x*len_y + 1e-6)
        err = torch.acos(err)
        err = torch.mean(err)
        return err

loss = angle_loss()
y = torch.tensor([1.0, 0.0, 0.0], requires_grad = True)
x_init = torch.tensor([0.00000001, 0.00000001, 0.0], requires_grad = True)
x_init.register_hook(print)
len_x_init = torch.sqrt(torch.sum(x_init*x_init)+1e-12)
x = 0.0+x_init/len_x_init
# x.register_hook(lambda grad: torch.tensor([0.0, 0.0, 0.0]) if grad[0] != grad[0] else grad)
x.register_hook(print)
# x = torch.tensor([-0.0000001, 0.000, 0.0], requires_grad = True)

# * Use clamp? Using clamp(-1.0 + eps, 1.0 - eps)
# * If length is 0, add random numbers? 
# * is setting gradient to 0 same as ignoring the bad input?
dot_prods = torch.sum(x*y)
len_x = torch.sqrt(torch.sum(x*x))
len_y = torch.sqrt(torch.sum(y*y))

err1 = dot_prods/(len_x*len_y)
print('err1: %.2e'%(err1))
err1.register_hook(lambda grad: torch.tensor(0.0) if grad == torch.tensor(-np.inf) else grad)
err1.register_hook(print)

err2 = torch.acos(err1)
err2.register_hook(print)

err3 = torch.mean(err2)
err3.register_hook(print)

err3.backward()
print(0.1**10)
# print(x.grad, x_init.grad)
# subprocess.run(['git', 'add', '-f', model_name+'.dvc'], cwd=model_dir)
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
