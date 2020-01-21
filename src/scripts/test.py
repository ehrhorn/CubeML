#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess
from multiprocessing import Pool, cpu_count

# from src.modules.classes import *
import src.modules.loss_funcs as lf
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.constants import *
from src.modules.classes import *
import src.modules.preprocessing as pp

def make_feature_histogram(package):
    # * Unpack
    key, d, clip_dict, file_list, \
    n_wanted_sample, n_wanted_histogram, particle_code = package

    # * Read some data
    all_data = []
    for file in file_list:
        # * once enough data has been read, break out
        if len(all_data)>n_wanted_sample:
            break
        data = read_h5_dataset(file, key, prefix='raw/')
        if data[0].shape:
            for entry in data:
                all_data.extend(entry)
        else:
            all_data.extend(data)
    
    # * Data read. Now draw a random sample
    indices = list(range(len(all_data)))
    random.shuffle(indices)
    random_subsample = sorted(indices[:n_wanted_histogram])

    # * Draw histogram and save it.
    plot_data = sorted(np.array(all_data)[random_subsample])
    if clip_dict:
        minimum = clip_dict['min']
        maximum = clip_dict['max']
        plot_data = np.clip(plot_data, minimum, maximum)
    d['data'] = [plot_data]
    d['title'] = key

    path = get_project_root() + '/plots/features/'
    d['savefig'] = path + particle_code + '_' + key + '.png'
    fig = rpt.make_plot(d)

data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2_copy'
particle_code = '140000'

files = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5' and confirm_particle_type(particle_code, file)])
random.shuffle(files)    

keys = pp.get_feature_keys()
dicts = pp.get_feature_plot_dicts()
clip_dicts = pp.get_feature_clip_dicts()

n_wanted_sample = 1e7
n_wanted_histogram = 50e3
dicts = [dicts[key] for key in keys]
clip_dicts = [clip_dicts[key] for key in keys]
files_list = [files]*len(keys)
n_wanted_sample = [n_wanted_sample for key in keys]
n_wanted_histogram = [n_wanted_histogram for key in keys]
particle_code = [particle_code for key in keys]


packages = [entry for entry in zip(keys, dicts, clip_dicts, files_list, n_wanted_sample, n_wanted_histogram, particle_code)]

# * Use multiprocessing for parallelizing the job.
available_cores = cpu_count()
with Pool(available_cores) as p:
    p.map(make_feature_histogram, packages)

# tot = []
# for entry in d['dom_charge_over_vertex']:
#     tot.extend(entry)

# tot = sorted(tot)
# tot = np.array(tot)
# print('we done here too')

# tot = (tot - np.median(tot))/calc_iqr(tot)
# path = get_project_root() + '/plots/dom_d_mink_to_prev.png'
# title = r'$d_{Minkowski}$ from $DOM_{t-1}$ to $DOM_{t} $'
# pd = {'data': [tot[:167000]], 'log': [False]}#, 'title': title, 'savefig': path}
# f = rpt.make_plot(pd)
# print(tot.shape)
# tot = []
# for entry in d['dom_t']:
#     tot.extend(entry)

# tot = sorted(tot)
# pd = {'data': [tot], 'log': [False]}#, 'title': title, 'savefig': path}
# f = rpt.make_plot(pd)
