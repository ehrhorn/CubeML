#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess
import multiprocessing

# from src.modules.classes import *
import src.modules.loss_funcs as lf
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.constants import *
from src.modules.classes import *

def get_tot_charge(dataset, d):
    # * SEEMS TO WORK
    charge = 'raw/dom_charge'
    d['tot_charge'] = np.array([0.0]*dataset['meta/events'][()])
    for i, event in enumerate(dataset[charge]):
        d['tot_charge'][i] = np.sum(event)
    return d

def get_charge_significance(dataset, d):
     # * SEEMS TO WORK
    charge = 'raw/dom_charge'
    key = 'dom_charge_significance'
    if not 'tot_charge' in d:
        d = get_tot_charge(dataset, d)
    
    d[key] = [[]]*dataset['meta/events'][()]
    for i_event, event in enumerate(dataset[charge]):
        n_doms = len(dataset[charge][i_event])
        # * charge_frac.shape = (n_doms,)
        d[key][i_event] = n_doms*event/d['tot_charge'][i_event]
    
    return d

def get_frac_of_n_doms(dataset, d):
    # * SEEMS TO WORK
    charge = 'raw/dom_charge'
    key = 'dom_frac_of_n_doms'
    d[key] = [[]]*dataset['meta/events'][()]
    for i_event, event in enumerate(dataset[charge]):
        # * charge_frac.shape = (n_doms,)
        n_doms = event.shape[0]
        d[key][i_event] = np.arange(1, n_doms+1)/n_doms

    return d

def get_d_to_prev(dataset, d):
    x, y, z = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z'
    key = 'dom_d_to_prev'
    n_events = dataset['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):
        # n_doms = event.shape[0]
        x_diff = dataset[x][i_event][1:] - dataset[x][i_event][:-1]
        y_diff = dataset[y][i_event][1:] - dataset[y][i_event][:-1]
        z_diff = dataset[z][i_event][1:] - dataset[z][i_event][:-1]
        dists = np.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        dists = np.append([0.0], dists)
        d[key][i_event] = dists
    return d

def get_v_from_prev(dataset, d):
    t = 'raw/dom_time'
    key = 'dom_v_from_prev'
    if 'dom_d_to_prev' not in d:
        d = get_d_to_prev(dataset, d)

    n_events = dataset['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):
        t_diff = dataset[t][i_event][1:] - dataset[t][i_event][:-1]

        # * Time has discrete values due to the clock on the electronics + the pulse extraction algorithm bins the pulses in time --> more discreteness.
        t_diff = np.where(t_diff==0, 1.0, t_diff)
        t_diff = np.append([np.inf], t_diff)
        
        d[key][i_event] = d['dom_d_to_prev'][i_event]/t_diff
    return d

def get_d_minkowski_to_prev(dataset, d, n=1.309):
    # * See subsection https://en.wikipedia.org/wiki/Minkowski_space --> Minkowski Metric
    x, y, z, t = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z', 'raw/dom_time'
    key = 'dom_d_minkowski_to_prev'
    n_events = dataset['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):

        x_diff_sqr = (dataset[x][i_event][1:] - dataset[x][i_event][:-1])
        x_diff_sqr = x_diff_sqr*x_diff_sqr
        
        y_diff_sqr = (dataset[y][i_event][1:] - dataset[y][i_event][:-1])
        y_diff_sqr = y_diff_sqr*y_diff_sqr
        
        z_diff_sqr = (dataset[z][i_event][1:] - dataset[z][i_event][:-1])
        z_diff_sqr = z_diff_sqr*z_diff_sqr

        t_diff_sqr = (dataset[t][i_event][1:] - dataset[t][i_event][:-1])
        t_diff_sqr = t_diff_sqr*t_diff_sqr

        c_ns = 3e8 * 1e-9 / n
        spacetime_interval = ((c_ns**2) * t_diff_sqr) - x_diff_sqr - y_diff_sqr - z_diff_sqr
        abs_interval_root = np.sqrt(np.abs(spacetime_interval))

        spacetime_interval_root = np.where(spacetime_interval>0, abs_interval_root, -abs_interval_root)
        d[key][i_event] = spacetime_interval_root
        
    return d

def sqr_dist(data1, data2):
    diff = data1-data2
    return diff*diff

def tile_diff_sqr(dataset):
    n_doms = len(dataset)
    tiled = np.tile(dataset, (n_doms, 1))
    diff_sqr = sqr_dist(dataset, tiled.transpose()) 
    
    return diff_sqr

def get_d_closest(dataset, d):
    x, y, z, t = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z', 'raw/dom_time'
    key = 'dom_d_closest'
    n_events = dataset['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):
        n_doms = len(dataset[x][i_event])
        min_dists = np.zeros(n_doms)
        x_diff_sqr2 = tile_diff_sqr(dataset[x][i_event])
        y_diff_sqr2 = tile_diff_sqr(dataset[y][i_event])
        z_diff_sqr2 = tile_diff_sqr(dataset[z][i_event])
        tot_dist = x_diff_sqr2 + y_diff_sqr2 + z_diff_sqr2
        np.fill_diagonal(tot_dist, np.inf)
        min_dists_all = np.sqrt(np.min(tot_dist, axis=1))
        d[key][i_event] = min_dists_all
        
    return d

def get_d_minkowski_closest(dataset, d, n=1.309):
    # * See subsection https://en.wikipedia.org/wiki/Minkowski_space --> Minkowski Metric
    x, y, z, t = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z', 'raw/dom_time'
    key = 'dom_d_minkowski_closest'
    n_events = dataset['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):
        n_doms = len(dataset[x][i_event])
        min_dists = np.zeros(n_doms)
        x_diff_sqr2 = tile_diff_sqr(dataset[x][i_event])
        y_diff_sqr2 = tile_diff_sqr(dataset[y][i_event])
        z_diff_sqr2 = tile_diff_sqr(dataset[z][i_event])
        t_diff_sqr2 = tile_diff_sqr(dataset[t][i_event])

        c_ns = 3e8 * 1e-9 / n
        tot_dist = c_ns*c_ns*t_diff_sqr2 - x_diff_sqr2 - y_diff_sqr2 - z_diff_sqr2
        tot_dist_abs = np.abs(tot_dist)
        np.fill_diagonal(tot_dist_abs, np.inf)
        indices = np.argmin(tot_dist_abs, axis=1)
        # * Row i is the i'th DOMs distance to the other doms. Therefore, the minimum distance for DOM i is located at (range[i], indices[i])
        closest_squared = tot_dist_abs[list(range(n_doms)), indices]
        closest = tot_dist[list(range(n_doms)), indices]
        closest = np.where(closest>0, np.sqrt(closest_squared), -np.sqrt(closest_squared))
        d[key][i_event] = closest
        
    return d

def get_d_vertex(dataset, d):
    # * Use crs_prefits prediction - they are more similar to ours
    x, y, z = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z'
    x_pred, y_pred, z_pred = 'raw/retro_crs_prefit_x', 'raw/retro_crs_prefit_y', 'raw/retro_crs_prefit_z'
    key = 'dom_d_vertex'
    n_events = dataset['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):
        x_diff_sqr = sqr_dist(dataset[x][i_event], dataset[x_pred][i_event])
        y_diff_sqr = sqr_dist(dataset[y][i_event], dataset[y_pred][i_event])
        z_diff_sqr = sqr_dist(dataset[z][i_event], dataset[z_pred][i_event])
        tot_dist = np.sqrt(x_diff_sqr+y_diff_sqr+z_diff_sqr)
        d[key][i_event] = tot_dist
    
    return d

def get_d_minkowski_vertex(dataset, d, n=1.309):
    # * Use crs_prefits prediction - they are more similar to ours
    x, y, z, t = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z', 'raw/dom_time'
    x_pred, y_pred, z_pred, t_pred = 'raw/retro_crs_prefit_x', 'raw/retro_crs_prefit_y', 'raw/retro_crs_prefit_z', 'raw/retro_crs_prefit_time'
    key = 'dom_d_minkowski_vertex'
    n_events = dataset['meta/events'][()]
    d[key] = [[]]*n_events
    for i_event in range(n_events):
        x_diff_sqr = sqr_dist(dataset[x][i_event], dataset[x_pred][i_event])
        y_diff_sqr = sqr_dist(dataset[y][i_event], dataset[y_pred][i_event])
        z_diff_sqr = sqr_dist(dataset[z][i_event], dataset[z_pred][i_event])
        t_diff_sqr = sqr_dist(dataset[t][i_event], dataset[t_pred][i_event])

        c_ns = 3e8 * 1e-9 / n

        tot_sqr = c_ns*c_ns*t_diff_sqr - x_diff_sqr - y_diff_sqr - z_diff_sqr
        tot_abs = np.sqrt(abs(tot_sqr))
        tot_mink_dist = np.where(tot_sqr>0, tot_abs, -tot_abs)
        d[key][i_event] = tot_mink_dist
    
    return d

def get_charge_over_d_vertex(dataset, d):
    Q = 'raw/dom_charge'
    key = 'dom_charge_over_vertex'
    key2 = 'dom_charge_over_vertex_sqr'
    n_events = dataset['meta/events'][()]
    d[key] = [[]]*n_events
    d[key2] = [[]]*n_events

    if 'dom_d_vertex' not in d:
        d = get_d_vertex(dataset, d)

    for i_event in range(n_events):
        d[key][i_event] = dataset[Q][i_event]/d['dom_d_vertex'][i_event]
        d[key2][i_event] = dataset[Q][i_event]/(d['dom_d_vertex'][i_event]*d['dom_d_vertex'][i_event])

    return d

def main(data_dir, particle_code=None):
    if particle_code:
        file_list = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5' and confirm_particle_type(particle_code, file)])
    else:
        file_list = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5'])

    N_FILES = len(file_list)
    
    import time
    start = time.time()
    # * For every datafile, make a new datafile to not fuck shit up
    for i_file, file in enumerate(file_list):

        # * open file and calc important stuff
        d = {}
        with h5.File(file, 'r') as f:
            d = get_tot_charge(f, d)
            d = get_charge_significance(f, d)
            d = get_frac_of_n_doms(f, d)
            d = get_d_to_prev(f, d)
            d = get_v_from_prev(f, d)
            d = get_d_minkowski_to_prev(f, d)
            d = get_d_closest(f, d)
            d = get_d_minkowski_closest(f, d)
            d = get_d_vertex(f, d)
            d = get_d_minkowski_vertex(f, d)
            d = get_charge_over_d_vertex(f, d)
        
        # * Append our calculations to the file
        with h5.File(file, 'a') as f:
            # * Make a 'raw/'-group if it doesnt exist
            if 'raw' not in f:
                raw = f.create_group("raw")

            # * Now make the datasets
            for key, data in d.items():
                dataset_path = 'raw/'+key
                # * Check if it is a DOM-variable or global event-variable
                if data[0].shape:
                    # * If dataset already exists, delete it first
                    if dataset_path in f:
                        del f[dataset_path]
                    f.create_dataset(dataset_path, data=data, dtype=h5.special_dtype(vlen=data[0][0].dtype))

                else:
                    # * If dataset already exists, delete it first
                    if dataset_path in f:
                        del f[dataset_path]
                    f.create_dataset(dataset_path, data=data, dtype=data[0].dtype)
        # * Print progress for our sanity..
        print_progress(start, i_file+1, N_FILES)

if __name__ =='__main__':
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2_copy'
    particle_code = '140000'
    # main(data_dir)
    print(multiprocessing.cpu_count())

# # * put in new file and save it
        # path_obj = Path(file)
        # new_path = dir_path+'/'+path_obj.name
        # new_path_obj = Path(new_full_path)
        # mode = 'a' if new_path_obj.is_file() else 'w'

    # # * Make new directory
    # data_path = hf.get_project_root() + hf.get_path_from_root(data_dir)
    # name = hf.get_dataset_name(data_dir)

    # dir_path = hf.get_project_root() + '/data/2.0_'+name
    # if not Path(dir_path).is_dir():
    #     Path(dir_path).mkdir(parents=True)

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
