import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess
from multiprocessing import Pool, cpu_count
from scipy import interpolate

from src.modules.classes import *
import src.modules.loss_funcs as lf
import src.modules.helper_functions as hf
from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.constants import *
from src.modules.classes import *
import src.modules.preprocessing as pp
import sys

N_BINS_WEIGHTS = 24
PRINT_EVERY = 100000

def calc_weights_multiprocess(pack):
    indices, interpolator, key, path, n_per_dir, subprocess_id = pack
  
    weights =[-1]*len(indices)
    n_indices = len(indices)
    for i_index, index in enumerate(indices):
        # * Check each file.
        full_path = path+'/pickles/'+str(index//n_per_dir)+'/'+str(index)+'.pickle'
        
        event = pickle.load(open(full_path, "rb" ))
        energy = event['raw']['true_primary_energy']
        weights[i_index] = interpolator(energy)
        
        if (i_index)%PRINT_EVERY == 0:
            print(get_time(), 'Subprocess %d: Processed %d of %d'%(subprocess_id, i_index, n_indices))
            sys.stdout.flush()
    
    return weights

def geomean_energy_entry_weights(masks, dataset_path, multiprocess=True, from_frac=0.0, to_frac=1.0):
    """Given a pickled dataset, a weight is calculated for each event. The weight is calculated (using a quadratic spline) as 

    w = (n_events*icecube_performance)**-0.5.

    In other words, the geometric mean of the inverse of the number of events in a certain energy range multiplied by Icecubes performance in the same range. It can be chosen to only use a fraction of the dataset for the creation of the quadratic spline. If an event is not in the mask, it is assigned a nan as weight.

    The weights are normalized such that the average weight of an event in a batch is 1. 

    Arguments:
        masks {list} -- Masknames for the data to calculate weights on
        dataset_path {str} -- path to dataset
    
    Keyword Arguments:
        multiprocess {bool} -- Whether or not to use multiprocessing in calculating weights for each event (default: {True})
        from_frac {float} -- Lower limit of the amount of data to use to calculate the spline (default: {0.0})
        to_frac {float} -- Upper limit of the amount of data to use to calculate the spline (default: {1.0})
    
    Returns:
        list -- Weights for each event
    """    

    # * Get mask
    mask_all = np.array(load_pickle_mask(dataset_path, masks))
    
    # * Get indices used for interpolator-calculation
    n_events = len(mask_all)
    indices = get_indices_from_fraction(n_events, from_frac, to_frac)
    indices_interpolator = mask_all[indices]    
    keys = ['retro_crs_prefit_energy', 'true_primary_energy']
    data_d = read_pickle_data(dataset_path, indices_interpolator, keys)
    retro_key = keys[0]
    true_key = keys[1]

    # * Calculate performance
    true = data_d['true_primary_energy']
    retro_dict = {'E': data_d[retro_key]}
    true_dict = {'logE': data_d[true_key]}
    perf = get_retro_crs_prefit_relE_error(retro_dict, true_dict, reporting=False)

    # * Sort w.r.t. true energy and bin
    bin_edges = np.linspace(0.0, 4.0, N_BINS_WEIGHTS+1)
    counts, bin_edges = np.histogram(true, bins=bin_edges)

    # * Calculate performance and weights
    retro_sigmas, _ = calc_perf2_as_fn_of_energy(true, perf, bin_edges)
    x = calc_bin_centers(bin_edges)
    geomean = np.sqrt(1/(counts*retro_sigmas))
    
    # * Normalize the weights. We want the average weight of a batch-entry to be 1
    # * Therefore: Calculate the mean weight in a batch and normalize by it
    ave_weight = np.sum(geomean*counts/np.sum(counts))
    geomean_normed = geomean/ave_weight

    # * Calculate spline
    interpolator_quadratic = interpolate.interp1d(x, geomean_normed, fill_value="extrapolate", kind='quadratic')
    
    # * Loop over all events using multiprocessing
    if multiprocess:
        available_cores = cpu_count()
        n_tot = get_n_tot_pickles(dataset_path)
        
        # * Create packs - loop over all events
        chunks = np.array_split(mask_all, available_cores)
        interpolator_list = [interpolator_quadratic]*len(chunks)
        keys_list = [true_key]*len(chunks)
        paths = [dataset_path]*len(chunks)
        n_per_dir_list = [get_n_events_per_dir(dataset_path)]*len(chunks)
        subprocess_id = list(range(available_cores))
        packed = zip(chunks, interpolator_list, keys_list, paths, n_per_dir_list, subprocess_id)
        
        # * Multiprocess and recombine
        with Pool(available_cores) as p:
            weights = p.map(calc_weights_multiprocess, packed)
        weights_combined = flatten_list_of_lists(weights)
        
        # * Make a list of weights - put nan where mask didn't apply
        weights_list = np.array([np.nan]*n_tot)
        weights_list[mask_all] = np.array(weights_combined)
    
    return weights_list

if __name__ == '__main__':
    # * Choose dataset, masks and size of subset to calculate weights from
    dataset_path = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2'
    masks = ['muon_neutrino', 'dom_interval_min0_max200']
    name = 'geomean_energy_entry.pickle'

    # * Ensure weight directory exists
    weights_dir = dataset_path+'/weights/'
    if not Path(weights_dir).exists():
        Path(weights_dir).mkdir()

    # * from and to are used for spline calculation
    from_, to_ = 0.8, 1.0
    weights = geomean_energy_entry_weights(masks, dataset_path, from_frac=from_, to_frac=to_)

    # * Save weights as a pickle
    weight_d = {'masks': masks, 'weights': weights}
    pickle.dump(weight_d, open(weights_dir+name, 'wb'))

