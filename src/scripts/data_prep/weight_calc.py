import numpy as np
import sys
import pickle
import argparse

from multiprocessing import Pool, cpu_count
from scipy import interpolate
from pathlib import Path
t
from src.modules.helper_functions import get_time, get_project_root, load_pickle_mask, get_indices_from_fraction, read_pickle_data, calc_perf2_as_fn_of_energy, calc_bin_centers, get_n_tot_pickles, get_n_events_per_dir, flatten_list_of_lists
from src.modules.eval_funcs import get_retro_crs_prefit_relE_error
from src.modules.reporting import make_plot

description = 'Calculates weights for a dataset.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-d', '--dev', action='store_true', help='Initiates developermode - weights are not saved, and only weights for a small subset of the dataset is calculated')
parser.add_argument('--from_frac', default=0.8, type=float, help='Sets the lower bound for amount of data used to calculate interpolator.')
parser.add_argument('--to_frac', default=1.0, type=float, help='Sets the upper bound for amount of data used to calculate interpolator.')
parser.add_argument('--name', nargs='+', type=str, help='Sets the kind of weights to calculate.')

args = parser.parse_args()

N_BINS_WEIGHTS = 24
PRINT_EVERY = 100000

def assign_energy_weights_multiprocess(masks, dataset_path, interpolator_quadratic, true_key='true_primary_energy', debug=False):
    available_cores = cpu_count()
    n_tot = get_n_tot_pickles(dataset_path)
    
    # * Create packs - loop over all events
    # * Get mask

    mask_all = np.array(load_pickle_mask(dataset_path, masks))
    if debug:
        mask_all = mask_all[:1000]
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

def calc_energy_performance_weights(masks, dataset_path, from_frac=0.0, to_frac=1.0):
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
    bin_centers = calc_bin_centers(bin_edges)
    
    return bin_centers, counts, retro_sigmas

def geomean_muon_energy_entry_weights(masks, dataset_path, multiprocess=True, from_frac=0.0, to_frac=1.0, debug=False):
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
    x, counts, retro_sigmas = calc_energy_performance_weights(masks, dataset_path, from_frac=from_frac, to_frac=to_frac)
    geomean = np.sqrt(1/(counts*retro_sigmas))
    
    # * Calculate spline
    interpolator_quadratic = make_scaled_interpolator(geomean, counts, x)
    
    # * Loop over all events using multiprocessing
    if multiprocess:
        weights_list = assign_energy_weights_multiprocess(masks, dataset_path, interpolator_quadratic, debug=debug)
    
    return weights_list, interpolator_quadratic

def inverse_performance_muon_energy(masks, dataset_path, multiprocess=True, from_frac=0.0, to_frac=1.0, debug=False):
    """Given a pickled dataset, a weight is calculated for each event. The weight is calculated (using a quadratic spline) as 

    w = (icecube_performance)**-0.5.

    In other words, the inverse of Icecubes performance in each energy range. It can be chosen to only use a fraction of the dataset for the creation of the quadratic spline. If an event is not in the mask, it is assigned a nan as weight.

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
    x, counts, retro_sigmas = calc_energy_performance_weights(masks, dataset_path, from_frac=from_frac, to_frac=to_frac)
    weights_unscaled = 1.0/np.array(retro_sigmas)
    
    interpolator_quadratic = make_scaled_interpolator(weights_unscaled, counts, x)

    # * Loop over all events using multiprocessing
    if multiprocess:
        weights_list = assign_energy_weights_multiprocess(masks, dataset_path, interpolator_quadratic, debug=debug)
    
    return weights_list, interpolator_quadratic

def make_scaled_interpolator(weights, counts, bin_centers):
    # * Normalize the weights. We want the average weight of a batch-entry to be 1
    # * Therefore: Calculate the mean weight in a batch and normalize by it
    ave_weight = np.sum(weights*counts/np.sum(counts))
    weights_scaled = weights/ave_weight

    # * Calculate spline
    interpolator = interpolate.interp1d(bin_centers, weights_scaled, fill_value="extrapolate", kind='quadratic')
    
    return interpolator

def make_weights(name, masks, dataset_path, from_frac=0.0, to_frac=1.0, debug=False):
    
    if name == 'geomean_muon_energy_entry':
        weights, interpolator = geomean_muon_energy_entry_weights(masks, dataset_path, from_frac=from_frac, to_frac=to_frac, debug=debug)
    elif name == 'inverse_performance_muon_energy':
        weights, interpolator = inverse_performance_muon_energy(masks, dataset_path, from_frac=from_frac, to_frac=to_frac, debug=debug)

    return weights, interpolator 

if __name__ == '__main__':
    
    # ! Can use 2*n_cpus - only ~45 % of processors are used
    # ! Update: Seems like with 2*n_cpus, ~45 % of procesors are also used.
    # * Choose dataset, masks and size of subset to calculate weights from
    dataset_path = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2'
    masks = ['muon_neutrino']
    names = args.name
    if not names:
        raise KeyError('Names must be supplied!')

    # * Ensure weight directory exists
    weights_dir = dataset_path+'/weights/'
    if not Path(weights_dir).exists():
        Path(weights_dir).mkdir()

    # * from and to are used for spline calculation
    from_frac, to_frac = args.from_frac, args.to_frac
    if args.dev:
        from_frac, to_frac = 0.8, 0.81
        PRINT_EVERY = 100
    
    for name in names:
        weights, interpolator = make_weights(name, masks, dataset_path, from_frac=from_frac, to_frac=to_frac, debug=args.dev)

        # * Save weights as a pickle
        if not args.dev:
            weight_d = {'masks': masks, 'weights': weights, 'interpolator': interpolator}
            pickle.dump(weight_d, open(weights_dir+name+'.pickle', 'wb'))
            print(get_time(), 'Saved weights at %s'%(weights_dir+name+'.pickle'))
        else:
            x = np.linspace(0.0, 4.0)
            y = interpolator(x)
            d = {'x': [x], 'y': [y]}
            d['savefig'] = get_project_root()+'/WEIGHT_TEST.png'
            d['yscale'] = 'log'
            _ = make_plot(d)