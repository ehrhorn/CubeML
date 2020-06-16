import numpy as np
import sys
import argparse
import joblib

from multiprocessing import Pool, cpu_count
from scipy import interpolate
from pathlib import Path

from src.modules.helper_functions import get_time, get_project_root, load_pickle_mask, get_indices_from_fraction, read_pickle_data, calc_width_as_fn_of_data, calc_bin_centers, get_n_tot_pickles, get_n_events_per_dir, flatten_list_of_lists, make_multiprocess_pack
from src.modules.eval_funcs import retro_relE_error
from src.modules.reporting import make_plot
from src.modules.classes import SqliteFetcher
from src.modules.constants import *

description = 'Calculates weights for a dataset.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-d', '--dev', action='store_true', help='Initiates developermode - weights are not saved, and only weights for a small subset of the dataset is calculated')
parser.add_argument(
    '--from_frac', 
    default=0.8, 
    type=float, 
    help='Sets the lower bound for amount of data used to calculate interpolator.')
parser.add_argument(
    '--to_frac', 
    default=1.0, 
    type=float, 
    help='Sets the upper bound for amount of data used to calculate interpolator.')
parser.add_argument(
    '--name', 
    nargs='+', 
    type=str, 
    help='Sets the kind of weights to calculate. Options:'
    'geomean_muon_energy_entry, inverse_performance_muon_energy emu_balanced'
    )
parser.add_argument(
    '--masks', 
    nargs='+', 
    type=str, 
    help='Sets the kind of weights to calculate. Options: geomean_muon_energy_entry, inverse_performance_muon_energy')

parser.add_argument(
    '--make_plot', 
    action='store_true', 
    help='Sets the kind of weights to calculate. Options: geomean_muon_energy_entry, inverse_performance_muon_energy')

args = parser.parse_args()

N_BINS_WEIGHTS = 24
PRINT_EVERY = 100000
USE_N_EVENTS = 500000
AVAILABLE_CORES = cpu_count()

def assign_energy_weights_multiprocess(
    ids, 
    db, 
    interpolator_quadratic, 
    true_key=['true_primary_energy'], 
    debug=False
    ):
    
    # Create packs - loop over all events
    n_chunks = len(ids)//AVAILABLE_CORES
    chunks = np.array_split(ids, n_chunks)
    packed = make_multiprocess_pack(
        chunks, interpolator_quadratic, true_key, db
    )

    # Multiprocess and recombine
    with Pool(AVAILABLE_CORES) as p:
        weights = p.map(calc_weights_multiprocess, packed)
    weights_combined = flatten_list_of_lists(weights)
    
    # # Make a dictionary with weights - put nan where mask didn't apply
    # event_ids = db.ids
    # all_weights_dict = {str(event_id): np.nan for event_id in event_ids}
    # weights_dict = {
    #     str(event_id): weight for event_id, weight in zip(ids, weights_combined)
    # }
    # all_weights_dict.update(weights_dict)
    
    return weights_combined

def assign_uniform_direction_weights_multiprocess(
    ids, 
    db, 
    interpolator_quadratic, 
    true_key=['true_primary_energy'], 
    debug=False
    ):
    
    # Create packs - loop over all events
    # n_chunks = len(ids)//AVAILABLE_CORES
    chunks = np.array_split(ids, AVAILABLE_CORES)
    packed = make_multiprocess_pack(
        chunks, interpolator_quadratic, true_key, db
    )

    # Multiprocess and recombine
    with Pool(AVAILABLE_CORES) as p:
        weights = p.map(calc_weights_multiprocess, packed)
    weights_combined = flatten_list_of_lists(weights)
    
    # # Make a dictionary with weights - put nan where mask didn't apply
    # event_ids = db.ids
    # all_weights_dict = {str(event_id): np.nan for event_id in event_ids}
    # weights_dict = {
    #     str(event_id): weight for event_id, weight in zip(ids, weights_combined)
    # }
    # all_weights_dict.update(weights_dict)
    
    return weights_combined

def calc_weights_multiprocess(pack):
   
    # Unpack
    indices, interpolator, true_key, db = pack
    # Fetch required data
    dicted_data = db.fetch_features(indices, scalar_features=true_key)
    arr_data = np.array([
        data[true_key[0]] for event_id, data in dicted_data.items()
    ])

    # Inverse transform it
    transformer_path = '/'.join([PATH_DATA_OSCNEXT, 'sqlite_transformers.pickle'])
    transformers = joblib.load(open(transformer_path, 'rb'))
    try:
        transformer = transformers[true_key[0]]
    except KeyError:
        transformer = None
    
    # inverse transform
    if transformer is not None:
        transformed_data = [
            np.squeeze(
                transformer.inverse_transform(
                    arr_data.reshape(-1, 1)
                )
            )
        ]
        weights = interpolator(transformed_data)[0]
    else:
        transformed_data = arr_data
        weights = interpolator(transformed_data)

    # Calculate interpolated value
    return weights

def calc_energy_performance_weights(ids, db):

    # Load data
    keys = ['retro_crs_prefit_energy', 'true_primary_energy']
    retro_key = keys[0]
    true_key = keys[1]
    all_data = db.fetch_features(ids, keys)

    retro_E = {
        retro_key: [
            data[retro_key] for index, data in all_data.items()
            ]
        }
    true_logE_transformed =np.array([
            data[true_key] for index, data in all_data.items()
            ])

    # Transform true logE. Load first
    transformer_path = '/'.join([PATH_DATA_OSCNEXT, 'sqlite_transformers.pickle'])
    transformers = joblib.load(open(transformer_path, 'rb'))
    transformer = transformers[true_key]
    
    # inverse transform
    true_logE = {
        true_key: np.squeeze(
            transformer.inverse_transform(
                true_logE_transformed.reshape(-1, 1)
            )
        )
    }

    # Calculate performance
    perf = retro_relE_error(retro_E, true_logE, reporting=False)
    
    # Sort w.r.t. true energy and bin
    bin_edges = np.linspace(0.0, 4.0, N_BINS_WEIGHTS+1)
    counts, bin_edges = np.histogram(true_logE[true_key], bins=bin_edges)

    # Calculate performance and weights
    retro_sigmas, _, _, _, _ = calc_width_as_fn_of_data(true_logE[true_key], perf, bin_edges)
    bin_centers = calc_bin_centers(bin_edges)
    
    return bin_centers, counts, retro_sigmas

def calc_uniform_direction_weights(ids, db):

    # Load data
    keys = ['true_primary_direction_z']
    true_key = keys[0]
    all_data = db.fetch_features(ids, keys)

    z_dir =np.array([
            data[true_key] for index, data in all_data.items()
            ])
    
    # Sort w.r.t. true energy and bin
    bin_edges = np.linspace(-1.0, 1.0, N_BINS_WEIGHTS+1)
    counts, bin_edges = np.histogram(z_dir, bins=bin_edges)

    bin_centers = calc_bin_centers(bin_edges)
    
    return bin_centers, counts

def geomean_muon_energy_entry_weights(masks, dataset_path, multiprocess=True,  debug=False):
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
    
    # Calculate spline
    interpolator_quadratic = make_scaled_interpolator(geomean, counts, x)
    
    # Loop over all events using multiprocessing
    if multiprocess:
        weights_list = assign_energy_weights_multiprocess(masks, dataset_path, interpolator_quadratic, debug=debug)
    
    return weights_list, interpolator_quadratic

def inverse_performance_muon_energy(name, 
    ids, 
    db, 
    multiprocess=True, 
    debug=False,
    interpolator=None
    ):
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
        dict -- Weights for each event
    """ 

    # Get indices used for interpolator-calculation
    if not interpolator:
        n_events = min(len(ids), USE_N_EVENTS)
        event_ids = ids[:n_events]
        print(get_time(), 'Calculating performance..')
        x, counts, retro_sigmas = calc_energy_performance_weights(event_ids, db)
        weights_unscaled = 1.0/np.array(retro_sigmas)
        print(get_time(), 'Performance calculated!')
        
        print(get_time(), 'Fitting interpolator')
        interpolator= make_scaled_interpolator(weights_unscaled, counts, x)
        print(get_time(), 'Interprolator fitted!')

    # Loop over all events using multiprocessing
    print(get_time(), 'Assigning energy weights...')
    if multiprocess:
        weights_dict = assign_energy_weights_multiprocess(
            ids, db, interpolator, debug=debug
        )
    print(get_time(), 'Energy weights assigned!')

    return weights_dict, interpolator

def make_scaled_interpolator(weights, counts, bin_centers):
    # Normalize the weights. We want the average weight of a batch-entry to be 1
    # Therefore: Calculate the mean weight in a batch and normalize by it
    ave_weight = np.sum(weights*counts/np.sum(counts))
    weights_scaled = weights/ave_weight

    # Calculate spline
    interpolator = interpolate.interp1d(bin_centers, weights_scaled, fill_value="extrapolate", kind='quadratic')
    
    return interpolator

def make_weights(name, masked_ids, db, debug=False, interpolator=None):
    
    if name == 'geomean_muon_energy_entry':
        weights, interpolator = geomean_muon_energy_entry_weights(
            name, masked_ids, db, debug=debug, interpolator=interpolator
            )
    elif name == 'inverse_performance_muon_energy':
        weights, interpolator = inverse_performance_muon_energy(
            name, masked_ids, db, debug=debug, interpolator=interpolator
            )
    elif name == 'uniform_direction_weights':
        weights, interpolator = uniform_direction(
            name, masked_ids, db, debug=debug, interpolator=interpolator
            )
    elif name == 'nue_numu_balanced':
        weights, interpolator = nue_numu_balanced(
            name, masked_ids, db, debug=debug
        )

    return weights, interpolator 

def nue_numu_balanced(name, masked_ids, db, debug=False, interpolator=None):
    # Load data
    meta_key = 'particle_code'
    all_data = db.fetch_features(
        all_events=masked_ids,
        meta_features=[meta_key]
    )
    
    if not interpolator:
        # Count number of nue and numu
        particle_codes = np.array(
            [data[meta_key] for ev_id, data in all_data.items()]
        )
        n_nue = np.sum(
            np.where(particle_codes == 120000)
        )
        n_numu = np.sum(
            np.where(particle_codes == 140000)
        )

        tot = n_nue + n_numu
        interpolator = {
            str(120000): tot/(2*n_nue),
            str(140000): tot/(2*n_numu)
        }
    
    # weights = np.empty(len(masked_ids))
    weights = [
        interpolator[str(all_data[entry][meta_key])] for entry in masked_ids
    ]

    return weights, interpolator

def uniform_direction(
    name, 
    ids, 
    db, 
    multiprocess=True, 
    debug=False,
    interpolator=None
    ):

    # Get indices used for interpolator-calculation
    if not interpolator:
        n_events = min(len(ids), USE_N_EVENTS)
        event_ids = ids[:n_events]
        print(get_time(), 'Calculating direction bins..')
        x, counts = calc_uniform_direction_weights(event_ids, db)
        weights_unscaled = 1.0/np.array(counts)
        print(get_time(), 'Bins calculated!')
        
        print(get_time(), 'Fitting interpolator')
        interpolator= make_scaled_interpolator(weights_unscaled, counts, x)
        print(get_time(), 'Interprolator fitted!')

    # Loop over all events using multiprocessing
    print(get_time(), 'Assigning energy weights...')
    if multiprocess:
        weights_dict = assign_uniform_direction_weights_multiprocess(
            ids, db, interpolator, true_key=['true_primary_direction_z'], debug=debug
        )
    print(get_time(), 'Energy weights assigned!')

    return weights_dict, interpolator

if __name__ == '__main__':
    
    print(get_time(), 'Weight calculation initiated')

    # Choose dataset, masks and size of subset to calculate weights from
    # masks = ['muon_neutrino']
    names = args.name
    if not names:
        raise KeyError('Names must be supplied!')

    # Ensure weight directory exists
    weights_dir = '/'.join([PATH_DATA_OSCNEXT, 'weights'])
    if not Path(weights_dir).exists():
        Path(weights_dir).mkdir()

    for name in names:
        interpolator = None
        all_weights = {}
        for path, keyword in zip(
            [PATH_TRAIN_DB, PATH_VAL_DB], 
            ['train', 'val'],
        ):  
            # Get DB and mask
            db = SqliteFetcher(path)
            db_specific_masks = [e+'_'+keyword for e in args.masks]
            ids = load_pickle_mask(PATH_DATA_OSCNEXT, db_specific_masks)
            ids = [
                str(i) for i in ids
            ]
            
            # If developing, use less data
            if args.dev:
                USE_N_EVENTS = 1000
                PRINT_EVERY = 100
                ids = ids[:1000]
                
            
            # Calculate weights and potentially interpolator
            if not interpolator:
                weights, interpolator = make_weights(
                    name, ids, db, debug=args.dev
                )
            else:
                weights, interpolator = make_weights(
                name, ids, db, debug=args.dev, interpolator=interpolator
            )

            # Save in DB
            ids_strings = [str(idx) for idx in ids]
            print(get_time(), 'Writing %s to database'%(name))
            db.write('scalar', name, ids_strings, weights)
            print(get_time(), 'Weights saved!')


        # Save a figure of the weights
        if args.make_plot:
            if name == 'uniform_direction_weights':
                x = np.linspace(-1.0, 1.0)
            else:
                x = np.linspace(0.0, 4.0)
            y = interpolator(x)
            d = {'x': [x], 'y': [y]}
            d['savefig'] = '/'.join([get_project_root(), 'reports/plots', name+'.png'])
            d['yscale'] = 'log'
            _ = make_plot(d)
