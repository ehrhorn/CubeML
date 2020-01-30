import h5py as h5
import numpy as np
from pathlib import Path
import subprocess
import pickle
import sys
import argparse
from multiprocessing import cpu_count, Pool

from src.modules.helper_functions import get_project_root, get_path_from_root, get_time, flatten_list_of_lists, get_particle_code

PRINT_EVERY = 10000

description = 'Creates masks for pickled Icecube data.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--name', default='None', type=str, help='Sets the name of the mask.')

args = parser.parse_args()

def make_mask(data_path, dirs, mask_name='all', min_doms=0, max_doms=np.inf, min_energy=-np.inf, max_energy=np.inf):
    # * make mask directory if it doesn't exist
    data_path = get_project_root() + get_path_from_root(data_path)

    if mask_name == 'dom_interval':
        mask_path = make_dom_interval_mask(data_path, dirs, min_doms, max_doms)
    
    elif mask_name == 'all':
        mask_path = make_all_mask(data_path, dirs)
    
    elif mask_name == 'muon_neutrino':
        mask_path = make_particle_mask(data_path, dirs, mask_name)
    
    elif mask_name == 'energy_interval':
        mask_path = make_energy_interval_mask(data_path, dirs, min_energy, max_energy)

    return mask_path

def make_dom_interval_mask(data_path, dirs, min_doms, max_doms, multiprocess=True):
    
    # * Split the candidates into chunks for multiprocessing
    if multiprocess:
        available_cores = cpu_count()
        dirs_chunked = np.array_split(dirs, available_cores)
        min_doms_list = [min_doms]*len(dirs_chunked)
        max_doms_list = [max_doms]*len(dirs_chunked)
        packed = zip(dirs_chunked, min_doms_list, max_doms_list)
        
        with Pool(available_cores) as p:
            accepted_lists = p.map(find_dom_interval_passed_cands, packed)
        
        # * Combine again
        mask = sorted(flatten_list_of_lists(accepted_lists))

    else:
        raise ValueError('make_dom_interval_mask: Only multiprocessing solution implemented')
    
    # * save it
    mask_path = data_path+'/masks/dom_interval_min%d_max%d.pickle'%(min_doms, max_doms)
    pickle.dump(mask, open(mask_path, 'wb'))
    
    return mask_path

def make_energy_interval_mask(data_path, dirs, min_energy, max_energy, multiprocess=True):

    # * Split the candidates into chunks for multiprocessing
    if multiprocess:
        available_cores = cpu_count()
        dirs_chunked = np.array_split(dirs, available_cores)
        min_energy_list = [min_energy]*len(dirs_chunked)
        max_energy_list = [max_energy]*len(dirs_chunked)
        packed = zip(dirs_chunked, min_energy_list, max_energy_list)
        
        with Pool(available_cores) as p:
            accepted_lists = p.map(find_energy_interval_passed_cands, packed)
        
        # * Combine again
        mask = sorted(flatten_list_of_lists(accepted_lists))

    else:
        raise ValueError('make_energy_interval_mask: Only multiprocessing solution implemented')
    
    # * save it
    mask_path = data_path+'/masks/energy_interval_min%.1f_max%.1f.pickle'%(min_energy, max_energy)
    pickle.dump(mask, open(mask_path, 'wb'))
    
    return mask_path

def make_particle_mask(data_path, dirs, particle, multiprocess=True):
    
    particle_code = get_particle_code(particle)

    # * Split the candidates into chunks for multiprocessing
    if multiprocess:
        available_cores = cpu_count()
        dirs_chunked = np.array_split(dirs, available_cores)
        particle_codes = [particle_code]*len(dirs_chunked)
        packed = zip(dirs_chunked, particle_codes)
        
        with Pool(available_cores) as p:
            accepted_lists = p.map(find_particles, packed)
        
        # * Combine again
        mask = sorted(flatten_list_of_lists(accepted_lists))

    else:
        raise ValueError('make_particle_mask: Only multiprocessing solution implemented')
    
    # * save it
    mask_path = data_path+'/masks/%s.pickle'%(particle)
    pickle.dump(mask, open(mask_path, 'wb'))
    
    return mask_path

def find_dom_interval_passed_cands(pack):
    # * Unpack
    dirs, min_doms, max_doms = pack
    
    accepted = []
    i_file = 0
    
    # * Loop over the given directories
    for directory in dirs:

        # * Loop over the events in the subdirectory
        for file in directory.iterdir():
            
            # * Check each file.
            event = pickle.load(open(file, "rb" ))
            n_doms = event['raw']['dom_charge'].shape[0]
            if min_doms <= n_doms <= max_doms:
                accepted.append(int(file.stem))

            # * Print for sanity
            i_file += 1
            if (i_file)%PRINT_EVERY == 0:
                print(get_time(), 'Subprocess: Processed %d'%(i_file))
                sys.stdout.flush()

    return accepted

def find_energy_interval_passed_cands(pack):
    # * Unpack
    dirs, min_energy, max_energy = pack
    
    accepted = []
    i_file = 0
    
    # * Loop over the given directories
    for directory in dirs:

        # * Loop over the events in the subdirectory
        for file in directory.iterdir():
            
            # * Check each file.
            event = pickle.load(open(file, "rb" ))
            energy = event['raw']['true_primary_energy']
            
            if min_energy <= energy <= max_energy:
                accepted.append(int(file.stem))

            # * Print for sanity
            i_file += 1
            if (i_file)%PRINT_EVERY == 0:
                print(get_time(), 'Subprocess: Processed %d'%(i_file))
                sys.stdout.flush()

    return accepted

def find_particles(pack):
    # * Unpack
    dirs, particle_code = pack
    
    accepted = []
    i_file = 0
    
    # * Loop over the given directories
    for directory in dirs:

        # * Loop over the events in the subdirectory
        for file in directory.iterdir():
            
            # * Check each file.
            event = pickle.load(open(file, "rb" ))
            if particle_code == event['meta']['particle_code']:
                accepted.append(int(file.stem))

            # * Print for sanity
            i_file += 1
            if (i_file) % PRINT_EVERY == 0:
                print(get_time(), 'Subprocess: Processed %d'%(i_file))
                sys.stdout.flush()
    
    return accepted

def make_all_mask(data_path, dirs):
    # * Check number of events
    n_events = 0
    for directory in dirs:
        n_events += len([file for file in Path(directory).iterdir()])
    
    # * Make mask - a list of indices
    mask = np.arange(n_events)
    
    # * save it
    mask_path = data_path + '/masks/all.pickle'
    pickle.dump(mask, open(mask_path, 'wb'))
    
    return mask_path


if __name__ == '__main__':
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2'
    # * Options: particle_name (e.g. muon_neutrino), dom_interval, energy_interval
    mask_name = args.name
    if mask_name == 'None':
        raise KeyError('Must parse a name!')

    min_doms = 0
    max_doms = 200
    
    min_energy = 0.0
    max_energy = 3.0
    mask_dict = {'mask_name': mask_name, 'min_doms':
        min_doms, 'max_doms': max_doms, 'min_energy': min_energy, 'max_energy': max_energy}

    # * If maskdirectory doesn't exist, make it
    mask_dir = data_dir + '/masks'
    if not Path(mask_dir).exists(): 
        Path(mask_dir).mkdir()

    dirs = [file for file in Path(data_dir + '/pickles').iterdir()]
    # * For debugging on local.
    # dirs = [file for file in Path(data_dir).iterdir() if file.is_dir() and file.name != 'masks' and file.name != 'transformers']
    
    print(get_time(), 'Mask calculation begun.')
    mask_path = make_mask(data_dir, dirs, **mask_dict)
    print(get_time(), 'Mask created', mask_path)
