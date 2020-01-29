import h5py as h5
import numpy as np
from pathlib import Path
import subprocess
import pickle
import sys
from multiprocessing import cpu_count, Pool
from src.modules.helper_functions import get_project_root, get_path_from_root, get_time, flatten_list_of_lists, get_particle_code

PRINT_EVERY = 10000


def make_mask(data_path, dirs, mask_name='all', min_doms=0, max_doms=np.inf):
    # * make mask directory if it doesn't exist
    data_path = get_project_root() + get_path_from_root(data_path)

    if mask_name == 'dom_interval':
        mask_path = make_dom_interval_mask(data_path, dirs, min_doms, max_doms)
    
    elif mask_name == 'all':
        mask_path = make_all_mask(data_path, dirs)
    
    elif mask_name == 'muon_neutrino':
        mask_path = make_particle_mask(data_path, dirs, mask_name)

    return mask_path
    # # * Make a .dvc-file to track mask
    # dvc_path = get_project_root() + '/data'
    # subprocess.run(['dvc', 'add', 'masks'], cwd=dvc_path)

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
    mask_name = 'muon_neutrino'
    min_doms = 0
    max_doms = 200
    mask_dict = {'mask_name': mask_name, 'min_doms':
        min_doms, 'max_doms': max_doms}

    # * If maskdirectory doesn't exist, make it
    mask_dir = get_project_root() + '/masks'
    if not Path(mask_dir).is_dir(): 
        Path(mask_dir).mkdir()

    dirs = [file for file in Path(data_dir + '/pickles').iterdir()]
    
    print(get_time(), 'Mask calculation begun.')
    mask_path = make_mask(data_dir, dirs, **mask_dict)
    print(get_time(), 'Mask created', mask_path)
