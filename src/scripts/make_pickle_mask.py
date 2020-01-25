import h5py as h5
import numpy as np
from pathlib import Path
import subprocess
import pickle
from multiprocessing import cpu_count, Pool
from src.modules.helper_functions import get_project_root, get_path_from_root, get_dataset_name, get_time, flatten_list_of_lists

def make_mask(data_path, mask_name='all', min_doms=0, max_doms=np.inf):
    # * make mask directory if it doesn't exist
    data_path = get_project_root() + get_path_from_root(data_path)

    if mask_name == 'dom_interval':
        mask_path = make_dom_interval_mask(data_path, min_doms, max_doms)
    
    elif mask_name == 'all':
        mask_path = make_all_mask(data_path)

    return mask_path
    # # * Make a .dvc-file to track mask
    # dvc_path = get_project_root() + '/data'
    # subprocess.run(['dvc', 'add', 'masks'], cwd=dvc_path)

def make_dom_interval_mask(data_path, min_doms, max_doms, multiprocess=True):
    
    # * Loop over all events - print every once in a while as a sanity check
    candidates = [file for file in Path(data_path).iterdir() if file.suffix == '.pickle']
    
    # * Split the candidates into chunks for multiprocessing
    if multiprocess:
        available_cores = cpu_count()
        cands_chunked = np.array_split(candidates, available_cores)
        min_doms_list = [min_doms]*len(cands_chunked)
        max_doms_list = [max_doms]*len(cands_chunked)
        packed = zip(cands_chunked, min_doms_list, max_doms_list)
        
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

def find_dom_interval_passed_cands(pack):
    # * Unpack
    candidates, min_doms, max_doms = pack
    print_every = 1000
    # * Check each file.
    accepted = []
    for i_file, file in enumerate(candidates):
        if (i_file+1)%print_every == 0:
            print(get_time(), 'Processed %d'%(i_file+1))
        event = pickle.load( open( file, "rb" ) )
        n_doms = event['raw']['dom_charge'].shape[0]
        if min_doms <= n_doms <= max_doms:
            accepted.append(int(file.stem))
    
    return accepted

def make_all_mask(data_path):
    # * Check number of events
    n_events = len([file for file in Path(data_path).iterdir() if file.suffix == '.pickle'])
    
    # * Make mask - a list of indices
    mask = np.arange(n_events)

    # * save it
    mask_path = data_path + '/masks/all.pickle'
    pickle.dump(mask, open(mask_path, 'wb'))
    
    return mask_path

if __name__ == '__main__':
    data_dir = '/data/oscnext-genie-level5-v01-01-pass2/140000'
    mask_name = 'dom_interval'
    min_doms = 0
    max_doms = 200
    mask_dict = {'mask_name': mask_name, 'min_doms': min_doms, 'max_doms': max_doms}

    # * If maskdirectory doesn't exist, make it
    mask_dir = get_project_root() + data_dir + '/masks'
    if not Path(mask_dir).is_dir(): 
        Path(mask_dir).mkdir()

    print(get_time(), 'Mask calculation begun.')
    mask_path = make_mask(data_dir, **mask_dict)
    print(get_time(), 'Mask created', mask_path)
