from pathlib import Path
import numpy as np
import h5py as h5
import pickle
import joblib
from time import time
import random
from multiprocessing import cpu_count, Pool

from src.modules.helper_functions import get_dataset_size, get_particle_code, get_n_events_in_h5, confirm_particle_type

def get_project_root():
    """Finds absolute path to project root - useful for running code on different machines.
    
    Returns:
        str -- path to project root
    """    
    
    # * Find project root
    current_dir_splitted = str(Path.cwd()).split('/')
    i = 0
    while current_dir_splitted[i] != 'CubeML':
        i += 1
    return '/'.join(current_dir_splitted[:i+1]) 

def empty_pickle_event():
    pickle_event = {}
    groups = wanted_groups()
    for group in groups:
        pickle_event[group] = {}
    return pickle_event

def wanted_groups():
    return ['raw', 'transform1', 'masks']

def pickle_events(pack):
    # * Unpack - assumes multiprocessing
    fname, new_names, data_dir = pack

    # * Loop over events in file - each event is turned into a .pickle
    with h5.File(fname, 'r') as f:
        n_events = f['meta/events'][()]

        for i_event, new_name in zip(range(n_events), new_names):
            event = empty_pickle_event()

            # * Fill the pickle file.
            for group in event:
                for key, data in f[group].items():

                    # * Save in numpy.float32 format - this is the format used in models anyways.
                    if group != 'masks':
                        event[group][key] = data[i_event].astype(np.float32)
                    else:
                        event[group][key] = data[i_event]
            
            # * Assign metavalues - where is event from?
            event['meta'] = {}
            event['meta']['file'] = Path(fname).name
            event['meta']['index'] = i_event

            # * Save it
            new_name = data_dir + '/' + str(new_name)+'.pickle'
            pickle.dump(event, open(new_name, 'wb'))

if __name__ == '__main__':
    # * Setup - where to load data, how many events
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2'
    particle = 'muon_neutrino'
    particle_code = get_particle_code(particle)
    new_data_dir = data_dir + '/' + particle_code
    
    seed = 2912
    n_files, ave_pr_file, std = get_dataset_size(data_dir, particle=particle)
    n_events = int(n_files*ave_pr_file+0.1)

    # * If new data directory does not exist, make it
    if not Path(new_data_dir).is_dir(): 
        Path(new_data_dir).mkdir()

    # * Each event is given a new name: A number. All events across all files are shuffled
    indices_shuffled = np.arange(n_events)
    random.seed(seed)
    random.shuffle(indices_shuffled)
   
    # * Get filepaths and retrieve the new names for the files in each file
    h5_files = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5' and confirm_particle_type(particle_code, file)])
    n_events_file = [get_n_events_in_h5(path) for path in h5_files]
    
    from_to = np.cumsum(n_events_file)
    from_to = np.append([0], from_to)
    new_names = []
    for from_, to_ in zip(from_to[:-1], from_to[1:]):
        name_indices = np.arange(from_, to_)
        new_names.append(indices_shuffled[name_indices])

    # * Zip and multiprocess
    new_data_dirs = [new_data_dir]*len(new_names)
    packed = [entry for entry in zip(h5_files, new_names, new_data_dirs)]
    
    available_cores = cpu_count()
    with Pool(available_cores) as p:
        p.map(pickle_events, packed)