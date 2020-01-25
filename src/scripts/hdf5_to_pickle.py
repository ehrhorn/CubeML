from pathlib import Path
import numpy as np
import h5py as h5
import pickle
import joblib
from time import time
import random
from multiprocessing import cpu_count, Pool

from src.modules.helper_functions import get_dataset_size, get_particle_code, get_n_events_in_h5, confirm_particle_type, get_particle_code_from_h5, get_project_root

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
    fname, new_names, data_dir, particle_code, n_per_dir = pack

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
            
            # * Assign metavalues - where is event from, what kind of particle?
            event['meta'] = {}
            event['meta']['file'] = Path(fname).name
            event['meta']['index'] = i_event
            event['meta']['particle_code'] = particle_code

            # * Save it in subdirs - put n_per_dir in each directory
            dir_name = str(new_name//n_per_dir)
            if not Path(data_dir+'/'+dir_name).is_dir(): 
                Path(data_dir+'/'+dir_name).mkdir()
            new_name = data_dir+'/'+dir_name+'/'+str(new_name)+'.pickle'
            pickle.dump(event, open(new_name, 'wb'))

if __name__ == '__main__':
    # * Setup - where to load data, how many events
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2'
    particles = ['electron_neutrino', 'muon_neutrino', 'tau_neutrino']
    n_per_dir = 10000
    particle_codes = [get_particle_code(particle) for particle in particles]
    
    seed = 2912
    n_files, ave_pr_file, std = get_dataset_size(data_dir)
    n_events = int(n_files*ave_pr_file+0.1)

    # * Each event is given a new name: A number. All events across all files are shuffled
    indices_shuffled = np.arange(n_events)
    random.seed(seed)
    random.shuffle(indices_shuffled)
   
    # * Get filepaths and retrieve the particle code for each file
    h5_files = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5'])
    particle_codes = [get_particle_code_from_h5(file, particle_codes) for file in h5_files]
    n_events_file = [get_n_events_in_h5(path) for path in h5_files]
    
    # * Create the new names for the events in each file
    from_to = np.cumsum(n_events_file)
    from_to = np.append([0], from_to)
    new_names = []
    for from_, to_ in zip(from_to[:-1], from_to[1:]):
        name_indices = np.arange(from_, to_)
        new_names.append(indices_shuffled[name_indices])

    # * Zip and multiprocess
    new_data_dirs = [data_dir]*len(new_names)
    n_per_dir_list = [n_per_dir]*len(new_names)
    packed = [entry for entry in zip(h5_files, new_names, new_data_dirs, particle_codes, n_per_dir_list)]
    
    available_cores = cpu_count()
    with Pool(available_cores+2) as p:
        p.map(pickle_events, packed)