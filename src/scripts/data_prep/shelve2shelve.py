from pathlib import Path
import numpy as np
import shelve
import pickle
from time import time
import random
from multiprocessing import cpu_count, Pool

from src.modules.helper_functions import get_dataset_size, get_particle_code, get_n_events_in_i3files, confirm_particle_type, get_particle_code_from_i3name, get_project_root, get_time

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
    print(get_time(), 'Pickling %s'%(Path(fname).name))
    
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
            new_name = data_dir+'/'+dir_name+'/'+str(new_name)+'.pickle'
            pickle.dump(event, open(new_name, 'wb'))

def write_n_events_shelve(data_dir):
    with shelve.open(data_dir, writeback=True) as f:
        n_keys = len([key for key in f])
        for i_key, key in enumerate(f):
            n_events = len(f[key]['dom_charge'])
            f[key]['n_events'] = n_events
            print(get_time(), 'progress %d/%d'%(i_key+1, n_keys))


def make_transformers(data_dir, indices):

    transformer_path = data_dir + '/transformers/' + particle_code + '_' + prefix +'.pickle'

    files = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5' and confirm_particle_type(particle_code, file)])
    random.shuffle(files)    

    keys = pp.get_feature_keys()
    dicts = pp.get_feature_plot_dicts()
    clip_dicts = pp.get_feature_clip_dicts()
    transformer_dict = pp.get_feature_transformers()

    n_wanted_sample = 10e6
    n_wanted_histogram = 1e6
    dicts = [dicts[key] for key in keys]
    clip_dicts = [clip_dicts[key] for key in keys]
    files_list = [files]*len(keys)
    n_wanted_sample = [n_wanted_sample for key in keys]
    n_wanted_histogram = [n_wanted_histogram for key in keys]
    particle_code = [particle_code for key in keys]
    transformers = [transformer_dict[key] for key in keys]

    packages = [entry for entry in zip(keys, dicts, clip_dicts, files_list, n_wanted_sample, n_wanted_histogram, particle_code, transformers)]

    # * Use multiprocessing for parallelizing the job.
    available_cores = cpu_count()
    with Pool(available_cores) as p:
        transformers = p.map(pp.fit_feature_transformers, packages)

    # * Update or create a transformer-pickle
    if Path(transformer_path).is_file():
        transformers_combined= joblib.load(open(transformer_path, "rb"))
    else:
        transformers_combined = {}
    
    # * Combine transformers    
    for entry in transformers:
        transformers_combined.update(entry)
    
    # * Save it again
    joblib.dump(transformers_combined, open(transformer_path, 'wb'))
    print('Updated transformers saved at:')
    print(transformer_path)


if __name__ == '__main__':
    # * Setup - where to load data, how many events
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2/shelve/oscnext-genie-level5-v01-01-pass2'
    print('whatup')
    with shelve.open(data_dir, writeback=True) as f:
        i3_files = sorted([str(file) for file in f])
    
    particles = ['electron_neutrino', 'muon_neutrino', 'tau_neutrino']
    particle_codes = [get_particle_code(particle) for particle in particles]
    
    # * Get filepaths and retrieve the particle code for each file
    particle_codes = [get_particle_code_from_i3name(file, particle_codes) for file in i3_files]
    n_events_file = get_n_events_in_i3files(data_dir)
    print(n_events_file)
    
    # * Create the new names for the events in each file
    from_to = np.cumsum(n_events_file)
    from_to = np.append([0], from_to)
    n_events = from_to[-1]

    indices_shuffled = np.arange(n_events)
    seed = 2912
    random.seed(seed)
    random.shuffle(indices_shuffled)

    new_names = []
    for from_, to_ in zip(from_to[:-1], from_to[1:]):
        name_indices = np.arange(from_, to_)
        new_names.append(indices_shuffled[name_indices])
    
    # * Match new names to their respective places in I3-files
    # * Write new names and save.
    print(len(new_names), len(i3_files))
    names_dict = dict(zip(new_names, i3_files))
    for key, items in names_dict.items():
        print(items)
        print(key)
        break

    # * Make transformers

    # * Make new DB
    # # * Make new directories
    # # * Save it in subdirs - put n_per_dir in each directory

    # if not Path(data_dir+'/pickles').exists():
    #     Path(data_dir+'/pickles').mkdir()

    # dir_names = np.arange(1 + n_events//n_per_dir)
    # for dir_name in dir_names:
    #     if not Path(data_dir+'/pickles/'+str(dir_name)).exists(): 
    #         Path(data_dir+'/pickles/'+str(dir_name)).mkdir()

    # # * Zip and multiprocess
    # new_data_dirs = [data_dir+'/pickles/']*len(new_names)
    # n_per_dir_list = [n_per_dir]*len(new_names)

    # packed = [entry for entry in zip(h5_files, new_names, new_data_dirs, particle_codes, n_per_dir_list)]
    
    # available_cores = cpu_count()
    # with Pool(available_cores) as p:
    #     p.map(pickle_events, packed)
