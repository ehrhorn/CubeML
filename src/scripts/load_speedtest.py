from pathlib import Path
import numpy as np
import h5py as h5
import pickle
import joblib
from time import time

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

def load_batch_h5(fname, prefix, keys, indices):
    # * Find right file and get sorted indices to load
    batch_size = len(indices)
    n_seq_features = len(keys)
    
    # * Extract relevant data from h5-file
    seq_features = {}
    
    with h5.File(fname, 'r', swmr=True) as f:
        
        # * If key does not exist, it means the key hasn't been transformed - it is therefore located at raw/key
        for key in  keys:
            path =  prefix + '/' + key
            seq_features[key] = f[path][indices]
    
    # * Prepare batch for collate_fn
    batch = []
    for i_batch in range(batch_size):
        
        seq_array = np.empty(( n_seq_features, seq_features[ keys[0]][i_batch].shape[0]))
        for i_seq, key in enumerate(seq_features):
            seq_array[i_seq, :] = seq_features[key][i_batch]
        
        batch.append(seq_array)
    
    return batch

def make_pickle_events(batch, keys, pickle_dir):
    
    # * Save each batch-element as an individual event in a .pickle
    for i_event, event in enumerate(batch):
        
        # * Make an event-dictionary
        pickle_event = {}
        for key, feat in zip(keys, event):

            pickle_event[key] = feat
        
        # * Save as a pickle
        path = pickle_dir+'/'+str(i_event)+'.pickle'
        pickle.dump(pickle_event, open(path, 'wb'))

def load_batch_pickle(indices, keys, pickle_dir):

    # * Find right file and get sorted indices to load
    batch_size = len(indices)
    n_seq_features = len(keys)
    
    # * Extract relevant data from h5-file
    batch = []
    for index in indices:
        path = pickle_dir + '/' + str(index) +'.pickle'
        event = pickle.load( open( path, "rb" ) )

        seq_features = {}
        for key in keys:
            seq_features[key] = event[key]
        
        seq_array = np.empty(( n_seq_features, seq_features[ keys[0]].shape[0]))
        for i_seq, key in enumerate(seq_features):
            seq_array[i_seq, :] = seq_features[key]
        
        batch.append(seq_array)
    
    return batch

def load_transform_batch_pickle(indices, keys, pickle_dir, transformer):

    # * Find right file and get sorted indices to load
    batch_size = len(indices)
    n_seq_features = len(keys)
    
    # * Extract relevant data from h5-file
    batch = []
    for index in indices:
        path = pickle_dir + '/' + str(index) +'.pickle'
        event = pickle.load( open( path, "rb" ) )

        seq_features = {}
        for key in keys:
            try:
                seq_features[key] = transformer[key].transform(event[key].reshape(-1, 1)).flatten()
            except KeyError:
                seq_features[key] = event[key]
        seq_array = np.empty(( n_seq_features, seq_features[ keys[0]].shape[0]))
        for i_seq, key in enumerate(seq_features):
            seq_array[i_seq, :] = seq_features[key]
        
        batch.append(seq_array)
    
    return batch

if __name__ == '__main__':
    """Script for profiling batching via hdf5 or via single event pickles.

    To profile it: Run 
    > python -m cProfile -o speedtest.profile load_speedtest.py
    After the profiling, run
    > export DISPLAY=localhost:0.0
    > snakeviz speedtest.profile

    Result: Pickling is atleast times faster, but pickling + transform is 7 times slower than h5.
    """    

    # * Setup - where to load data, which keys in hdf5, how big batches
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2'
    keys = ['dom_charge', 'dom_x', 'dom_y', 'dom_z', 'dom_time', 'dom_charge_significance', 'dom_frac_of_n_doms', 'dom_d_to_prev', 'dom_v_from_prev', 'dom_d_minkowski_to_prev', 'dom_d_closest', 'dom_d_minkowski_closest', 'dom_d_vertex', 'dom_d_minkowski_vertex', 'dom_charge_over_vertex']
    prefix = 'raw'
    indices = np.arange(64)
    n_experiments = 100
    transformer_path = data_dir + '/transformers/' + '140000_transform1.pickle'
    transformer = joblib.load(open(transformer_path, "rb"))
    
    # * Load a batch such that the pickle-directory can be made
    h5_file = [file for file in Path(data_dir).iterdir()][0]
    batch_h5 = load_batch_h5(h5_file, prefix, keys, indices)

    # * Make a subdir with picklefiles. Must contain the same as the hdf5-file
    pickle_directory = get_project_root() + '/src/scripts/pickle_files'
    if not Path(pickle_directory).is_dir():
        Path(pickle_directory).mkdir()
    make_pickle_events(batch_h5, keys, pickle_directory)

    start = time()
    # * Load a batch N_experiments times in both ways.
    for i_exp in range(n_experiments):
        batch_h5 = load_batch_h5(h5_file, prefix, keys, indices)
    print('h5 time: %.1f'%(time()-start))

    start = time()
    # * Load a batch N_experiments times in both ways.
    for i_exp in range(n_experiments):
        batch_pickle = load_batch_pickle(indices, keys, pickle_directory)
    print('pickle time: %.1f'%(time()-start))

    start = time()
    # * Load a batch N_experiments times AND ADD TRANSFORM.
    for i_exp in range(n_experiments):
        batch_pickle = load_transform_batch_pickle(indices, keys, pickle_directory, transformer)
    print('pickle+transform time: %.1f'%(time()-start))
    
