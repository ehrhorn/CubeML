import src.modules.helper_functions as hf
import src.modules.reporting as rpt
import h5py as h5
import numpy as np
from pathlib import Path
import random
from time import time
from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler

def calc_features(functions, file):
    """Calculates features induced by functions for all events in file.
    
    Arguments:
        functions {list} -- list of feature-calculating functions.
        file {str} -- Absolute path to a h5-datafile.
    
    Returns:
        dict -- Dictionary with features for each event.
    """    

    # * open file and calc important stuff
    d = {}
    with h5.File(file, 'r') as f:
        n_events = f['meta/events'][()]

        # * Loop over events
        for i_event in range(n_events):

            # * Calculate each feature for this event
            for func in functions:
                d = func(f, d, i_event, n_events)
        
    return d

def get_feature_keys():
    keys = [
        'dom_atwd',
        'dom_pulse_width',
        'dom_charge',
        'true_primary_time',
        'dom_n_hit_multiple_doms',
        'dom_time', 
        'dom_x', 
        'dom_y',
        'dom_z', 
        'dom_timelength_fwhm',
        'linefit_point_on_line_x',
        'linefit_point_on_line_y',
        'linefit_point_on_line_z',
        'toi_evalratio',
        'toi_point_on_line_x',
        'toi_point_on_line_y',
        'toi_point_on_line_z',
        'true_primary_energy',
        'true_primary_position_x',
        'true_primary_position_y',
        'true_primary_position_z',
        'tot_charge',
        'dom_charge_significance',
        'dom_frac_of_n_doms',
        'dom_d_to_prev',
        'dom_v_from_prev',
        'dom_d_minkowski_to_prev',
        'dom_d_closest',
        'dom_d_minkowski_closest',
        'dom_d_vertex',
        'dom_d_minkowski_vertex',
        'dom_charge_over_vertex',
        'dom_charge_over_vertex_sqr'
    ]
    return keys

def get_feature_plot_dicts():
    d  = {
        'tot_charge': {},
        'dom_charge_significance': {
            'log': [True]},
        'dom_frac_of_n_doms': {},
        'dom_d_to_prev': {},
        'dom_v_from_prev': {
            'log': [True]},
        'dom_d_minkowski_to_prev': {},
        'dom_d_closest': {},
        'dom_d_minkowski_closest': {},
        'dom_d_vertex': {},
        'dom_d_minkowski_vertex': {},
        'dom_charge_over_vertex': {
            'log': [True]},
        'dom_charge_over_vertex_sqr': {
            'log': [True]}
    }
    return d

def get_feature_transformers():
    d  = {
        'dom_atwd': None,
        'dom_pulse_width': RobustScaler(),
        'dom_charge': QuantileTransformer(output_distribution='normal'),
        'true_primary_time': QuantileTransformer(output_distribution='normal'),
        'dom_n_hit_multiple_doms': RobustScaler(),
        'dom_time': RobustScaler(),
        'dom_x': RobustScaler(),
        'dom_y': RobustScaler(),
        'dom_z': RobustScaler(),
        'dom_timelength_fwhm': RobustScaler(),
        'linefit_point_on_line_x': RobustScaler(),
        'linefit_point_on_line_y': RobustScaler(),
        'linefit_point_on_line_z': RobustScaler(),
        'toi_evalratio': RobustScaler(),
        'toi_point_on_line_x': RobustScaler(),
        'toi_point_on_line_y': RobustScaler(),
        'toi_point_on_line_z': RobustScaler(),
        'true_primary_energy': RobustScaler(),
        'true_primary_position_x': RobustScaler(),
        'true_primary_position_y': RobustScaler(),
        'true_primary_position_z': RobustScaler(),
        'tot_charge': StandardScaler(),
        'dom_charge_significance': StandardScaler(),
        'dom_frac_of_n_doms': RobustScaler(),
        'dom_d_to_prev': RobustScaler(),
        'dom_v_from_prev': QuantileTransformer(output_distribution='normal'),
        'dom_d_minkowski_to_prev': RobustScaler(),
        'dom_d_closest': StandardScaler(),
        'dom_d_minkowski_closest': StandardScaler(),
        'dom_d_vertex': RobustScaler(),
        'dom_d_minkowski_vertex': RobustScaler(),
        'dom_charge_over_vertex': QuantileTransformer(output_distribution='normal'),
        'dom_charge_over_vertex_sqr': QuantileTransformer(output_distribution='normal')
    }
    return d

def get_feature_clip_dicts():
    d  = {
        'tot_charge': None,
        'dom_charge_significance': {'min': None, 'max': 15},
        'dom_frac_of_n_doms': None,
        'dom_d_to_prev': None,
        'dom_v_from_prev': {'min': None, 'max': 0.4e12},
        'dom_d_minkowski_to_prev': None,
        'dom_d_closest': None,
        'dom_d_minkowski_closest': None,
        'dom_d_vertex': None,
        'dom_d_minkowski_vertex': None,
        'dom_charge_over_vertex': {'min': None, 'max': 0.4},
        'dom_charge_over_vertex_sqr': {'min': None, 'max': 0.04}

    }
    return d

def get_wanted_feature_engineers():
    """Helperfunction for feature_engineer. Predefined list of functions, which calculate the desired features.
    
    Returns:
        list -- list of functions.
    """    
    functions = [
        get_tot_charge,
        get_charge_significance,
        get_frac_of_n_doms,
        get_d_to_prev,
        get_v_from_prev,
        get_d_minkowski_to_prev,
        get_d_closest,
        get_d_minkowski_closest,
        get_d_vertex,
        get_d_minkowski_vertex,
        get_charge_over_d_vertex
    ]

    return functions

def feature_engineer(pack):
    """Calculates desired features for a h5-datafile and appends the new datasets to the file. Multiprocessing-friendly.
    
    Arguments:
        
        packed {tuple} -- a tuple containing:
            i_file {int} -- Filenumber i of N_FILES - to track progress
            file {str} -- absolute path to h5-datafile.
            N_FILES {int} -- Total number of files to process (via multi- or singleprocesing).
    """    
    
    # * Unpack. One input is expected to be compatible with multiprocessing
    i_file, file, N_FILES = pack
    name = Path(file).name
    
    # * Print progress for our sanity..
    print(hf.get_time(), 'Processing %s (file %d of %d)'%(name, i_file+1, N_FILES))
    
    # * Retrieve wanted engineers - they have to be predefined in get_wanted_feature_engineers (for now)
    functions = get_wanted_feature_engineers()

    # * Now calculate the features on a per event basis.
    d = calc_features(functions, file)

    # * Append our calculations to the datafile
    with h5.File(file, 'a') as f:
        # * Make a 'raw/'-group if it doesnt exist
        if 'raw' not in f:
            raw = f.create_group("raw")

        # * Now make the datasets
        for key, data in d.items():
            dataset_path = 'raw/'+key
            # * Check if it is a DOM-variable or global event-variable
            if data[0].shape:
                # * If dataset already exists, delete it first
                if dataset_path in f:
                    del f[dataset_path]
                f.create_dataset(dataset_path, data=data, dtype=h5.special_dtype(vlen=data[0][0].dtype))

            else:
                # * If dataset already exists, delete it first
                if dataset_path in f:
                    del f[dataset_path]
                f.create_dataset(dataset_path, data=data, dtype=data[0].dtype)

def fit_feature_transformers(pack):
    # * Unpack
    key, d, clip_dict, file_list, \
    n_wanted_sample, n_wanted_histogram, particle_code, transformer = pack

    # * Read some data
    all_data = []
    for file in file_list:
        # * once enough data has been read, break out
        if len(all_data)>n_wanted_sample:
            break
        data = hf.read_h5_dataset(file, key, prefix='raw/')
        if data[0].shape:
            for entry in data:
                all_data.extend(entry)
        else:
            all_data.extend(data)
    
    # * Data read. Now draw a random sample
    indices = np.array(range(len(all_data)))
    random.shuffle(indices)
    random_subsample = sorted(indices[:min(len(indices), int(n_wanted_histogram))])

    # * Draw histogram and save it.
    plot_data = np.array(sorted(np.array(all_data)[random_subsample]))
    plot_data_unclipped = np.array(sorted(np.array(all_data)[random_subsample]))
    if clip_dict:
        minimum = clip_dict['min']
        maximum = clip_dict['max']
        plot_data = np.clip(plot_data, minimum, maximum)
    d['data'] = [plot_data]
    d['title'] = key + '- Entries = %.1e'%(plot_data_unclipped.shape[0])

    path = hf.get_project_root() + '/reports/plots/features/'
    d['savefig'] = path + particle_code + '_' + key + '.png'
    fig = rpt.make_plot(d)

    # * Fit a transformer/scaler
    transformer.fit(plot_data_unclipped.reshape(-1, 1))

    # * Transform plot data
    plot_data_transformed = transformer.transform(plot_data_unclipped.reshape(-1, 1))    

    # * Plot and save
    d_transformed = {'data': [plot_data_transformed]}
    d_transformed['title'] = key+' transformed - Entries = %.1e'%(plot_data_unclipped.shape[0])
    d_transformed['savefig'] = path + particle_code + '_transformed_' + key + '.png'
    fig = rpt.make_plot(d_transformed)

    d_transformer = {key: transformer}

    return d_transformer

def transform_features(pack):
    file, transformers, keys, prefix = pack
    start = time()
    name = Path(file).name

    with h5.File(file, 'a') as f:
        n_events = f['meta/events'][()]
        
        # * Loop over keys and do all transformations for the whole file.
        # * scikits transformers expect 2D-arrays, hence we reshape into 2D-array and flatten again.
        d = {}
        for key in keys:
           
           # * For each key, check if already transformed - if yes, don't do it again
            if f['raw/'+key]:# and prefix+'/'+key not in f:
                transformer = transformers[key]
                
                # * Prepare an empty dataset
                if f['raw/'+key][0].shape:
                    d[key] = [[]]*n_events

                    # * We must loop due to the sequential nature of DOM sequences
                    for i_event, event in enumerate(f['raw/'+key]):
                        d[key][i_event] = transformer.transform(event.reshape(-1, 1)).flatten()
                
                else:   
                    # * For non-sequential data, we can transform entire set in one go
                    d[key] = transformer.transform(f['raw/'+key][:].reshape(-1, 1)).flatten()
        
        # * Now save
        for key, data in d.items():
            dataset_path = prefix + '/' + key

            # * Check if it is a DOM-variable or global event-variable
            if data[0].shape:
                
                # * If dataset already exists, delete it first
                if dataset_path in f:
                    del f[dataset_path]
                
                f.create_dataset(dataset_path, data=data, dtype=h5.special_dtype(vlen=data[0][0].dtype))

            else:
                
                # * If dataset already exists, delete it first
                if dataset_path in f:
                    del f[dataset_path]
                
                f.create_dataset(dataset_path, data=data, dtype=data[0].dtype)
    
    # * Print progress for sanity...
    finish_time = time()-start
    print(hf.get_time(),'Finished %s in %.0f seconds'%(name, finish_time))
    print('Speed: %.0f Events per second\n'%(n_events/finish_time))