import src.modules.helper_functions as hf
import h5py as h5
import numpy as np
from pathlib import Path

def prep_dict(key, d, n_events, datatype='sequence'):
    """Small helper function for feature_engineer. Checks if a key is in the given dictionary and adds it if it is not.
    
    Arguments:
        key {str} -- variable name.
        d {dict} -- Dictionary containing the engineered features.
        n_events {int} -- total number of events.
    
    Keyword Arguments:
        datatype {str} -- Whether the engineered feature is a sequence or a scalar (default: {'sequence'})
    
    Returns:
        dict -- updated dictionary
    """    
    if key not in d:
        if datatype == 'sequence':
            d[key] = [[]]*n_events
        elif datatype == 'scalar':
            d[key] = np.array([0.0]*n_events)
    return d

def get_tot_charge(dataset, d, i_event, n_events):
    """Calculates the total charge of an event and adds it to a dictionary.

    The variablename assigned is: tot_charge
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """    
    charge = 'raw/dom_charge'
    key = 'tot_charge'
    d = prep_dict(key, d, n_events, datatype='scalar')
    
    d[key][i_event] = np.sum(dataset[charge][i_event])
    return d

def get_charge_significance(dataset, d, i_event, n_events):
    """Calculates the charge significance per DOM in an event and adds it to a dictionary.

    The dom significance is given by: n_DOMS*DOM_charge / total_charge. The variablename assigned is: dom_charge_significance
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    charge = 'raw/dom_charge'
    key = 'dom_charge_significance'
    d = prep_dict(key, d, n_events)

    if not 'tot_charge' in d:
        d = get_tot_charge(dataset, d, i_event, n_events)
    
    event = dataset[charge][i_event]
    n_doms = len(event)
    d[key][i_event] = n_doms*event/d['tot_charge'][i_event]
    
    return d

def get_frac_of_n_doms(dataset, d, i_event, n_events):
    """Calculates the fractional amount of DOMs seen including the current DOM in an event and adds it to a dictionary.

    The fractional amount is given by: i_DOM / n_DOMs. The variablename given is: dom_frac_of_n_doms
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    charge = 'raw/dom_charge'
    key = 'dom_frac_of_n_doms'
    d = prep_dict(key, d, n_events)

    event = dataset[charge][i_event]
    n_doms = event.shape[0]
    d[key][i_event] = np.arange(1, n_doms+1)/n_doms

    return d

def get_d_to_prev(dataset, d, i_event, n_events):
    """Calculates the the Euclidian distance to the previous DOM in an event and adds it to a dictionary. The first DOMs distance is given the value 0.0.

    The variablename given is: dom_d_to_prev
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    x, y, z = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z'
    key = 'dom_d_to_prev'
    d = prep_dict(key, d, n_events)

    x_diff = dataset[x][i_event][1:] - dataset[x][i_event][:-1]
    y_diff = dataset[y][i_event][1:] - dataset[y][i_event][:-1]
    z_diff = dataset[z][i_event][1:] - dataset[z][i_event][:-1]
    dists = np.sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff)
    dists = np.append([0.0], dists)
    d[key][i_event] = dists
    return d

def get_v_from_prev(dataset, d, i_event, n_events):
    """Calculates the signal speed from the previous DOM to the current in an event and adds it to a dictionary. The first DOMs value is set to 0.0.

    It is calculated as: Dist(DOM_i, DOM_{i-1})/(DOM_i(t) - DOM_{i-1}). The variablename given is: dom_v_from_prev
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    t = 'raw/dom_time'
    key = 'dom_v_from_prev'
    d = prep_dict(key, d, n_events)

    if 'dom_d_to_prev' not in d:
        d = get_d_to_prev(dataset, d, i_event, n_events)

    t_diff = dataset[t][i_event][1:] - dataset[t][i_event][:-1]

    # * Time has discrete values due to the clock on the electronics + the pulse extraction algorithm bins the pulses in time --> more discreteness.
    t_diff = np.where(t_diff==0, 1.0, t_diff)
    t_diff = t_diff*1e-9
    t_diff = np.append([np.inf], t_diff)
    
    d[key][i_event] = d['dom_d_to_prev'][i_event]/t_diff
    return d

def get_d_minkowski_to_prev(dataset, d, i_event, n_events, n=1.309):
    """Calculates the Minkowski distance (See subsection 'Minkowski Metric' at https://en.wikipedia.org/wiki/Minkowski_space) from the previous DOM to the current in an event and adds it to a dictionary. The first DOMs value is set to 0.0.

    It is calculated as: Dist_Minkowski(DOM_i, DOM_{i-1}) with metric (+---) and with v_{light} = c/n, where n is the index of refraction for ice. The variablename given is: dom_d_minkowski_to_prev
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    
    x, y, z, t = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z', 'raw/dom_time'
    key = 'dom_d_minkowski_to_prev'
    d = prep_dict(key, d, n_events)

    x_diff_sqr = (dataset[x][i_event][1:] - dataset[x][i_event][:-1])
    x_diff_sqr = x_diff_sqr*x_diff_sqr
    
    y_diff_sqr = (dataset[y][i_event][1:] - dataset[y][i_event][:-1])
    y_diff_sqr = y_diff_sqr*y_diff_sqr
    
    z_diff_sqr = (dataset[z][i_event][1:] - dataset[z][i_event][:-1])
    z_diff_sqr = z_diff_sqr*z_diff_sqr

    t_diff_sqr = (dataset[t][i_event][1:] - dataset[t][i_event][:-1])
    t_diff_sqr = t_diff_sqr*t_diff_sqr

    c_ns = 3e8 * 1e-9 / n
    spacetime_interval = ((c_ns**2) * t_diff_sqr) - x_diff_sqr - y_diff_sqr - z_diff_sqr
    abs_interval_root = np.sqrt(np.abs(spacetime_interval))

    spacetime_interval_root = np.where(spacetime_interval>0, abs_interval_root, -abs_interval_root)
    spacetime_interval_root = np.append([0.0], spacetime_interval_root)

    d[key][i_event] = spacetime_interval_root
        
    return d

def get_d_closest(dataset, d, i_event, n_events):
    r"""Calculates the Euclidian distance to the closest activated DOM in an event and adds it to a dictionary.

    It is calculated as: min(Dist(DOM_i, DOM_{j})), i \neq j.. The variablename given is: dom_d_closest
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    x, y, z, t = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z', 'raw/dom_time'
    key = 'dom_d_closest'
    d = prep_dict(key, d, n_events)

    n_doms = len(dataset[x][i_event])
    min_dists = np.zeros(n_doms)
    x_diff_sqr2 = tile_diff_sqr(dataset[x][i_event])
    y_diff_sqr2 = tile_diff_sqr(dataset[y][i_event])
    z_diff_sqr2 = tile_diff_sqr(dataset[z][i_event])
    tot_dist = x_diff_sqr2 + y_diff_sqr2 + z_diff_sqr2
    np.fill_diagonal(tot_dist, np.inf)
    min_dists_all = np.sqrt(np.min(tot_dist, axis=1))
    d[key][i_event] = min_dists_all
        
    return d

def get_d_minkowski_closest(dataset, d, i_event, n_events, n=1.309):
    r"""Calculates the Minkowski distance (See subsection 'Minkowski Metric' at https://en.wikipedia.org/wiki/Minkowski_space) closest to 0 for all DOMs in an event and adds it to a dictionary. 

    It is calculated as: min(Dist_Minkowski(DOM_i, DOM_j)), i \neq j with metric (+---) and with v_{light} = c/n, where n is the index of refraction for ice. The variablename given is: dom_d_minkowski_closest
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    # * See subsection https://en.wikipedia.org/wiki/Minkowski_space --> Minkowski Metric
    x, y, z, t = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z', 'raw/dom_time'
    key = 'dom_d_minkowski_closest'
    d = prep_dict(key, d, n_events)

    n_doms = len(dataset[x][i_event])
    min_dists = np.zeros(n_doms)
    x_diff_sqr2 = tile_diff_sqr(dataset[x][i_event])
    y_diff_sqr2 = tile_diff_sqr(dataset[y][i_event])
    z_diff_sqr2 = tile_diff_sqr(dataset[z][i_event])
    t_diff_sqr2 = tile_diff_sqr(dataset[t][i_event])

    c_ns = 3e8 * 1e-9 / n
    tot_dist = c_ns*c_ns*t_diff_sqr2 - x_diff_sqr2 - y_diff_sqr2 - z_diff_sqr2
    tot_dist_abs = np.abs(tot_dist)
    np.fill_diagonal(tot_dist_abs, np.inf)
    indices = np.argmin(tot_dist_abs, axis=1)
    # * Row i is the i'th DOMs distance to the other doms. Therefore, the minimum distance for DOM i is located at (range[i], indices[i])
    closest_squared = tot_dist_abs[list(range(n_doms)), indices]
    closest = tot_dist[list(range(n_doms)), indices]
    closest = np.where(closest>0, np.sqrt(closest_squared), -np.sqrt(closest_squared))
    d[key][i_event] = closest
        
    return d

def get_d_vertex(dataset, d, i_event, n_events):
    r"""Calculates the Euclidian distance for each DOM to a predicted interaction vertex and adds it to a dictionary. For now, the prediction of retro_crs_prefit is used.

    The variablename given is: dom_d_vertex
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """

    # * Use crs_prefits prediction - they are more similar to ours
    x, y, z = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z'
    x_pred, y_pred, z_pred = 'raw/retro_crs_prefit_x', 'raw/retro_crs_prefit_y', 'raw/retro_crs_prefit_z'
    key = 'dom_d_vertex'
    d = prep_dict(key, d, n_events)

    x_diff_sqr = sqr_dist(dataset[x][i_event], dataset[x_pred][i_event])
    y_diff_sqr = sqr_dist(dataset[y][i_event], dataset[y_pred][i_event])
    z_diff_sqr = sqr_dist(dataset[z][i_event], dataset[z_pred][i_event])
    tot_dist = np.sqrt(x_diff_sqr+y_diff_sqr+z_diff_sqr)
    d[key][i_event] = tot_dist
    
    return d

def get_d_minkowski_vertex(dataset, d, i_event, n_events, n=1.309):
    r"""Calculates the Minkowski distance for each DOM to a predicted interaction vertex and adds it to a dictionary. For now, the prediction of retro_crs_prefit is used.

    The Minkowski metric with signature (+---) is used. The variablename given is: dom_d_minkowski_vertex
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    # * Use crs_prefits prediction - they are more similar to ours
    x, y, z, t = 'raw/dom_x', 'raw/dom_y', 'raw/dom_z', 'raw/dom_time'
    x_pred, y_pred, z_pred, t_pred = 'raw/retro_crs_prefit_x', 'raw/retro_crs_prefit_y', 'raw/retro_crs_prefit_z', 'raw/retro_crs_prefit_time'
    key = 'dom_d_minkowski_vertex'
    d = prep_dict(key, d, n_events)
    
    x_diff_sqr = sqr_dist(dataset[x][i_event], dataset[x_pred][i_event])
    y_diff_sqr = sqr_dist(dataset[y][i_event], dataset[y_pred][i_event])
    z_diff_sqr = sqr_dist(dataset[z][i_event], dataset[z_pred][i_event])
    t_diff_sqr = sqr_dist(dataset[t][i_event], dataset[t_pred][i_event])

    c_ns = 3e8 * 1e-9 / n

    tot_sqr = c_ns*c_ns*t_diff_sqr - x_diff_sqr - y_diff_sqr - z_diff_sqr
    tot_abs = np.sqrt(abs(tot_sqr))
    tot_mink_dist = np.where(tot_sqr>0, tot_abs, -tot_abs)
    d[key][i_event] = tot_mink_dist
    
    return d

def get_charge_over_d_vertex(dataset, d, i_event, n_events):
    r"""For each DOM, the DOM charge divided by the Euclidian distance and squared distance to a predicted interaction vertex is calculated and adds it to a dictionary. For now, the prediction of retro_crs_prefit is used.

    The variablenames given are: dom_charge_over_vertex, dom_charge_over_vertex_sqr
    
    Arguments:
        dataset {h5-file} -- An open h5-file.
        d {dict} -- Dictionary containing variable names and their values.
        i_event {int} -- event number in a file.
        n_events {int} -- total number of events in file.
    
    Returns:
        dict -- Updated dictionary
    """
    
    Q = 'raw/dom_charge'
    key = 'dom_charge_over_vertex'
    key2 = 'dom_charge_over_vertex_sqr'
    d = prep_dict(key, d, n_events)
    d = prep_dict(key2, d, n_events)

    d[key][i_event] = dataset[Q][i_event]/d['dom_d_vertex'][i_event]
    d[key2][i_event] = dataset[Q][i_event]/(d['dom_d_vertex'][i_event]*d['dom_d_vertex'][i_event])

    return d

def sqr_dist(data1, data2):
    """Helper function for feature_engineer. Subtracts to matrices and squares each entry.
    
    Arguments:
        data1 {array} -- array 1
        data2 {array} -- array 2
    
    Returns:
        array -- squared difference.
    """    
    diff = data1-data2
    return diff*diff

def tile_diff_sqr(vector):
    """Helperfunction for feature_engineer. Tiles a vector n_doms times such that an array is converted from (length,) to (length, n_doms), then computes the square of the difference each vector entry with all other vector entries, i.e.

    array[i, j] = (vector[i] - vector[j])**2
    
    Arguments:
        vector {array} -- Vector to turn into a matrix.
    
    Returns:
        array -- array of squared differences.
    """    
    n_doms = len(vector)
    tiled = np.tile(vector, (n_doms, 1))
    diff_sqr = sqr_dist(vector, tiled.transpose()) 
    return diff_sqr

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

def feature_engineer(packed):
    """Calculates desired features for a h5-datafile and appends the new datasets to the file. Multiprocessing-friendly.
    
    Arguments:
        
        packed {tuple} -- a tuple containing:
            i_file {int} -- Filenumber i of N_FILES - to track progress
            file {str} -- absolute path to h5-datafile.
            N_FILES {int} -- Total number of files to process (via multi- or singleprocesing).
    """    
    
    # * Unpack. One input is expected to be compatible with multiprocessing
    i_file, file, N_FILES = packed
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
    