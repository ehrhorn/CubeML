import shelve
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import sys
import dbm.dumb as dumbdbm

from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from src.modules.helper_functions import get_project_root, get_time, get_path_from_root
from src.modules.reporting import make_plot

def get_tot_charge(event):
    """Calculates the total charge of an event and adds it to a dictionary.

    The variablename assigned is: tot_charge
    
    """    
    charge = 'dom_charge'
    key = 'tot_charge'
    
    val = np.sum(event[charge])
    return val

def get_charge_significance(event):
    """Calculates the charge significance per DOM in an event and adds it to a dictionary.

    The dom significance is given by: n_DOMS*DOM_charge / total_charge. The variablename assigned is: dom_charge_significance

    """
    charge = 'dom_charge'
    key = 'dom_charge_significance'
    
    n_doms = len(event)
    val = n_doms*event[charge]/np.sum(event[charge])
    
    return val

def get_frac_of_n_doms(event):
    """Calculates the fractional amount of DOMs seen including the current DOM in an event and adds it to a dictionary.

    The fractional amount is given by: i_DOM / n_DOMs. The variablename given is: dom_frac_of_n_doms
    
    """
    charge = 'dom_charge'
    n_doms = event[charge].shape[0]
    val = np.arange(1, n_doms+1)/n_doms

    return val

def get_d_to_prev(event):
    """Calculates the the Euclidian distance to the previous DOM in an event and adds it to a dictionary. The first DOMs distance is given the value 0.0.

    The variablename given is: dom_d_to_prev
    
    """
    x, y, z = 'dom_x', 'dom_y', 'dom_z'

    x_diff = event[x][1:] - event[x][:-1]
    y_diff = event[y][1:] - event[y][:-1]
    z_diff = event[z][1:] - event[z][:-1]
    dists = np.sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff)
    dists = np.append([0.0], dists)
    val = dists
    return val

def get_v_from_prev(event):
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
    x, y, z, t = 'dom_x', 'dom_y', 'dom_z','dom_time'

    x_diff = event[x][1:] - event[x][:-1]
    y_diff = event[y][1:] - event[y][:-1]
    z_diff = event[z][1:] - event[z][:-1]
    dists = np.sqrt(x_diff*x_diff + y_diff*y_diff + z_diff*z_diff)
    dists = np.append([0.0], dists)

    t_diff = event[t][1:] - event[t][:-1]

    # * Time has discrete values due to the clock on the electronics + the pulse extraction algorithm bins the pulses in time --> more discreteness.
    t_diff = np.where(t_diff==0, 1.0, t_diff)
    t_diff = t_diff*1e-9
    t_diff = np.append([np.inf], t_diff)
    
    val = dists/t_diff
    
    return val

def get_d_minkowski_to_prev(event, n=1.309):
    """Calculates the Minkowski distance (See subsection 'Minkowski Metric' at https://en.wikipedia.org/wiki/Minkowski_space) from the previous DOM to the current in an event and adds it to a dictionary. The first DOMs value is set to 0.0.

    It is calculated as: Dist_Minkowski(DOM_i, DOM_{i-1}) with metric (+---) and with v_{light} = c/n, where n is the index of refraction for ice. The variablename given is: dom_d_minkowski_to_prev
    
    """
    
    x, y, z, t = 'dom_x', 'dom_y', 'dom_z', 'dom_time'

    x_diff_sqr = (event[x][1:] - event[x][:-1])
    x_diff_sqr = x_diff_sqr*x_diff_sqr
    
    y_diff_sqr = (event[y][1:] - event[y][:-1])
    y_diff_sqr = y_diff_sqr*y_diff_sqr
    
    z_diff_sqr = (event[z][1:] - event[z][:-1])
    z_diff_sqr = z_diff_sqr*z_diff_sqr

    t_diff_sqr = (event[t][1:] - event[t][:-1])
    t_diff_sqr = t_diff_sqr*t_diff_sqr

    c_ns = 3e8 * 1e-9 / n
    spacetime_interval = ((c_ns**2) * t_diff_sqr) - x_diff_sqr - y_diff_sqr - z_diff_sqr
    abs_interval_root = np.sqrt(np.abs(spacetime_interval))

    spacetime_interval_root = np.where(spacetime_interval>0, abs_interval_root, -abs_interval_root)
    spacetime_interval_root = np.append([0.0], spacetime_interval_root)

    val = spacetime_interval_root
        
    return val

def get_d_closest(event):
    r"""Calculates the Euclidian distance to the closest activated DOM in an event and adds it to a dictionary.

    It is calculated as: min(Dist(DOM_i, DOM_{j})), i \neq j.. The variablename given is: dom_d_closest
    
    """
    x, y, z, t = 'dom_x', 'dom_y', 'dom_z', 'dom_time'

    n_doms = len(event[x])
    min_dists = np.zeros(n_doms)
    x_diff_sqr2 = tile_diff_sqr(event[x])
    y_diff_sqr2 = tile_diff_sqr(event[y])
    z_diff_sqr2 = tile_diff_sqr(event[z])
    tot_dist = x_diff_sqr2 + y_diff_sqr2 + z_diff_sqr2
    np.fill_diagonal(tot_dist, np.inf)
    min_dists_all = np.sqrt(np.min(tot_dist, axis=1))
    val = min_dists_all
        
    return val

def get_d_minkowski_closest(event, n=1.309):
    r"""Calculates the Minkowski distance (See subsection 'Minkowski Metric' at https://en.wikipedia.org/wiki/Minkowski_space) closest to 0 for all DOMs in an event and adds it to a dictionary. 

    It is calculated as: min(Dist_Minkowski(DOM_i, DOM_j)), i \neq j with metric (+---) and with v_{light} = c/n, where n is the index of refraction for ice. The variablename given is: dom_d_minkowski_closest
    
    """
    # * See subsection https://en.wikipedia.org/wiki/Minkowski_space --> Minkowski Metric
    x, y, z, t = 'dom_x', 'dom_y', 'dom_z', 'dom_time'
    key = 'dom_d_minkowski_closest'

    n_doms = len(event[x])
    min_dists = np.zeros(n_doms)
    x_diff_sqr2 = tile_diff_sqr(event[x])
    y_diff_sqr2 = tile_diff_sqr(event[y])
    z_diff_sqr2 = tile_diff_sqr(event[z])
    t_diff_sqr2 = tile_diff_sqr(event[t])

    c_ns = 3e8 * 1e-9 / n
    tot_dist = c_ns*c_ns*t_diff_sqr2 - x_diff_sqr2 - y_diff_sqr2 - z_diff_sqr2
    tot_dist_abs = np.abs(tot_dist)
    np.fill_diagonal(tot_dist_abs, np.inf)
    indices = np.argmin(tot_dist_abs, axis=1)
    # * Row i is the i'th DOMs distance to the other doms. Therefore, the minimum distance for DOM i is located at (range[i], indices[i])
    closest_squared = tot_dist_abs[list(range(n_doms)), indices]
    closest = tot_dist[list(range(n_doms)), indices]
    closest = np.where(closest>0, np.sqrt(closest_squared), -np.sqrt(closest_squared))
    val = closest
        
    return val

def get_d_vertex(event):
    r"""Calculates the Euclidian distance for each DOM to a predicted interaction vertex and adds it to a dictionary. For now, the prediction of retro_crs_prefit is used.

    The variablename given is: dom_d_vertex
    """

    # * Use crs_prefits prediction - they are more similar to ours
    x, y, z = 'dom_x', 'dom_y', 'dom_z'
    x_pred, y_pred, z_pred = 'retro_crs_prefit_x', 'retro_crs_prefit_y', 'retro_crs_prefit_z'
    key = 'dom_d_vertex'

    x_diff_sqr = sqr_dist(event[x], event[x_pred])
    y_diff_sqr = sqr_dist(event[y], event[y_pred])
    z_diff_sqr = sqr_dist(event[z], event[z_pred])
    tot_dist = np.sqrt(x_diff_sqr+y_diff_sqr+z_diff_sqr)
    val = tot_dist
    
    return val

def get_d_minkowski_vertex(event, n=1.309):
    r"""Calculates the Minkowski distance for each DOM to a predicted interaction vertex and adds it to a dictionary. For now, the prediction of retro_crs_prefit is used.

    The Minkowski metric with signature (+---) is used. The variablename given is: dom_d_minkowski_vertex
    """
    # * Use crs_prefits prediction - they are more similar to ours
    x, y, z, t = 'dom_x', 'dom_y', 'dom_z', 'dom_time'
    x_pred, y_pred, z_pred, t_pred = 'retro_crs_prefit_x', 'retro_crs_prefit_y', 'retro_crs_prefit_z', 'retro_crs_prefit_time'
    key = 'dom_d_minkowski_vertex'
    
    x_diff_sqr = sqr_dist(event[x], event[x_pred])
    y_diff_sqr = sqr_dist(event[y], event[y_pred])
    z_diff_sqr = sqr_dist(event[z], event[z_pred])
    t_diff_sqr = sqr_dist(event[t], event[t_pred])

    c_ns = 3e8 * 1e-9 / n

    tot_sqr = c_ns*c_ns*t_diff_sqr - x_diff_sqr - y_diff_sqr - z_diff_sqr
    tot_abs = np.sqrt(abs(tot_sqr))
    tot_mink_dist = np.where(tot_sqr>0, tot_abs, -tot_abs)
    val = tot_mink_dist
    
    return val

def get_charge_over_d_vertex(event):
    r"""For each DOM, the DOM charge divided by the Euclidian distance and squared distance to a predicted interaction vertex is calculated and adds it to a dictionary. For now, the prediction of retro_crs_prefit is used.

    The variablenames given are: dom_charge_over_vertex, dom_charge_over_vertex_sqr
    """
    
    charge = 'dom_charge'
    d_vertex = get_d_vertex(event)
    val = event[charge]/d_vertex

    return val

def get_charge_over_d_vertex_sqr(event):
    r"""For each DOM, the DOM charge divided by the Euclidian distance and squared distance to a predicted interaction vertex is calculated and adds it to a dictionary. For now, the prediction of retro_crs_prefit is used.

    The variablenames given are: dom_charge_over_vertex, dom_charge_over_vertex_sqr
    """
    
    charge = 'dom_charge'
    d_vertex = get_d_vertex(event)
    val = event[charge]/(d_vertex*d_vertex)

    return val

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

def wanted_groups():
    return ['raw', 'transform1', 'masks']

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

def get_feature_dicts():
    d  = {
        'retro_crs_prefit_x': 
            {'transformer': None,
            'feature_calculator': None},
        
        'retro_crs_prefit_y': 
            {'transformer': None,
            'feature_calculator': None},

        'retro_crs_prefit_z': 
            {'transformer': None,
            'feature_calculator': None},
        
        'retro_crs_prefit_azimuth': 
            {'transformer': None,
            'feature_calculator': None},

        'retro_crs_prefit_zenith': 
            {'transformer': None,
            'feature_calculator': None},

        'retro_crs_prefit_time': 
            {'transformer': None,
            'feature_calculator': None},

        'retro_crs_prefit_energy': 
            {'transformer': None,
            'feature_calculator': None},

        'true_primary_direction_x': 
            {'transformer': None,
            'feature_calculator': None},

        'true_primary_direction_y': 
            {'transformer': None,
            'feature_calculator': None},

        'true_primary_direction_z': 
            {'transformer': None,
            'feature_calculator': None},
        
        'true_primary_speed': 
            {'transformer': None,
            'feature_calculator': None},
        
        'dom_atwd': 
            {'transformer': None,
            'feature_calculator': None},
        
        'dom_pulse_width': 
            {'transformer': StandardScaler(),
            'feature_calculator': None},
        
        'dom_charge': 
            {'transformer': QuantileTransformer(output_distribution='normal'),
            'feature_calculator': None},
        
        'true_primary_time': 
            {'transformer': QuantileTransformer(output_distribution='normal'),
            'feature_calculator': None},
        
        'dom_n_hit_multiple_doms': 
            {'transformer': StandardScaler(),
            'feature_calculator': None},
        
        'dom_time': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'dom_x': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'dom_y': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'dom_z': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'dom_timelength_fwhm': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'linefit_direction_x': 
            {'transformer': None,
            'feature_calculator': None},

        'linefit_direction_y': 
            {'transformer': None,
            'feature_calculator': None},

        'linefit_direction_z': 
            {'transformer': None,
            'feature_calculator': None},

        'linefit_point_on_line_x': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'linefit_point_on_line_y': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'linefit_point_on_line_z': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'toi_direction_x': 
            {'transformer': None,
            'feature_calculator': None},
        
        'toi_direction_y': 
            {'transformer': None,
            'feature_calculator': None},
        
        'toi_direction_z': 
            {'transformer': None,
            'feature_calculator': None},

        'toi_evalratio': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'toi_point_on_line_x': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'toi_point_on_line_y': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'toi_point_on_line_z': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'true_primary_energy': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'true_primary_position_x': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'true_primary_position_y': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'true_primary_position_z': 
            {'transformer': RobustScaler(),
            'feature_calculator': None},
        
        'tot_charge': 
            {'transformer': StandardScaler(),
            'feature_calculator': get_tot_charge},
        
        'dom_charge_significance': 
            {'transformer': StandardScaler(),
            'feature_calculator': get_charge_significance,
            'clip': {'min': None, 'max': 15}},
        
        'dom_frac_of_n_doms': 
            {'transformer': RobustScaler(),
            'feature_calculator': get_frac_of_n_doms},
        
        'dom_d_to_prev': 
            {'transformer': RobustScaler(),
            'feature_calculator': get_d_to_prev},
        
        'dom_v_from_prev': 
            {'transformer': QuantileTransformer(output_distribution='normal'),
            'feature_calculator': get_v_from_prev,
            'clip': {'min': None, 'max': 0.4e12}},
        
        'dom_d_minkowski_to_prev': 
            {'transformer': RobustScaler(),
            'feature_calculator': get_d_minkowski_to_prev},
        
        'dom_d_closest': 
            {'transformer': StandardScaler(),
            'feature_calculator': get_d_closest},
        
        'dom_d_minkowski_closest': 
            {'transformer': StandardScaler(),
            'feature_calculator': get_d_minkowski_closest},
        
        'dom_d_vertex': 
            {'transformer': RobustScaler(),
            'feature_calculator': get_d_vertex},
        
        'dom_d_minkowski_vertex': 
            {'transformer': RobustScaler(),
            'feature_calculator': get_d_minkowski_vertex},
        
        'dom_charge_over_vertex': 
            {'transformer': QuantileTransformer(output_distribution='normal'),
            'feature_calculator': get_charge_over_d_vertex,
            'clip': {'min': None, 'max': 0.4}},
        
        'dom_charge_over_vertex_sqr': 
            {'transformer': QuantileTransformer(output_distribution='normal'),
            'feature_calculator': get_charge_over_d_vertex_sqr,
            'clip': {'min': None, 'max': 0.04}},
    }
    return d

def get_geom_features(n_nearest=2):

    d = {}
    for i in range(n_nearest):
        x_name = 'dom_closest%d_x'%(i+1)
        y_name = 'dom_closest%d_y'%(i+1)
        z_name = 'dom_closest%d_z'%(i+1)
        q_name = 'dom_closest%d_charge'%(i+1)
        t_name = 'dom_closest%d_time'%(i+1)
        
        d[x_name] = {'transformer': 'dom_x',
                     'n_nearest': n_nearest}
        d[y_name] = {'transformer': 'dom_y',
                     'n_nearest': n_nearest}
        d[z_name] = {'transformer': 'dom_z',
                     'n_nearest': n_nearest}
        d[q_name] = {'transformer': 'dom_charge',
                     'n_nearest': n_nearest}
        d[t_name] = {'transformer': 'dom_time',
                     'n_nearest': n_nearest}
    
    return d

def get_n_nearest_data(db_path, id_chunk, geom_features, geom_dict_path):
    """Finds and extracts data from the nearest n DOMs
    
    Parameters
    ----------
    db_path : str
        Absolute path to the Shelve-database
    id_chunk : list
        list of event IDs to extract data for
    geom_features : dict
        What geometry data to extract, e.g. nearest DOMs x-value
    geom_dict_path : str
        full path to geometry dictionary (dictionary containing nearest DOMs for each DOM)
    
    Returns
    -------
    dict
        Data of nearest N doms for each event ID
    """    
    # * Chunk ID's for multiprocessing
    n_chunks = cpu_count()
    id_chunks = np.array_split(id_chunk, n_chunks)

    # * Multiprocess
    db_list = [db_path]*n_chunks
    geom_features_list = [geom_features]*n_chunks
    geom_dict_list = [geom_dict_path]*n_chunks
    packed = zip(id_chunks, db_list, geom_features_list, geom_dict_list)
    with Pool() as p:
        data = p.map(get_n_nearest_data_multiprocess, packed)
    
    # * Unpack
    all_events = {}
    for events in data:
        all_events.update(events)

    return all_events

def get_n_nearest_data_multiprocess(pack):
    # * Unpack
    ids, db_path, geom_features, geom_dict_path = pack

    # * Load geometry dictionary
    with open(geom_dict_path, 'rb') as f:
        geom_dict = pickle.load(f)

    # * Extract how many neighbors wanted
    dummy = next(iter(geom_features))
    n_nearest = geom_features[dummy]['n_nearest']
    events = {index: {} for index in ids}
    
    # * We open the database inside each process - opening outside and passing opened file seems to produce bugs...
    with shelve.open(path_db, 'r') as db:
        
        # * Loop over events
        for index in ids:
            raw_event = db[index]['raw']

            # * Loop over DOMs in each event
            n_doms = len(raw_event['dom_key'])
            for i_dom, dom_id in enumerate(raw_event['dom_key']):

                # * Get nearest n doms
                closest_doms = geom_dict[dom_id]['closest'][:n_nearest]
                for i_closest, closest_dom in enumerate(closest_doms):
                    
                    # * Check if closest lit up aswell
                    where = np.where(raw_event['dom_key'] == closest_dom)[0]
                    if where.shape[0]>0:
                        
                        # * Sort w.r.t. absolute distance in time.
                        t = raw_event['dom_time'][i_dom]
                        t_dist = np.abs(raw_event['dom_time'][where]-t)
                        # * Oneliner: zips, Sorts w.r.t. distance in time and extracts index of closest
                        closest_index = sorted(zip(where, t_dist), key=lambda x: x[1])[0][0]

                        # * Assign Q and t-value
                        q = raw_event['dom_charge'][closest_index]
                        t = raw_event['dom_time'][closest_index]
                    
                    # * If not, assign Q = 0.0, time = -12.000 ns.
                    # * -12.000 is chosen such that DOMs which did not light up gets a unique time (only times >-10k are in dataset)
                    else:
                        q = 0.0
                        t = -12000
                    
                    # * Get x, y, z-values aswell from the geometry dictionary
                    x = geom_dict[dom_id]['coordinates'][0]
                    y = geom_dict[dom_id]['coordinates'][1]
                    z = geom_dict[dom_id]['coordinates'][2]

                    # * make new name: Closest DOM charge is named 'dom_closest1_charge' and so on.
                    x_name = 'dom_closest%d_x'%(i_closest+1)
                    y_name = 'dom_closest%d_y'%(i_closest+1)
                    z_name = 'dom_closest%d_z'%(i_closest+1)
                    q_name = 'dom_closest%d_charge'%(i_closest+1)
                    t_name = 'dom_closest%d_time'%(i_closest+1)

                    # * If first dom: Make new key
                    if i_dom == 0:
                        events[index][x_name] = np.zeros(n_doms) 
                        events[index][y_name] = np.zeros(n_doms) 
                        events[index][z_name] = np.zeros(n_doms) 
                        events[index][q_name] = np.zeros(n_doms) 
                        events[index][t_name] = np.zeros(n_doms) 
                    
                    events[index][x_name][i_dom] = x 
                    events[index][y_name][i_dom] = y 
                    events[index][z_name][i_dom] = z 
                    events[index][q_name][i_dom] = q 
                    events[index][t_name][i_dom] = t
    
    return events

def load_and_fit_transformer(pack):
    ids, (key, feature_dict), db_path, n_data = pack

    with shelve.open(db_path, 'r') as db:
        id_iter = iter(ids)
        data =np.array([])
        
        loaded = 0
        transformer = feature_dict['transformer']
        clip_d = feature_dict.get('clip', None)

        # * If we are dealing with a feature that needs to be transforme, make the transformer!
        if transformer:

            # * Extract the function needed for derived features
            fnc = feature_dict['feature_calculator']
            
            # * Loop until we have enough samples for the transformer
            while loaded < n_data:
                
                # * If we iterated over all data, thats it - just exit loop.
                try:
                    event = db[next(id_iter)]['raw']
                except StopIteration:
                    break

                # * If dealing with a derived feature, calculate it!
                if fnc:
                    new_data = fnc(event)
                
                # * If not, just load it
                else:
                    new_data = event[key]
                
                data = np.append(data, new_data)
                if isinstance(new_data, np.ndarray):
                    loaded += new_data.shape[0]
                elif isinstance(new_data, (float, int)):
                    loaded += 1
                else:
                    raise ValueError('load_and_fit_transformer: Unknown type (%s) encountered'%(type(new_data)))

            # * Save plot of pre-transformed data
            path = get_project_root()+'/reports/shelve_data'
            if not Path(path).exists():
                Path(path).mkdir()
            plot_d = {'data': [data], 'savefig': path+'/%s.png'%key}
            _ = make_plot(plot_d)
            
            # * Now fit a transformer
            transformer.fit(data.reshape(-1, 1))
            
            # * save plot of transformed data
            if clip_d:
                data_transformed = np.clip(data, clip_d['min'], clip_d['max'])
                data_transformed = transformer.transform(data_transformed.reshape(-1, 1))
            else:
                data_transformed = transformer.transform(data.reshape(-1, 1))
            plot_d = {'data': [data_transformed], 'savefig': path+'/%s_transformed.png'%key}
            _ = make_plot(plot_d)
    
    return {key: transformer}

def fit_transformers(db_path, n_data, feature_dicts):
    # * Assumes a RANDOMIZED DB!
    ids = [str(i) for i in range(n_data)]

    # * Load/calculate features, then transform
    keys = [key for key in feature_dicts]
    
    # * Multiprocess
    db_list = [db_path]*len(keys)
    ids_list = [ids]*len(keys)
    n_data_list = [n_data]*len(keys)
    packed = zip(ids_list, feature_dicts.items(), db_list, n_data_list)
    with Pool() as p:
        transformers_list = p.map(load_and_fit_transformer, packed)

    # * Make a dictionary with the transformers
    transformers = {}
    for transformer in transformers_list:
        transformers.update(transformer)
    
    return transformers

def make_multiprocess_pack(discriminator, stuff):
    """Packs the object to multiprocess with copies of important stuff put in a list.
    
    Parameters
    ----------
    discriminator : list
        Objects to multiprocess over.
    stuff : list
        List of objects to make copies of.
    
    Returns
    -------
    zip-iterator
        A zip-object with packed stuff.
    """    
    lists = [discriminator]
    n_copies = len(discriminator)
    for entry in stuff:
        lists.append([entry]*n_copies)
    return zip(*lists)

def transform_events(db_path, ids, feature_dicts, transformers, n_nearest_data, geom_features):
    """Transforms events.

    For each ID, the data induced by feature_dicts is calculated and/or transformed and placed in a a dictionary under event ID --> transformed. Furthermore, meta-information and masks are saved aswell.
    
    Parameters
    ----------
    db_path : str
        FUll path to Shelve database
    ids : list
        List of ids to process
    feature_dicts : dict
        dictionary containing dictionaries with e.g. which transformer to use
    transformers : dict
        Dictionary containing info on which transformer to use
    n_nearest_data : dict
        Dictionary containing the geometry data to transform and add to the event.
    geom_features : dict
        Dictionary containing the informtion required to transform the n_nearest_data
    
    Returns
    -------
    dict
        Dictionary containing transformed events.
    """    
    # * Chunk ID's for multiprocessing
    n_chunks = cpu_count()
    id_chunks = np.array_split(ids, n_chunks)
    
    # * Repack the n_nearest_data so that IDs matches
    n_nearest_chunks = [{event_id: n_nearest_data[event_id] for event_id in chunk} for chunk in id_chunks]

    # * Multiprocess - prep by zipping all the required stuff for each process
    db_path_list = [db_path]*n_chunks
    transformers_list = [transformers]*n_chunks
    feature_dicts_list = [feature_dicts]*n_chunks
    geom_features_list = [geom_features]*n_chunks
    packed = zip(id_chunks, n_nearest_chunks, db_path_list, transformers_list, feature_dicts_list, geom_features_list)
    with Pool() as p:
        events_transformed = p.map(transform_events_multiprocess, packed)
    events_unpacked = {}
    for events in events_transformed:
        events_unpacked.update(events)   
    
    return events_unpacked

def transform_events_multiprocess(pack):
    
    # * Unpack
    ids, n_nearest, path_db, transformers, feature_dicts, geom_features = pack
    events = {}
    
    with shelve.open(path_db, 'r') as db:
        # * Loop over events and transform
        for index in ids:
            raw_event = db[index]['raw']
            transformed_event = {}

            # * Save meta and masks aswell
            events[index] = {'meta': db[index]['meta']}
            events[index].update({'masks': db[index]['masks']})
            
            # * For each key in transformer, create transformed data and put in dict
            for key, transformer in transformers.items():
                
                # * If key is not in raw_event, it is a derived feature. Calculate it.
                try:
                    data = raw_event[key]
                except KeyError:
                    fnc = feature_dicts[key]['feature_calculator']
                    data = fnc(raw_event)
                
                if transformer:
                    if isinstance(data, np.ndarray):
                        transformed = transformer.transform(data.reshape(-1, 1)).flatten()
                    elif isinstance(data, (float, int)):
                        transformed = transformer.transform(np.array(data).reshape(-1, 1)).flatten()[0]
                    else:
                        raise ValueError('transform_events_multiprocess: Unknown type (%s) encountered'%(type(data)))
                else:
                    transformed = data

                # * Save it in 32bit format
                try:
                    transformed_event[key] = transformed.astype(np.float32)
                except AttributeError:
                    if isinstance(transformed, float):
                        transformed_event[key] = transformed
                    else:
                        raise AttributeError('Unknown format encountered(%s)!'%(type(transformed)))
            
            # * Transform geometric data aswell
            for geo_key, data in n_nearest[index].items():
                geo_transformer_key = geom_features[geo_key]['transformer']
                transformer = transformers[geo_transformer_key]
                if isinstance(data, np.ndarray):
                    transformed_geo_data = transformer.transform(data.reshape(-1, 1)).flatten()
                else:
                    raise ValueError('transform_events_multiprocess: Unknown type (%s) encountered'%(type(data)))
                transformed_event[geo_key] = transformed_geo_data.astype(np.float32)
            
            # * Append to all events
            events[index].update({'transformed': transformed_event})

    return events

if __name__ == '__main__':

    description = 'Converts raw data in a shelve-database to transformed data in a new shelve-database'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--chunksize', default=1000, type=int, help='Sets the amount of events to process and save at a time. Splits the entire data into chunks of this size.')
    parser.add_argument('--n_transform', default=1000, type=int, help='Sets the amount of datapoints to use in approximating their distribution during fitting of transformer')
    parser.add_argument('--path', default='None', type=str, help='Data directory.')
    parser.add_argument('--fit_transformers', action='store_true', help='Whether or not to fit new transformers.')
    parser.add_argument('--new_name', default='None', type=str, help='Sets the new databases name.')

    args = parser.parse_args()

    if args.path == 'None':
        raise KeyError(r'A path must be supplied! Use flag --path')

    if args.new_name == 'None':
        raise KeyError(r'A new name must be supplied! Use flag --new_name')

    # * Setup - where to load data, how many events
    path_db = Path(get_project_root()+'/'+get_path_from_root(args.path))
    path_geom_dict = str(path_db.parent)+'/dom_geom.pickle'
    path_transformer = str(path_db.parent)+'/transformers.pickle'
    path_new_db = str(path_db.parent)+'/'+args.new_name

    n_data = args.n_transform
    chunksize = args.chunksize
    feature_dicts = get_feature_dicts()
    geom_features = get_geom_features()
    
    # * Fit and save transformers
    if args.fit_transformers:
        print(get_time(), 'Fitting transformers...')
        transformers = fit_transformers(path_db, n_data, feature_dicts)
        with open(transformer_path, 'wb') as f:
            pickle.dump(transformers, f)
        print(get_time(), 'Transformers fitted!')

    with open(path_transformer, 'rb') as f:
        transformers = pickle.load(f)
    
    # * Chunk IDs and process it chunk by chunk
    with shelve.open(path_db, 'r') as f:
        ids = [index for index in f]

    # * Chunk can't be smaller than 1
    n_chunks = max(1, len(ids)//chunksize)
    id_chunks = np.array_split(ids, n_chunks)

    # * Make database
    print(get_time(), 'Creating database')
    with dumbdbm.open(path_new_db, 'n') as f:
        db_train = shelve.Shelf(f)
    print(get_time(), 'Database created!')

    # * Loop over chunks and save then one at a time
    for i_chunk, id_chunk in enumerate(id_chunks):
        print('')
        print(get_time(), 'Processing chunk %d of %d'%(i_chunk+1, n_chunks))
        # * For each chunk, first retrieve data on the n nearest neighbors. 
        n_nearest_data = get_n_nearest_data(path_db, id_chunk, geom_features, geom_dict_path)
        
        # * Now transform all data
        events = transform_events(path_db, id_chunk, feature_dicts, transformers, n_nearest_data, geom_features)

        print(get_time(), 'Saving chunk %d of %d'%(i_chunk+1, n_chunks))
        with shelve.open(new_db_path, 'w') as db:
            for event, d in events.items():
                db[event] = d
        print(get_time(), 'Saved chunk!')
        