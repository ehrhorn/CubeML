import numpy as np
import shelve
import argparse
import sys
import pickle

from multiprocessing import cpu_count, Pool
from pathlib import Path
from src.modules.helper_functions import get_project_root, get_time, get_path_from_root, calc_l2_dist


def find_unique_ids(pack):
    # * Unpack and notify
    datafile, path = pack
    print(get_time(), 'Processing %s'%(datafile))
    sys.stdout.flush()

    # * all_doms will be a dictionary with dom_id: coordinates.
    all_doms = {}

    # * Retrieve DOM-ID and coordinates of each event
    # * len(keys) = N_events in file, len(keys[0]) = NUmber of DOMs in event 0
    with shelve.open(path) as f:
        keys = f[datafile]['dom_key']
        dom_xs = f[datafile]['dom_x']
        dom_ys = f[datafile]['dom_y']
        dom_zs = f[datafile]['dom_z']
    
    for key, dom_x, dom_y, dom_z in zip(keys, dom_xs, dom_ys, dom_zs):
        # * Convert x, y, z into one coordinate entry as a np-array
        coords = [{'coordinates': np.array([x, y, z])} for x, y, z in zip(dom_x, dom_y, dom_z)]

        # * Update the dictionary over all events
        all_doms.update(zip(key, coords))
    
    return all_doms

def make_geom_dict(data_dir_path=get_project_root()+'/data/oscnext-genie-level5-v01-01-pass2/',
                   multiprocess=True,
                   d_name='dom_geom.pickle'):
    
    print(get_time(), 'Making geometry dictionary...')
    shelve_path = data_dir_path+'shelve/oscnext-genie-level5-v01-01-pass2'
    
    # * Get filenames
    with shelve.open(shelve_path) as f:
        filenames = [key for key in f]

    # * Prepare for multiprocessing
    path_list = [shelve_path]*len(filenames)
    packed = [entry for entry in zip(filenames, path_list)]
    
    # * Multiprocess
    if multiprocess:
        with Pool() as p:
            all_dicts = p.map(find_unique_ids, packed)

        # * Combine dictionaries
        print(get_time(), 'Combining dictionaries...')
        dom_geom_dict = {}
        for d in all_dicts:
            dom_geom_dict.update(d)
        print(get_time(), 'Dictionaries combined!')
        
    else:
        dom_geom_dict = {}
        for pack in packed:
            dom_geom_dict.update(find_unique_ids(pack))
    
    return dom_geom_dict

def find_nearest_doms(data_dir_path=get_project_root()+'/data/oscnext-genie-level5-v01-01-pass2/',
                      multiprocess=True,
                      d_name='dom_geom.pickle'):
    
    # * Load precalculated geometry dictionary
    d_geom = pickle.load(open(data_dir_path+d_name, 'rb'))

    # * For each entry, calculate distances to all other DOMs
    # * Extract coordinates and pair with ID
    dom_ids = [dom_id for dom_id in d_geom]
    coords = {key: items['coordinates'] for key, items in d_geom.items()}
    own_coords = [items['coordinates'] for key, items in d_geom.items()]
    
    print(get_time(), 'Calculation of nearest DOMs begun...')
    if multiprocess:
        # * prepare for multiprocessing - we loop over DOM IDs
        coords_list = [coords]*len(dom_ids)
        packed = [pack for pack in zip(dom_ids, own_coords, coords_list)]

        with Pool() as p:
            dicts = p.map(find_nearest_doms_multi, packed)
    else:
        raise ValueError('Only multiprocessing implemented!')
    print(get_time(), 'Calculation finished!')
    
    # * Update the geometry dictionary with the closest DOMs
    for dom_id, d in zip(dom_ids, dicts):
        d_geom[dom_id].update(d)
    
    return d_geom

def find_nearest_doms_multi(pack):

    # * Unpack. dom_id = str, own_coords = array w. shape(3,), coords = dict with dom_id: coords for all DOMs
    own_id, own_coords, coords= pack

    # * Calculate distances to all DOMs
    dists = {dom_id: calc_l2_dist(own_coords, dom_coords) for dom_id, dom_coords in coords.items()}

    # * Sort IDs wrt distance and put in dict 
    # * drop the first entry, since this is itself
    d = {'closest': [key for key, value in sorted(dists.items(), key=lambda kv: kv[1])][1:]}

    return d

if __name__ == '__main__':

    # * Parse arguments!
    description = 'Creates a dictionary of DOM-IDs and their positions by looping over all DOMs.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--multiprocess', action='store_true', help='Enables multiprocessing.')
    parser.add_argument('--create_geom_dict', action='store_true', help='Calculates a geometry dictionary.')
    parser.add_argument('--find_nearest', action='store_true', help='Finds the nearest DOMs for each DOM.')
    args = parser.parse_args()

    data_dir_path = get_project_root()+'/data/oscnext-genie-level5-v01-01-pass2/'
    d_name = 'dom_geom.pickle'

    # * If geometry dictionary does not exist, make it first
    if args.create_geom_dict or not Path(data_dir_path+d_name).exists():
        dom_geom_dict = make_geom_dict(data_dir_path=data_dir_path, multiprocess=args.multiprocess, d_name=d_name)
        
        # * Save geometry as a dict with DOM ID: np.array([x, y, z])
        n_doms_found = len([key for key in dom_geom_dict])
        pickle.dump(dom_geom_dict, open(data_dir_path+d_name, 'wb'))
        print(get_time(), 'Found %d DOMs in total.'%(n_doms_found))
        print(get_time(), 'Saved file at %s'%(get_path_from_root(data_dir_path+d_name)))

    # * Calculate distance to all other DOMs if it doesn't already exist
    if args.find_nearest:
        d_geom = find_nearest_doms(data_dir_path=data_dir_path, multiprocess=args.multiprocess, d_name=d_name)

        # * Save it 
        pickle.dump(d_geom, open(data_dir_path+d_name, 'wb'))
        print(get_time(), 'Saved file at %s'%(get_path_from_root(data_dir_path+d_name)))

    
   