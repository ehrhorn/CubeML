import numpy as np
import argparse
import sys
import pickle
import sqlite3
import joblib
import time

from multiprocessing import cpu_count, Pool
from pathlib import Path
from src.modules.helper_functions import get_project_root, get_time, get_path_from_root, calc_l2_dist, make_multiprocess_pack, flatten_list_of_lists
from src.modules.constants import *
from src.modules.classes import SqliteFetcher

def find_unique_ids(pack):
    # Unpack and notify
    ids, db = pack

    # all_doms will be a dictionary with dom_id: coordinates.
    all_doms = {}

    seq = [
        'dom_x',
        'dom_y',
        'dom_z'
    ]
    fetched = db.fetch_features(
        all_events=ids,
        seq_features=seq
    )
    x = flatten_list_of_lists(
        [fetched[idx]['dom_x'] for idx in ids]
    )
    y = flatten_list_of_lists(
        [fetched[idx]['dom_y'] for idx in ids]
    )
    z = flatten_list_of_lists(
        [fetched[idx]['dom_z'] for idx in ids]
    )
    coords = list(zip(x, y, z))
    print(coords[0], len(coords))
    unique_coords = list(set(coords))
    print(len(unique_coords))
    a+=1
    
    return all_doms

def make_geom_dict(d_name='dom_geom.pickle'):
    
    print(get_time(), 'Making geometry dictionary...')
    db = SqliteFetcher(PATH_TRAIN_DB)
    ids = db.ids[:10000]
    chunk_size = 40000
    n_chunks = 1 + len(ids)//chunk_size
    chunks = np.array_split(ids, n_chunks)
    packed = make_multiprocess_pack(chunks, db)
    
    with Pool() as p:
        all_dicts = p.map(find_unique_ids, packed)

    # Combine dictionaries
    print(get_time(), 'Combining dictionaries...')
    dom_geom_dict = {}
    for d in all_dicts:
        dom_geom_dict.update(d)
    print(get_time(), 'Dictionaries combined!')
        
    
    return dom_geom_dict

def find_nearest_doms(db,
                      multiprocess=True,
                      d_name='dom_geom.pickle'):
    
    # Load precalculated geometry dictionary
    d_geom = pickle.load(open(PATH_DATA_OSCNEXT+d_name, 'rb'))

    # For each entry, calculate distances to all other DOMs
    # Extract coordinates and pair with ID
    dom_ids = [dom_id for dom_id in d_geom]
    coords = {key: items['coordinates'] for key, items in d_geom.items()}
    own_coords = [items['coordinates'] for key, items in d_geom.items()]
    PATH
    print(get_time(), 'Calculation of nearest DOMs begun...')
    if multiprocess:
        # prepare for multiprocessing - we loop over DOM IDs
        coords_list = [coords]*len(dom_ids)
        packed = [pack for pack in zip(dom_ids, own_coords, coords_list)]

        with Pool() as p:
            dicts = p.map(find_nearest_doms_multi, packed)
    else:
        raise ValueError('Only multiprocessing implemented!')
    print(get_time(), 'Calculation finished!')
    
    # Update the geometry dictionary with the closest DOMs
    for dom_id, d in zip(dom_ids, dicts):
        d_geom[dom_id].update(d)
    
    return d_geom

def find_nearest_doms_multi(pack):

    # Unpack. dom_id = str, own_coords = array w. shape(3,), coords = dict with dom_id: coords for all DOMs
    own_id, own_coords, coords= pack

    # Calculate distances to all DOMs
    dists = {dom_id: calc_l2_dist(own_coords, dom_coords) for dom_id, dom_coords in coords.items()}

    # Sort IDs wrt distance and put in dict 
    # drop the first entry, since this is itself
    d = {'closest': [key for key, value in sorted(dists.items(), key=lambda kv: kv[1])][1:]}

    return d

def find_indices(pack):
    i_coords, coords = pack
    chunksize = i_coords.shape[0]
    diff = i_coords.reshape((chunksize, 1, 3))-coords
    dists = np.sum(diff*diff, axis=2)
    indices = np.argmin(dists,axis=1)
    return indices

if __name__ == '__main__':

    # Parse arguments!
    description = 'Creates a dictionary of DOM-IDs and their positions by looping over all DOMs.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--find_ids', 
        action='store_true', 
        help='Finds IDs of all DOMs in a DB and saves them. Takes time.'
    )
    parser.add_argument(
        '--db',
        default='train',
        type=str,
        help='Sets which database to iterate over. Options: "train", "val", "test".'
    )

    parser.add_argument(
        '--find_nearest', 
        action='store_true', 
        help='Finds the nearest DOMs for each DOM.'
    )

    parser.add_argument(
        '--convert_pulse_width', 
        action='store_true', 
        help='Potentially fits a transformer and converts keys'
    )


    args = parser.parse_args()

    if args.db == 'train':
        path = PATH_TRAIN_DB
    elif args.db == 'val':
        path = PATH_VAL_DB
    elif args.db == 'test':
        path = PATH_TEST_DB
    else:
        raise ValueError('A database must be chosen!')

    d_name = 'dom_geom.pickle'
    geom_d_path = PATH_DATA_OSCNEXT+'/'+d_name
    geom_dict = pickle.load(open(geom_d_path, 'rb'))
    db = SqliteFetcher(path)

    if args.find_ids:

        print(get_time(), 'Fetching rows')
        chunksize = 50000
        all_rows = db.rows
        # all_rows = [str(e) for e in range(120000)]
        n_rows = len(all_rows)
        i_rows = 0
        ave_time = None
        n_chunks = 1 + n_rows//chunksize
        chunks = np.array_split(all_rows, n_chunks)
        print(get_time(), 'Rows fetched. Processing begun.')
        
        seq = [
            'row',
            'dom_x',
            'dom_y',
            'dom_z',
            'dom_key'
        ]
        dom_ids = np.array([key for key in geom_dict])
        coords = np.array([data['coordinates'] for key, data in geom_dict.items()])
        
        transformers = joblib.load(open(PATH_TRANSFORMERS, 'rb'))
        with sqlite3.connect(path) as db:
            cursor = db.cursor()
        
            for i_rows, rows in enumerate(chunks):
                print('')
                print(get_time(), 'Processing chunk %d/%d'%(i_rows+1, n_chunks))
                print(get_time(), 'Fetching DOMs from Sqlite...')

                n_ids = len(rows)
                query = 'SELECT {features} FROM sequential WHERE row IN ({rows})'.format(
                    features=', '.join(seq),
                    rows=', '.join(['?'] * n_ids)
                )
                cursor.execute(query, rows)
                fetched = cursor.fetchall()
                if fetched[-1][4] != 'NotConvertedYet':
                    print('ALREADY PROCESSED')
                    continue
                
                # Time it for progress printing
                start_time = time.time()
                print(get_time(), 'DOMs fetched. Inverse transforming...')
                row = np.array([e[0] for e in fetched])
                x = np.array([e[1] for e in fetched])
                y = np.array([e[2] for e in fetched])
                z = np.array([e[3] for e in fetched])

                # Inverse transform
                x_t = np.squeeze(
                    transformers['dom_x'].inverse_transform(
                        x.reshape(-1, 1)
                    )
                )

                y_t = np.squeeze(
                    transformers['dom_y'].inverse_transform(
                        y.reshape(-1, 1)
                    )
                )

                z_t = np.squeeze(
                    transformers['dom_z'].inverse_transform(
                        z.reshape(-1, 1)
                    )
                )

                print(get_time(), 'Inverse transform finished. Finding DOM ID...')

                # Find ID
                i_coords = np.append(x_t.reshape(-1, 1), y_t.reshape(-1, 1), axis=1)
                i_coords = np.append(i_coords, z_t.reshape(-1, 1), axis=1)
                i_coords_split = np.array_split(i_coords, AVAILABLE_CORES, axis=0)
                
                packed = make_multiprocess_pack(i_coords_split, coords)
                with Pool() as p:
                    indices = flatten_list_of_lists(
                        p.map(find_indices, packed)
                    )

                dom_id_matched = dom_ids[indices]
            
                # Save IDs
                print(get_time(), 'DOM IDs found. Writing DOM IDs to database...')
                query = 'UPDATE {table} SET {name}=? WHERE row=?'.format(
                    table='sequential',
                    name='dom_key',
                )
                cursor.executemany(
                    query, [
                        [e[0], str(e[1])] for e in zip(dom_id_matched, row)
                    ]
                )
                db.commit()
                print(get_time(), 'IDs saved.')
                

                chunk_time = time.time()-start_time
                if not ave_time:
                    ave_time = 1.0*chunk_time
                else:
                    ave_time = 0.9*ave_time + 0.1*chunk_time
                
                remaining = ((n_chunks-1-i_rows)*ave_time)/3600
                print(get_time(), 'Average conversion time: %.1f seconds.'%(ave_time))
                print(get_time(), 'Time remaining: %.2f hours.'%(remaining))

    if args.find_nearest:
        
        from src.modules.preprocessing import (
            get_n_nearest_data_sqlite_multi,
            get_geom_features
        )
        
        ids = db.ids
        # ids = [str(i) for i in range(10000)]
        chunksize = 10000
        n_chunks = 1 + len(ids)//chunksize
        chunks = np.array_split(ids, n_chunks)
        ave_time = None
        
        transformers = joblib.load(open(PATH_TRANSFORMERS, 'rb'))
        geom_feats = get_geom_features()

        with Pool() as p:
            for i_chunk, chunk in enumerate(chunks):
                
                # Time it for progress printing
                start_time = time.time()
                print('')
                print(
                    get_time(), 'Processing chunk %d/%d. Finding nearest...'%(
                        i_chunk+1, n_chunks
                    )
                )

                chunk_split = np.array_split(chunk, AVAILABLE_CORES)
                packed = make_multiprocess_pack(
                    chunk_split,
                    db,
                    geom_dict,
                    transformers
                )
                
                data_ds = p.map(
                        get_n_nearest_data_sqlite_multi, packed
                    )
                
                # Unpack dictionary
                data_d = data_ds[0]
                for i_d, d in enumerate(data_ds):
                    if i_d == 0:
                        continue
                    for key, data in d.items():
                        data_d[key] = np.append(data_d[key], data)
                end = time.time()
                
                # Save
                print(get_time(), 'Nearest found. Saving...')

                with sqlite3.connect(path) as con:
                    cursor = con.cursor()
                    rows = [str(e) for e in data_d['row']]
                    for name in geom_feats:
                        data = data_d[name]
                        # Check if column exists - if not, create it.
                        if not name in db.tables['sequential']:
                            query = 'ALTER TABLE {table} ADD COLUMN {name} {astype}'.format(
                                table='sequential',
                                name=name,
                                astype='REAL'
                            )
                            cursor.execute(query)
                        
                        # Write data to column
                        query = 'UPDATE {table} SET {name}=? WHERE row=?'.format(
                            table='sequential',
                            name=name,
                        )
                        cursor.executemany(query, [[e[0], e[1]] for e in zip(data, rows)])
                    con.commit()
                

                print(get_time(), 'Saved chunk.')
                chunk_time = time.time()-start_time
                if not ave_time:
                    ave_time = 1.0*chunk_time
                else:
                    ave_time = 0.9*ave_time + 0.1*chunk_time
                
                remaining = ((n_chunks-1-i_chunk)*ave_time)/3600
                print(get_time(), 'Average conversion time: %.1f seconds.'%(ave_time))
                print(get_time(), 'Time remaining: %.2f hours.'%(remaining))

    if args.convert_pulse_width:

        from src.modules.preprocessing import (
            get_feature_dicts
        )
        feature_dicts = get_feature_dicts()
        seq = [
            'dom_pulse_width',
        ]
        fit_on_n_events = 1000
        with sqlite3.connect(path) as con:
            cursor = con.cursor()
            cursor.execute('SELECT min(event_no), max(event_no) FROM meta')
            ids = cursor.fetchall()
        
        minimum, maximum = ids[0][0], ids[0][1]
        
        # Load transformers
        transformers = joblib.load(open(PATH_TRANSFORMERS, 'rb'))
        ids = [str(minimum+i) for i in range(fit_on_n_events)]
        events = db.fetch_features(all_events=ids, seq_features=seq)
        for key, item in feature_dicts.items():
            # if there is not transformer, load n events and fit transformer
            if key in seq and key not in transformers:
                transformer = item['transformer']
                data = np.array(
                    flatten_list_of_lists(
                        [events[idx][key] for idx in ids]
                    )
                )
                transformer.fit(data.reshape(-1, 1))
                # data_transformed = np.squeeze(
                #     transformer.transform(
                #         data.reshape(-1, 1)
                #     )
                # )
                transformers[key] = transformer

        # Save transformers with newly fitted ones    
        with open(PATH_TRANSFORMERS, 'wb') as f:
            joblib.dump(transformers, f)
        transformer = transformers['dom_pulse_width']
                
        # Now convert
        print(get_time(), 'Fetching rows')
        chunksize = 100000
        # all_rows = [str(e) for e in range(2000000)]
        n_rows = db.n_rows
        with sqlite3.connect(path) as con:
            cursor = con.cursor()
            cursor.execute('SELECT min(row), max(row) FROM sequential')
            ids = cursor.fetchall()
        
        minimum, maximum = ids[0][0], ids[0][1]
        # all_rows = [str(e) for e in range(n_rows)]
        ave_time = None
        n_chunks = 1 + n_rows//chunksize
        # chunks = np.array_split(all_rows, n_chunks)
        print(get_time(), 'Rows fetched. Processing begun.')
        
        with sqlite3.connect(path) as db:
            cursor = db.cursor()
            i_chunk = 0
            while i_chunk < n_chunks:
                chunk = [
                    str(e) for e in np.arange(
                        minimum + i_chunk*chunksize, minimum + (i_chunk+1)*chunksize
                    )
                ]
                i_chunk += 1
                
                print('')
                print(get_time(), 'Processing chunk %d/%d'%(
                    i_chunk,
                    n_chunks
                ))
                # Fetch
                n_ids = len(chunk)
                query = 'SELECT {features} FROM sequential WHERE row IN ({rows})'.format(
                    features=', '.join(['row', 'dom_pulse_width']),
                    rows=', '.join(['?'] * n_ids)
                )
                cursor.execute(query, chunk)
                fetched = cursor.fetchall()
                
                row_all = np.array([e[0] for e in fetched])
                width_all = np.array([e[1] for e in fetched])

                # Ensure it hasn't already been transformed
                mask = width_all>=1.0
                row = row_all[mask]
                width = width_all[mask]

                if width.shape[0] == 0:
                    print('ALREADY TRANSFORMED')
                    continue
                
                start_time = time.time()
                
                # Transform
                width_transformed = np.squeeze(
                    transformer.transform(
                        width.reshape(-1, 1)
                    )
                )
                # Save transformed
                query = 'UPDATE {table} SET {name}=? WHERE row=?'.format(
                    table='sequential',
                    name='dom_pulse_width',
                )
                cursor.executemany(query, [[e[0], str(e[1])] for e in zip(width_transformed, row)])
                db.commit()

                print(get_time(), 'Saved chunk.')
                chunk_time = time.time()-start_time
                if not ave_time:
                    ave_time = 1.0*chunk_time
                else:
                    ave_time = 0.9*ave_time + 0.1*chunk_time
                
                remaining = ((n_chunks-i_chunk)*ave_time)/3600
                print(get_time(), 'Average conversion time: %.1f seconds.'%(ave_time))
                print(get_time(), 'Time remaining: %.2f hours.'%(remaining))
