import shelve
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import sys
import sqlite3

from sklearn.preprocessing import QuantileTransformer, RobustScaler, StandardScaler
from src.modules.helper_functions import get_project_root, get_time, get_path_from_root, make_multiprocess_pack, convert_keys
from src.modules.reporting import make_plot
from src.modules.preprocessing import *

if __name__ == '__main__':

    description = 'Converts raw data in a shelve-database to transformed data in a new shelve-database'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dev', action='store_true', help='Initiates developermode')
    parser.add_argument('--chunksize', default=20000, type=int, help='Sets the amount of events to process and save at a time. Splits the entire data into chunks of this size.')
    parser.add_argument('--n_transform', default=500000, type=int, help='Sets the amount of datapoints to use in approximating their distribution during fitting of transformer')
    parser.add_argument('--bs', default=1000, type=int, 
    help='Sets the amount of events to load at a time during fitting of transformers.'
    )
    parser.add_argument('--n_cpus', default=cpu_count(), type=int, help='Sets the amount of datapoints to use in approximating their distribution during fitting of transformer')
    parser.add_argument('--path', default='None', type=str, help='Path to shelve-file.')
    parser.add_argument('--fit_transformers', action='store_true', help='Whether or not to fit new transformers.')
    parser.add_argument('--new_name', default='None', type=str, help='Sets the new databases name.')

    args = parser.parse_args()

    if args.path == 'None':
        raise KeyError(r'A path must be supplied! Use flag --path')

    if args.new_name == 'None':
        raise KeyError(r'A new name must be supplied! Use flag --new_name')
    print(get_time(), 'Database creation initiated.')
    # * Setup - where to load data, how many events
    path_db = Path(''.join((get_project_root(), get_path_from_root(args.path))))
    path_geom_dict = str(path_db.parent)+'/dom_geom.pickle'
    path_transformer = str(path_db.parent)+'/sqlite_transformers.pickle'
    path_new_db = str(path_db.parent)+'/'+args.new_name

    # * pass to old script
    use_n_data_transform = args.n_transform if not args.dev else 100
    chunksize = args.chunksize if not args.dev else 100
    n_cpus = args.n_cpus# if not args.dev else 2
    feature_dicts = get_feature_dicts()
    geom_features = get_geom_features()
    mask = ['split_in_ice_pulses_event_length']
    BATCH_SIZE = args.bs if not args.dev else 100
    db = SqlFetcher(path_db, feature_dicts, mask=mask)
    
    # * Fit and save transformers
    if args.fit_transformers:
        transformers = fit_transformers(
            db, use_n_data_transform, feature_dicts, n_cpus=n_cpus
            )
        with open(path_transformer, 'wb') as f:
            pickle.dump(transformers, f)

    with open(path_transformer, 'rb') as f:
        transformers = pickle.load(f)
    
    # * Make database - first fetch info from raw_db
    tables_data = db.tables()

    # ! Rename stuff since they are ambiguous
    tables_data['sequential'] = convert_keys(
        tables_data['sequential'], ['event_no'], ['event']
    )

    feature_dicts_all = feature_dicts.copy()
    make_new_sql(path_new_db, tables_data, feature_dicts_all, geom_features)
    new_db = SqlFetcher(path_new_db, feature_dicts_all, mask=mask)
    
    # * Chunk IDs and process it chunk by chunk
    bounds = db.get_id_bounds()
    ids = np.arange(bounds[0], bounds[1])
    n_chunks = max(1, len(ids)//chunksize)
    chunks = np.array_split(ids, n_chunks)

    # * Extract the tables and its columns
    tables = new_db.tables()
    
    for i_chunk, id_chunk in enumerate(chunks):
        print('')
        print(get_time(), 'Processing chunk %d of %d'%(i_chunk+1, n_chunks))

        # * For each chunk, first retrieve data on the n nearest neighbors. 
        n_nearest_data = get_n_nearest_data(db, id_chunk, geom_features, path_geom_dict, n_cpus=n_cpus)

        # * Now transform data
        events = transform_events(db, id_chunk, feature_dicts, transformers, n_nearest_data, geom_features, n_cpus=n_cpus)

        print(get_time(), 'Saving chunk %d of %d'%(i_chunk+1, n_chunks))
        new_db.write_from_dict(tables, events)
        print(get_time(), 'Saved chunk!')
        
        if args.dev:
            break

    # * Finally, do some black magic that makes sqlite fast.
    # * It causes the load time to run in O(log(N))
    with sqlite3.connect(path_new_db) as db:
        cursor = db.cursor()
        cursor.execute('''CREATE INDEX sequential_idx ON sequential(event)''')
        cursor.execute('''CREATE UNIQUE INDEX scalar_idx ON scalar(event_no)''')
        cursor.execute('''CREATE UNIQUE INDEX meta_idx ON meta(event_no)''')
    print('')
    print(get_time(), 'Database creation finished!')