from src.modules.helper_functions import get_time, make_multiprocess_pack, flatten_list_of_lists
import numpy as np
from src.modules.constants import *
import time
import sys
from multiprocessing import Pool
import sqlite3
import pickle
from functools import reduce

PATH_META_DB = PATH_DATA_OSCNEXT + '/epsilon_bjorn.db'
PRINT_EVERY = 1000

def exec_query(db_path, query, ids=None):
    with sqlite3.connect(db_path) as db:
        cursor = db.cursor()
        cursor.execute(query)
        data = cursor.fetchall()

    return data

def multiprocess_match_indices(pack):
    event_nos, data_meta, subprocess_id = pack
    
    x_meta = data_meta[:, 0]
    y_meta = data_meta[:, 1]
    z_meta = data_meta[:, 2]
    event_no_meta = data_meta[:, 3]
    
    wanted_scalar_val = [
        'true_primary_direction_x',
        'true_primary_direction_y',
        'true_primary_direction_z',
        'event_no'
    ]

    with sqlite3.connect(PATH_VAL_DB) as db:
        query = 'SELECT {features} FROM scalar WHERE event_no IN ({events})'.format(
        features=', '.join(wanted_scalar_val),
        events=', '.join(['?'] * len(event_nos))
        )
        cursor = db.cursor()
        cursor.execute(query, [str(e) for e in event_nos])
        data_val_tupled = cursor.fetchall()

        data_val = np.array(data_val_tupled)
        x_val = data_val[:, 0]
        y_val = data_val[:, 1]
        z_val = data_val[:, 2]
        event_no_val = data_val[:, 3]

    # Loop and identify
    n_val = len(event_no_val)
    val_to_meta = [(-1, -1) for i in range(n_val)]
    for i_val in range(n_val):
        if (i_val)%PRINT_EVERY == 0 and subprocess_id == 0:
            print(
                get_time(), 'Subprocess %d: Processed %d of %d'%(subprocess_id, i_val, n_val)
            )
            sys.stdout.flush()

        res1 = np.flatnonzero(
            x_val[i_val] == x_meta
        )
        if len(res1) == 1:
            val_to_meta[i_val] = (int(event_no_val[i_val]), int(event_no_meta[i_val]))
            continue
        
        res2 = np.flatnonzero(
            y_val[i_val] == y_meta
        )
        if len(res2) == 1:
            val_to_meta[i_val] = (int(event_no_val[i_val]), int(event_no_meta[i_val]))
            continue

        res3 = np.flatnonzero(
            y_val[i_val] == y_meta
        )
        if len(res3) == 1:
            val_to_meta[i_val] = (int(event_no_val[i_val]), int(event_no_meta[i_val]))
            continue
        
        res4 = reduce(np.intersect1d, (res1, res2, res3))
        
        if len(res4) == 1:
            val_to_meta[i_val] = (int(event_no_val[i_val]), int(event_no_meta[i_val]))
    
    return val_to_meta


wanted_scalar_val = [
    'true_primary_direction_x',
    'true_primary_direction_y',
    'true_primary_direction_z',
]

wanted_meta_val = [
    'event_no'
]

query = 'SELECT event_no FROM meta'
data_val = [e[0] for e in exec_query(PATH_VAL_DB, query)]

# Create packs - loop over all events
event_no_val_chunks = np.array_split(data_val, AVAILABLE_CORES)

# Fetch from meta and prep for multiprocess
wanted = [
    'true_primary_direction_x',
    'true_primary_direction_y',
    'true_primary_direction_z',
    'event_no',
]
query = 'SELECT {features} FROM features'.format(
    features=', '.join(wanted)
)

start = time.time()
data_meta_tupled = exec_query(PATH_META_DB, query)
data_meta = np.array(data_meta_tupled)
end = time.time()
print(get_time(), 'Time spent fetching from meta: %d seconds'%(end-start))
packed = make_multiprocess_pack(event_no_val_chunks, data_meta, enumerate_processes=True)

with Pool(AVAILABLE_CORES) as p:
    matches = flatten_list_of_lists(
        p.map(multiprocess_match_indices, packed)
    )

with open(PATH_DATA_OSCNEXT + '/matched_val.pickle', 'wb') as f:
    pickle.dump(matches, f)

end = time.time()
print(get_time(), 'Time spent matching: %d seconds'%(end-start))