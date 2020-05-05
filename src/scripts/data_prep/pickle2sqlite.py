import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import sys
import sqlite3
import pandas as pd
import time
import shutil

from src.modules.helper_functions import get_project_root, get_time, get_path_from_root, make_multiprocess_pack, convert_keys
from src.modules.reporting import make_plot
from src.modules.constants import *

def make_db(db_name, tables, rows):
    query = '{name} {format}'
    table_create_q = 'CREATE TABLE {tablename} ({queries})'

    for table, cols in zip(
        tables,
        rows
    ):
        
        # * Create new db
        queries = []
        
        for var in cols:
            q = query.format(
                name=var['name'],
                format=var['format']
            )
            queries.append(q)

        with sqlite3.connect(db_name) as db:
            cursor = db.cursor()
            q = table_create_q.format(
                tablename=table,
                queries=', '.join(queries)
            )

            cursor.execute(q)

sequential_keys = [
    {
        'name': 'row',
        'format': 'INTEGER PRIMARY KEY'
    },
    {
        'name': 'dom_x',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_y',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_z',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_charge',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_time',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_atwd',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_pulse_width',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_charge_significance',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_frac_of_n_doms',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_d_to_prev',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_v_from_prev',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_d_minkowski_to_prev',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_d_closest',
        'format': 'REAL NOT NULL'
    },
    
    {
        'name': 'dom_d_minkowski_closest',
        'format': 'REAL NOT NULL'
    },

    {
        'name': 'event',
        'format': 'INTEGER NOT NULL'
    },
    {
        'name': 'dom_key',
        'format': 'TEXT NOT NULL'
    },
    {
        'name': 'pulse_no',
        'format': 'INTEGER NOT NULL'
    },
    {
        'name': 'SplitInIcePulses',
        'format': 'INTEGER NOT NULL'
    },
    {
        'name': 'SRTInIcePulses',
        'format': 'INTEGER NOT NULL'
    }
]

scalar_keys = [
    {
        'name': 'event_no',
        'format': 'INTEGER PRIMARY KEY'
    },
    {
        'name': 'dom_timelength_fwhm',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'true_primary_direction_x',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'true_primary_direction_y',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'true_primary_direction_z',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'true_primary_position_x',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'true_primary_position_y',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'true_primary_position_z',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'true_primary_time',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'true_primary_energy',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'linefit_direction_x',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'linefit_direction_y',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'linefit_direction_z',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'linefit_point_on_line_x',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'linefit_point_on_line_y',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'linefit_point_on_line_z',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'toi_direction_x',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'toi_direction_y',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'toi_direction_z',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'toi_point_on_line_x',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'toi_point_on_line_y',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'toi_point_on_line_z',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'toi_evalratio',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'retro_crs_prefit_x',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'retro_crs_prefit_y',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'retro_crs_prefit_z',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'retro_crs_prefit_time',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'retro_crs_prefit_energy',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'retro_crs_prefit_azimuth',
        'format': 'REAL NOT NULL'
    },
    {
        'name': 'retro_crs_prefit_zenith',
        'format': 'REAL NOT NULL'
    },
]

meta_keys = [
    {
        'name': 'event_no',
        'format': 'INTEGER PRIMARY KEY'
    },
    {
        'name': 'file',
        'format': 'TEXT NOT NULL'
    },
    {
        'name': 'idx',
        'format': 'INTEGER NOT NULL'
    },
    {
        'name': 'particle_code',
        'format': 'INTEGER NOT NULL'
    },
    {
        'name': 'level',
        'format': 'INTEGER NOT NULL'
    },
    {
        'name': 'split_in_ice_pulses_event_length',
        'format': 'INTEGER NOT NULL'
    },
    {
        'name': 'srt_in_ice_pulses_event_length',
        'format': 'INTEGER NOT NULL'
    },
]

PICKLES_PATH = Path(PATH_DATA_OSCNEXT+'/pickles')
# geom_dict = pickle.load(open(PATH_DATA_OSCNEXT+'/dom_geom.pickle', 'rb'))

# loop over pickle directories and prepare for multiprocessing
row = 0
n_per_dir = 10000
all_event_nos = []
for pickle_dir in PICKLES_PATH.iterdir():
    from_ = n_per_dir*int(pickle_dir.stem)
    to_ = n_per_dir*(int(pickle_dir.stem)+1)
    all_event_nos.append(np.arange(from_, to_))
all_event_nos.sort(key=lambda x: x[0])

db_names = [
    PATH_DATA_OSCNEXT+'/train_transformed.db',
    PATH_DATA_OSCNEXT+'/val_transformed.db',
    PATH_DATA_OSCNEXT+'/test_transformed.db'
]

n_dirs = len(all_event_nos)
event_nos_split = [
    all_event_nos[0:int(0.8*n_dirs)],
    all_event_nos[int(0.8*n_dirs):int(0.9*n_dirs)],
    all_event_nos[int(0.9*n_dirs):]
]

for DB_NAME, event_nos in zip(db_names, event_nos_split):
    n_chunks = len(event_nos)
    make_db(
        DB_NAME, 
        ['sequential', 'scalar', 'meta'],
        [sequential_keys, scalar_keys, meta_keys]
    )
    ave_time = None
    for i_chunk, events in enumerate(event_nos):
        start_time = time.time()
        seq = {d['name']: [] for d in sequential_keys}
        scalar = {d['name']: [] for d in scalar_keys}
        meta = {d['name']: [] for d in meta_keys}
        print('')
        print(get_time(), 'Loading pickles...')
        
        for i_event, event in enumerate(events):
            path = '/'.join(
                [str(PICKLES_PATH), str(events[0]//n_per_dir), str(event)+'.pickle']
            )
            
            try:
                loaded = pickle.load(open(path, 'rb'))
            except FileNotFoundError:
                break

            seq_len = loaded['transform1']['dom_charge'].shape[0]
            srt_mask = np.isin(
                        np.arange(seq_len), loaded['masks']['SRTInIcePulses']
                    ).astype(int)

            for key in seq:
                
                if key in loaded['transform1']:
                    seq[key].extend(loaded['transform1'][key])
                elif key in loaded['raw']:
                    seq[key].extend(loaded['raw'][key])
                elif key == 'pulse_no':
                    seq[key].extend(
                        np.arange(seq_len)
                    )
                elif key == 'event':
                    seq[key].extend(
                        np.array([event]*seq_len)
                    )
                elif key == 'row':
                    pass
                elif key == 'SplitInIcePulses':
                    seq[key].extend(np.ones(seq_len))
                elif key == 'SRTInIcePulses':
                    seq[key].extend(srt_mask)
                elif key == 'dom_key':
                    seq[key].extend(['NotConvertedYet']*seq_len)
                else:
                    if loaded['meta']['particle_code'] in ['120000', '160000']:
                        seq[key].extend([1337.0]*seq_len)
                    else:
                        raise KeyError(
                            'A key is missing from sequential'
                            'which is not accounted for'
                        )
            
            for key in scalar:
                if key in loaded['transform1']:
                    scalar[key].append(loaded['transform1'][key])
                elif key in loaded['raw']:
                    scalar[key].append(loaded['raw'][key])
                elif key == 'event_no':
                    scalar[key].append(event)
                else:
                    raise KeyError(
                            'A key is missing from scalar'
                            'which is not accounted for'
                        )
            
            for key in meta:
                if key == 'event_no':
                    meta[key].append(event)
                elif key == 'particle_code':
                    meta[key].append(loaded['meta']['particle_code'])
                elif key == 'level':
                    meta[key].append(5)
                elif key == 'file':
                    meta[key].append(loaded['meta']['file'])
                elif key == 'idx':
                    meta[key].append(loaded['meta']['index'])
                elif key == 'split_in_ice_pulses_event_length':
                    meta[key].append(seq_len)
                elif key == 'srt_in_ice_pulses_event_length':
                    meta[key].append(np.sum(srt_mask))
                else:
                    raise KeyError(
                            'A key is missing from meta'
                            'which is not accounted for'
                        )
        

        seq['row'] = np.arange(len(seq['dom_charge']))+row
        
        # Update which row we are at. 
        row = seq['row'][-1] + 1

        # Convert to dataframe and add to database
        print(get_time(), 'Pickles loaded. Writing chunk %d/%d to database...'%(i_chunk+1, n_chunks))

        seq_df = pd.DataFrame.from_dict(seq)
        with sqlite3.connect(DB_NAME) as con:
            seq_df.to_sql('sequential', con=con, if_exists='append', index=False)

        scalar_df = pd.DataFrame.from_dict(scalar)
        with sqlite3.connect(DB_NAME) as con:
            scalar_df.to_sql('scalar', con=con, if_exists='append', index=False)
        meta_df = pd.DataFrame.from_dict(meta)

        with sqlite3.connect(DB_NAME) as con:
            meta_df.to_sql('meta', con=con, if_exists='append', index=False)
        
        print(get_time(), 'Writing finished. Deleting pickle-events.')
        path = '/'.join(
                [str(PICKLES_PATH), str(events[0]//n_per_dir)]
            )
        shutil.rmtree(path)

        chunk_time = time.time()-start_time
        if not ave_time:
            ave_time = 1.0*chunk_time
        else:
            ave_time = 0.8*ave_time + 0.2*chunk_time
        remaining = ((n_chunks-1-i_chunk)*ave_time)/3600
        print(get_time(), 'Pickles deleted.')
        print(get_time(), 'Average conversion time: %.1f seconds.'%(ave_time))
        print(get_time(), 'Time remaining: %.2f hours.'%(remaining))

    # * Finally, do some black magic that makes sqlite fast.
    # * It causes the load time to run in O(log(N))
    with sqlite3.connect(DB_NAME) as db:
        cursor = db.cursor()
        cursor.execute('''CREATE INDEX sequential_idx ON sequential(event)''')
        cursor.execute('''CREATE UNIQUE INDEX scalar_idx ON scalar(event_no)''')
        cursor.execute('''CREATE UNIQUE INDEX meta_idx ON meta(event_no)''')
    print('')
    print(get_time(), 'Database creation finished!')

