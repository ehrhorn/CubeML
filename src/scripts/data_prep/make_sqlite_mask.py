import numpy as np
from pathlib import Path
import subprocess
import pickle
import sys
import argparse
from multiprocessing import cpu_count, Pool

from src.modules.constants import *
from src.modules.classes import SqliteFetcher
from src.modules.helper_functions import get_project_root, get_path_from_root, get_time, flatten_list_of_lists, get_particle_code, make_multiprocess_pack
from src.modules.preprocessing import DomChargeScaler, EnergyNoLogTransformer
from src.modules.reporting import make_plot

PRINT_EVERY = 10000
CHUNK_SIZE = 20000
AVAILABLE_CORES = cpu_count()

description = 'Creates masks for Icecube data in a custom Sqlite-database.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--name', default='None', type=str, help='Sets the name of the mask. Options: particle_name (e.g. muon_neutrino), dom_interval, dom_interval_SRTInIcePulses, energy_interval')
parser.add_argument('--min_energy', default=0.0, type=float, help='Sets the minimum log(energy/GeV) wrt energy_interval mask creation.')
parser.add_argument('--max_energy', default=3.0, type=float, help='Sets the maximum log(energy/GeV) wrt energy_interval mask creation.')
parser.add_argument('--min_doms', default=0, type=int, help='Sets the minimum amount of DOMs wrt dom_interval mask creation.')
parser.add_argument('--max_doms', default=200, type=int, help='Sets the minimum amount of DOMs wrt dom_interval mask creation.')

args = parser.parse_args()

def make_mask(db, mask_name='all', min_doms=0, max_doms=np.inf, min_energy=-np.inf, max_energy=np.inf):

    # * Retrieve event IDs
    ids = db.ids

    # * Create mask
    if mask_name == 'dom_interval':
        mask, mask_name= make_dom_interval_mask(db, ids, min_doms, max_doms)
    
    if mask_name == 'dom_interval_SRTInIcePulses':
        mask, mask_name= make_dom_interval_mask(db, ids, min_doms, max_doms, dom_mask='SRTInIcePulses')

    elif mask_name == 'all':
        mask, mask_name= make_all_mask(db, ids)
    
    elif (
        mask_name == 'muon_neutrino' or
        mask_name == 'electron_neutrino' or
        mask_name == 'tau_neutrino'
        ):
        mask, mask_name= make_particle_mask(db, ids, mask_name)
    
    elif mask_name == 'energy_interval':
        mask, mask_name= make_energy_interval_mask(db, ids, min_energy, max_energy)

    return mask, mask_name

def make_dom_interval_mask(db, ids, min_doms, max_doms, multiprocess=True, dom_mask='SplitInIcePulses'):
    
    # * Split the candidates into chunks for multiprocessing
    if multiprocess:
        ids_chunked = np.array_split(ids, AVAILABLE_CORES)
        packed = make_multiprocess_pack(ids_chunked, db, min_doms, max_doms, dom_mask)
        
        with Pool(AVAILABLE_CORES) as p:
            accepted_lists = p.map(find_dom_interval_passed_cands, packed)
        
        # * Combine again
        mask = sorted(flatten_list_of_lists(accepted_lists))

    else:
        raise ValueError('make_dom_interval_mask: Only multiprocessing solution implemented')
    
    # * save it
    mask_name = 'dom_interval_%s_min%d_max%d'%(dom_mask, min_doms, max_doms)
    
    return mask, mask_name

def make_energy_interval_mask(db, ids, min_energy, max_energy, multiprocess=True):

    # * Split the candidates into chunks for multiprocessing
    if multiprocess:
        ids_chunked = np.array_split(ids, AVAILABLE_CORES)
        packed = make_multiprocess_pack(ids_chunked, db, min_energy, max_energy)
        
        with Pool(AVAILABLE_CORES) as p:
            accepted_lists = p.map(find_energy_interval_passed_cands, packed)
        
        # * Combine again
        mask = sorted(flatten_list_of_lists(accepted_lists))

    else:
        raise ValueError('make_energy_interval_mask: Only multiprocessing solution implemented')
    print(len(mask)/len(ids))
    
    # * save it
    mask_name = 'energy_interval_min%.1f_max%.1f'%(min_energy, max_energy)
    
    return mask, mask_name

def make_particle_mask(db, ids, particle, multiprocess=True):
    
    # * find the particle code to make mask for
    particle_code = get_particle_code(particle)
    
    # * Split the candidates into chunks for multiprocessing
    if multiprocess:
        ids_chunked = np.array_split(ids, AVAILABLE_CORES)
        packed = make_multiprocess_pack(ids_chunked, db, particle_code)
        with Pool(AVAILABLE_CORES) as p:
            accepted_lists = p.map(find_particles, packed)
        
        # * Combine again
        mask = sorted(flatten_list_of_lists(accepted_lists))
        # x = np.arange(len(mask))
        # d = {'x': [x], 'y': [mask]}
        # d['savefig'] = get_project_root()+'/MASK'+str(len(mask))+'_TESTLOL.png'
        # _ = make_plot(d)
    else:
        raise ValueError('make_particle_mask: Only multiprocessing solution implemented')
    
    mask_name = particle

    return mask, mask_name

def find_dom_interval_passed_cands(pack):
    # * Unpack
    ids, db, min_doms, max_doms, dom_mask = pack

    accepted = []
    
    # * Split into chunks
    n_chunks = len(ids)//CHUNK_SIZE
    chunks = np.array_split(ids, n_chunks)

    # * Loop over chunks
    for i_chunk, chunk in enumerate(chunks):

        # * Retrieve the '<MASK_NAME>_event_length' - 
        # * this value is the number of DOMs in an event
        if dom_mask == 'SplitInIcePulses':
            len_key = 'split_in_ice_pulses_event_length'
        elif dom_mask == 'SRTInIcePulses':
            len_key = 'srt_in_ice_pulses_event_length'
        
        data_dict = db.fetch_features(all_events=chunk, meta_features=[len_key])
        
        for event_id, event_dict in data_dict.items():
            n_doms = event_dict[len_key]
            if min_doms <= n_doms <= max_doms:
                accepted.append(int(event_id))

        # * Print for sanity
        print(get_time(), 'Processed chunk %d of %d'%(i_chunk+1, n_chunks))
        sys.stdout.flush()

    return accepted

def find_energy_interval_passed_cands(pack):
    # * Unpack
    ids, db, min_energy, max_energy = pack
    
    accepted = []
    energy_key = 'true_primary_energy'

    # * Split into chunks
    n_chunks = len(ids)//CHUNK_SIZE
    chunks = np.array_split(ids, n_chunks)

    # * Load transformer
    transformer_path = '/'.join([PATH_DATA_OSCNEXT, 'sqlite_transformers.pickle'])
    transformers = pickle.load(open(transformer_path, 'rb'))
    transformer = transformers[energy_key]

    # * Loop over chunks
    for i_chunk, chunk in enumerate(chunks):
        
        # * Fetch energy
        data_dict = db.fetch_features(all_events=chunk, scalar_features=[energy_key])
        energies_transformed = np.array(
            [data_d[energy_key] for event_id, data_d in data_dict.items()]
        )

        # * inverse transform
        energies = np.squeeze(
            transformer.inverse_transform(
                energies_transformed.reshape(-1, 1)
                )
            )

        # * add or discard
        for event_id, energy in zip(data_dict.keys(), energies):
            if min_energy <= energy <= max_energy:
                accepted.append(int(event_id))

        # * Print for sanity
        print(get_time(), 'Processed chunk %d of %d'%(i_chunk+1, n_chunks))
        sys.stdout.flush()

    return accepted

def find_particles(pack):
    # * Unpack
    ids, db, particle_code = pack
    
    accepted = []

    # * Split into chunks
    n_chunks = len(ids)//CHUNK_SIZE
    chunks = np.array_split(ids, n_chunks)

    # * Loop over chunks
    for i_chunk, chunk in enumerate(chunks):

        # * Retrieve the 'particle_code' from meta -
        # * this value determines the particle
        code_name = 'particle_code'
        data_dict = db.fetch_features(all_events=chunk, meta_features=[code_name])
        for event_id, event_dict in data_dict.items():
            code = event_dict[code_name]
            if str(code) == particle_code:
                accepted.append(int(event_id))

        # * Print for sanity
        print(get_time(), 'Processed chunk %d of %d'%(i_chunk+1, n_chunks))
        sys.stdout.flush()

    return accepted

def make_all_mask(db, ids):
    # * Make mask - a list of indices
    mask = [int(e) for e in ids]
    mask_name = 'all'
    
    return mask, mask_name

if __name__ == '__main__':
    
    print(get_time(), 'Mask creation initiated.')
    
    # * Extract all parsed information

    # * Options: particle_name (e.g. muon_neutrino), dom_interval, energy_interval
    mask_name = args.name
    if mask_name == 'None':
        raise KeyError('Must parse a name!')
    
    min_doms = args.min_doms
    max_doms = args.max_doms
    
    min_energy = args.min_energy
    max_energy = args.max_energy
    mask_dict = {'mask_name': mask_name, 'min_doms':
        min_doms, 'max_doms': max_doms, 'min_energy': min_energy, 'max_energy': max_energy}

    # * If maskdirectory doesn't exist, make it
    mask_dir = '/'.join([PATH_DATA_OSCNEXT, 'masks'])
    if not Path(mask_dir).exists(): 
        Path(mask_dir).mkdir()

    # * Loop over different DBs
    for path, ext in zip(
        [PATH_TRAIN_DB, PATH_VAL_DB], 
        ['_train.pickle', '_val.pickle']
    ):
        
        db = SqliteFetcher(path)
        print(get_time(), '%s mask calculation begun.'%(mask_dict['mask_name']))
        mask, mask_name = make_mask(db, **mask_dict)
        mask_path = '/'.join([mask_dir, mask_name+ext])
        with open(mask_path, 'wb') as f:
            pickle.dump(mask, f)
        print(get_time(), 'Mask created at', mask_path, '\n')
