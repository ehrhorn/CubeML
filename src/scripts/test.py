# import torch
# from matplotlib import pyplot as plt
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# import h5py as h5
# from time import time
# from scipy.stats import norm
# import subprocess
# from multiprocessing import Pool, cpu_count

from src.modules.classes import *
from pathlib import Path
# import src.modules.loss_funcs as lf
# import src.modules.helper_functions as hf
# from src.modules.eval_funcs import *
# import src.modules.reporting as rpt
# from src.modules.classes import *
# from src.modules.preprocessing import *
# # import src.modules.preprocessing as pp
# from src.modules.main_funcs import *
# import shelve
# import sys
# from time import sleep

path_pickles = Path('/'.join([PATH_DATA_OSCNEXT, 'pickles', '757']))
events = sorted([entry.stem for entry in path_pickles.iterdir()])
# events = [entry.stem for entry in path_pickles.iterdir()]
events = events[105:200]
db = SqliteFetcher(PATH_TRAIN_DB)

seq_feats = ['dom_charge', 
            'dom_x', 
            'dom_y', 
            'dom_z', 
            'dom_time', 
            'dom_charge_significance',
            'dom_frac_of_n_doms',
            'dom_d_to_prev',
            'dom_v_from_prev',
            'dom_d_minkowski_to_prev',
            'dom_d_closest',
            'dom_d_minkowski_closest',
    ]

scalar_feats = ['tot_charge',
                'dom_timelength_fwhm',
    ]

targets = ['true_primary_energy', 'true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z', 'true_primary_time', 'true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z']

batch = db.make_batch(
    ids=events, 
    scalars=scalar_feats, 
    seqs=seq_feats, 
    targets=targets,
    weights=['inverse_performance_muon_energy'], 
    mask=['SplitInIcePulses']
)

for i_event, event in enumerate(events):
    pickle_path = str(path_pickles)+'/'+event+'.pickle'
    with open(pickle_path, 'rb') as f:
        pickle_event = pickle.load(f)
    if pickle_event['meta']['particle_code'] != '140000':
        continue
    transformed_pickle = pickle_event['transform1']
    break
pickle_charge = transformed_pickle['dom_charge']
pickle_energy = transformed_pickle['true_primary_energy']
print(pickle_charge.shape)
sql_charge = batch[0][0][0, :]
sql_energy = batch[0][2][0]
print(sql_charge, pickle_charge-sql_charge)
print(sql_energy-pickle_energy)

#     for key in seq_feats:
#         # for key2 in sqlite_event:
#             # print(key2)
#         # a+=1
#         d1 = sqlite_event[key] 
#         d2 = transformed_pickle[key]
#         data[key].append(d1-d2)
#         random.shuffle(d1)
#         data_random[key].append(d1-d2)

# for key in seq_feats:
#     data[key] = flatten_list_of_lists(data[key])
#     data_random[key] = flatten_list_of_lists(data_random[key])
#     d = {'data': [np.clip(data[key], -5, 5), np.clip(data_random[key], -5, 5)]}
#     d['savefig'] = get_project_root() +'/'+key
#     _ = make_plot(d)
#     for i_seq_feat, seq_feat in enumerate(seq_feats):

# for key in seq_feat:

#     # Check if linearly dependent.
#     # Sort
#     zipped = [e for e in zip(d1, d2)]
#     # sorted_batch = sorted(zipped, key=lambda x: x[0])
#     # d1 = [e[0] for e in sorted_batch]
#     # d2 = [e[1] for e in sorted_batch]
#     d = {'x': [d1], 'y': [d2]}
#     d['xlabel'] = 'Sql'
#     d['ylabel'] = 'Pickle'
#     d['title'] = key
#     d['savefig'] = get_project_root()+'/reports/sqlite_pickle_comparison/'+key+'.png'
#     _ = make_plot(d)
    
    # print(key)
    # for e1, e2 in zip(d1, d2):
    #     print(round(e1, 3), round(e2, 3))
    
    # # diff = (d1-d2)/d1
    # # print(d1)
    # # print(d2)
    # print('')
