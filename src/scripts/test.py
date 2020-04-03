# import torch
# from matplotlib import pyplot as plt
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# import h5py as h5
# from time import time
# from scipy.stats import norm
# import subprocess
# from multiprocessing import Pool, cpu_count

from src.modules.classes import *
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

# seq_feat = ['dom_charge', 
#             'dom_x', 
#             'dom_y', 
#             'dom_z', 
#             'dom_time', 
#             'dom_charge_significance',
#             'dom_frac_of_n_doms',
#             'dom_d_to_prev',
#             'dom_v_from_prev',
#             'dom_d_minkowski_to_prev',
#             'dom_d_closest',
#             'dom_d_minkowski_closest',
#     ]

# targets = ['true_primary_energy', 'true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z', 'true_primary_time', 'true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z']
# np.set_printoptions(precision=3)
# event = 10000001
# pickle_path = PATH_DATA_OSCNEXT + '/10000001.pickle'
# db = SqliteFetcher(PATH_VAL_DB)
# sqlite = db.fetch_features(all_events=[event], scalar_features=targets, seq_features=seq_feat)
# sqlite_event = sqlite[str(event)]
# with open(pickle_path, 'rb') as f:
#     pickle_event = pickle.load(f)

# transformed_pickle = pickle_event['transform1']

# for key in seq_feat:
#     d1, d2 = sqlite_event[key], transformed_pickle[key]

#     # Check if linearly dependent.
#     # Sort
#     zipped = [e for e in zip(d1, d2)]
#     sorted_batch = sorted(zipped, key=lambda x: x[0])
#     d1_sorted = [e[0] for e in sorted_batch]
#     d2_sorted = [e[1] for e in sorted_batch]
#     d = {'x': [d1_sorted], 'y': [d2_sorted]}
#     d['xlabel'] = 'Sql'
#     d['ylabel'] = 'Pickle'
#     d['title'] = key
#     d['savefig'] = get_project_root()+'/reports/sqlite_pickle_comparison/'+key+'.png'
#     _ = make_plot(d)
    
#     # print(key)
#     # for e1, e2 in zip(d1, d2):
#     #     print(round(e1, 3), round(e2, 3))
    
#     # # diff = (d1-d2)/d1
#     # # print(d1)
#     # # print(d2)
#     # print('')

print('whatup')