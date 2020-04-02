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

seq_feat = ['dom_charge', 
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

targets = ['true_primary_energy', 'true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z', 'true_primary_time', 'true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z']

pickle_path = PATH_DATA_OSCNEXT + '/0.pickle'
db = SqliteFetcher(PATH_TRAIN_DB)
tables = db.tables
for key, table in tables.items():
    print(key)
    for key in table:
        print(key)
    print('')

sqlite = db.fetch_features(all_events=[0], scalar_features=targets, seq_features=seq_feat)
sqlite_event = sqlite['0']
with open(pickle_path, 'rb') as f:
    pickle_event = pickle.load(f)

transformed_pickle = pickle_event['transform1']

for key in transformed_pickle:
    print(key)