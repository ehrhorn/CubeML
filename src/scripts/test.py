# import torch
# from matplotlib import pyplot as plt
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
# from time import time
# from scipy.stats import norm
# import subprocess
# from multiprocessing import Pool, cpu_count

# from src.modules.classes import *
# from pathlib import Path
# from src.modules.reporting import *
# from src.modules.constants import *
# import src.modules.loss_funcs as lf
from src.modules.helper_functions import *
# from src.modules.eval_funcs import *
from src.modules.reporting import *
from src.modules.classes import *
# from src.modules.preprocessing import *
# # import src.modules.preprocessing as pp
# from src.modules.main_funcs import *
# import shelve
# import sys
# from time import sleep
import numpy as np
import pickle

from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
from src.modules.thesis_plotting import *
import os
from src.modules.retro_funcs import *
import time

PATH_META_DB = PATH_DATA_OSCNEXT + '/epsilon_bjorn.db'
with open(PATH_DATA_OSCNEXT + '/matched_val.pickle', 'rb') as f:
    found = pickle.load(f)

muons_path = PATH_DATA_OSCNEXT + '/masks/muon_neutrino_val.pickle'
with open(muons_path, 'rb') as f:
    muons = [str(e) for e in pickle.load(f)]

# new_mask = []
# for e in muons:
#     if e in found:
#         if found[e]['interaction_type'] == 1: # CC
#             new_mask.append(int(e))

# muons_CC_val_path = PATH_DATA_OSCNEXT + '/masks/muon_CC_neutrino_val.pickle'
# with open(muons_CC_val_path, 'wb') as f:
#     pickle.dump(new_mask, f)
val_ids = []
meta_ids = []
for key, data in found.items():
    val_ids.append(key)
    meta_ids.append(data['event_no_meta'])

print(len(val_ids)/len(muons))

# wanted = [
#     'true_primary_direction_x',
#     'true_primary_direction_y',
#     'true_primary_direction_z',
#     'event_no'
# ]
# with sqlite3.connect(PATH_VAL_DB) as db:
#     query = 'SELECT {features} FROM scalar WHERE event_no IN ({events})'.format(
#         features=', '.join(wanted),
#         events=', '.join(['?'] * len(val_ids))
#         )
#     cursor = db.cursor()
#     cursor.execute(query, val_ids)
#     data_val_tupled = cursor.fetchall()

# with sqlite3.connect(PATH_META_DB) as db:
#     query = 'SELECT {features} FROM features WHERE event_no IN ({events})'.format(
#         features=', '.join(wanted),
#         events=', '.join(['?'] * len(meta_ids))
#         )
#     cursor = db.cursor()
#     cursor.execute(query, meta_ids)
#     data_meta_tupled = cursor.fetchall()

# for val, meta in zip(data_val_tupled, data_meta_tupled):
#     print(val, meta)
#     print('')