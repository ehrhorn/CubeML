import numpy as np
import sqlite3
import time

n_tests = 100
batch_size = 128
path_train_db = '/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/train_transformed.db'

seq =  [
    'dom_charge', 
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
    'dom_d_minkowski_closest'
    ]

scalar = [
    'dom_timelength_fwhm',
    'tot_charge'
    ]

target = [
    'true_primary_energy', 
    'true_primary_position_x', 
    'true_primary_position_y', 
    'true_primary_position_z', 
    'true_primary_time', 
    'true_primary_direction_x', 
    'true_primary_direction_y', 
    'true_primary_direction_z'
    ]

events = [str(e) for e in np.arange(batch_size*n_tests)]

query = 'SELECT {features} FROM {table} WHERE {discriminator} IN ({events})'

start = time.time()
seq_time, scalar_time, target_time = 0.0, 0.0, 0.0
for i_test in range(n_tests):

    ids = events[i_test*batch_size:(i_test+1)*batch_size]
    with sqlite3.connect(path_train_db) as db:

        cursor = db.cursor()

        seq_start = time.time()
        query = query.format(
            features=', '.join(seq),
            table='sequential',
            discriminator='event',
            events=', '.join(['?'] * len(ids))
            )
        cursor.execute(query, ids)
        fetched_seq = cursor.fetchall()
        seq_end = time.time()
        seq_time += seq_end-seq_start

        scalar_start = time.time()
        query = query.format(
            features=', '.join(scalar),
            table='scalar',
            discriminator='event_no',
            events=', '.join(['?'] * len(ids))
            )
        cursor.execute(query, ids)
        fetched_scalar = cursor.fetchall()
        scalar_end = time.time()
        scalar_time += scalar_end-scalar_start

        target_start = time.time()
        query = query.format(
            features=', '.join(target),
            table='scalar',
            discriminator='event_no',
            events=', '.join(['?'] * len(ids))
            )
        cursor.execute(query, ids)
        fetched_target = cursor.fetchall()
        target_end = time.time()
        target_time += target_end-target_start


end = time.time()
ave = (end-start)/n_tests
events_per_seq = batch_size/ave

ave_seq = seq_time/n_tests
ave_scalar = scalar_time/n_tests
ave_target = target_time/n_tests
print('Ran %d experiments with batchsize %d.'%(n_tests, batch_size))
print('Average fetch time: %.4f seconds'%(ave))
print('Fetched events per sec: %d'%(events_per_seq))
print('')
print('Average seq fetch time: %.4f seconds'%(ave_seq))
print('Average scalar fetch time: %.4f seconds'%(ave_scalar))
print('Average target fetch time: %.4f seconds'%(ave_target))


# # import torch
# # from matplotlib import pyplot as plt
# # from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# # import h5py as h5
# # from time import time
# # from scipy.stats import norm
# # import subprocess
# # from multiprocessing import Pool, cpu_count

# # from src.modules.classes import *
# # import src.modules.loss_funcs as lf
# # import src.modules.helper_functions as hf
# # from src.modules.eval_funcs import *
# # import src.modules.reporting as rpt
# # from src.modules.classes import *
# # from src.modules.preprocessing import *
# # # import src.modules.preprocessing as pp
# # from src.modules.main_funcs import *
# # import shelve
# # import sys
# # from time import sleep
# import numpy as np
# import sqlite3
# import time

# n_tests = 20
# batch_size = 256
# path_train_db = '/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/train_transformed.db'

# seq =  [
#     'dom_charge', 
#     'dom_x', 
#     'dom_y', 
#     'dom_z', 
#     'dom_time', 
#     'dom_charge_significance',
#     'dom_frac_of_n_doms',
#     'dom_d_to_prev',
#     'dom_v_from_prev',
#     'dom_d_minkowski_to_prev',
#     'dom_d_closest',
#     'dom_d_minkowski_closest'
#     ]

# scalar = [
#     'dom_timelength_fwhm',
#     'tot_charge'
#     ]

# target = [
#     'true_primary_energy', 
#     'true_primary_position_x', 
#     'true_primary_position_y', 
#     'true_primary_position_z', 
#     'true_primary_time', 
#     'true_primary_direction_x', 
#     'true_primary_direction_y', 
#     'true_primary_direction_z'
#     ]
# meta = ['split_in_ice_pulses_event_length']

# events = [str(e) for e in np.arange(batch_size*n_tests)]

# scalar_query = 'SELECT {features} FROM scalar INNER JOIN meta ON scalar.event_no=meta.event_no WHERE scalar.event_no IN ({events})'

# seq_query = 'SELECT {features} FROM sequential WHERE event IN ({events})'

# start = time.time()
# seq_time, scalar_time, target_time = 0.0, 0.0, 0.0

# feats = ['scalar.'+e for e in target] + ['meta.'+e for e in meta]

# for i_test in range(n_tests):

#     ids = events[i_test*batch_size:(i_test+1)*batch_size]
#     with sqlite3.connect(path_train_db) as db:

#         cursor = db.cursor()

#         seq_start = time.time()
#         query = seq_query.format(
#             features=', '.join(seq),
#             events=', '.join(['?'] * len(ids))
#             )
#         cursor.execute(query, ids)
#         fetched_seq = cursor.fetchall()
#         seq_end = time.time()
#         seq_time += seq_end-seq_start

#         scalar_start = time.time()
#         query = scalar_query.format(
#             features=', '.join(feats),
#             events=', '.join(['?'] * len(ids))
#             )
#         cursor.execute(query, ids)
#         fetched_scalars = cursor.fetchall()
#         scalar_end = time.time()
#         scalar_time += scalar_end-scalar_start

# end = time.time()
# ave = (end-start)/n_tests
# events_per_seq = batch_size/ave

# ave_seq = seq_time/n_tests
# ave_scalar = scalar_time/n_tests
# ave_target = target_time/n_tests
# print('Ran %d experiments with batchsize %d.'%(n_tests, batch_size))
# print('Average fetch time: %.4f seconds'%(ave))
# print('Fetched events per sec: %d'%(events_per_seq))
# print('')
# print('Average seq fetch time: %.4f seconds'%(ave_seq))
# print('Average scalar fetch time: %.4f seconds'%(ave_scalar))
# print('Average target fetch time: %.4f seconds'%(ave_target))