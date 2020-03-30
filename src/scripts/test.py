# import torch
# from matplotlib import pyplot as plt
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# import h5py as h5
# from time import time
# from scipy.stats import norm
# import subprocess
# from multiprocessing import Pool, cpu_count

# from src.modules.classes import *
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
import numpy as np
import sqlite3
import time
from src.modules.classes import SqliteFetcher

batch_size = 256
n_experiments = 20
path_train_db = '/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/train_transformed.db'

db = SqliteFetcher(path_train_db)

tables = db.tables
for table in tables:
    print(table)
    for key in tables[table]:
        print(key)
    print('')