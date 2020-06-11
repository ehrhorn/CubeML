# import torch
# from matplotlib import pyplot as plt
# from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
# import h5py as h5
# from time import time
# from scipy.stats import norm
# import subprocess
# from multiprocessing import Pool, cpu_count

# from src.modules.classes import *
from pathlib import Path
# from src.modules.reporting import *
# from src.modules.constants import *
# import src.modules.loss_funcs as lf
import src.modules.helper_functions as hf
import subprocess
# from src.modules.eval_funcs import *
# from src.modules.preprocessing import *
# # import src.modules.preprocessing as pp
# from src.modules.main_funcs import *
# import shelve
# import sys
# from time import sleep
for path in Path(hf.get_project_root() + '/reports/thesis_plots').iterdir():
    if path.is_dir():
        if path.name == 'all_pgf':
            continue
        print(hf.get_time(), 'Running', path.name)
        runpath = str(path) + '/script.py'
        subprocess.call(['python', runpath])
        print('')
