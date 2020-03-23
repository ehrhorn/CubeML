import argparse
import pickle
import h5py as h5
import wandb

from src.modules.helper_functions import find_files, get_time
from src.modules.main_funcs import (calc_predictions_pickle, 
load_model_pars, get_project_root)

description = 'Runs all experiments saved in "/experiments" starting with the oldest.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', nargs='+', metavar='', type=str, help='Paths to model directories')


args = parser.parse_args()

if __name__ == '__main__':

    for model in args.path:
        
        # * Locate the model directory
        paths = find_files(model)
        for path in paths:
            if path.split('/')[-1] == model:
                break
        hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(path)
        WANDB_DIR = get_project_root()+'/models'
        PROJECT = meta_pars['project']
        WANDB_ID = path.split('/')[-1]
        wandb.init(resume=True, id=WANDB_ID, dir=WANDB_DIR, project=PROJECT)

        # * Load the model and check all predictions are there
        perf_class_path = path +'/data/Performance.pickle'
        perf_class = pickle.load( open( perf_class_path, "rb" ) )
        perf_class.update_onenumber_performance()
        perf = perf_class.onenumber_performance
        print(get_time(), 'Onenumber performance: %.3f'%(perf))

        wandb.config.update({'Performance': perf}, allow_val_change=True)
        wandb.log()
        wandb.join()


    