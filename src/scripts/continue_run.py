import json
from pathlib import Path
import argparse

from time import localtime, strftime
from src.modules.helper_functions import locate_model, get_project_root
from src.modules.main_funcs import run_experiment

description = 'Continues a training - a crashed or non-finished model.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--tags', nargs='+', default='', type=str, help='Tags a run for easier comparisons on W&B')
parser.add_argument('--continue_crashed', default='None', type=str, help='Sets the model to continue training a crashed run.')
parser.add_argument('--continue_training', default='None', type=str, help='Sets the model to continue training a crashed run.')
parser.add_argument('-r', '--run', action='store_true', help='Runs experiment immediately.')

args = parser.parse_args()

if __name__ == '__main__':

    #* ======================================================================== 
    #* DEFINE SCRIPT OBJECTIVE
    #* ========================================================================

    # * Options: 'train_new', 'continue_training', 'explore_lr', 'continue_crashed'
    objective = 'train_new'
    if args.continue_crashed != 'None':
        objective = 'continue_crashed'
    elif args.continue_training != 'None':
        objective = 'continue_training'
    else:
        raise KeyError('An objective must be set!')

    meta_pars = {'objective':            objective,
                'pretrained_path':      locate_model(args.continue_training),
                'crashed_path':         locate_model(args.continue_crashed),
                }

    #* ======================================================================== 
    #* SAVE SETTINGS
    #* ========================================================================

    json_dict = {'hyper_pars': {}, 'data_pars': {}, 'arch_pars': {}, 'meta_pars': meta_pars}
    exp_dir = get_project_root() + '/experiments/'

    # * Finally! Make model directory 
    base_name = strftime("%Y-%m-%d-%H.%M.%S", localtime())
    exp_name = exp_dir+base_name+'.json'
    with open(exp_name, 'w') as name:
        json.dump(json_dict, name)
    
    if args.run:
        run_experiment(exp_name)

