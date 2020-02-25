# from src.modules.main_funcs import evaluate_model
from src.modules.helper_functions import get_time, locate_model
from src.modules.reporting import FeaturePermutationImportance
import argparse
from pathlib import Path

description = 'Loops over a directory containing a dataset of h5-files and reports the total number of events.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', nargs='+', metavar='', type=str, help='Paths to model directories')
parser.add_argument('--predict', action='store_true', help='Whether to predict with the trained model (default: False)')
parser.add_argument('--wandb', type=int, default=0, help='Whether to log (1) to W&B or not (0) (default: 0)')
args = parser.parse_args()

if __name__ == '__main__':
    model_dirs = args.path
    if len(model_dirs) == 0:
        raise ValueError('No models supplied!')
    for model_dir in model_dirs:  
        # * Locate the model directory
        model = locate_model(model_dir)

        print('')
        print(get_time(), 'Used model: %s. Calculating Permutation Feature Importance.'%(Path(model_dir).name))
        if args.wandb:
            wandb_ID = model.split('/')[-1]
        else:
            wandb_ID = None


        # save_dir = '/home/bjoern/Thesis/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/energy_reg/test_2020.02.21-12.21.14'
        fpi = FeaturePermutationImportance(model)
        fpi.calc_all_seq_importances()
        fpi.save()
        # features =  [['dom_x'], ['dom_charge'],['dom_z'],['dom_y'], ['dom_time']]
        # a.calc_all_seq_importances()