from src.modules.main_funcs import evaluate_model
from src.modules.helper_functions import get_time, find_files
import argparse
from pathlib import Path

description = 'Loops over a directory containing a dataset of h5-files and reports the total number of events.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', nargs='+', metavar='', type=str, help='Paths to model directories')
parser.add_argument('--predict', action='store_true', help='Whether to predict with the trained model (default: False)')
parser.add_argument('--wandb', type=int, default=1, help='Whether to log (1) to W&B or not (0) (default: 1)')
args = parser.parse_args()

if __name__ == '__main__':
    model_dirs = args.path
    if len(model_dirs) == 0:
        raise ValueError('No models supplied!')
    for model_dir in model_dirs:
        # * Locate the model directory
        paths = find_files(model_dir)
        for path in paths:
            if path.split('/')[-1] == model_dir:
                model = path
                break
        
        print('')
        print(get_time(), 'Used model: %s'%(Path(model_dir).name))
        if args.wandb:
            wandb_ID = model.split('/')[-1]
        else:
            wandb_ID = None
        evaluate_model(model, wandb_ID=wandb_ID, predict=args.predict)