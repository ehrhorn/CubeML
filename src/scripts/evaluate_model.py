from src.modules.main_funcs import evaluate_model
from src.modules.helper_functions import find_files
import argparse

description = 'Loops over a directory containing a dataset of h5-files and reports the total number of events.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', metavar='', type=str, help='Path to model directory')
args = parser.parse_args()

if __name__ == '__main__':
    if args.path != None:
        model_dir = args.path
    if 'model_dir' not in locals():
        raise ValueError('No path supplied!')
    
    model_dir = find_files(model_dir)
    
    if len(model_dir ) > 1:
        for name in model_dir:
            print(name)
        raise ValueError('Several models with name %s'%(args.path))
    else:
        model_dir = model_dir[0]
    print(model_dir)
    wandb_ID = model_dir.split('/')[-1]
    evaluate_model(model_dir, wandb_ID=wandb_ID)