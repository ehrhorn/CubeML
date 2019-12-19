import src.modules.main_funcs as mf
import src.modules.helper_functions as hf
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
    
    #* Locate the model directory
    paths = hf.find_files(model_dir)
    for path in paths:
        if path.split('/')[-1] == model_dir:
            model = path
            break
    
    print(model)
    wandb_ID = model.split('/')[-1]
    print(wandb_ID)
    mf.evaluate_model(model, wandb_ID=wandb_ID)