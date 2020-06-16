from src.modules.main_funcs import calc_raw_predictions
from src.modules.helper_functions import (
    get_time, locate_model, convert_keys, remove_dots_and_lines
)
from src.modules.classes import SqliteFetcher
import argparse
from src.modules.constants import *
from pathlib import Path
import numpy as np

description = 'Loops over a directory containing a dataset of h5-files and reports the total number of events.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    '-p', 
    '--path', 
    nargs='+', 
    metavar='', 
    type=str, 
    help='Paths to model directories'
    )
parser.add_argument(
    '--prediction_keys', 
    nargs='+', 
    metavar='', 
    type=str, 
    help='Paths to model directories'
    )
parser.add_argument(
    '-n', 
    '--n_predictions_wanted', 
    type=int,
    default=np.inf, 
    help='Number of wanted predictions'
    )
args = parser.parse_args()

if __name__ == '__main__':
    
    model_dirs = args.path
    
    if len(model_dirs) == 0:
        raise ValueError('No models supplied!')
    if args.prediction_keys is None:
        raise ValueError('Wanted prediction keys must be supplied!')

    for model_dir in model_dirs:
        
        # Locate the model directory
        model = locate_model(model_dir)
        model_name = Path(model_dir).name
        
        print('')
        print(get_time(), 'Used model: %s'%(model_name))

        
        for path in [PATH_TRAIN_DB, PATH_VAL_DB]:
            preds, indices = calc_raw_predictions(
                model, 
                n_predictions_wanted=args.n_predictions_wanted, 
                db_path=path
                )
            
            predictions = {}
            for key in args.prediction_keys:
                predictions[key] = preds[key]

            indices = [str(entry) for entry in indices]
            db = SqliteFetcher(path)
            keys = [key for key in predictions]
            new_keys = [key + '_' + remove_dots_and_lines(model_name) for key in predictions]
            predictions_newnames = convert_keys(predictions, keys, new_keys)
            print(get_time(), 'Saving to db...')
            for name, values in predictions_newnames.items():
                db.write('scalar', name, indices, values, astype='REAL')
            print(get_time(), 'Data saved.')
        
        
        