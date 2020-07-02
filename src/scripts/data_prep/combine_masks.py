from src.modules.constants import *
from src.modules.helper_functions import load_pickle_mask, mask_union

import argparse
import pickle

description = 'Combines masks to make a new mask.'
parser = argparse.ArgumentParser(description=description)
# parser.add_argument('-d', '--dev', action='store_true', help='Initiates developermode - weights are not saved, and only weights for a small subset of the dataset is calculated')

parser.add_argument(
    '--masks', 
    nargs='+', 
    type=str, 
    help='Sets which masks to combine'
    ) 

parser.add_argument(
    '--name', 
    type=str, 
    help='Sets name of new mask'
    ) 


args = parser.parse_args()

if __name__ == '__main__':
    if args.name is None:
        raise NameError('A name must be set for the new mask')

    combined = []
    for path, keyword in zip(
        [PATH_TRAIN_DB, PATH_VAL_DB], 
        ['train', 'val'],
    ):  
        
        # Create union
        db_specific_mask = [mask+'_'+keyword for mask in args.masks]
        combined = mask_union(PATH_DATA_OSCNEXT, db_specific_mask)
        
        # Save new mask
        new_name = (
            PATH_DATA_OSCNEXT + 
            '/masks/' + 
            args.name + '_' + keyword + '.pickle'
            )
        with open(new_name, 'wb') as f:
            pickle.dump(combined, f)
