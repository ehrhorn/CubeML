# from src.modules.main_funcs import evaluate_model
from src.modules.helper_functions import get_time, locate_model
from src.modules.reporting import FeaturePermutationImportance
import argparse
from pathlib import Path
from numpy import inf

description = 'Loops over a directory containing a dataset of h5-files and reports the total number of events.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', nargs='+', metavar='', type=str, help='Paths to model directories')
parser.add_argument('-n', '--n_predictions', type=int, default=-1, help='Number of predictions to use in calculating FPI')
parser.add_argument(
    '--seq_features', 
    nargs='+', 
    default='None', 
    type=str, 
    help='Sets the masks to choose data. Options:'\
    'dom_interval_SplitInIcePulses_min0_max200,'\
    'dom_interval_SRTInIcePulses_min0_max200, muon_neutrino,'\
    'energy_interval_min0.0_max3.0')
parser.add_argument(
    '--seq_permutation', 
    action='store_true',
    help='Randomizes the DOMs in the interval [from_frac, to_frac] in sequences.'\
         'From_frac and to_frac must be supplied')

parser.add_argument('--steps', type=int, default=10, help='The fractional size of the scanning window')

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
        
        # Default to calculate for all features
        n_wanted = args.n_predictions if args.n_predictions != -1 else inf
        fpi = FeaturePermutationImportance(model)
        
        if args.seq_features == 'all':
            fpi.calc_all_seq_importances(n_predictions_wanted=n_wanted)
            fpi.save()
        elif args.seq_features != 'None':
            for feature in args.seq_features:
                fpi.calc_seq_feature_importance(
                    n_predictions_wanted=n_wanted, feature=feature)
            
            fpi.save()
        elif args.seq_permutation != 'None':
            fpi.calc_seq_permutation_importance(
                steps=args.steps,
                n_predictions_wanted=n_wanted
            )
            fpi.save()