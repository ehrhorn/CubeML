from src.modules.helper_functions import get_dataset_size, PATH_DATA
import argparse
import numpy as np

description = 'Loops over a directory containing a dataset of h5-files and reports the total number of events.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-d', '--dataset', metavar='', type=str, default='oscnext-genie-level5-v01-01-pass2', help='Name of dataset (default: oscnext-genie-level5-v01-01-pass2).')
parser.add_argument('-t', '--particle_type', metavar='', type=str, default='any', help='Particle type (e.g. tau_neutrino)')
parser.add_argument('--mask', metavar='', type=str, default='all', help='Which mask to be applied to data. Options: all, dom_interval_min<NUM>_max<NUM>.')

args = parser.parse_args()

if __name__ == '__main__':
    
    data_dir = PATH_DATA + args.dataset +'/'
    n_files, mean, std = get_dataset_size(data_dir, particle=args.particle_type, mask_name=args.mask)
    print('Dataset:', args.dataset)
    print('# of files: %d'%(n_files))
    print('# of events: %d'%(n_files*mean))
    print('# of events pr. file: %.0f +- %.0f'%(mean, std))