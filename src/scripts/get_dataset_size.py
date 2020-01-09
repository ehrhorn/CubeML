from src.modules.helper_functions import get_dataset_size
import argparse
import numpy as np

description = 'Loops over a directory containing a dataset of h5-files and reports the total number of events.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', metavar='', type=str, help='Path to dataset')
parser.add_argument('-t', '--particle_type', metavar='', type=str, default='any', help='Particle type (e.g. tau_neutrino)')
parser.add_argument('--mask', metavar='', type=str, default='all', help='Which mask to be applied to data. Some masks require additional keyword arguments. Options: all, dom_interval.')
parser.add_argument('--min_doms', metavar='', type=int, default=0, help='Minimum number of activated DOMs in an event for the event to be considered')
parser.add_argument('--max_doms', metavar='', type=int, default=np.inf, help='Maximum number of activated DOMs in an event for the event to be considered')


args = parser.parse_args()

if __name__ == '__main__':
    if args.path != None:
        data_dir = args.path
    if 'data_dir' not in locals():
        raise ValueError('No path supplied!')
    
    mask_dict = {'mask_name': args.mask, 'min_doms': args.min_doms, 'max_doms': args.max_doms}

    dataset_name = data_dir.split('/')[-1]
    n_files, mean, std = get_dataset_size(data_dir, particle=args.particle_type, mask_dict=mask_dict)
    
    print('Dataset:', dataset_name)
    print('# of files: %d'%(n_files))
    print('# of events: %d'%(n_files*mean))
    print('# of events pr. file: %.0f +- %.0f'%(mean, std))