from src.modules.helper_functions import get_dataset_size
import argparse

description = 'Loops over a directory containing a dataset of h5-files and reports the total number of events.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', metavar='', type=str, help='Path to dataset')
args = parser.parse_args()

if __name__ == '__main__':
    if args.path != None:
        data_dir = args.path
    if 'data_dir' not in locals():
        raise ValueError('No path supplied!')

    dataset_name = data_dir.split('/')[-1]
    n_files, mean, std = get_dataset_size(data_dir)

    print('Dataset:', dataset_name)
    print('# of files: %d'%(n_files))
    print('# of events: %d'%(n_files*mean))
    print('# of events pr. file: %.0f +- %.0f'%(mean, std))