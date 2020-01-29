from pathlib import Path
from time import time
from multiprocessing import cpu_count, Pool
import sys
import subprocess

from src.modules.helper_functions import get_project_root, get_time

def make_tar(pack):
    pickle_dir, tar_dir = pack

    print(get_time(), 'Making tar of %s'%(pickle_dir))
    sys.stdout.flush()

    tar_path = tar_dir + '/' + pickle_dir.name + '.tar'
    subprocess.run(['tar', '-cf', tar_path, pickle_dir])


if __name__ == '__main__':
    # * Setup - where to load data, how many events
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2/'
    from_ = data_dir + 'pickles'
    to_ = data_dir + 'tarballs'
    pickle_dirs = [path for path in Path(from_).iterdir()]

    # * Zip and multiprocess
    to_list = [to_]*len(pickle_dirs)
    packed = [entry for entry in zip(pickle_dirs, to_list)]

    available_cores = cpu_count()
    with Pool(available_cores+2) as p:
        p.map(make_tar, packed)
    
    print(get_time(), 'Finished making tarballs!')