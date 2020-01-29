from multiprocessing import Pool, cpu_count
from src.modules.helper_functions import get_project_root, get_time
import subprocess
from pathlib import Path
import pickle
import sys

def move_tar(pack):
    from_hep, to_gpu = pack

    print(get_time(), 'Moving %s'%(pickle_dir))
    sys.stdout.flush()
    command = 'rsync'
    print(command, from_hep, to_gpu)
    # subprocess.run([command, from_hep, to_gpu])


if __name__ == '__main__':
    n_pickle_dirs = 1131
    # * Setup - where to load data, how many events
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2/'
    from_ = 'bjoernhm@hep03.hpc.ku.dk:/groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/tarballs/'
    to_ = data_dir + 'tarballs/'

    from_tarballs = [from_+str(i)+'.tar' for i in range(n_pickle_dirs)]
    to_list = [to_+str(i)+'.tar' for i in range(n_pickle_dirs)]

    # * Zip and multiprocess
    packed = [entry for entry in zip(from_tarballs, to_list)]
    for pack in packed:
        move_tar(pack)
    # available_cores = cpu_count()
    # with Pool(available_cores+8) as p:
    #     p.map(make_tar, packed)
    
    # print(get_time(), 'Finished making tarballs!')
    
    
