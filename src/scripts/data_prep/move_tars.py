from multiprocessing import Pool, cpu_count
from src.modules.helper_functions import get_project_root, get_time
import subprocess
from pathlib import Path
import sys

def move_tar(pack):
    from_hep, to_gpu = pack
    if Path(to_gpu).exists():
        pass
    else:
        print(get_time(), 'Copying %s'%(from_hep))
        sys.stdout.flush()
        command = 'rsync'
        subprocess.run([command, from_hep, to_gpu])

def move_tars():
    """Scripts used to move tarballs of rpickled data from HEP to gpulab.

    Script must be run on gpulab - cannot ssh from HEP to gpulab, only other way around.

    Uses rsync to move tarballs. WHere, to and how many must be hardcoded for now.
    """    
    
    # * Setup - where to load data, how many events
    n_pickle_dirs = 1131
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2/'
    if not Path(data_dir).exists():
        Path(data_dir).mkdir()
        print(get_time(), 'Created directory %s'%(data_dir))
    from_ = 'bjoernhm@hep03.hpc.ku.dk:/groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/tarballs/'
    to_ = data_dir + 'tarballs/'
    if not Path(to_).exists():
        Path(to_).mkdir()
        print(get_time(), 'Created directory %s'%(to_))

    from_tarballs = [from_+str(i)+'.tar' for i in range(n_pickle_dirs)]
    to_list = [to_+str(i)+'.tar' for i in range(n_pickle_dirs)]

    # * Zip and multiprocess
    packed = [entry for entry in zip(from_tarballs, to_list)]
    available_cores = cpu_count()
    with Pool(available_cores+8) as p:
        p.map(move_tar, packed)
    
    print(get_time(), 'Finished copying tarballs!')
    
if __name__ == '__main__':
    move_tars()
    
