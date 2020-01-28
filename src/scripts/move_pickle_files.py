from multiprocessing import Pool, cpu_count
from src.modules.helper_functions import get_project_root, get_time
import subprocess
from pathlib import Path
import pickle
import sys

def move_pickle(pack):
    integer, hep_dir, gpu_dir = pack
    n_per_dir = 10000
    path = hep_dir + str(integer)
    name_range = range(integer*n_per_dir, (integer+1)*n_per_dir)

    print(get_time(), 'Moving %s'%(path))
    sys.stdout.flush()
    command = 'scp'
    for name in name_range:
        from_ = 'bjoernhm@hep03.hpc.ku.dk:' + hep_dir + str(integer) + '/' + str(name) +'.pickle'
        to = gpu_dir + str(integer) + '/' + str(name) +'.pickle'
        subprocess.run([command, from_, to])
    
        event = pickle.load(open(to, "rb"))
        if event['meta']['particle_code'] != '140000':
            Path(to).unlink()


if __name__ == '__main__':
    n_dirs = 1131

    # * Now, prep a list of from-paths
    hep_dir = '/groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/'
    gpu_dir = '/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/'
    dir_names = list(range(n_dirs))
    hep_dir_list = [hep_dir]*len(dir_names)
    gpu_dir_list = [gpu_dir]*len(dir_names)

    packed = [entry for entry in zip(dir_names, hep_dir_list, gpu_dir_list)]
    
    available_cores = cpu_count()
    with Pool(available_cores+6) as p:
        p.map(move_pickle, packed)
