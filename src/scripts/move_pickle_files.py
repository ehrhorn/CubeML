from multiprocessing import Pool, cpu_count
from src.modules.helper_functions import get_project_root, get_time
import subprocess
from pathlib import Path
import pickle
import sys

def move_pickle(pack):
    integer, hep_dir, gpu_dir = pack
    path = hep_dir + str(integer)
    

    print(get_time(), 'Moving %s'%(path))
    sys.stdout.flush()
    for entry in Path(path).iterdir():
        event = pickle.load(open(entry, "rb"))
        if event['meta']['particle_code'] == '140000':
            from_ = hep_dir + str(integer) + '/' + entry.name
            destination =  'bjoernhm@gpulab.hepexp.nbi.dk:' + gpu_dir + str(integer) + '/' + entry.name
            command = 'scp'
            subprocess.run([command, from_, destination])
            # call(command + from_ + destination)

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
    with Pool(available_cores+2) as p:
        p.map(move_pickle, packed)
