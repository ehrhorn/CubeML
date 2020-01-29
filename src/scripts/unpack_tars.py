from multiprocessing import Pool, cpu_count
from src.modules.helper_functions import get_project_root, get_time
import subprocess
from pathlib import Path
import sys

def unpack_tar_remove(pack):
    tarball, path = pack
    
    command = 'tar'
    flags_tar = '-xf'
    flags_dir = '-C'
    # where_in_tar = 'lustre/hpc/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/pickles'

    # * The tar was created in a silly way - it is deeply nested in 
    # * lustre/hpc/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/pickles/.
    # * This is unwanted. Therefore, standing in ../pickles run:
    # * mv lustre/hpc/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/pickles/* .

    subprocess.run([command, flags_tar, tarball, flags_dir, path])
    
    # * Remove the tarball
    tarball.unlink()

    print(get_time(), 'Unpacked and removed %s'%(tarball))
    sys.stdout.flush()
   
if __name__ == '__main__':
    # * Setup - where to load data, how many events
    tarball_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2/tarballs'
    tarballs = [path for path in Path(tarball_dir).iterdir()]
    
    pickle_dir =  get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2/pickles/'
    if not Path(pickle_dir).exists():
        Path(pickle_dir).mkdir()
    
    pickle_dir_list = [pickle_dir]*len(tarballs)
    packed = [entry for entry in zip(tarballs, pickle_dir_list)]
    available_cores = cpu_count()
    
    with Pool(available_cores+2) as p:
        p.map(unpack_tar_remove, packed)
    print(get_time(), 'Finished unpacking tarballs!')

    
    
    
    
