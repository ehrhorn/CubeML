import h5py as h5
import numpy as np
import src.modules.helper_functions as hf
from pathlib import Path
import subprocess

def make_mask(data_path, mask_name='any', min_doms=0, max_doms=np.inf):
    # * make mask directory if it doesn't exist
    data_path = hf.get_project_root() + hf.get_path_from_root(data_path)
    name = hf.get_dataset_name(data_path)
    
    dir_path = hf.get_project_root() + '/data/masks/'+name
    if not Path(dir_path).is_dir():
        Path(dir_path).mkdir(parents=True)

    if mask_name == 'dom_interval':
        make_dom_interval_mask(data_path, dir_path, min_doms, max_doms)

    # * Make a .dvc-file to track mask
    dvc_path = hf.get_project_root() + '/data'
    subprocess.run(['dvc', 'add', 'masks'], cwd=dvc_path)

def make_dom_interval_mask(data_path, own_path, min_doms, max_doms):
    file_name = own_path+'/dom_interval_min%d_max%d.h5'%(min_doms, max_doms)
    
    # * Make file
    with h5.File(file_name, 'w') as f:
        grp = f.create_group('indices')
        
        # * Loop over all data - print every once in a while as a sanity check
        i_file = 1
        n_files = len([file for file in Path(data_path).iterdir() if file.suffix == '.h5'])
        for file in Path(data_path).iterdir():
            if file.suffix == '.h5':

                name = file.stem
                print('File %d of %d: Processing %s'%(i_file, n_files, name))
                i_file += 1
                # * Make a dataset of indices for each file in dataset
                indices = hf.apply_mask(file, mask_name='dom_interval', min_doms=min_doms, max_doms=max_doms)
                grp.create_dataset(name, data=indices)
    
if __name__ == '__main__':
    data_dir = '/data/oscnext-genie-level5-v01-01-pass2'
    mask_name, minimum, maximum = 'dom_interval', 0, 200
    mask_dict = {'mask_name': mask_name, 'min_doms': minimum, 'max_doms': maximum}
    make_mask(data_dir, **mask_dict)
