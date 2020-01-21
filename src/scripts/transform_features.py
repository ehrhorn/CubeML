import joblib
from pathlib import Path
from multiprocessing import Pool, cpu_count

import src.modules.helper_functions as hf
import src.modules.preprocessing as pp

if __name__ =='__main__':
    # * For every datafile, make a new datafile to not fuck shit up
    data_dir = hf.get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2_copy'
    particle_code = '140000'
    prefix = 'transform1'

    # * Load transformers, keys and prepare filenames
    transformer_path = data_dir + '/transformers/' + particle_code + '_' + prefix +'.pickle'
    transformers = joblib.load(open(transformer_path, 'rb'))
    file_list = [str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5' and hf.confirm_particle_type(particle_code, file)]
    keys = pp.get_feature_keys()

    # * Pack each filepath with transformers and keys for multiprocessing
    N_FILES = len(file_list)
    
    transformers_list = [transformers for i_file in range(N_FILES)]
    keys_list = [keys for i_file in range(N_FILES)]
    prefix_list = [prefix for i_file in range(N_FILES)]

    packed = [entry for entry in zip(file_list, transformers_list, keys_list, prefix_list)]

    # * Use multiprocessing for parallelizing the job.
    available_cores = cpu_count()
    with Pool(available_cores) as p:
        print(hf.get_time(), 'Transformation begun')
        p.map(pp.transform_features, packed)