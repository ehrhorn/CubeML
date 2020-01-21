import src.modules.preprocessing as pp
from pathlib import Path
import random
import joblib
from src.modules.helper_functions import get_project_root, confirm_particle_type
from multiprocessing import cpu_count, Pool

if __name__ == '__main__':
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2_copy'
    particle_code = '140000'
    prefix = 'transform1'

    transformer_path = data_dir + '/transformers/' + particle_code + '_' + prefix +'.pickle'

    files = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5' and confirm_particle_type(particle_code, file)])
    random.shuffle(files)    

    keys = pp.get_feature_keys()
    dicts = pp.get_feature_plot_dicts()
    clip_dicts = pp.get_feature_clip_dicts()
    transformer_dict = pp.get_feature_transformers()

    n_wanted_sample = 10e6
    n_wanted_histogram = 100e3
    dicts = [dicts[key] for key in keys]
    clip_dicts = [clip_dicts[key] for key in keys]
    files_list = [files]*len(keys)
    n_wanted_sample = [n_wanted_sample for key in keys]
    n_wanted_histogram = [n_wanted_histogram for key in keys]
    particle_code = [particle_code for key in keys]
    transformers = [transformer_dict[key] for key in keys]

    packages = [entry for entry in zip(keys, dicts, clip_dicts, files_list, n_wanted_sample, n_wanted_histogram, particle_code, transformers)]

    # * Use multiprocessing for parallelizing the job.
    available_cores = cpu_count()
    with Pool(available_cores) as p:
        transformers = p.map(pp.fit_feature_transformers, packages)

    # * Update or create a transformer-pickle
    if Path(transformer_path).is_file():
        transformers_combined= joblib.load(open(transformer_path, "rb"))
    else:
        transformers_combined = {}
    
    # * Combine transformers    
    for entry in transformers:
        transformers_combined.update(entry)
    
    # * Save it again
    joblib.dump(transformers_combined, open(transformer_path, 'wb'))
    print('Updated transformers saved at:')
    print(transformer_path)
