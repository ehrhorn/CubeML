from src.modules.helper_functions import get_project_root
from src.modules.preprocessing import feature_engineer
from pathlib import Path
from multiprocessing import cpu_count, Pool

if __name__ =='__main__':
    # * For every datafile, make a new datafile to not fuck shit up
    data_dir = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2_copy'

    file_list = sorted([str(file) for file in Path(data_dir).iterdir() if file.suffix == '.h5'])

    N_FILES = len(file_list)
    packed = [entry for entry in zip(range(N_FILES), file_list, [N_FILES]*N_FILES)]

    # * Use multiprocessing for parallelizing the job.
    available_cores = cpu_count()
    with Pool(available_cores) as p:
        p.map(feature_engineer, packed)