from pathlib import Path
from multiprocessing import cpu_count

def get_project_root():
    """Finds absolute path to project root - useful for running code on different machines.
    
    Returns:
        str -- path to project root
    """    
    
    # * Find project root
    current_dir_splitted = str(Path.cwd()).split('/')
    i = 0
    while current_dir_splitted[i] != 'CubeML':
        i += 1
    return '/'.join(current_dir_splitted[:i+1]) 

AVAILABLE_CORES = cpu_count()

PATH_THESIS_PLOTS = get_project_root() + '/reports/thesis_plots/'

PATH_MASKS = get_project_root() + '/data/masks/'

PATH_DATA = get_project_root() + '/data/'

PATH_DATA_OSCNEXT = get_project_root() + '/data/oscnext-genie-level5-v01-01-pass2'

PATH_MODELS = get_project_root() + '/models/'

PATH_TRAIN_DB = '/'.join([
    PATH_DATA, '/oscnext-genie-level5-v01-01-pass2/train_transformed.db']
)

PATH_VAL_DB = '/'.join([
    PATH_DATA, '/oscnext-genie-level5-v01-01-pass2/val_transformed.db']
)

PATH_TEST_DB = '/'.join([
    PATH_DATA, '/oscnext-genie-level5-v01-01-pass2/test_transformed.db']
)

PATH_TRANSFORMERS = PATH_DATA_OSCNEXT+'/sqlite_transformers.pickle'
N_BINS_PERF_PLOTS = 18

