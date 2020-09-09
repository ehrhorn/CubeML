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

# PATH_TEST_DB = '/'.join([
#     PATH_DATA, '/oscnext-genie-level5-v01-01-pass2/test_transformed.db']
# )

PATH_TRANSFORMERS = PATH_DATA_OSCNEXT+'/sqlite_transformers.pickle'
N_BINS_PERF_PLOTS = 18
N_BINS_CLASS_PLOTS = 6

# Set ensemble
ENSEMBLE = {
    'direction_reg': [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-28-16.25.11?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-28-22.42.34?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-01-11.14.08?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-02-00.33.27?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-02-13.20.30?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-02-19.57.38?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-03-06.05.26?workspace=user-bjoernmoelvig'
    ],

    'energy_vertex_reg': [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-21-15.26.17?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-30-19.47.16?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-31-04.03.42?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-31-11.21.01?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-31-18.21.51?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-03-14.53.28?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-03-22.12.38?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-04-05.14.55?workspace=user-bjoernmoelvig'
    ],

    'energy_reg': [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-21-15.26.17?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-30-19.47.16?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-31-04.03.42?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-31-11.21.01?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-08-31-18.21.51?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-03-14.53.28?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-03-22.12.38?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-09-04-05.14.55?workspace=user-bjoernmoelvig'
    ]

}

CLASSIFICATION = ['nue_numu', 'nue_numu_nutau']