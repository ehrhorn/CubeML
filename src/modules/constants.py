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

# Set ensemble
ENSEMBLE = {
    'direction_reg': [
        # https://app.wandb.ai/cubeml/cubeml/runs/2020-05-29-16.57.10/overview?workspace=user-bjoernmoelvig
        '2020-05-29-16.57.10',

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-05-30-12.10.48/overview?workspace=user-bjoernmoelvig
        '2020-05-30-12.10.48', 

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-04-23-13.25.26?workspace=user-bjoernmoelvig
        '2020-04-23-13.25.26',

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-04-23-00.02.48/overview?workspace=user-bjoernmoelvig
        '2020-04-23-00.02.48',

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-05-30-11.12.08/overview?workspace=user-bjoernmoelvig
        '2020-05-30-11.12.08'
    ],

    'energy_reg': [
        # https://app.wandb.ai/cubeml/cubeml/runs/2020-04-20-08.19.33?workspace=user-bjoernmoelvig
        '2020-04-20-08.19.33',

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-04-16-11.34.16?workspace=user-bjoernmoelvig
        '2020-04-16-11.34.16',

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-03-20-11.43.00?workspace=user-bjoernmoelvig
        '2020-03-20-11.43.00',

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-03-08-07.39.49?workspace=user-bjoernmoelvig
        '2020-03-08-07.39.49',

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-03-05-21.09.41?workspace=user-bjoernmoelvig
        '2020-03-05-21.09.41'
    ]
}

CLASSIFICATION = ['nue_numu']