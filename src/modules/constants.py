from pathlib import Path

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

PATH_MASKS = get_project_root() + '/data/masks/'
PATH_DATA = get_project_root() + '/data/'
PATH_MODELS = get_project_root() + '/models/'