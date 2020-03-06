import argparse
import multiprocessing
from src.modules.main_funcs import run_experiment
from src.modules.helper_functions import get_project_root
from pathlib import Path

if __name__ == '__main__':
    description = 'Runs an experiment from the experiments folder.'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--gpu', nargs='+', default='0', type=str, help='Sets the IDs of the GPUs to use')

    args = parser.parse_args()
    
    # * Fetch an experiment - run the oldest first.
    exp_dir = get_project_root() + '/experiments'
    exps = sorted(Path(exp_dir).glob('*.json'))
    
    # ! Someone online set to add next line to ensure CUDA works...
    multiprocessing.set_start_method('spawn')
    print('WHAT THE FUCK IS UP')
    # run_experiment(exps[0], gpu_id=args.gpu[0])

