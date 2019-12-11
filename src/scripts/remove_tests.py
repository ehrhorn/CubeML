from shutil import rmtree
from pathlib import Path
import json

from src.modules.helper_functions import get_project_root, remove_tests_modeldir, remove_tests_wandbdir, delete_nohup_file
import argparse

description = 'Deletes tests and unused wandb-directories'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-a', '--all', action='store_true', help='Removes all wandb-directories, including killed experiments')
args = parser.parse_args()


if __name__ == '__main__':
    # Remove tests from wandb-directory
    if args.all:
        remove_tests_wandbdir(rm_all=True)
    else:
        remove_tests_wandbdir()
    
    # Remove tests from models-directory
    remove_tests_modeldir()

    # Remove nohup.out from src/scripts
    delete_nohup_file()


