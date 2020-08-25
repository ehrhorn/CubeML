import json
from pathlib import Path
from src.modules.main_funcs import run_experiments
from src.modules.helper_functions import delete_nohup_file
from src.modules.preprocessing import DomChargeScaler, EnergyNoLogTransformer
import argparse

description = 'Runs all experiments saved in "/experiments" starting with the oldest.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument(
    '--newest_first', 
    action='store_true', 
    help='Runs the newest experiments first.'
    )

parser.add_argument(
    '--gpu', 
    default=0, 
    type=int, 
    help='Sets which experiments to run - from gpu0-folder or gpu1-folder.'
    )
args = parser.parse_args()

if __name__ == '__main__':
    run_experiments(newest_first=args.newest_first, gpu=args.gpu)