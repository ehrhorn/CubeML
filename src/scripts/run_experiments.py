import json
from pathlib import Path
import src.modules.main_funcs as mf
import argparse

description = 'Runs all experiments saved in "/experiments" starting with the oldest.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('--newest_first', action='store_true', help='Runs the newest experiments first.')

args = parser.parse_args()

if __name__ == '__main__':
    mf.run_experiments(newest_first=args.newest_first)