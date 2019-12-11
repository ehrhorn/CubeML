#!/usr/bin/env conda run -n cubeml python
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
from argparse import ArgumentParser
from src.modules.transform_funcs import (
    reader,
    update_hdf5,
    clean_lvl_1,
    transform_lvl_1
)
import traceback
import pickle

parser = ArgumentParser()
parser.add_argument(
    '-f',
    '--file',
    dest='filename',
    help='JSON file with transforms'
)
parser.add_argument(
    '-d',
    '--directory',
    dest='directory',
    help='Directory containing HDF5 files'
)
args = parser.parse_args()
json_file = args.filename
hdf5_files_path = Path(args.directory)
transformers_save_path = hdf5_files_path.joinpath('transformers')
transformers_save_path.mkdir(parents=True, exist_ok=True)

# hdf5_files = sorted(hdf5_files_path.glob('*.h5'))
hdf5_files = [
    f for f in hdf5_files_path.glob('**/*.h5') if f.is_file() and '120000' not in f.name and '160000' not in f.name
]
hdf5_files = sorted(hdf5_files)
hdf5_files = hdf5_files[0:37]

with open(json_file, 'r') as read_file:
    transforms = json.load(read_file)

for file in tqdm(hdf5_files):
    print('Processing file:', file)
    try:
        for transform, functions in transforms.items():
            print('\nTransform:', transform)
            data = reader(str(file), 'raw')
            transformed_data = []
            for i, function in enumerate(functions['transforms']):
                data, transformers = globals()[function](data)
                transformed_data.append(data)
                if i == 0:
                    transformer_save_address = str(
                        transformers_save_path.joinpath(transform + '.pickle')
                    )
            for out_data in reversed(transformed_data):
                for key in out_data:
                    if key not in transformed_data[-1]:
                        transformed_data[-1][key] = out_data[key]
            histograms = {}
            for key in transformed_data[-1]:
                histograms[key] = np.hstack(transformed_data[-1][key])
                pickle.dump(transformers, open(transformer_save_address, 'wb'))
            update_hdf5(
                path=str(file),
                data=transformed_data[-1],
                parent='/',
                group_name=transform,
                group_description=functions['group_description'],
                )
            update_hdf5(
                path=str(file),
                data=histograms,
                parent='/histograms',
                group_name=transform,
                group_description='Histograms for' + transform
            )
    except Exception:
        traceback.print_exc()
        pass

print('Done!')
