from tables import *
from pathlib import Path
import numpy as np
from scipy.stats import iqr
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
import os
import psutil
import joblib
import os
import psutil


def bjoern_transform(x):
    return (x - 150) / 150


def file_reader(file, group, BANNED_GROUPS):
    dictionary = {}
    with File(file, 'r') as f:
        array_iter = f.root._f_get_child(group).__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                dictionary[array._v_name] = array.read()
    return dictionary


def transformer_transform(hist_dict, transform_dict):
    final_dict = {}
    for key in transform_dict:
        if type(hist_dict[key][0]) == np.ndarray:
            final_dict[key] = []
            for i in range(len(hist_dict[key])):
                transformed_data = transform_dict[key].transform(
                    hist_dict[key][i].reshape(-1, 1)
                )
                flat_list = [
                    item for sublist in transformed_data for item in sublist
                ]
                final_dict[key].append(flat_list)
        else:
            transformed_data = transform_dict[key].transform(
                np.array([hist_dict[key]]).reshape(-1, 1)
            )
            flat_list = [
                item for sublist in transformed_data for item in sublist
            ]
            final_dict[key] = flat_list
    return final_dict


def scaled_data_saver(file, data, transform_level):
    with File(file, 'a') as f:
        transformed_data_group = f.create_group(
            where='/',
            name=transform_level
        )
        for key in data:
            if type(data[key][0]) == list:
                vlarray = f.create_vlarray(
                    where=transformed_data_group,
                    name=key,
                    atom=Float64Atom(shape=())
                )
                for i in range(len(data[key])):
                    vlarray.append(data[key][i])
            else:
                f.create_array(
                    where=transformed_data_group,
                    name=key,
                    obj=data[key]
                )


PARTICLE_TYPES = ['140000']

DATA_DIR = Path(
    '/groups/hep/ehrhorn/oscnext-genie-level5-v01-01-pass2_new'
)
# DATA_DIR = Path(
#     '/groups/hep/ehrhorn/transform_test'
# )
SCALER_DIR = Path(
    '/groups/hep/ehrhorn/oscnext-genie-level5-v01-01-pass2_new/'
    'transformers'
)
# SCALER_DIR = Path(
#     '/groups/hep/ehrhorn'
# )

BANNED_GROUPS = [
    'dom_atwd',
    'dom_fadc',
    'dom_lc',
    'dom_pulse_width',
    'secondary_track_length',
    'true_primary_speed'
]

TRANSFORM = 'transform1'

process = psutil.Process(os.getpid())

for particle_type in PARTICLE_TYPES:
    data_files = sorted([
        f for f in DATA_DIR.glob('**/*.h5') if f.is_file()
            and particle_type in f.name
    ])
    scalers = joblib.load(
        SCALER_DIR.joinpath(
            particle_type + '_' + TRANSFORM + '.pickle'
            )
        )
    for i, data_file in enumerate(data_files):
        print('Handling particle {}, file {}, RAM used {} GB, {}/{}'.format(
            particle_type,
            data_file.stem.split('.')[-1],
            round(process.memory_info().rss / 1073741824, 2),
            i + 1,
            len(data_files)
        ))
        dictionary = file_reader(data_file, 'raw', BANNED_GROUPS)
        final_dict = transformer_transform(dictionary, scalers)
        scaled_data_saver(data_file, final_dict, TRANSFORM)

print('Done')
