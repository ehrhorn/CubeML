from tables import *
from pathlib import Path
import numpy as np
from scipy.stats import iqr
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PowerTransformer
import os
import psutil
import joblib


def bjoern_transform(x):
    return (x - 150) / 150


def histogram_reader(file, dictionary, group, BANNED_GROUPS):
    with File(file, 'r') as f:
        array_iter = f.root._f_get_child(group).__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                obj = array.read()
                if type(obj[0]) == np.ndarray:
                    obj = np.concatenate(obj).ravel()
                dictionary[array._v_name] = np.append(
                    dictionary[array._v_name],
                    obj
                )
    return dictionary


def groups_reader(file, group, BANNED_GROUPS):
    group_list = []
    with File(file, 'r') as f:
        array_iter = f.root._f_get_child(group).__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                group_list.append(array._v_name)
    return group_list


def transformer_fit(hist_dict, ROBUST_KEYS, QUANTILE_KEYS, GEOMETRY_KEYS):
    transformer_dict = {}
    geometry_array = np.empty(1)
    for key in hist_dict:
        if key in ROBUST_KEYS:
            transformer_dict[key] = RobustScaler()
        elif key in QUANTILE_KEYS:
            samples = hist_dict[key].reshape(-1, 1).shape[0]
            transformer_dict[key] = QuantileTransformer(
                n_quantiles=100000,
                output_distribution='normal',
                subsample=100000
            )
            # transformer_dict[key] = PowerTransformer(
            #     method='box-cox'
            # )
        elif key in GEOMETRY_KEYS:
            transformer_dict[key] = RobustScaler()
            geometry_array = np.append(geometry_array, hist_dict[key].reshape(-1, 1))
        else:
            continue
        transformer_dict[key].fit(hist_dict[key].reshape(-1, 1))
    for key in GEOMETRY_KEYS:
        transformer_dict[key].fit(geometry_array.reshape(-1, 1))
    return transformer_dict


DATA_DIR = Path(
    '/groups/hep/ehrhorn/oscnext-genie-level5-v01-01-pass2_new'
)
# DATA_DIR = Path(
#     '/groups/hep/ehrhorn/transform_test'
# )
OUT_DIR = Path(
    '/groups/hep/ehrhorn/oscnext-genie-level5-v01-01-pass2_new/'
    'transformers'
)
# OUT_DIR = Path(
#     '/groups/hep/ehrhorn/'
# )

PARTICLE_TYPES = ['140000']
BANNED_GROUPS = [
    'dom_atwd',
    'dom_fadc',
    'dom_lc',
    'dom_pulse_width',
    'secondary_track_length',
    'true_primary_speed'
]
QUANTILE_KEYS = [
    'dom_charge',
    'true_primary_time'
]
ROBUST_KEYS = [
    'dom_n_hit_multiple_doms',
    'dom_time',
    'dom_timelength_fwhm',
    'linefit_point_on_line_x',
    'linefit_point_on_line_y',
    'linefit_point_on_line_z',
    'toi_evalratio',
    'toi_point_on_line_x',
    'toi_point_on_line_y',
    'toi_point_on_line_z',
    'true_primary_energy',
    'true_primary_position_x',
    'true_primary_position_y',
    'true_primary_position_z'
]
GEOMETRY_KEYS = [
    'dom_x',
    'dom_y',
    'dom_z'
]

process = psutil.Process(os.getpid())

TRANSFORM = 'transform1'

for particle_type in PARTICLE_TYPES:
    out_file = OUT_DIR.joinpath(particle_type + '_' + TRANSFORM + '.pickle')
    if out_file.is_file():
        continue
    DATA_FILES = sorted([
        f for f in DATA_DIR.glob('**/*.h5') if f.is_file()
            and particle_type in f.name
    ])

    group_list = groups_reader(DATA_FILES[0], 'raw', BANNED_GROUPS)

    hist_dict = {key: np.empty(0) for key in group_list}

    for i, data_file in enumerate(DATA_FILES):
        if i % 20 == 0:
            print('Handling particle {}, file {}, RAM used {} GB, {}/{}'.format(
                particle_type,
                data_file.stem.split('.')[-1],
                round(process.memory_info().rss / 1073741824, 2),
                i + 1,
                len(DATA_FILES)
            ))
        hist_dict = histogram_reader(
            data_file,
            hist_dict,
            'raw',
            BANNED_GROUPS
        )
    print('Fitting particle {}'.format(particle_type))
    transformer_dict = transformer_fit(
        hist_dict,
        ROBUST_KEYS,
        QUANTILE_KEYS,
        GEOMETRY_KEYS
    )
    joblib.dump(transformer_dict, out_file)

print('Done')
