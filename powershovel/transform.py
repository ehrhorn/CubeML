from tables import *
from pathlib import Path
import numpy as np
from scipy.stats import iqr
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
import os
import psutil
import joblib

def histogram_reader(file, dictionary, group, BANNED_GROUPS):
    with File(file, 'r') as f:
        array_iter = f.root.histograms._f_get_child(group).__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                dictionary[array._v_name] = np.append(
                    dictionary[array._v_name],
                    array.read()
                )
    return dictionary


def groups_reader(file, group, BANNED_GROUPS):
    group_list = []
    with File(file, 'r') as f:
        array_iter = f.root.histograms._f_get_child(group).__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                group_list.append(array._v_name)
    return group_list


def transformer_fit(hist_dict, transformer_dict):
    for key in transformer_dict:
        transformer_dict[key].fit(hist_dict[key].reshape(-1, 1))
    return transformer_dict


def transformer_transform(hist_dict, transformer_dict):
    transformed_dict = {}
    for key in transformer_dict:
        transformed_dict[key] = transformer_dict
    return transformed_dict

DATA_DIR = Path(
    '/groups/hep/ehrhorn/files/icecube/hdf5_files/'
    'oscnext-genie-level5-v01-01-pass2/'
)
# DATA_DIR = Path(
#     '/groups/hep/ehrhorn/transform_test'
# )
PARTICLE_TYPES = ['120000', '140000', '160000']
BANNED_GROUPS = [
    'dom_atwd',
    'dom_fadc',
    'dom_lc',
    'dom_pulse_width',
    'secondary_track_length'
]
QUANTILE_KEYS = ['dom_charge']
ROBUST_KEYS = [
    'dom_n_hit_multiple_doms',
    'dom_time',
    'dom_timelength_fwhm',
    'dom_x',
    'dom_y',
    'dom_z',
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

process = psutil.Process(os.getpid())
OUT_DIR = Path('/groups/hep/ehrhorn')

for particle_type in PARTICLE_TYPES:
    out_file = OUT_DIR.joinpath(particle_type + '.pkl')
    if out_file.is_file():
        continue
    DATA_FILES = [
        f for f in DATA_DIR.glob('**/*.h5') if f.is_file()
            and particle_type in f.name
    ]
    DATA_FILES = sorted(DATA_FILES)

    group_list = groups_reader(DATA_FILES[0], 'raw', BANNED_GROUPS)

    hist_dict = {key: np.empty(0) for key in group_list}

    transformer_dict = {}
    for key in group_list:
        if key in ROBUST_KEYS:
            transformer_dict[key] = RobustScaler()
        elif key in QUANTILE_KEYS:
            transformer_dict[key] = QuantileTransformer()

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
    transformer_dict = transformer_fit(hist_dict, transformer_dict)
    joblib.dump(transformer_dict, out_file)

print('Done')
