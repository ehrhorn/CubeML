from tables import *
from pathlib import Path
import numpy as np
from scipy.stats import iqr
import gc
from datetime import datetime
import os
import psutil


def dict_creator(data_file, group):
    BANNED_GROUPS = [
        'dom_atwd',
        'dom_fadc',
        'dom_lc',
        'dom_pulse_width',
        'secondary_track_length',
        'true_primary_speed'
    ]
    dictionary = {}
    with File(data_file, 'r') as f:
        array_iter = f.root._f_get_child(group).__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                dictionary[array._v_name] = []
    return dictionary


def file_reader(data_file, group):
    BANNED_GROUPS = [
        'dom_atwd',
        'dom_fadc',
        'dom_lc',
        'dom_pulse_width',
        'secondary_track_length',
        'true_primary_speed'
    ]
    dictionary = {}
    with File(data_file, 'r') as f:
        array_iter = f.root._f_get_child(group).__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                obj = array.read()
                if type(obj[0]) == np.ndarray:
                    obj = np.concatenate(obj).ravel()
                dictionary[array._v_name] = obj
    return dictionary


def iqr_calculator(dictionary, iqr_dict):
    for key in dictionary:
        iqr_dict[key].append(iqr(dictionary[key]))
    return iqr_dict


def n_calculator(dictionary, n_dict):
    for key in dictionary:
        n_dict[key].append(len(dictionary[key]))
    return n_dict


def min_max_calculator(dictionary, min_max_dict):
    for key in dictionary:
        min_max_dict[key].append(np.min(dictionary[key]))
        min_max_dict[key].append(np.max(dictionary[key]))
    return min_max_dict


def histogram_calculator(dictionary, hist_dict, bins):
    bin_edges = {}
    for key in dictionary:
        histogram, bin_edges[key] = np.histogram(dictionary[key], bins=bins[key])
        hist_dict[key] += histogram
    return hist_dict, bin_edges


def hist_saver(out_file, hist_dict, bin_edges, transform):
    if out_file.is_file():
        mode = 'a'
    else:
        mode = 'w'
    with File(out_file, mode=mode) as f:
        if mode == 'w':
            hist_group = f.create_group(
                where='/',
                name='histograms'
            )
        raw_group = f.create_group(
            where='/histograms',
            name=transform
        )
        edges_group = f.create_group(
            where=raw_group,
            name='edges'
        )
        values_group = f.create_group(
            where=raw_group,
            name='values'
        )
        for key in dictionary:
            f.create_array(
                where=edges_group,
                name=key,
                obj=bin_edges[key]
            )
            f.create_array(
                where=values_group,
                name=key,
                obj=hist_dict[key]
            )

process = psutil.Process(os.getpid())

TRANSFORMS = ['transform1']
PARTICLE_TYPES = ['140000']

DATA_DIR = Path(
    '/groups/hep/ehrhorn/oscnext-genie-level5-v01-01-pass2_new'
)
# DATA_DIR = Path(
#     '/groups/hep/ehrhorn/transform_test'
# )

for transform in TRANSFORMS:
    for particle_type in PARTICLE_TYPES:
        data_files = [
            f for f in DATA_DIR.glob('**/*.h5') if f.is_file() and particle_type in f.name
        ]
        data_files = sorted(data_files)
        out_file = Path('/groups/hep/ehrhorn/repos/CubeML/powershovel').joinpath(
            particle_type + '.h5'
        )

        iqr_dict = dict_creator(data_files[0], transform)
        n_dict = dict_creator(data_files[0], transform)
        min_max_dict = dict_creator(data_files[0], transform)

        for i, data_file in enumerate(data_files):
            if i % 20 == 0:
                print('''\nAt timestamp {} I handled:\n
    A file of particle type {}, with file name {}, using transform {}.\n
    I used {} GB memory at the moment.\n
    This was file number {} out of {} for this particle type.'''
                    .format(
                        datetime.now(),
                        particle_type,
                        data_file.stem.split('.')[-1],
                        transform,
                        round(process.memory_info().rss / 1073741824, 2),
                        i + 1,
                        len(data_files)
                    )
                )
            hist_dict = file_reader(data_file, transform)
            iqr_dict = iqr_calculator(hist_dict, iqr_dict)
            n_dict = n_calculator(hist_dict, n_dict)
            min_max_dict = min_max_calculator(hist_dict, min_max_dict)

        iqr_mean = {key: np.mean(iqr_dict[key]) for key in iqr_dict}
        n_sum = {key: np.sum(n_dict[key]) for key in n_dict}
        min_val = {key: np.min(min_max_dict[key]) for key in min_max_dict}
        max_val = {key: np.max(min_max_dict[key]) for key in min_max_dict}
        bins = {}
        hist_dict = {}
        for key in iqr_mean:
            hist_dict[key] = 0

        for key in iqr_dict:
            h = 2 * iqr_mean[key] / (n_sum[key])**(1 / 3)
            bins[key] = int(
                round(
                    (max_val[key] - min_val[key]) / h, 0
                )
            )
            if bins[key] > 2000:
                bins[key] = 2000

        for data_file in data_files:
            dictionary = file_reader(data_file, transform)
            hist_dict, bin_edges = histogram_calculator(dictionary, hist_dict, bins)

        hist_saver(out_file, hist_dict, bin_edges, transform)

print('Done.')
