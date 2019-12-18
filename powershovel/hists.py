from tables import *
from pathlib import Path
import numpy as np
from scipy.stats import iqr
import gc


def dict_creator(data_file):
    BANNED_GROUPS = [
        'dom_atwd',
        'dom_fadc',
        'dom_lc',
        'dom_pulse_width',
        'secondary_track_length'
    ]
    dictionary = {}
    with File(data_file, 'r') as f:
        array_iter = f.root.histograms.raw.__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                dictionary[array._v_name] = []
    return dictionary


def file_reader(data_file):
    BANNED_GROUPS = [
        'dom_atwd',
        'dom_fadc',
        'dom_lc',
        'dom_pulse_width',
        'secondary_track_length'
    ]
    dictionary = {}
    with File(data_file, 'r') as f:
        array_iter = f.root.histograms.raw.__iter__()
        for array in array_iter:
            if array._v_name not in BANNED_GROUPS:
                dictionary[array._v_name] = array.read()
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


def h5_saver(OUT_FILE, hist_dict, bin_edges):
    with File(OUT_FILE, mode='w') as f:
        hist_group = f.create_group(
            where='/',
            name='histograms'
        )
        raw_group = f.create_group(
            where=hist_group,
            name='raw'
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


DATA_DIR = Path('/groups/hep/ehrhorn/files/icecube/hdf5_files/oscnext-genie-level5-v01-01-pass2')
DATA_FILES = [
    f for f in DATA_DIR.glob('**/*.h5') if f.is_file() and '120000' in f.name
]
DATA_FILES = sorted(DATA_FILES)[0:10]
OUT_FILE = Path('/groups/hep/ehrhorn/').joinpath('hists.h5')

iqr_dict = dict_creator(DATA_FILES[0])
n_dict = dict_creator(DATA_FILES[0])
min_max_dict = dict_creator(DATA_FILES[0])

for data_file in DATA_FILES:
    print('Handling file', data_file.name)
    hist_dict = file_reader(data_file)
    iqr_dict = iqr_calculator(hist_dict, iqr_dict)
    n_dict = n_calculator(hist_dict, n_dict)
    min_max_dict = min_max_calculator(hist_dict, min_max_dict)
    gc.collect()

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

for data_file in DATA_FILES:
    dictionary = file_reader(data_file)
    hist_dict, bin_edges = histogram_calculator(dictionary, hist_dict, bins)

h5_saver(OUT_FILE, hist_dict, bin_edges)