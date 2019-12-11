from tables import *
from astropy.stats import bayesian_blocks
from astropy.visualization import hist
from pathlib import Path
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import shutil


def dict_maker(data_file):
    hist_dict = {}
    with File(data_file, 'r') as f:
        group_iter = f.root.histograms.__iter__()
        for group in group_iter:
            hist_dict[group._v_name] = {}
            array_iter = group.__iter__()
            for array in array_iter:
                hist_dict[group._v_name][array._v_name] = np.empty(0)
    return hist_dict


def dict_updater(data_file):
    hist_dict = dict_maker(data_file)
    with File(data_file, 'r') as f:
        group_iter = f.root.histograms.__iter__()
        for group in group_iter:
            array_iter = group.__iter__()
            for array in array_iter:
                hist_dict[group._v_name][array._v_name] = np.append(
                    hist_dict[group._v_name][array._v_name],
                    array.read()
                )
    return hist_dict


def h5_saver(OUT_FILE, hist_dict):
    if not OUT_FILE.is_file():
        with open_file(OUT_FILE, mode='w') as f:
            for key in hist_dict:
                group = f.create_group(
                            where='/',
                            name=key,
                        )
                for variable in hist_dict[key]:
                    f.create_earray(
                        where=group,
                        name=variable,
                        obj=hist_dict[key][variable]
                    )
    else:
        with open_file(OUT_FILE, mode='a') as f:
            for key in hist_dict:
                for variable in hist_dict[key]:
                    array = f.root.__getattr__(key).__getattr__(variable)
                    array.append(hist_dict[key][variable])


DATA_DIR = Path(__file__).resolve().parent.parent.joinpath('data')
DATA_SETS = [d for d in DATA_DIR.iterdir() if d.is_dir()]
BANNED_GROUPS = ['histograms', 'meta']

for j, data_set in enumerate(DATA_SETS):
    OUT_FILE = Path(__file__).resolve().parent.joinpath(data_set.stem + '.h5')
    data_files = [f for f in data_set.glob('**/*.h5') if f.is_file()
        and '120000' not in f.name and '160000' not in f.name]
    data_files = sorted(data_files)
    data_files = data_files[0:37]
    for i, data_file in enumerate(data_files):
        print('Handling file', data_file.name)
        hist_dict = dict_updater(data_file)
        h5_saver(OUT_FILE, hist_dict)
        if j == 1:
            OUT_FILE_COPY = Path('/home/mads/temp/' + data_file.name)
            shutil.copy(str(data_file), str(OUT_FILE_COPY))
