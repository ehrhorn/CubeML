import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.stats import zscore
from numpy.linalg import norm
from tables import *
import time
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

good_keys = [
    'charge',
    'dom_x',
    'dom_y',
    'dom_z',
    'time',
    'toi_point_on_line_x',
    'toi_point_on_line_y',
    'toi_point_on_line_z',
    'true_muon_energy',
    'true_muon_entry_position_x',
    'true_muon_entry_position_y',
    'true_muon_entry_position_z'
]


def dist_plots(keys, group):
    files_path = Path('../data/MuonGun_Level2_139008/').glob('*')
    events_files = sorted([f for f in files_path if f.is_file() and f.suffix == '.h5'])
    events = {}
    for key in keys:
        events[key] = []
    for file in events_files:
        print(file)
        hdf5_file = str(file)
        with File(hdf5_file, 'r') as f:
            data = f.root.__getattr__(group)
            for array in data.__iter__():
                # if array.name == 'no_of_doms':
                #     continue
                if array.name in keys:
                    events[array.name].append(np.hstack(array.read()))
            for array in f.root.raw.__iter__():
                if array.name in keys and array.name not in events:
                    events[array.name].append(np.hstack(array.read()))
    for key in events.keys():
        events[key] = np.hstack(np.array(events[key]))
    dist_fig, ax = plt.subplots(
        nrows=6,
        ncols=2,
        figsize=(10, 30)
    )
    ax = ax.ravel()
    for i, key in enumerate(keys):
        ax[i].hist(
            events[key],
            bins=30,
            histtype='step'
        )
        if key == 'charge':
            ax[i].set_yscale('log')
        ax[i].set(title=key)
    return dist_fig


def space_time_cleanup(activations, clean_distance):
    cols = ['dom_x', 'dom_y', 'dom_z', 'time']
    activations_np = activations[cols].values
    dom_spacetime_distance_table = cdist(activations_np, activations_np)
    np.fill_diagonal(dom_spacetime_distance_table, np.nan)
    good_doms = np.nanmin(dom_spacetime_distance_table, axis=0)
    good_doms = np.where(good_doms < int(clean_distance))
    activations = activations.iloc[good_doms]
    return activations


def custom_distance_metric(activations, geom, n_nearest, m):
    left_idx = ['dom_x', 'dom_y', 'dom_z']
    right_idx = ['x', 'y', 'z']
    geom['idx'] = geom.index.values
    indexed_activations = activations.merge(
        geom,
        left_on=left_idx,
        right_on=right_idx
    )
    indexed_activations_np = indexed_activations[right_idx].values
    geom_np = geom[right_idx].values
    distance_to_other_active_doms = cdist(
        indexed_activations_np,
        indexed_activations_np
        )
    inverse_distance_to_other_active_doms = np.divide(
        1,
        distance_to_other_active_doms**m,
        where=[distance_to_other_active_doms > 0]
    )
    numerator = np.sum(inverse_distance_to_other_active_doms, axis=0)
    # z_score = zscore(distance_to_other_active_doms)
    # # print(z_score)
    # z_score_sum = np.sum(z_score, axis=1)
    # distance_metric = abs(z_score_sum)
    distance_to_all_other_doms = cdist(
        geom_np,
        indexed_activations_np
    )[0:n_nearest, :]
    inverse_distance_to_all_other_doms = np.divide(
        1,
        distance_to_all_other_doms**m,
        where=[distance_to_all_other_doms > 0]
    )
    denominator = np.sum(inverse_distance_to_all_other_doms[0], axis=0)
    distance_metric = numerator / denominator
    return distance_metric


def toi_distance(activations, toi_points, max_distance):
    p1 = np.asarray(toi_points['toi'])[:, 0]
    p2 = np.asarray(toi_points['toi'])[:, 1]
    p3 = np.zeros((len(activations), 3))
    p3[:, 0] = activations.dom_x.values
    p3[:, 1] = activations.dom_y.values
    p3[:, 2] = activations.dom_z.values
    distance_to_toi = norm(np.cross(p2 - p1, p3 - p1) / norm(p2 - p1), axis=1)
    activations = activations.iloc[distance_to_toi < max_distance]
    return activations
