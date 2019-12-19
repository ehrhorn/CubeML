from tables import *
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
import pickle


def reader(path, group):
    events = {}
    with File(path, 'r') as f:
        data = f.root.__getattr__(group)
        for array in data.__iter__():
            events[array.name] = array.read()
    return events


def update_hdf5(path, data, parent, group_name, group_description):
        with open_file(path, mode='a') as f:
            new_group = f.create_group(
                where=parent,
                name=group_name,
                title=group_description
            )
            for key in data:
                if type(data[key][0]) == np.ndarray:
                    variable = data[key]
                    if key == 'dom_x':
                        no_of_doms = [
                            len(
                                data[key][i]
                            )
                            for i in range(len(data[key]))
                        ]
                        f.create_array(
                            where=new_group,
                            name='no_of_doms',
                            obj=no_of_doms
                        )
                    vlarray = f.create_vlarray(
                        where=new_group,
                        name=key,
                        atom=Float64Atom(shape=())
                    )
                    for i in range(len(variable)):
                        vlarray.append(variable[i])
                else:
                    f.create_array(
                        where=new_group,
                        name=key,
                        obj=data[key]
                    )


def clean_lvl_1(events):
    for key in events:
        no_of_events = len(events[key])
        break
    clean_keys = ['dom_x', 'dom_y', 'dom_z', 'time', 'charge']
    clean_distance = 300
    clean_data = {}
    # Init the new dict with empty lists
    for name in clean_keys:
        clean_data[name] = []
    # Loop over all events in HDF5 file
    for i in range(no_of_events):
        no_of_doms = len(events['dom_x'][i])
        # Init array for space-time cleaning
        activations = np.zeros((no_of_doms, len(clean_keys)))
        # Loop over the keys that need cleaning
        for j, key in enumerate(clean_keys):
            activations[:, j] = events[key][i]
        # Array with the features needed for space-time cleaning calc
        activations_to_calc = activations[:, 0:4]
        # Distance table
        dom_spacetime_distance_table = cdist(
            activations_to_calc,
            activations_to_calc
        )
        # Fill the diagonal of the distance matrix, because zeros mess shit up
        np.fill_diagonal(dom_spacetime_distance_table, np.nan)
        # Get the minimums of each distance, ignoring nans
        good_activation_mins = np.nanmin(dom_spacetime_distance_table, axis=0)
        # Get the activations where the min is less than the clean distance
        cleaned_activations = activations[
            good_activation_mins < int(clean_distance)
        ]
        for k, key in enumerate(clean_keys):
            clean_data[key].append(cleaned_activations[:, k])
        transformers = {}
        for key in events:
            transformers[key] = None
    return clean_data, transformers


def transform_lvl_1(events):
    quantile_keys = ['dom_charge']
    robust_keys = [
        'dom_x',
        'dom_y',
        'dom_z',
        'dom_time',
        'toi_point_on_line_x',
        'toi_point_on_line_y',
        'toi_point_on_line_z',
        'true_primary_energy',
        'true_primary_entry_position_x',
        'true_primary_entry_position_y',
        'true_primary_entry_position_z'
    ]
    no_transform = [
        'toi_direction_x',
        'toi_direction_y',
        'toi_direction_z',
        'toi_evalratio',
        'true_primary_direction_x',
        'true_primary_direction_y',
        'true_primary_direction_z'
    ]
    do_not_use = [
        'linefit_direction_x',
        'linefit_direction_y',
        'linefit_direction_z',
        'linefit_point_on_line_x',
        'linefit_point_on_line_y',
        'linefit_point_on_line_z'
    ]
    hists = {}
    data = {}
    # Convert hdf5 to histograms
    for key in events:
        if type(events[key][0]) == np.ndarray:
            for var_len_data in events[key]:
                try:
                    hists[key] = np.append(hists[key], var_len_data[:])
                except KeyError:
                    hists[key] = var_len_data[:]
        else:
            hists[key] = np.array(events[key])
        data[key] = np.array(events[key])
    # Transform data using RobustScaler, Quantile Transformer to a norm dist
    transformers = {}
    d_transformed = {}
    n_events = data['charge'].shape[0]

    hists_transformed = {}

    for key, vals in data.items():
        if key in do_not_use: continue
        if key in no_transform: 
            # d_transformed[key] = vals
            hists_transformed[key] = hists[key] 
            transformers[key] = None               
        if key in quantile_keys:
            # Initialize and fit to a normal distribution
            transformers[key] = QuantileTransformer(
                output_distribution='normal'
            )
            transformers[key].fit(hists[key].reshape(-1, 1), )
            hists_transformed[key] = transformers[key].transform(
                hists[key].reshape(-1, 1)
            )      
            # If entries are of variable length, transform each individually
            if vals[0].shape:
                d_transformed[key] = [[]] * n_events
                for i_event in range(n_events):
                    d_transformed[key][i_event] = transformers[key].transform(
                        vals[i_event].reshape(-1, 1)
                    ).reshape(vals[i_event].shape[0], )
            else:
                d_transformed[key] = transformers[key].transform(
                    vals.reshape(-1, 1)
                ).reshape(n_events, )
        if key in robust_keys:

            # Initialize RobustScaler - docs at
            # https://scikit-learn.org/stable/modules/generated/
            # sklearn.preprocessing.RobustScaler.html
            transformers[key] = RobustScaler()
            transformers[key].fit(hists[key].reshape(-1, 1), )
            hists_transformed[key] = transformers[key].transform(
                hists[key].reshape(-1, 1)
            )

            # If entries are of variable length, transform each individually
            if vals[0].shape:
                d_transformed[key] = [[]] * n_events
                for i_event in range(n_events):
                    d_transformed[key][i_event] = transformers[key].transform(
                        vals[i_event].reshape(-1, 1)
                    ).reshape(vals[i_event].shape[0], )
            else:
                d_transformed[key] = transformers[key].transform(
                    vals.reshape(-1, 1)
                ).reshape(n_events, )
    return d_transformed, transformers
