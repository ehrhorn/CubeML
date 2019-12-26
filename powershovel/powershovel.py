"""streamlit dashboard for IceCube event viewing.

Install streamlit with `pip install streamlit`.

How to run: `streamlit run powershovel.py`

Created by Mads Ehrhorn 19/10/2019.
"""
import streamlit as st
import pandas as pd
from tables import *
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from pathlib import Path
import numpy as np
from operator import itemgetter

# Columns to drop from geom frame
drop_cols = ['pmt', 'om', 'string', 'area', 'omtype', 'ux', 'uy', 'uz']


# @st.cache
def read_geom_file(geom_file, drop_cols):
    """Read the geometry file, output as dictionary of DataFrames.

    Function also deletes columns that aren't used.

    Args:
        geom_file (str): HDF5 geometry file path.

    Returns:
        geom (pandas.DataFrame): Dictionary containing geometry info.

    """
    with File(geom_file, 'r') as f:
        out = {}
        pmt_geom = f.root.pmt_geom
        col_names = pmt_geom.colnames
        for col_name in col_names:
            out[col_name] = pmt_geom.col(col_name)
        geom = pd.DataFrame.from_dict(out)
    geom_clean = geom.loc[geom.omtype == 20].copy()
    geom_clean.drop(drop_cols, axis=1, inplace=True)
    return geom_clean


# @st.cache
def meta_data(events_file, group):
    """Calculate meta data from event file.

    Args:
        None

    Returns:
        no_of_doms (list): Number of DOMs involved in each event.
        energy (pandas.Series): Energy for each event.
        events (list): Events to include.
        integrated_charge (list): Total charge involved in each event.

    """
    with File(events_file, 'r') as f:
        data = f.root.__getattr__(group)
        all_integrated_charge = data.dom_charge.read()
        integrated_charge = list(
            map(
                sum,
                all_integrated_charge
            )
        )
        max_charge = max(np.concatenate(all_integrated_charge).ravel())
        min_charge = min(np.concatenate(all_integrated_charge).ravel())
        no_of_doms = data.no_of_doms.read()
        try:
            all_energy = f.root.raw.true_muon_energy.read()
        except:
            all_energy = f.root.raw.true_primary_energy.read()
        toi_eval_ratios = f.root.raw.toi_evalratio.read()
        toi_ratios = pd.Series(toi_eval_ratios)
        energy = pd.Series(all_energy)
        out = pd.DataFrame(
            {
                'toi_ratio': toi_ratios,
                'energy': energy,
                'integrated_charge': integrated_charge,
                'no_of_doms': no_of_doms,
                'max_charge': max_charge
            }
        )
    return out


def read_event(events_file, event_no, group):
    """Read an hdf5 file and output activations and truth as DataFrame.

    Args:
        events_file (str): HDF5 events file path.
        event_no (int): Event number.
        truth_cols (list): List of truth variable columns names.

    Returns:
        activations (pandas.DataFrame): DataFrame containing event activations.
        truth (pandas.Series): Series containing truth variables.

    """
    with File(events_file, 'r') as f:
        event = {}
        vlen_dict = {}
        scalar_dict = {}
        data = f.root.__getattr__(group)
        all_integrated_charge = data.dom_charge.read()
        integrated_charge = list(
            map(
                sum,
                all_integrated_charge
            )
        )
        max_charge = max(np.concatenate(all_integrated_charge).ravel())
        min_charge = min(np.concatenate(all_integrated_charge).ravel())
        for array in data.__iter__():
            event[array.name] = array.read(
                start=event_no,
                stop=event_no + 1
            )
        for array in f.root.raw.__iter__():
            if array.name not in event:
                scalar_dict[array.name] = array.read(
                    start=event_no,
                    stop=event_no + 1
                )
        for key in event:
            if key == 'no_of_doms':
                scalar_dict[key] = event[key]
            elif type(event[key][0]) == np.ndarray:
                vlen_dict[key] = event[key][0]
            else:
                scalar_dict[key] = event[key]
        activations = pd.DataFrame.from_dict(vlen_dict)
        temp_charge = np.array(f.root.raw.dom_charge.read(
            start=event_no,
            stop=event_no + 1
        ))
        activations.dom_charge.loc[activations.dom_charge > 20] = 20
        activations.dom_charge = (
            (
                activations.dom_charge - min_charge
            )
            / (
                max_charge - min_charge
            )
        )
        activations.dom_time = (
            (
                activations.dom_time - activations.dom_time.min()
            )
            / (
                activations.dom_time.max() - activations.dom_time.min()
            )
        )
        truth = pd.DataFrame.from_dict(scalar_dict)
    return activations, truth


def direction_vectors(truth, scale):
    """Calculate direction vectors.

    Args:
        truth (pandas.Series): Series containing truth variables.
        key1 (str): String with name of first key.
        key2 (str): String with name of second key.

        Returns:
            x (list): List with two x points.
            y (list): List with two y points.
            z (list): List with two z points.

    """
    output = {'true': [], 'toi': []}
    keys = []
    keys.append(['true_primary_position', 'true_primary_direction'])
    keys.append(['toi_point_on_line', 'toi_direction'])
    keys_iter = iter(keys)
    for output_type in output:
        key = next(keys_iter)
        pol_x = truth[key[0] + '_x'].values[0]
        pol_y = truth[key[0] + '_y'].values[0]
        pol_z = truth[key[0] + '_z'].values[0]
        dir_x = truth[key[1] + '_x'].values[0] * scale
        dir_y = truth[key[1] + '_y'].values[0] * scale
        dir_z = truth[key[1] + '_z'].values[0] * scale
        x = [pol_x - dir_x, dir_x + pol_x]
        y = [pol_y - dir_y, dir_y + pol_y]
        z = [pol_z - dir_z, dir_z + pol_z]
        output[output_type] = (x, y, z)
    output['entry'] = (
        truth['true_primary_position_x'].values[0],
        truth['true_primary_position_y'].values[0],
        truth['true_primary_position_z'].values[0]
    )
    return output


def direction_vectors2(truth, scale):
    """Calculate direction vectors.

    Args:
        truth (pandas.Series): Series containing truth variables.
        key1 (str): String with name of first key.
        key2 (str): String with name of second key.

        Returns:
            x (list): List with two x points.
            y (list): List with two y points.
            z (list): List with two z points.

    """
    output = {'true': []}
    keys = []
    keys.append(['true_primary_entry_position', 'true_primary_direction'])
    keys_iter = iter(keys)
    for output_type in output:
        key = next(keys_iter)
        pol_x = truth[key[0] + '_x'].values[0]
        pol_y = truth[key[0] + '_y'].values[0]
        pol_z = truth[key[0] + '_z'].values[0]
        dir_x = truth[key[1] + '_x'].values[0] * scale
        dir_y = truth[key[1] + '_y'].values[0] * scale
        dir_z = truth[key[1] + '_z'].values[0] * scale
        x = [pol_x - dir_x, dir_x + pol_x]
        y = [pol_y - dir_y, dir_y + pol_y]
        z = [pol_z - dir_z, dir_z + pol_z]
        output[output_type] = (x, y, z)
    return output


def create_animation(data, template):
    """Create plotly express object with animation.

    Args:
        data (pandas.DataFrame): DataFrame to plot.

    Returns:
        fig (plotly.ExpressFigure): Plotly Express object.

    """
    fig = px.scatter_3d(
        data,
        x='dom_x',
        y='dom_y',
        z='dom_z',
        animation_frame='bin',
        size='dom_charge',
        template=template,
        range_x=[-axis_lims, axis_lims],
        range_y=[-axis_lims, axis_lims],
        range_z=[-650, 650]
    )
    return fig


def create_static(
    activations,
    geom,
    truth,
    predict,
    template,
    superpose,
    time_range
):
    """Create plotly express object without animation.

    Args:
        data (pandas.DataFrame): DataFrame to plot.

    Returns:
        fig (plotly.ExpressFigure): Plotly Express object.

    """
    # Create plot of activations
    vectors = direction_vectors(truth, 2000)
    fig = px.scatter_3d(
        activations,
        x='dom_x',
        y='dom_y',
        z='dom_z',
        size='dom_charge',       
        color='dom_time',
        template=template,
        range_x=[-axis_lims, axis_lims],
        range_y=[-axis_lims, axis_lims],
        range_z=[-axis_lims, axis_lims],
        color_continuous_scale=px.colors.diverging.RdYlBu,
        range_color=time_range,
        size_max=20
    )
    st.write(truth)
    # Add DOM geometry
    fig.add_scatter3d(
        x=geom.x,
        y=geom.y,
        z=geom.z,
        mode='markers',
        marker={'size': 0.8, 'color': 'black', 'opacity': 0.2},
        name='DOM'
    )
    if 'Truth' in superpose:
        fig.add_scatter3d(
            x=vectors['true'][0],
            y=vectors['true'][1],
            z=vectors['true'][2],
            mode='lines',
            name='Truth',
            marker={'size': 6.0, 'color': 'red'}
        )
        fig.add_scatter3d(
            x=[vectors['entry'][0]],
            y=[vectors['entry'][1]],
            z=[vectors['entry'][2]],
            marker={'color': 'red'},
            mode='markers',
            name='Entry'
        )
        fig.add_trace(
            go.Cone(
                x=[vectors['entry'][0]],
                y=[vectors['entry'][1]],
                z=[vectors['entry'][2]],
                u=100 * truth.true_primary_direction_x,
                v=100 * truth.true_primary_direction_y,
                w=100 * truth.true_primary_direction_z
            )
        )

    if 'ToI' in superpose:
        fig.add_scatter3d(
            x=vectors['toi'][0],
            y=vectors['toi'][1],
            z=vectors['toi'][2],
            mode='lines',
            name='ToI',
            marker={'size': 6.0, 'color': 'orange'}
        )
    if 'Predict' in superpose:
        fig.add_scatter3d(
            x=predict['true'][0],
            y=predict['true'][1],
            z=predict['true'][2],
            mode='lines',
            name='Prediction',
            marker={'size': 6.0, 'color': 'orange'}
        )
    return fig


def selection_func(choice):
    """Label selections with energy instead of event number.

    Args:
        input (int): Event number.

    Returns:
        energy_label (str): Stringified energy value.

    """
    label = str(round(10**select_meta.loc[choice], 2))
    return label


# @st.cache
def create_template():
    """Create a template for a plotly plot.

    Args:
        None

    Returns:
        icecube_template (plotly.graph_objects.layout.Template): Template.

    """
    icecube_template = go.layout.Template(
        layout=go.Layout(
            {
                'scene': {
                    'xaxis': {
                        'range': [-axis_lims, axis_lims],
                        'backgroundcolor': 'grey',
                        'showbackground': True,
                        'showgrid': False,
                        'showticklabels': False,
                        'title': 'x'
                    },
                    'yaxis': {
                        'range': [-axis_lims, axis_lims],
                        'backgroundcolor': 'grey',
                        'showbackground': True,
                        'showgrid': False,
                        'showticklabels': False,
                        'title': 'y'
                    },
                    'zaxis': {
                        'range': [-axis_lims, axis_lims],
                        'backgroundcolor': 'grey',
                        'showbackground': True,
                        'showgrid': False,
                        'showticklabels': False,
                        'title': 'z'
                    },
                    'aspectmode': 'cube',
                },
                'legend_orientation': 'h'
            }
        )
    )
    return icecube_template


def hist_maker(events_file, group, key):
    with open_file(events_file, 'r') as f:
        raw = f.root.histograms.raw
        group = f.root.histograms._f_get_child(group)
        if group.__contains__(key):
            array = group._f_get_child(key).read()
        else:
            array = raw._f_get_child(key).read()
    return array


def file_finder(data_set, index=None):
    events_path = files_path.joinpath(data_set)
    events_path = events_path.absolute()
    data_set_files = sorted(
        [f for f in events_path.iterdir() if f.is_file() and f.suffix == '.h5']
    )
    if index == None:
        return data_set_files
    else:
        return data_set_files[index]


def key_finder(data_set):
    events_file = file_finder(data_set, index=0)
    key_df = pd.DataFrame()
    with File(events_file, 'r') as f:
        array_iter = f.root.histograms.raw.__iter__()
        for array in array_iter:
            temp_df = pd.DataFrame(
                {
                    'key': array.name,
                    'length': array.nrows
                },
                index=[0]
            )
            key_df = key_df.append(temp_df)
    return key_df


def group_finder(data_set):
    groups = []
    events_file = file_finder(data_set, index=0)
    with File(events_file, 'r') as f:
        group_iter = f.root.histograms.__iter__()
        for group in group_iter:
            groups.append(group._v_name)
    return groups


def valid_predict_files_finder(predict_run_file):
    with File(predict_run_file, 'r') as f:
        valid_predict_files = list(f.root._v_children.keys())
    return valid_predict_files


def predictions(predict_file, file):
    predict_dict = {}
    with File(predict_file, 'r') as f:
        predict_group_iter = f.root._f_get_child(file).__iter__()
        for predict_group in predict_group_iter:
            predict_dict[predict_group.name] = predict_group.read()
    return predict_dict


particle_type_dict = {
    'oscnext-genie-level5-v01-01-pass2': 'neutrino',
    'MuonGun_Level2_139008': 'muon'
}

# Path to HDF5 event file
files_path = Path('../data/')

# Path to HDF5 geometry file
geom_file = './geom.h5'

# Sidebar title
st.sidebar.markdown('# *Powershovel^{TM}*')
st.sidebar.markdown('## IceCube event viewer')

show_dists = st.sidebar.selectbox(
    'View type',
    options=['Events', 'Distributions'],
    index=0
)

data_sets = [d.name for d in files_path.iterdir() if d.is_dir()]
data_set = st.sidebar.selectbox(
    'Select dataset',
    options=data_sets,
    index=0
)

# Columns containing truth
truth_cols = [
    'toi_point_on_line_x',
    'toi_point_on_line_y',
    'toi_point_on_line_z',
    'true_primary_direction_x',
    'true_primary_direction_y',
    'true_primary_direction_z',
    'true_primary_energy',
    'true_primary_position_x',
    'true_primary_position_y',
    'true_primary_position_z',
    'toi_direction_x',
    'toi_direction_y',
    'toi_direction_z',
    'toi_evalratio'
]

transformed_cols = [
    'charge',
    'dom_x',
    'dom_y',
    'dom_z',
    'time',
    'toi_point_on_line_x',
    'toi_point_on_line_y',
    'toi_point_on_line_z',
    'true_primary_energy',
    'true_primary_position_x',
    'true_primary_position_y',
    'true_primary_position_z'
]

groups = group_finder(data_set)
group = st.sidebar.selectbox(
    'Select key',
    options=groups
)

if show_dists == 'Events':
    predict_path = Path('../models').joinpath(data_set).joinpath('regression').joinpath('volapyk')
    if predict_path.exists():
        predict_types = [d for d in predict_path.iterdir() if d.is_dir()]
        predict_type = st.sidebar.selectbox(
            'Select prediction type',
            options=predict_types,
            format_func=lambda x: x.name,
            index=0
        )

        predict_runs = sorted(
            [
                d for d in predict_type.iterdir() if d.is_dir()
                and d.joinpath('data').exists()
            ]
        )
        st.write(str(predict_runs[0]))

        predict_run = st.sidebar.selectbox(
            'Select prediction run',
            options=predict_runs,
            format_func=lambda x: x.name,
            index=0
        )
        prediction_file = predict_run.joinpath('data/')
        prediction_file = [
            f for f in prediction_file.iterdir() if f.is_file() and f.suffix == '.h5'
        ]
        st.write(prediction_file)
        events_files = file_finder(data_set)
        events_names = [f.stem for f in events_files]
        valid_predict_files = valid_predict_files_finder(prediction_file[0])
        events_intersection = sorted(
            list(set(events_names) & set(valid_predict_files))
        )
        events_file = st.sidebar.selectbox(
            'Select file',
            options=events_intersection,
            index=0
        )
        events_file = files_path.joinpath(data_set).joinpath(events_file + '.h5')
        meta = meta_data(str(events_file), group)
        predict_dict = predictions(prediction_file[0], events_file.stem)
        predict_df = pd.DataFrame().from_dict(predict_dict)
        predict_df.drop(
            [
                'azi_error',
                'directional_error',
                'polar_error'
            ],
            axis=1,
            inplace=True
        )
    else:
        events_files = file_finder(data_set)
        events_file = st.sidebar.selectbox(
            'Select file',
            options=events_files,
            format_func=lambda x: x.stem,
            index=0
        )
        meta = meta_data(str(events_file), group)

# TODO block that only runs when user wants to see predictions

# TODO restrict file choices to only those that have prediction available

# TODO restrict event choices to only those that have prediction available

# TODO only show prediction options when predictions contain directions

# TODO histogram comparisons

# TODO bayesian blocks in hists

# Read HDF5 geometry file
geom = read_geom_file(geom_file, drop_cols)
axis_lims = geom.x.max() + 0.4 * geom.x.max()
template = create_template()

if show_dists == 'Events':
    manual = st.sidebar.radio(
        'Input event or browse?',
        options=['Input', 'Browse'],
        index=0
    )

    if manual == 'Input':
        event_no = st.sidebar.text_input(
            'Enter event number:',
            0
        )
        event_no = int(event_no)
    else:
        browse_type = st.sidebar.selectbox(
            'Browsing type',
            [
                'energy',
                'toi_ratio',
                'no_of_doms',
                'integrated_charge'
            ],
            index=0
        )
        sort = st.sidebar.radio(
            'Sort by',
            ['High', 'Low'],
            0
        )
        if sort == 'High':
            meta = meta.sort_values(browse_type, ascending=False)
        elif sort == 'Low':
            meta = meta.sort_values(browse_type, ascending=True)
        select_meta = meta[browse_type].iloc[0:100]
        event_no = st.sidebar.selectbox(
            'Select {} value of event'.format(browse_type),
            options=select_meta.index,
            format_func=selection_func
        )

    # Get activations and truth from selected event number
    activations, truth = read_event(
        str(events_file),
        event_no,
        group,
    )
    true_muon_entry_pos = truth[
        [
            'true_primary_position_x',
            'true_primary_position_y',
            'true_primary_position_z',
        ]
    ]
    if predict_path.exists():
        true_muon_direction = predict_df.loc[predict_df.index == event_no]
        true_muon_direction.drop(['index'], axis=1, inplace=True)
        true_muon_direction.reset_index(inplace=True)
        event_prediction_df = pd.concat(
            [
                true_muon_direction,
                true_muon_entry_pos
            ],
            axis=1
        )
        event_prediction_df.drop(['index'], axis=1, inplace=True)
        event_prediction_vector = direction_vectors2(event_prediction_df, 1000)

    activations = activations.sort_values('dom_time')
    time_range = [0, 1]
    st.write(
        'Selected event no. {}, length of event {}, ToI ratio {}, energy {}'
        .format(
            event_no,
            len(activations),
            round(truth.toi_evalratio.values[0], 3),
            float(round(10**truth['true_primary_energy'], 3))
        )
    )
    if predict_path.exists():
        superpose = st.multiselect(
            'Choose wisely',
            options=['Truth', 'ToI', 'Predict'],
        )
    else:
        superpose = st.multiselect(
            'Choose wisely',
            options=['Truth', 'ToI'],
        )
        event_prediction_vector = None
    # Create plotly figure
    fig = create_static(
        activations,
        geom,
        truth,
        event_prediction_vector,
        template,
        superpose,
        time_range
    )
    # Plot plotly figure
    st.plotly_chart(fig, width=0, height=700)
    st.text('Event dataframe:')
    st.write(activations.sort_values('dom_time'))
    doms_fig = px.histogram(meta, x='no_of_doms', log_y=False)
    st.plotly_chart(doms_fig)
    st.write('Mean number of DOMs:', meta.no_of_doms.mean())
elif show_dists == 'Distributions':
    events_files = file_finder(data_set)
    key_df = key_finder(data_set)
    hist_key = st.selectbox(
        'Select key',
        list(key_df.key.values),
        index=0
    )
    hists = []
    for events_file in events_files:
        hists.append(
            hist_maker(events_file, group, hist_key))
    hists = np.array(hists)
    hists = np.concatenate(hists).ravel()
    freqs, bins = np.histogram(hists, bins=300)
    width = 1 * (bins[1] - bins[0])
    centers = (bins[:-1] + bins[1:]) / 2
    fig = go.Figure(
        [
            go.Bar(
                x=centers,
                y=freqs,
                width=width
            )
        ]
    )
    if hist_key == 'charge':
        fig.update_layout(yaxis_type='log')
    st.plotly_chart(fig)