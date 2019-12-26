import streamlit as st
from tables import *
from pathlib import Path
import plotly.graph_objects as go
#import matplotlib.pyplot as plt
#plt.rcParams["figure.figsize"] = (10, 12)


def h5_file_reader(data_set, group, variable):
    with File(data_set, 'r') as f:
        hist = f.root.histograms._f_get_child(group).values._f_get_child(variable).read()
        edges = f.root.histograms._f_get_child(group).edges._f_get_child(variable).read()
    return hist, edges


def h5_groups_reader(data_file):
    groups = []
    with File(data_file, 'r') as f:
        group_iter = f.root.histograms.__iter__()
        for group in group_iter:
            groups.append(group._v_name)
    return groups


def h5_variables_reader(data_set, group):
    variables = {}
    variables[data_set.stem] = []
    with File(data_set, 'r') as f:
        array_iter = f.root.histograms._f_get_child(group).values.__iter__()
        for array in array_iter:
            variables[data_set.stem].append(array._v_name)
    return variables


HISTS_DIR = Path(__file__).resolve().parent
DATA_SETS = sorted([
    f for f in HISTS_DIR.glob('*.h5') if f.is_file() and f.stem != 'geom'
])

data_set = st.sidebar.selectbox(
    label='Chose particle type',
    options=DATA_SETS,
    format_func=lambda x: x.stem
)

groups = h5_groups_reader(data_set)

group = st.sidebar.selectbox(
    label='Choose transform',
    options=groups
)

variables = h5_variables_reader(data_set, group)

variable = st.sidebar.selectbox(
    label='Choose variable',
    options=variables[data_set.stem],
    format_func=lambda x: x.replace('muon', 'particle')
)

hist, edges = h5_file_reader(data_set, group, variable)
centers = (edges[:-1] + edges[1:]) / 2
width = edges[1] - edges[0]
bins = len(edges) - 1

st.write('No. of bins:', bins)

fig = go.Figure(
    data=go.Bar(
        x=centers,
        y=hist,
        width=width
    )
)

fig.update_layout(
    updatemenus=[
        go.layout.Updatemenu(
            buttons=list(
                [
                    dict(
                        label="Linear",
                        method="update",
                        args=[
                            {
                                'visible': [
                                    True,
                                    False
                                ]
                            },
                            {
                                'yaxis': {
                                    'type': 'linear'
                                }
                            }
                        ]
                    ),
                    dict(
                        label="Log",
                        method="update",
                        args=[
                            {
                                'visible': [
                                    True,
                                    True
                                ]
                            },
                            {
                                'yaxis': {
                                    'type': 'log'
                                }
                            }
                        ]
                    )
                ]
            ),
            pad={"r": 10, "t": 10},
            showactive=True,
        )
    ]
)

fig.update_traces(
    marker_color='black',
    marker_line_color='black',
)

st.plotly_chart(fig)
