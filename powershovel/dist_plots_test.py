import streamlit as st
from tables import *
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 12)


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


def h5_variables_reader(DATA_SETS, group):
    variables = {}
    for data_set in DATA_SETS:
        variables[data_set.stem] = []
        with File(data_set, 'r') as f:
            array_iter = f.root.histograms._f_get_child(group).values.__iter__()
            for array in array_iter:
                variables[data_set.stem].append(array._v_name)
    return variables


HISTS_DIR = Path('/home/mads/')
DATA_SETS = [
    f for f in HISTS_DIR.glob('*.h5') if f.is_file() and f.stem != 'geom'
]

groups = h5_groups_reader(DATA_SETS[0])

group = st.sidebar.selectbox(
    label='Choose transform',
    options=groups
)

variables = h5_variables_reader(DATA_SETS, group)

variable = st.sidebar.selectbox(
    label='Choose variable',
    options=variables[DATA_SETS[0].stem],
    format_func=lambda x: x.replace('muon', 'particle')
)

hist, edges = h5_file_reader(DATA_SETS[0], group, variable)
centers = (edges[:-1] + edges[1:]) / 2

fig = go.Figure(
    data=go.Bar(
        x=centers,
        y=hist
    )
)
if variable == 'dom_charge':
    fig.update_layout(yaxis_type='log')
st.plotly_chart(fig)
