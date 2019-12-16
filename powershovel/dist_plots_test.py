import streamlit as st
from tables import *
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (10, 12)


def h5_file_reader(DATA_SETS, group, variable):
    hist = {}
    for data_set in DATA_SETS:
        hist[data_set.stem] = []
        with File(data_set, 'r') as f:
            if f.root._f_get_child(group).__contains__(variable):
                hist[data_set.stem] = f.root._f_get_child(group)._f_get_child(variable).read()
            else:
                variable = variable.replace('muon', 'neutrino')
                hist[data_set.stem] = f.root._f_get_child(group)._f_get_child(variable).read()
    return hist


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
            array_iter = f.root._f_get_child(group).__iter__()
            for array in array_iter:
                variables[data_set.stem].append(array._v_name)
    return variables


HISTS_DIR = Path('/groups/hep/ehrhorn/')
DATA_SETS = [
    f for f in HISTS_DIR.glob('**/*.h5') if f.is_file() and f.stem != 'geom'
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

hist = h5_file_reader(DATA_SETS, group, variable)

fig, axes = plt.subplots(nrows=2, ncols=1)
plot_colors = ['red', 'blue']

for i, ax in enumerate(axes):
    ax.hist(
        hist[DATA_SETS[i].stem],
        histtype='step',
        bins='fd',
        density=True,
        color=plot_colors[i]
    )
    if variable == 'dom_charge':
        ax.set_yscale('log')
    ax.set(
        xlabel=variable,
        ylabel='Density',
        title=DATA_SETS[i].stem
    )
st.pyplot(fig)

# fig = go.Figure(
#     data=go.Histogram(
#         x=hist
#     )
# )

# st.plotly_chart(fig)
