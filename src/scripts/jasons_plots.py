import src.modules.helper_functions as hf
import src.modules.reporting as rpt
import argparse
import pickle
from src.modules.constants import *
from pathlib import Path
import numpy as np

def t_plot(models, perf_classes, title=None, savefig=None):
    # * t plot
    edges, y, yerr, label = [], [], [], []
    data, bins, weights, histtype, log = [], [], [], [], []
    for model, pc in zip(models, perf_classes):
        pd = pc.get_t_dict()
        edges.extend(pd['edges'])
        y.extend(pd['y'])
        yerr.extend(pd['yerr'])
        label.append(model)

        pd_h = pc.get_energy_dict()
        data.extend(pd_h['data'])
        bins.extend(pd_h['bins'])
        weights.extend(pd_h['weights'])
        histtype.extend(pd_h['histtype'])
        log.extend(pd_h['log'])
        del pd_h['color']

    edges.append(pc.bin_edges)
    y.append(pc.t_crs_sigmas)
    yerr.append(pc.t_crs_errors)
    pd['edges'] = edges
    pd['y'] = y
    pd['yerr'] = yerr
    pd['xlabel'] = r'$\log_{10}$E [E/GeV]'
    pd['ylabel'] = r'$\sigma_{t}$ [ns]'
    pd['yrange'] = [0, 420]
    if savefig:
        pd['savefig'] = savefig

    fig = rpt.make_plot(pd)
    return fig

# * VERTEX REG WITH OR WITHOUT MASK? BEST MODELS 
# * 16.57.20 IS WITH MASK, 21.01.31 IS WITHOUT
models = ['2020-01-11-01.23.49']
# models = ['2020-01-08-13.54.40', ]

perf_classes = []
for model in models:
    
    # * Locate the model directory
    paths = hf.find_files(model)
    for path in paths:
        if path.split('/')[-1] == model:
            break
    
    perf_class_path = path +'/data/VertexPerformance.pickle'
    perf_class = pickle.load( open( perf_class_path, "rb" ) )
    perf_classes.append(perf_class)

path = get_project_root() + '/reports/plots/jason_t.pdf'

fig = t_plot(models, perf_classes, savefig=path)


def energy_plot(models, perf_classes, title=None, savefig=None):
    # * t plot
    edges, y, yerr, label = [], [], [], []
    data, bins, weights, histtype, log = [], [], [], [], []
    for model, pc in zip(models, perf_classes):
        pd = pc.get_relE_dict()

        pd_edges = [pd['edges'][0][:]]
        pd_y = [pd['y'][0][:]]
        pd_yerr = [pd['yerr'][0][:]]   

        edges.extend(pd_edges)
        y.extend(pd_y)
        yerr.extend(pd_yerr)
        label.append(model)


    edges.append(pc.bin_edges)
    y.append(pc.relE_crs_sigmas)
    yerr.append(pc.relE_crs_errors)
    label.append('Icecube')
    pd['edges'] = edges
    pd['y'] = y
    pd['yerr'] = yerr
    pd['xlabel'] = r'$\log_{10}$E [E/GeV]'
    pd['ylabel'] = r'Relative Error'
    if savefig:
        pd['savefig'] = savefig

    fig = rpt.make_plot(pd)
    return fig

# * ENERGY REG VS POINTNET 
# * Stacked LSTM 256 HUBER LOSS, L2 1024 LSTM, LSTM 512 HUBER LOSS
models = ['2020-01-19-22.00.11']

# models = ['2020-01-08-13.54.40', ]

perf_classes = []
for model in models:
    
    # * Locate the model directory
    paths = hf.find_files(model)
    for path in paths:
        if path.split('/')[-1] == model:
            break
    
    perf_class_path = path +'/data/EnergyPerformance.pickle'
    perf_class = pickle.load( open( perf_class_path, "rb" ) )
    perf_classes.append(perf_class)

path = get_project_root() + '/reports/plots/jason_energy.pdf'

fig = energy_plot(models, perf_classes, savefig=path)