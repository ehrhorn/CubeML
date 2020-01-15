import src.modules.helper_functions as hf
import src.modules.reporting as rpt
import argparse
import pickle
from src.modules.constants import *
from pathlib import Path
import numpy as np

def azi_plot(models, perf_classes, title=None, savefig=None):
    # * t plot
    edges, y, yerr, label = [], [], [], []
    data, bins, weights, histtype, log = [], [], [], [], []
    for model, pc in zip(models, perf_classes):
        pd = pc.get_azi_dict()
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
    y.append(pc.azi_crs_sigmas)
    yerr.append(pc.azi_crs_errors)
    label.append('Icecube')
    pd['edges'] = edges
    pd['y'] = y
    pd['yerr'] = yerr
    pd['label'] = label
    pd_h['data'] = data
    pd_h['bins'] = bins
    pd_h['weights'] = weights
    pd_h['histtype'] = histtype
    pd_h['log'] = log
    
    pd['grid'] = True
    pd['y_minor_ticks_multiple'] = 0.2
    if savefig:
        pd_h['savefig'] = savefig
    if title:
        pd_h['title'] = title

    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    return fig

def polar_plot(models, perf_classes, title=None, savefig=None):
    # * t plot
    edges, y, yerr, label = [], [], [], []
    data, bins, weights, histtype, log = [], [], [], [], []
    for model, pc in zip(models, perf_classes):
        pd = pc.get_polar_dict()
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
    y.append(pc.polar_crs_sigmas)
    yerr.append(pc.polar_crs_errors)
    label.append('Icecube')
    pd['edges'] = edges
    pd['y'] = y
    pd['yerr'] = yerr
    pd['label'] = label
    pd_h['data'] = data
    pd_h['bins'] = bins
    pd_h['weights'] = weights
    pd_h['histtype'] = histtype
    pd_h['log'] = log

    pd['grid'] = True
    pd['y_minor_ticks_multiple'] = 0.2
    if savefig:
        pd_h['savefig'] = savefig
    if title:
        pd_h['title'] = title

    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    return fig
# * VERTEX REG WITH OR WITHOUT MASK? BEST MODELS 
# * L2 loss simple model, squared_angle complex model
models = ['2020-01-14-19.48.33', '2020-01-15-00.45.13']
# models = ['2020-01-08-13.54.40', ]

perf_classes = []
for model in models:
    
    # * Locate the model directory
    paths = hf.find_files(model)
    for path in paths:
        if path.split('/')[-1] == model:
            break
    
    perf_class_path = path +'/data/AziPolarPerformance.pickle'
    perf_class = pickle.load( open( perf_class_path, "rb" ) )
    perf_classes.append(perf_class)

path = get_project_root() + '/plots/polar_L2_vs_sqr_angle.png'
title = 'polar: L2, 256 LSTM-size (blue). sqr. angle, 1028 LSTM-size (orange)'

fig = polar_plot(models, perf_classes, title=title, savefig=path)
