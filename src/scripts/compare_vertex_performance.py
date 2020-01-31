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

    if savefig:
        pd_h['savefig'] = savefig
    if title:
        pd_h['title'] = title

    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    return fig

def x_plot(models, perf_classes, title=None, savefig=None):
    # * t plot
    edges, y, yerr, label = [], [], [], []
    data, bins, weights, histtype, log = [], [], [], [], []
    for model, pc in zip(models, perf_classes):
        pd = pc.get_x_dict()
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
    y.append(pc.x_crs_sigmas)
    yerr.append(pc.x_crs_errors)
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

    if savefig:
        pd_h['savefig'] = savefig
    if title:
        pd_h['title'] = title
    
    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    
    return fig

def y_plot(models, perf_classes, title=None, savefig=None):
    # * t plot
    edges, y, yerr, label = [], [], [], []
    data, bins, weights, histtype, log = [], [], [], [], []
    for model, pc in zip(models, perf_classes):
        pd = pc.get_y_dict()
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
    y.append(pc.y_crs_sigmas)
    yerr.append(pc.y_crs_errors)
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
    if savefig:
        pd_h['savefig'] = savefig
    if title:
        pd_h['title'] = title
    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    return fig

def z_plot(models, perf_classes, title=None, savefig=None):
    # * t plot
    edges, y, yerr, label = [], [], [], []
    data, bins, weights, histtype, log = [], [], [], [], []
    for model, pc in zip(models, perf_classes):
        pd = pc.get_z_dict()
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
    y.append(pc.z_crs_sigmas)
    yerr.append(pc.z_crs_errors)
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
    if savefig:
        pd_h['savefig'] = savefig
    if title:
        pd_h['title'] = title
    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)

    # mod = pd['y'][0]
    # ice = pd['y'][2]
    # print(-(np.array(mod)-np.array(ice))/np.array(ice))
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

path = get_project_root() + '/plots/vertex_reg_z_with_without_t.png'
title = 'vertex z: Without t (blue) and with t (orange)'

fig = t_plot(models, perf_classes)#, title=title, savefig=path)
