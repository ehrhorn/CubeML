import src.modules.helper_functions as hf
import src.modules.reporting as rpt
import argparse
import pickle
from pathlib import Path

def t_plot(models, perf_classes):
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

    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    return fig

def x_plot(models, perf_classes):
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

    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    return fig

def y_plot(models, perf_classes):
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

    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    return fig

def z_plot(models, perf_classes):
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

    fig = rpt.make_plot(pd)
    fig = rpt.make_plot(pd_h, h_figure=fig, axes_index=0)
    return fig

models = ['2020-01-08-21.01.31', '2020-01-08-13.54.40']
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

fig = y_plot(models, perf_classes)
