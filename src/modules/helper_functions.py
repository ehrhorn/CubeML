from time import localtime, strftime
import random
from pathlib import Path
import pickle
from matplotlib import pyplot as plt
from torch import optim
import wandb
from math import ceil
import shutil
import torch
import h5py as h5
import numpy as np
import json
from ignite.engine import Events

import src.modules.loss_funcs


def append_list_and_save(list_address, item):
    """Opens or creates a .pickle-file containing a list and appends a number to it.
    
    Arguments:
        list_address {string} -- Full path to file of interest.
        item {float} -- Number to add to list.
    """    
    
    if Path(list_address).is_file():
        with open(list_address, 'rb') as f:
            to_update = pickle.load(f)
        
        to_update.append(item)
        pickle.dump(to_update, open(list_address, 'wb'))
    else:
        with open(list_address, 'wb') as f:
            pickle.dump([item], f)

def bin_data(l1, l2, bin_edges):
    """Expects sorted lists. Splits lists into several lists induced by l1 and edges.
    
    Arguments:
        l1 {array-like} -- Dataset of independent variable to create bins from
        l2 {array-like} -- Dataset of dependent variables.
        bin_edges {array-like} -- Array of bin edges
    
    Returns:
        l1_bins [List] -- List of lists containing the binned data
        l2_bins [List] -- List of lists containing the binned data
    """    
    l1_bins = []
    l2_bins = []
    n = 0
    for lower_bin, upper_bin in zip(bin_edges[:-1], bin_edges[1:]):
        lower_i = np.searchsorted(l1, lower_bin)
        upper_i = np.searchsorted(l1, upper_bin, side='right')
        l1_bins.append(l1[lower_i:upper_i])
        l2_bins.append(l2[lower_i:upper_i])

    return l1_bins, l2_bins

def delete_nohup_file(path='/src/scripts/nohup.out'):
    try:
        Path(get_project_root()+path).unlink()
        print('Deleted', get_project_root()+path)
    except FileNotFoundError:
        pass

def calc_bin_centers(edges):
    """Calculates the bin centers of a histogram given a list of bin widths.
    
    Arguments:
        edges {List} -- list of edges in a histogram
    
    Returns:
        List -- list of bin centers
    """    
    centers = []

    for lower, upper in zip(edges[:-1], edges[1:]):
        center = (upper+lower)/2.0
        centers.append(center)

    return centers

def calc_histogram(sorted_data, n_bins=10, mode='equal_amount'):
    '''Calculates the histogram with n_bins from a sorted list. Mode can be either equal_amount or same_width - equal_amount puts same, given number in each bin, whereas same_width gives every bin the same width

    input
    sorted_data: A sorted list of data

    Output
    entries: List of number of entries in each bin
    edges: List of bin edges.
    '''

    if mode == 'equal_amount':
        #*put equal amount in each bin
        edges = [sorted_data[0]]
        entries = []
        
        try:
            tot = sorted_data.shape[0]
        except AttributeError:
            tot = len(sorted_data)

        per_bin = (0.0+tot)/n_bins
        for i in range(1, n_bins):
            edges.append(sorted_data[int(per_bin*i)])
            entries.append(int(per_bin*i)-int(per_bin*(i-1)))
        
        edges.append(sorted_data[-1])
        entries.append(int(per_bin*(i+1))-int(per_bin*i))
    else:
        raise ValueError('Unknown mode given to calc_histogram!')

    return entries, edges

def calc_dists_to_binedges(edges):
    '''GIven a list of binedges, returns a list of each bincenters distance to the edges.
    '''
    dists = []

    for lower, upper in zip(edges[:-1], edges[1:]):
        dist = (upper-lower)/2
        dists.append(dist)

    return dists

def calc_iqr(data):
    '''Calculats the interquartile range of a dataset. Ignores nans.
    '''
    return np.nanpercentile(data, 75)-np.nanpercentile(data, 25)

def calc_MAEs(sorted_data, entries, error_measure='median'):
    '''Calculates the error-measure (mean or median, default = 'median') of the data in each bin induced by entries.

    input
    sorted_data: list of sorted data
    entries: list of number of entries in each bin

    output
    maes: list of MAE in each bin
    '''
    from_to = np.append(0, np.cumsum(entries))

    #*calculate MAE
    maes = []
    for lower, upper in zip(from_to[:-1], from_to[1:]):
        
        if error_measure == 'mean':
            maes.append(np.nanmean(sorted_data[lower:upper]))
        elif error_measure == 'median':
            maes.append(np.nanmedian(sorted_data[lower:upper]))
        else:
            raise ValueError('Unknown error measure encountered!')
    
    return maes

def calc_perf_as_fn_of_energy(energy, predictor_vals, n_bins=15):
    '''Calculates error histograms as a function of energy.

    Input
    true: list of true values
    predictor_vals: list of predictions

    returns: lists of edges, Median Absolute errors and interquartile errors.
    '''

    energy_sorted, predictor_vals_sorted = sort_pairs(energy, predictor_vals)

    entries, edges = calc_histogram(energy_sorted, n_bins=n_bins)
    maes = calc_MAEs(predictor_vals_sorted, entries)
    widths_lower, widths_upper = calc_widths(predictor_vals_sorted, entries)

    return edges, maes, [widths_lower, widths_upper]

def calc_perf2_as_fn_of_energy(energy, predictor_vals, bin_edges):
    '''Calculates error histograms as a function of energy.

    Input
    true: list of true values
    predictor_vals: list of predictions

    returns: lists of edges, Median Absolute errors and interquartile errors.
    '''

    energy_sorted, predictor_vals_sorted = sort_pairs(energy, predictor_vals)
    _, predictor_bins = bin_data(energy_sorted, predictor_vals_sorted, bin_edges)
    
    sigmas, e_sigmas = [], []
    for entry in predictor_bins:
        means, plussigmas, minussigmas = estimate_percentile(entry, [0.25, 0.75])
        e_quartiles = []
        e_quartiles.append((plussigmas[0]-minussigmas[0])/2)
        e_quartiles.append((plussigmas[1]-minussigmas[1])/2)

        #*Assume errors are symmetric - which they look to be (quick inspection)
        #*Look at plussigma[0]-mean[0], mean[0]-minussigma[0] for instance
        sigma, e_sigma = convert_iqr_to_sigma(means, e_quartiles)
        
        #*Ignore nans - it is due to too little statistics in a bin
        if e_sigma != e_sigma:
            sigma = np.nan
            
        sigmas.append(sigma)
        e_sigmas.append(e_sigma)

    return sigmas, e_sigmas

def calc_relative_error(l1, l2, e1=None, e2=None):
    """Calculates the relative error (l2-l1) / l1 wrt the values of l1 and propagates uncertainties if given.

    Arguments:
        l1 {array-like} -- list of values
        l2 {array_like} -- list of values, must be equal in length to l1

    Keyword Arguments:
        e1 {array_like} -- Potential list of errors (default: {None})
        e2 {array_like} -- potential list of errors (default: {None})
    
    Returns:
        [np.array] -- relative errors, error on relative errors
    """    
    
    if isinstance(l1, list):
        l1 = np.array(l1)
    if isinstance(l2, list):
        l2 = np.array(l2)  
    
    if isinstance(e1, list):
        e1 = np.array(e1) 
    elif e1 is None:
        e1 = np.zeros(l1.shape)

    if isinstance(e2, list):
        e2 = np.array(e2)
    elif e2 is None:
        e2 = np.zeros(l2.shape)    
    
    rel_e = (l2-l1)/l1

    term1 = (e2/l1)**2
    term2 = (e1*l2/l1**2)**2

    sigma_e = np.sqrt(term1 + term2)

    return rel_e, sigma_e

def calc_widths(sorted_data, entries, width_measure='iqr'):
    '''Calculates the width-measure ('std' or 'iqr', default = 'iqr', IGNORES NANS) of the data in each bin induced by entries.

    input
    sorted_data: list of sorted data
    entries: list of number of entries in each bin

    output
    widths: list of measures of width in each bin
    '''

    from_to = np.append(0, np.cumsum(entries))
    if width_measure == 'iqr':
        widths_lower, widths_upper = [], []
    else:
        widths = []
    for lower, upper in zip(from_to[:-1], from_to[1:]):
        
        if width_measure == 'std':
            widths.append(np.std(sorted_data[lower:upper]))
        elif width_measure == 'iqr':
            width_lower = np.nanpercentile(sorted_data[lower:upper], 50)- np.nanpercentile(sorted_data[lower:upper], 25)
            width_upper = np.nanpercentile(sorted_data[lower:upper], 75)- np.nanpercentile(sorted_data[lower:upper], 50)
            
            widths_lower.append(width_lower)
            widths_upper.append(width_upper)
        else:
            raise ValueError('Unknown width measure encountered!')
    
    if width_measure == 'iqr':
        return widths_lower, widths_upper
    else:
        return widths

def convert_id_to_int(file_path, id_str):
    '''A very non-modular conversion of a file ID.. For each new dataset, a new converter must be added.
    '''
    dataset = get_dataset_name(file_path)
    if dataset == 'oscnext-genie-level5-v01-01-pass2':
        id_stripped = id_str.split('.')[-1]
        id_int = int(id_stripped.split('__')[0])
    elif dataset == 'MuonGun_Level2_139008':
        id_int = int(id_str)
    
    return id_int

def convert_iqr_to_sigma(quartiles, e_quartiles):
    '''Converts 75th and 25th percentiles (with errors) to sigma with error.
    
    From https://en.wikipedia.org/wiki/Interquartile_range
    IQR = 1.349*sigma - therefore sigma = IQR/1.349
    '''
    factor = 1/1.349
    sigma = np.abs(quartiles[1]-quartiles[0])*factor
    e_sigma = factor*np.sqrt(e_quartiles[0]**2 + e_quartiles[1]**2)
    
    return sigma, e_sigma

def convert_key(d, old_key, new_key):
    d[new_key] = d[old_key]
    del d[old_key]
    return d

def convert_keys(d, old_keys, new_keys):
    
    for old_key, new_key in zip(old_keys, new_keys):
        d = convert_key(d, old_key, new_key)
    
    return d

def estimate_percentile(data, percentiles, n_bootstraps=1000):
    '''Uses bootstrapping and order statistics to estimate a percentile and its error.

    A confidence interval of +-1 sigma is created for each percentile 
    '''
    #*Convert to np-array and sort (if it hasnt been already)
    data = np.array(data)
    n = data.shape[0]
    data.sort()
    i_means, means = [], []
    i_plussigmas, plussigmas = [], []
    i_minussigmas, minussigmas = [], []


    for percentile in percentiles:
        #*THe order statistic is binomially distributed - we approximate it with a gaussian. 
        sigma = np.sqrt(percentile*n*(1-percentile))
        mean = n*percentile
        i_means.append(int(mean))
        i_plussigmas.append(int(mean+sigma+1))
        i_minussigmas.append(int(mean-sigma))
    
    
    #*bootstrap
    bootstrap_indices = np.random.choice(np.arange(0, n), size=(n, n_bootstraps))
    bootstrap_indices.sort(axis=0)
    bootstrap_samples = data[bootstrap_indices]

    #*An IndexError is due to too few events in a bin. Set these to NaN, we don't care about bins with very little data, so dont log the error.
    for i in range(len(i_means)):
        
        try:    
            mean = bootstrap_samples[i_means[i], :]
            means.append(np.mean(mean))

            plussigma = bootstrap_samples[i_plussigmas[i], :]
            plussigmas.append(np.mean(plussigma))

            minussigma = bootstrap_samples[i_minussigmas[i], :]
            minussigmas.append(np.mean(minussigma))
        except IndexError:
            means.append(np.nan)
            plussigmas.append(np.nan)
            minussigmas.append(np.nan)

    return means, plussigmas, minussigmas

def find_best_model_pars(model_dir):
    '''Scans through saved model parameters in a model directory and returns the parameter-file of the best model.
    '''
    best = -1
    for file in Path(model_dir).iterdir():
        loss = float(str(file).split('Loss=')[-1].split('.pth')[0])
        
        if loss < best or best == -1:
            best = loss
            best_pars = file

    return best_pars

def find_files(name, main_dir=None):
    '''Searches main_dir and its subdirectories for a files or directories with name in it

    Input
    name: string of file or directory wanted

    Output
    files_str: list containing strings of paths found.
    '''
    if main_dir == None:
        main_dir = get_project_root()+'/models/'
    name = name.split('/')[-1]
    main_dir = Path(main_dir)
    name_altered = '*'+name+'*'
    files = main_dir.rglob(name_altered)
    files_str = []
    for file in files:
        files_str.append(str(file))
    
    return files_str

def get_dataloader_params(batch_size, num_workers=8, shuffle=False, dataloader=None):

    if dataloader == 'FullBatchLoader':
        dataloader_params = {'batch_size': None, 'shuffle': False, 'num_workers': num_workers}
    else:
        dataloader_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
    
    return dataloader_params

def get_dataset_name(file_path):
    '''Given a path to a file, the dataset name is extracted
    '''
    from_root = get_path_from_root(file_path)
    from_root_splitted = from_root.split('/')
    
    #*Expects the format chosen for the framework
    if from_root_splitted[1] == 'data' or from_root_splitted[1] == 'models':
        name = from_root_splitted[2]
    else:
        'Unknown format given!'
    
    return name

def get_dataset_size(data_dir):
    """Loops over a data directory and returns the total number of events
    
    Arguments:
        data_dir {str} -- relative or full path to data directory
    
    Returns:
        float -- number of files, average number of events per file and std on number of events pr file
    """    
    
    n_events = 0.0
    n_events_sqr = 0.0
    path = get_project_root() + get_path_from_root(data_dir)
    n_files = 0.0
    for file in Path(path).iterdir():
        if file.suffix == '.h5':  
            n_files += 1.0  
            with h5.File(file, 'r') as f:
                n_events += f['meta/events'][()]
                n_events_sqr += f['meta/events'][()]**2
    mean = n_events/n_files
    std = np.sqrt(n_events_sqr/n_files - mean**2)
    return n_files, mean, std

def get_device(ID=0):
    cuda = 'cuda:'+str(ID)
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    return device

def get_indices_from_fraction(n, from_frac, to_frac):
    '''Converts the interval induced by from_frac and to_frac to a list of indices of a list of length n.
    '''
    frac = to_frac-from_frac
    
    n_below = int(from_frac*n +0.5)
    leftover = n_below - from_frac*n

    n_wanted = int(frac*n - leftover + 0.5)

    indices = list(range(n_below, n_below+n_wanted))
    
    return indices

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr_scheduler(lr_dict, optimizer, batch_size, n_train):

    #*Simply multiplies lr by 1 on every iteration - equal to no lr-schedule
    if lr_dict['lr_scheduler'] == None:
        lambda1 = lambda step: 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    elif lr_dict['lr_scheduler'] == 'CyclicLR':
        #*Some default values
        #*{'lr_scheduler':   'CyclicLR',
        #*'base_lr':        0.00001,
        #*'max_lr':         0.001,
        #*'period':         12, #*in epochs
        #*'cycle_momentum': False}

        if 'cycle_momentum' in lr_dict: 
            cycle_momentum = lr_dict['cycle_momentum']
        else: 
            cycle_momentum = True #*default value from Torch
        
        if 'period' in lr_dict: 
            n_steps = 0.5*n_train*lr_dict['period']/batch_size
            step_size_up = int(n_steps)
        else: step_size_up = 2000 #*default value from Torch

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_dict['base_lr'], max_lr = lr_dict['max_lr'], step_size_up=step_size_up, cycle_momentum=cycle_momentum)
    
    elif lr_dict['lr_scheduler'] == 'ReduceLROnPlateau':
        #*Some default values
        #*{'lr_scheduler':   'ReduceLROnPlateau',
        #*'factor':         0.1,
        #*'patience':       2,
        #*}

        pars = {}
        if 'factor' in lr_dict:
            pars['factor'] = lr_dict['factor']
        if 'patience' in lr_dict:
            pars['patience'] = lr_dict['patience']
        if 'min_lr' in lr_dict:
            pars['min_lr'] = lr_dict['min_lr']
        if 'cooldown' in lr_dict:
            pars['cooldown'] = lr_dict['cooldown']
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, **pars)
    
    return scheduler

def get_set_length(dataloader):
    '''Determines the set length given a custom setholder for Torch's dataloader. Defaults to returning len(dataloader).
    '''
    if str(dataloader) == 'FullBatchLoader':
        set_length = dataloader.batch_size*len(dataloader)
    else:
        set_length = len(dataloader)
    
    return set_length

def get_n_parameters(model):
    params = 0
    for parameter in model.parameters():
        layer_params = 1
        for entry in parameter.shape:
            layer_params *= entry

        params += layer_params
    return params

def get_n_train_val_test(n_data, train_frac, val_frac, test_frac):
    if train_frac+val_frac+test_frac > 1.0:
        raise ValueError('\nERROR: Train, validation and test fractions add up to more than one !\n')
        return 0, 0, 0

    n_train = int(train_frac*n_data +0.5)
    leftover = n_train - train_frac*n_data

    n_val = int(val_frac*n_data - leftover +0.5)
    leftover = n_val - (val_frac*n_data - leftover)

    n_test = int(test_frac*n_data - leftover +0.5)

    return n_train, n_val, n_test

def get_optimizer(model_pars, d_opt):
    '''Returns desired optimizer to train().
    
    input: Model parameters, dict with optimizer parameters
    returns: optimizer.
    '''
    if d_opt['optimizer'] == 'Adam':
        if 'lr' in d_opt: lr = d_opt['lr']
        else: lr = 0.001 #*standard Adam lr

        if 'betas' in d_opt: betas = d_opt['betas']
        else: betas = (0.9, 0.999) #*standard Adam par

        if 'eps' in d_opt: eps = d_opt['eps']
        else: eps = 1e-08 #*default Adam par

        if 'weight_decay' in d_opt: weight_decay = d_opt['eps']
        else: weight_decay = 0 #*default Adam par
        
        return optim.Adam(model_pars, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
    
    else:
        raise ValueError('Unknown optimizer chosen!')

def get_path_from_root(path):
    '''Given a path, get_path_from_root strips the potential path to the root directory.

    Input
    path: a string
    
    Output
    path_from_root: a string with the path from the root directory
    '''
    split = path.split('/')

    cond1 = split[0] == '' and split[1] == 'CubeML'
    cond2 = split[0] == 'CubeML'
    
    if cond1:
        path_from_root = '/'+'/'.join(split[2:])
    elif cond2:
        path_from_root = '/'+'/'.join(split[1:])
    elif 'CubeML' not in split:
        path_from_root = path      
    else:

        i = 0
        while True:
            if split[i] == 'CubeML':
                path_from_root = '/'+'/'.join(split[i+1:])
                break
            else:
                i += 1
    return path_from_root

def get_project_root():
    '''Finds project root - useful for running code on different machines.
    
    Inputs
    None

    Output
    path as a string to project root.
    '''
    #*Find project root
    current_dir_splitted = str(Path.cwd()).split('/')
    i = 0
    while current_dir_splitted[i] != 'CubeML':
        i += 1
    return '/'.join(current_dir_splitted[:i+1]) 

def get_train_val_test_indices(n_data, train_frac, val_frac, test_frac, shuffle = True):
    '''Split a dataset into a test-sample with test_frac*n_data datapoints, a training-sample with (1-test_frac)*train_frac*n_data datapoints and a validation set with (1-test_frac)*(1-train_frac)*n_data datapoints.
    '''
    
    n_train, n_val, n_test = get_n_train_val_test(n_data, train_frac, val_frac, test_frac)
    train_end = n_train 
    val_end = train_end + n_val
    test_end = val_end + n_test

    indices = list(range(test_end))

    return indices[:train_end], indices[train_end:val_end], indices[val_end:test_end]

def inverse_transform(data, model_dir):
    '''
    Input 
    data: a dictionary with matching keys between data and loaded transformer.
    model-dir: full path to used model

    Output: dictionary with transformed data
    '''

    try:
        transformers = pickle.load( open(model_dir+'/transformers.pickle', "rb"))
    except FileNotFoundError:
        transformers = None
    transformed = {}

    if transformers == None:
        for key in data:
            transformed[key] = data[key]
    
    else:
        for key in data:

            if transformers[key] == None: 
                transformed[key] = data[key]

            #*The reshape is required for scikit to function...
            else: 
                #*Input might be given as a dictionary of lists
                try: 
                    transformed[key] = transformers[key].inverse_transform(data[key].reshape(-1, 1))
                except AttributeError: 
                    transformed[key] = transformers[key].inverse_transform(np.array(data[key]).reshape(-1, 1))

    return transformed

#*TODO: delete inverse_transform_predictions and only use inverse_transform, predict() needs adjusting
def inverse_transform_predictions(preds, keys, model_dir):

    try:
        transformers = pickle.load( open(model_dir+'/transformers.pickle', "rb"))
    except FileNotFoundError:
        transformers = None
    transformed = {}
    
    if transformers == None:
        for key in keys:
            transformed[key] = preds[key]
    
    else:
        for key in keys:

            if transformers[key] == None: 
                transformed[key] = preds[key]

            #*The reshape is required for scikit to function...
            else: 
                #*Input might be given as a dictionary of lists
                try: 
                    transformed[key] = transformers[key].inverse_transform(preds[key].reshape(-1, 1))
                except AttributeError: 
                    transformed[key] = transformers[key].inverse_transform(np.array(preds[key]).reshape(-1, 1))

    return transformed

def load_model_pars(model_dir):
    '''loads and returns hyper-pars, data-pars, arch-pars and meta-pars from a directory

    Input:
    model_dir: Path to model directory as a string.
    '''    
    model_dir = get_path_from_root(model_dir)
    with open(get_project_root() + model_dir + "/architecture_pars.json", 'r') as f: 
        arch_pars = json.load(f)
    with open(get_project_root() + model_dir + "/data_pars.json", 'r') as f: 
        data_pars = json.load(f)
    with open(get_project_root() + model_dir + "/hyper_pars.json", 'r') as f: 
        hyper_pars = json.load(f)
    with open(get_project_root() + model_dir + "/meta_pars.json", 'r') as f: 
        meta_pars = json.load(f)
    
    return hyper_pars, data_pars, arch_pars, meta_pars

def log_weights_and_grads(i_layer, layer, step):
    '''Logs weight- and gradienthistograms to wandb
    '''
    if type(layer) == torch.nn.modules.linear.Linear:
        name = 'L'+str(i_layer)+'_'+str(layer).split('(')[0]
        
        weight = layer.weight.view(-1).detach().cpu().numpy()
        weight_grad = layer.weight.grad.view(-1).detach().cpu().numpy()

        bias = layer.bias.view(-1).detach().cpu().numpy()
        bias_grad = layer.bias.grad.view(-1).detach().cpu().numpy()
        
        wandb.log({'Weights/'+name+'_weights': wandb.Histogram(weight)}, step=step)
        wandb.log({'Weights/'+name+'_bias': wandb.Histogram(bias)}, step=step)

        wandb.log({'Gradients/'+name+'_weights': wandb.Histogram(weight_grad, num_bins = 256)}, step=step)
        wandb.log({'Gradients/'+name+'_bias': wandb.Histogram(bias_grad, num_bins = 256)}, step=step)
        
        return i_layer+1

    elif type(layer) == torch.nn.modules.rnn.LSTM:
        name = 'L'+str(i_layer)+'_'+str(layer).split('(')[0]

        weight_ih = layer.weight_ih_l0.view(-1).detach().cpu().numpy()
        weight_hh = layer.weight_hh_l0.view(-1).detach().cpu().numpy()
        bias_ih = layer.bias_ih_l0.view(-1).detach().cpu().numpy()
        bias_hh = layer.bias_hh_l0.view(-1).detach().cpu().numpy()

        weight_ih_grad = layer.weight_ih_l0.grad.view(-1).detach().cpu().numpy()
        weight_hh_grad = layer.weight_hh_l0.grad.view(-1).detach().cpu().numpy()
        bias_ih_grad = layer.bias_ih_l0.grad.view(-1).detach().cpu().numpy()
        bias_hh_grad = layer.bias_hh_l0.grad.view(-1).detach().cpu().numpy()

        wandb.log({'Weights/'+name+'_weight_ih': wandb.Histogram(weight_ih)}, step=step)
        wandb.log({'Weights/'+name+'_weight_hh': wandb.Histogram(weight_hh)}, step=step)
        wandb.log({'Weights/'+name+'_bias_ih': wandb.Histogram(bias_ih)}, step=step)
        wandb.log({'Weights/'+name+'_bias_hh': wandb.Histogram(bias_hh)}, step=step)

        wandb.log({'Gradients/'+name+'_weight_ih': wandb.Histogram(weight_ih_grad, num_bins = 256)}, step=step)
        wandb.log({'Gradients/'+name+'_weight_hh': wandb.Histogram(weight_hh_grad, num_bins = 256)}, step=step)
        wandb.log({'Gradients/'+name+'_bias_ih': wandb.Histogram(bias_ih_grad, num_bins = 256)}, step=step)
        wandb.log({'Gradients/'+name+'_bias_hh': wandb.Histogram(bias_hh_grad, num_bins = 256)}, step=step)
        
        return i_layer+1
    else:
        return i_layer 

def make_lr_dir(data_folder_address, project, batch_size):
    '''Makes a lr-finder folder at CubeML/models/<DATA_TRAINED_ON>/lr_finders/BS#_<WHEN_CREATED>.
    '''

    #*Get CubeML/models/ directory
    models_dir = get_project_root()+'/models/'

    #*See if model/dataset exists - if not, make it
    data_name = data_folder_address.split('/')
    if data_name[-1] == '': 
        data_name = data_name[-2]
    else: 
        data_name = data_name[-1]

    if not Path(models_dir+data_name).is_dir(): 
        Path(models_dir+data_name).mkdir()
    data_dir = models_dir+data_name+'/'

    #*Make lr-finder directory if it doesnt exist
    if not Path(data_dir+'lr_finders').is_dir(): 
        Path(data_dir+'lr_finders').mkdir()
    lr_finders_dir = data_dir+'lr_finders/'

    #*Finally! lr-finder dir
    if project.split('_')[-1] == 'test':
        base_name = 'test_BS'+str(batch_size)+'_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())
    else:
        base_name = 'BS'+str(batch_size)+'_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())

    Path(lr_finders_dir+base_name).mkdir()
    lr_dir = lr_finders_dir+base_name
    
    return lr_dir

def make_model_dir(reg_type, data_folder_address, clean_keys, project):
    '''Makes a model folder at CubeML/models/<DATA_TRAINED_ON>/regression/<REGRESSION_TYPE>/<WHEN_TRAINED>/ with subdirectories figures and data.
    '''

    #*Get CubeML/models/ directory
    models_dir = get_project_root()+'/models/'

    #*See if model/dataset exists - if not, make it
    data_name = data_folder_address.split('/')
    if data_name[-1] == '': data_name = data_name[-2]
    else: data_name = data_name[-1]

    if not Path(models_dir+data_name).is_dir(): Path(models_dir+data_name).mkdir()
    data_dir = models_dir+data_name+'/'

    #*Make regression directory if it doesnt exist
    if not Path(data_dir+'regression').is_dir(): Path(data_dir+'regression').mkdir()
    regression_dir = data_dir+'regression/'

    #*Make regression type directory if it doesnt exist
    if not Path(regression_dir+reg_type).is_dir(): Path(regression_dir+reg_type).mkdir()
    reg_type_dir = regression_dir+reg_type
    
    #*Finally! Make model directory
    if project.split('_')[-1] == 'test':
        base_name = 'test_'+strftime("%Y.%m.%d-%H.%M.%S", localtime())
    else:
        base_name = strftime("%Y-%m-%d-%H.%M.%S", localtime())

    Path(reg_type_dir+'/'+base_name).mkdir()
    model_dir = reg_type_dir+'/'+base_name
    
    #*Add subdirectories
    Path(model_dir+'/figures').mkdir()
    Path(model_dir+'/data').mkdir()

    #*Copy transform dicts
    transformer_address = get_project_root()+'/data/'+data_name+'/transformers/'+'transform'+str(clean_keys['transform'])+'.pickle'
    try:
        data_dir = shutil.copy(transformer_address, model_dir+'/transformers.pickle')
    except FileNotFoundError:
        pass

    return model_dir

def read_h5_dataset(file_address, key, prefix=None, from_frac=0, to_frac=1, indices=None):
    '''Reads a dataset from a h5-file induced by keys and prefix.

    Inputs
    file_address: full address as string to h5-file
    key: Key/name of variable to read.
    prefix: If required, a string to ensure correct path in file (path is prefix+/+key)
    from_frac: Used to calculate the index to read from; index = int( N_data_in_file*from_frac+0.5)
    to_frac: The index to read to. Calculated as from_frac.

    output: Array with desired dataset.
    '''

    with h5.File(file_address, 'r') as f:
        n_events = f['meta/events'][()]
        if indices == None:
            indices = get_indices_from_fraction(n_events, from_frac, to_frac)
        
        #*If not transformed, it is found under raw/
        try:
            path = prefix+'/'+key
            data = f[path][indices]
        except KeyError:
            path = 'raw/'+key

            data = f[path][indices]
    
    return data

def read_h5_directory(data_dir, keys, prefix=None, from_frac = 0, to_frac = 1):
    '''Loops over each h5-file in a directory and reads the datasets induced by keys and prefix.

    Inputs
    data_dir: path from project root to the data directory.
    keys: list of keys/variablenames to read.
    prefix: If required, a string to ensure correct path in file (path is prefix+/+key)
    from_frac: Used to calculate the index to read from; index = int( N_data_in_file*from_frac+0.5)
    to_frac: The index to read to. Calculated as from_frac.

    output: Dictionary with desired datasets.
    '''
    values = {key: np.array([]) for key in keys}
    values = {key: {} for key in keys}

    
    for file in Path(get_project_root()+data_dir).iterdir():
        
        if file.suffix == '.h5':
            for key in keys:
                values[key][file.stem] = read_h5_dataset(file, key, prefix, from_frac=from_frac, to_frac=to_frac)
    
    #*Sort wrt file index
    values_sorted = sort_wrt_file_id(str(file), values)

    return values_sorted

def read_predicted_h5_data(file_address, keys):
    '''Reads predictions from a h5-file.

    Inputs
    file_address: full address as string to h5-file with structure dfile_name/key for each dfile.
    keys: list of keys of interest

    output: dictionary with predictions

    '''
    
    preds = {key: {} for key in keys}
    
    with h5.File(file_address, 'r') as f:
        for dfile in f:
            for key in keys:
                preds[key][str(dfile)] = f[dfile+'/'+key][:]

    #*Sort wrt file index
    values_sorted = sort_wrt_file_id(file_address, preds) 

    return values_sorted

def remove_tests_modeldir(directory = get_project_root() + '/models/'):
    '''Deletes all cubeml_test-models and all models that failed during training.
    '''
    for file in Path(directory).iterdir():
        if Path(file).is_dir():
            curr_dir = str(file).split('/')[-1]
            name = curr_dir.split('_')[0]
            
            try:
                _, _, _, meta_pars = load_model_pars(str(file))
                if 'status' in meta_pars:
                    if meta_pars['status'] == 'Failed':
                        shutil.rmtree(file)
                        print('Deleted', str(file))
                if name == 'test': 
                    shutil.rmtree(file)
                    print('Deleted', str(file))
                continue
            except FileNotFoundError:
                pass
                
            if name == 'test':
                try:
                    shutil.rmtree(file)
                    print('Deleted', str(file))
                except FileNotFoundError:
                    pass
            else:
                remove_tests_modeldir(file)

def remove_tests_wandbdir(directory = get_project_root() + '/models/wandb/', rm_all=False):
    '''Deletes all wandb-folder which failed during training or was a test-run.
    '''
    #*Check each run - if test, delete it.
    for run in Path(directory).iterdir():
        
        remove = False
        if Path(run).is_dir():
            
            try:
                with open(str(run)+'/wandb-metadata.json') as f:
                    metadata = json.load(f)
                    cond1 = metadata['project'] == 'cubeml_test'
                    cond2 = metadata['state'] == 'failed'
                    
                    if rm_all:
                        remove = True
                    else:
                        if cond1 or cond2:
                            remove = True
            except FileNotFoundError:
                remove = True

        if remove:
            shutil.rmtree(run)
            print('Deleted', str(run))

def sort_pairs(l1, l2, reverse=False):
    '''Sorts lists l1 and l2 w.r.t. the l1-values
    '''
    
    l2_sorted = [x for _, x in sorted(zip(l1, l2), key=lambda pair: pair[0], reverse=reverse)]
    l1_sorted = sorted(l1, reverse=reverse)

    return l1_sorted, l2_sorted

def sort_wrt_file_id(file_path, values):
    '''Takes a dictionary with keys equal to variable of interest and values equal to a dictionary consisting of a file ID and the values of interest. Returns a new dictionary with the same keys as values and the sorted values wrt the file index
    '''
    values_sorted = {key: np.array([]) for key in values}
    for key, dicts in values.items():
        unsorted = []
        for ID, vals in dicts.items():
            id_int = convert_id_to_int(file_path, ID)
            unsorted.append((id_int, vals))
        
        sorted_list = sorted(unsorted, key=lambda x: x[0])
        for item in sorted_list:
            values_sorted[key] = np.append(values_sorted[key], item[1])

    return values_sorted

def show_train_val_error(x_address, train_address, val_address, save_address=None):
    
    #*Load pickle-files
    x = pickle.load( open( x_address, "rb" ) )
    train = pickle.load( open( train_address, "rb" ) )
    val = pickle.load( open( val_address, "rb" ) )

    alpha = 0.5
    plt.style.use('default')

    h_figure = plt.figure()
    h_subfig = plt.subplot(1, 1, 1)

    h_subfig.set_xlabel('Epoch [s]')
    h_subfig.set_ylabel('Mean error')

    plt.plot(x, train, label = 'Train error')
    plt.plot(x, val, label = 'Val. error')
    h_subfig.legend()
    h_subfig.grid(alpha = alpha)

    if save_address is not None:
        Path(save_address).mkdir(parents=True, exist_ok=True)
        h_figure.savefig(save_address+'train_val_e.pdf')
        print('Figure saved at: ')
        print(save_address+'train_val_e.pdf','\n')

def update_model_pars(new_hyper_pars, new_data_pars, meta_pars):
    '''Updates parameterdictionaries used for continuing training of a pretrained mode. Only certain keys can be updated (for instance, not model architecture.)

    Input:
    new_hyper_pars: Dictionary containing hyperparameters of new model
    new_data_pars: Dictionary containing dataparameters of new model.
    meta_pars: Dictionary containing metaparameters of new model.

    Output:
    Updated hyperparameter-, dataparameter and metaparameter-dictionaries.
    '''
    #*Load hyperparameters of experiment - metapars has to be path from project root
    model_dir = get_project_root() + meta_pars['pretrained_path']    
    hyper_pars, data_pars, arch_pars, _ = load_model_pars(model_dir)
    
    checkpoint_path = model_dir + '/backup.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    #*Get last epoch number
    hyper_pars['epochs_completed'] = checkpoint['epochs_completed']

    #*Update the old pars - do not update arch-pars: 
    for key, item in new_hyper_pars.items():
        
        #*Let optimizer be fixed
        if key == 'optimizer':
            continue

        hyper_pars[key] = item

    #*LR should be updated though
    hyper_pars['optimizer']['lr'] = new_hyper_pars['optimizer']['lr']    

    #*Update the old pars
    for key, item in new_data_pars.items():
        data_pars[key] = item

    return hyper_pars, data_pars, arch_pars
    
    
