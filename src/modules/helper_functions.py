from time import localtime, strftime, time
import random
import pickle
import joblib
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
from multiprocessing import cpu_count, Pool

import src.modules.loss_funcs
from src.modules.constants import *


class lr_watcher:

    def __init__(self, start_lr, max_lr, min_lr, n_rise, n_fall, batch_size, schedule='exp'):
        """Calculates the factor the initial learning rate should be multiplied with to get desired learning rate. Options: 'inverse', 'exp'
        
        Arguments:
            start_lr {float} -- initial learning rate
            max_lr {float} -- maximal learning rate during training
            min_lr {float} -- minimal/end learning rate during training
            n_rise {int} -- steps up from initial learning rate
            n_fall {int} -- steps down from max learning rate
            batch_size {int} -- used batch size
        
        Keyword Arguments:
            schedule {str} -- Keyword for factor calculation (default: {'exp'})
        """        

        self._steps_up = n_rise//batch_size
        self._steps_down = n_fall//batch_size
        self.gamma_up = (max_lr/start_lr)**(1/self._steps_up)
        self.gamma_down = (min_lr/max_lr)**(1/self._steps_down)

        self._start_lr = start_lr
        self._max_lr = max_lr
        self.schedule = schedule
        self.step = 1

        if schedule == 'inverse':
            # * 1/t decay
            frac = min_lr/max_lr
            self.s = self._steps_down*frac/(1-frac)

    def get_factor(self):

        if self.schedule == 'exp':
            if self.step < self._steps_up:
                factor = self.gamma_up**self.step
            else:
                factor = (self._max_lr/self._start_lr) * self.gamma_down**(self.step-self._steps_up)
            
            self.step += 1
        
        elif self.schedule == 'inverse':
            if self.step < self._steps_up:
                factor = self.gamma_up**self.step
            else:
                factor = (self._max_lr/self._start_lr) * self.s/(self.s+(self.step-self._steps_up))
            
            self.step += 1
        else:
            raise ValueError('lr_watcher: Unknown (%s) schedule given!'%(self.schedule))

        return factor

def apply_mask(file, mask_name='all', min_doms=0, max_doms=np.inf):
    """Given a path to a h5-file with data, the indices of the events satisfying a criterion are extracted.
    
    Arguments:
        file {str or pathlib.Path} -- absolute path to h5-datafile
    
    Keyword Arguments:
        mask_name {str} -- Name of mask. The mask is associated with a user-defined criterion (default: {'None'})
        min_doms {int} -- Minimum amount of activated doms for an event to pass the test (default: {0})
        max_doms {int} -- Maximum amount. (default: {Infinity})
    
    Raises:
        ValueError: If an unknown mask_name is given
    
    Returns:
        [list] -- indices of events satisfying criterion.
    """

    if mask_name == 'all':
        with h5.File(file, 'r') as f:
            n = f['meta/events'][()]
            indices = np.arange(n)
    
    elif mask_name == 'dom_interval':
        with h5.File(file, 'r') as f:
            n_doms = np.array([max_doms >= x.shape[0] >= min_doms for x in f['raw/dom_charge']])
            indices = np.where(n_doms)[0]
    else:
        raise ValueError('No mask supplied!')
    
    return indices

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
        # * put equal amount in each bin
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

    # * calculate MAE
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

        # * Assume errors are symmetric - which they look to be (quick inspection)
        # * Look at plussigma[0]-mean[0], mean[0]-minussigma[0] for instance
        sigma, e_sigma = convert_iqr_to_sigma(means, e_quartiles)
        
        # * Ignore nans - it is due to too little statistics in a bin
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

def confirm_particle_type(particle_code, file):
    """When loading data and looping over files, this function tries to 
    confirm if the file in question contains events with the right particle.
    
    Arguments:
        particle_code {str} -- 6-digit particle code of the desired particle
        file {pathlib.Path-object} -- Path-object of the file in question
    
    Returns:
        Bool -- True if file contains event of particle in question, False if not.
    """    
    file_splitted = str(file).split('.')
    if particle_code in file_splitted or particle_code == '-1':
        checker = True
    else:
        checker = False
    
    return checker

def convert_id_to_int(file_path, id_str):
    """Conversion of a file ID to a unique integer.
    
    Arguments:
        file_path {str} -- Absolute or relative path to dataset folder
        id_str {str} -- Name of h5-datafile
    
    Returns:
        int -- a unique integer
    """    
    
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

def convert_to_proper_list(l):
    """Converts a list of entries in an unwanted structure (such as a list of arrays with entry 1 in each array) to a proper list. Function is updated each time a new problem arises...
    
    Arguments:
        l {list} -- a list with weird entries
    
    Returns:
        list -- a proper list
    """    

    if type(l[0]) == np.ndarray:
        if l[0].shape[0] == 1:
            converted = [entry[0] for entry in l]
    elif type(l) == np.ndarray and len(l.shape) == 2:
        converted = l[:,0]
    else:
        converted = l
    
    return converted

def estimate_percentile(data, percentiles, n_bootstraps=1000):
    """Estimation of percentile of a dataset using bootstrapping and order statistics (see https://en.wikipedia.org/wiki/Order_statistic). A confidence interval of +-1 sigma is created for each percentile
    
    Arguments:
        data {array-like} -- Data in which to find percentiles.
        percentiles {list} -- list of wanted percentiles (between 0 and 1)
    
    Keyword Arguments:
        n_bootstraps {int} -- Number of bootstrap experiments to run (default: {1000})
    
    Returns:
        lists -- means, -sigma and +sigma, lists of same lengths as percentiles
    """    
     
    # * Convert to np-array and sort (if it hasnt been already)
    data = np.array(data)
    n = data.shape[0]
    data.sort()
    i_means, means = [], []
    i_plussigmas, plussigmas = [], []
    i_minussigmas, minussigmas = [], []


    for percentile in percentiles:
        # * THe order statistic is binomially distributed - we approximate it with a gaussian. 
        sigma = np.sqrt(percentile*n*(1-percentile))
        mean = n*percentile
        i_means.append(int(mean))
        i_plussigmas.append(int(mean+sigma+1))
        i_minussigmas.append(int(mean-sigma))
    
    
    # * bootstrap
    bootstrap_indices = np.random.choice(np.arange(0, n), size=(n, n_bootstraps))
    bootstrap_indices.sort(axis=0)
    bootstrap_samples = data[bootstrap_indices]

    # * An IndexError is due to too few events in a bin. Set these to NaN, we don't care about bins with very little data, so dont log the error.
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
    """Searches main_dir and its subdirectories for files or directories with name in it
    
    Arguments:
        name {string} -- Name of file or directory wanted
    
    Keyword Arguments:
        main_dir {str} -- The parent directory to scan - defaults to root if None (default: {None})
    
    Returns:
        list -- List of all found paths.
    """    
    
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

def flatten_list_of_lists(l):
    """Given a list of lists (e.g. from multiprocessing), unpacks the elements of the sublists into a single list.
    
    Arguments:
        l {list} -- List of lists
    
    Returns:
        list -- Unpacked sublists.
    """    
    return [item for sublist in l for item in sublist]

def get_dataloader_params(batch_size, num_workers=8, shuffle=False, dataloader=None):
    """A helper function for initializing of dataloader - the different dataloaders require different settings
    
    Arguments:
        batch_size {int} -- Desired batch size
    
    Keyword Arguments:
        num_workers {int} -- number of subprocesses to call torch.Dataloader with (default: {8})
        shuffle {bool} -- Whether batching should be shuffled or not (default: {False})
        dataloader {str} -- Name of wanted dataloader (default: {None})
    
    Returns:
        dict -- Dictionary with desired items.
    """    

    if dataloader == 'FullBatchLoader':
        dataloader_params = {'batch_size': None, 'shuffle': False, 'num_workers': num_workers}
    else:
        dataloader_params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers}
    
    return dataloader_params

def get_dataset_name(file_path):
    """Retrieves the dataset name from an absolute or relative path to the dataset
    
    Arguments:
        file_path {str} -- Absolute or relative path to dataset folder
    
    Returns:
        str -- Dataset name
    """    
    
    from_root = get_path_from_root(file_path)
    from_root_splitted = from_root.split('/')
    
    # * Expects the format chosen for the framework
    if from_root_splitted[1] == 'data' or from_root_splitted[1] == 'models':
        name = from_root_splitted[2]
    else:
        raise ValueError('Unknown format (%s) given!'%(file_path))
    
    return name

def get_dataset_size(data_dir, particle='any', mask_name='all'):
    """Loops over a data directory and returns the total number of events
    
    Arguments:
        data_dir {str} -- relative or full path to data directory
    
    Keyword Arguments:
        particle {str} -- name of particle (Default: 'any') 
    
    Returns:
        float -- number of files, average number of events per file and std on number of events pr file
    """    
    
    n_events = 0.0
    n_events_sqr = 0.0
    path = get_project_root() + get_path_from_root(data_dir)
    n_files = 0.0
    particle_code = get_particle_code(particle)

    for file in Path(path).iterdir():
        if file.suffix == '.h5' and confirm_particle_type(particle_code, file):
            n_files += 1.0
            indices = load_mask(file, mask_name)
            n_events += len(indices)
            n_events_sqr += len(indices)**2
            
    mean = n_events/n_files
    std = np.sqrt(n_events_sqr/n_files - mean**2)
    return n_files, mean, std

def get_device(ID=0):
    cuda = 'cuda:'+str(ID)
    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    return device

def get_indices_from_fraction(n, from_frac, to_frac, shuffle=False, file_name='None', dataset_path='None', seed=2912):
    """Converts the interval induced by n, from_frac, to_frac to a list of indices in ascending order. The option is given to get the indices of a SHUFFLED list instead.

    Furthermore, if a file_name and a dataset_path is given, a seed induced by the file_name is used when shuffling such that the same indices always are returned - useful when a dataset has to be split into train-, val.- and test-sets and the same splitting is wanted for every experiment.
    
    Arguments:
        n {int} -- length of array/list the indices are to be used for
        from_frac {float} -- number between 0 and 1.
        to_frac {float} -- number between 0 and 1 - must be larger than from_frac
    
    Keyword Arguments:
        shuffle {bool} -- Whether indices of a shuffled list should be returned (default: {False})
        file_name {str} -- Name of datafile used to make a unique seed for the RNG (default: {'None'})
        dataset_path {str} -- ABsolute or relative path to the dataset (default: {'None'})
        seed {int} -- Seed used by RNG (default: {2912})
    
    Returns:
        np.array -- array of indices in ascending order
    """    
    
    # * Calculate from and to indices
    
    frac = to_frac-from_frac
    n_below = int(from_frac*n +0.5)
    leftover = n_below - from_frac*n
    n_wanted = int(frac*n - leftover + 0.5)

    all_indices = np.arange(n)
    if shuffle:
        # * Use filename if given to generate seed such that it is shuffled the same way every time.
        if file_name != 'None':
            seed = convert_id_to_int(dataset_path, file_name)
            random.seed(seed)
        random.shuffle(all_indices)

        # * Make RNG random again..
        random.seed()
        random.seed(time()*1e7)

    indices = all_indices[n_below:n_below+n_wanted]
    
    return sorted(indices)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr_scheduler(hyper_pars, optimizer, batch_size, n_train):
    """Instantiates a torch.optim learning rate-scheduler and attaches it to an optimizer (see https://pytorch.org/docs/stable/optim.html)
    
    Arguments:
        hyper_pars {dict} -- A dictionary containing keywords and values required for training - in this dictionary, a lr_dict should be found containing keyword and values required to define the lr-schedule.
        optimizer {torch.optim-object} -- Optimizer to attach the scheduler to.
        batch_size {int} -- The used batch-size
        n_train {int} -- number of events in train set.
    
    Raises:
        ValueError: When an unknown scheduler is requested.
    
    Returns:
        torch.optim-object -- the desired lr-scheduler.
    """    

    # * Simply multiplies lr by 1 on every iteration - equal to no lr-schedule
    lr_dict = hyper_pars.get('lr_schedule', None)
    
    if lr_dict['lr_scheduler'] == None:
        lambda1 = lambda step: 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    
    elif lr_dict['lr_scheduler'] == 'CyclicLR':
        # * Some default values
        # * {'lr_scheduler':   'CyclicLR',
        # * 'base_lr':        0.00001,
        # * 'max_lr':         0.001,
        # * 'period':         12, # * in epochs
        # * 'cycle_momentum': False}

        if 'cycle_momentum' in lr_dict: 
            cycle_momentum = lr_dict['cycle_momentum']
        else: 
            cycle_momentum = True # * default value from Torch
        
        if 'period' in lr_dict: 
            n_steps = 0.5*n_train*lr_dict['period']/batch_size
            step_size_up = int(n_steps)
        else: step_size_up = 2000 # * default value from Torch

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr_dict['base_lr'], max_lr = lr_dict['max_lr'], step_size_up=step_size_up, cycle_momentum=cycle_momentum)
    
    elif lr_dict['lr_scheduler'] == 'ReduceLROnPlateau':
        # * Some default values
        # * {'lr_scheduler':   'ReduceLROnPlateau',
        # * 'factor':         0.1,
        # * 'patience':       2,
        # * }

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
    
    elif lr_dict['lr_scheduler'] == 'OneCycleLR':
        # * Some default values
        # {'lr_scheduler':   'OneCycleLR',
        # 'max_lr':          0.1,
        # 'min_lr':          1e-6,
        # 'pct_start':       0.3,
        # }

        pars = {}
        pars['max_lr'] = lr_dict['max_lr']
        pars['div_factor'] = pars['max_lr']/hyper_pars['optimizer']['lr']
        pars['final_div_factor'] = hyper_pars['optimizer']['lr']/lr_dict.get('min_lr', 1e-6)
        pars['pct_start'] = lr_dict.get('pct_start', 0.3)
        pars['epochs'] = hyper_pars['max_epochs']
        pars['steps_per_epoch'] = int(n_train/batch_size)
        pars['anneal_strategy'] = lr_dict.get('anneal_strategy', 'cos')
        pars['cycle_momentum'] = lr_dict.get('cycle_momentum', False)
        pars['base_momentum'] = lr_dict.get('base_momentum', 0.85)
        pars['max_momentum'] = lr_dict.get('max_momentum', 0.95)

        scheduler = optim.lr_scheduler.OneCycleLR(optimizer, **pars)
    
    elif lr_dict['lr_scheduler'] == 'CustomOneCycleLR':
        # * Some default values
        # {'lr_scheduler':   'ExpOneCycleLR',
        # 'max_lr':          1e-3,
        # 'min_lr':          1e-6,
        # 'frac_up':         0.2,
        # 'frac_down':       0.8,
        # }
        start_lr = hyper_pars['optimizer']['lr']
        max_lr = lr_dict['max_lr']
        min_lr = lr_dict['min_lr']
        schedule = lr_dict.get('schedule', 'exp')
        n_rise = int(lr_dict['frac_up']*lr_dict['train_set_size']*hyper_pars['max_epochs'])
        n_fall = int(lr_dict['frac_down']*lr_dict['train_set_size']*hyper_pars['max_epochs'])
        batch_size = hyper_pars['batch_size']
        # * A quick sanity check
        if lr_dict['frac_up'] + lr_dict['frac_down'] != 1.0:
            raise ValueError('CustomOneCycleLR frac_up and frac_down does not add up to 1 (adds to %.2f)'%(lr_dict['frac_up'] + lr_dict['frac_down']))

        lr_watch = lr_watcher(start_lr, max_lr, min_lr, n_rise, n_fall, batch_size, schedule=schedule)
        lambda1 = lambda step: lr_watch.get_factor()
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

    else:
        raise ValueError('get_lr_scheduler: Undefined lr_scheduler wanted!')

    return scheduler

def get_set_length(dataloader):
    '''Determines the set length given a custom setholder for Torch's dataloader. Defaults to returning len(dataloader).
    '''
    if str(dataloader) == 'FullBatchLoader':
        set_length = dataloader.batch_size*len(dataloader)
    else:
        set_length = len(dataloader)
    
    return set_length

def get_n_events_in_h5(file_path):
    """Retrieve the number of events in a h5-file
    
    Arguments:
        file_path {str} -- absolute path to file
    
    Returns:
        int -- number of events in file
    """    
    with h5.File(file_path, 'r') as f:
        n_events = int(f['meta/events'][()])
    return n_events

def get_n_parameters(model):
    params = 0
    for parameter in model.parameters():
        layer_params = 1
        for entry in parameter.shape:
            layer_params *= entry

        params += layer_params
    return params

def get_n_events_per_dir(data_dir):
    """Finds the number of pickled events in each subdirectory
    
    Arguments:
        data_dir {str} -- relative path to main data directory
    
    Returns:
        int -- Events per subdirectory
    """    
    return len([event for event in Path(data_dir+'/pickles/0').iterdir()])


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
        else: lr = 0.001 # * standard Adam lr

        if 'betas' in d_opt: betas = d_opt['betas']
        else: betas = (0.9, 0.999) # * standard Adam par

        if 'eps' in d_opt: eps = d_opt['eps']
        else: eps = 1e-08 # * default Adam par

        if 'weight_decay' in d_opt: weight_decay = d_opt['eps']
        else: weight_decay = 0 # * default Adam par
        
        return optim.Adam(model_pars, lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
    
    else:
        raise ValueError('Unknown optimizer (%s) chosen!'%(d_opt['optimizer']))

def get_particle_code(particle):
    """Retrieves the particle code (a 6-digit number) as a str for the desired particle.
    
    Arguments:
        particle {str} -- name of particle (for instance muon_neutrino)
    
    Raises:
        ValueError: If an unknown particle is given
    
    Returns:
        str -- the 6-digit particle code as a string
    """    

    if particle == 'electron':
        particle_code = '110000'
    elif particle == 'electron_neutrino':
        particle_code = '120000'
    elif particle == 'muon':
        particle_code = '130000'
    elif particle == 'muon_neutrino':
        particle_code = '140000'
    elif particle == 'tau':
        particle_code = '150000'
    elif particle == 'tau_neutrino':
        particle_code = '160000'
    elif particle == 'any':
        particle_code = '-1'
    else:
        raise ValueError('Unknown particle type (%s) given to get_particle_code!'%(particle))    

    return particle_code

def get_particle_code_from_h5(h5_file, codes):
    """Given an oscnext-h5-file, the particle code is retrieved from the filename
    
    Arguments:
        h5_file {str} -- path to or name of h5_file
        codes {list} -- List of particle-codes to check for in filename.
    
    Returns:
        str -- The particle code of the file.
    """    
    splitted = h5_file.split('.')
    for index, code in enumerate(codes):
        if code in splitted:
            break
    
    return codes[index]

def get_path_from_root(path):
    '''Given a path, get_path_from_root strips the potential path to the root directory.

    Input
    path: a string or Path-object
    
    Output
    path_from_root: a string with the path from the root directory
    '''
    try:
        split = path.split('/')
    except AttributeError:
        split = str(path).split('/')

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

def get_retro_crs_prefit_vertex_keys():
    """Simple function to retrieve Icecubes vertex reconstruction keys for vertex_reg
    
    Returns:
        list -- keys of Icecubes reconstruction
    """    
    reco_keys = ['retro_crs_prefit_x', 'retro_crs_prefit_y', 'retro_crs_prefit_z', 'retro_crs_prefit_time']
    return reco_keys

def get_target_keys(data_pars, meta_pars):
    """Given a dataset and a regression type, the desired target keys are returned.
    
    Arguments:
        data_pars {dict} -- a dictionary containing relevant data parameters
        meta_pars {dict} -- a dictionary with meta-information of the experiment
    
    Raises:
        ValueError: If unknown dataset encountered
        ValueError: If unknown regression type wanted
    
    Returns:
        list -- List with target variable keys
    """ 

    dataset_name = get_dataset_name(data_pars['data_dir'])
    
    if dataset_name == 'oscnext-genie-level5-v01-01-pass2' or dataset_name == 'oscnext-genie-level5-v01-01-pass2_copy':
        if meta_pars['group'] == 'direction_reg':
            target_keys = ['true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z']
        elif meta_pars['group'] == 'vertex_reg':
            target_keys = ['true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z', 'true_primary_time']
        elif meta_pars['group'] == 'vertex_reg_no_time':
            target_keys = ['true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z']
        elif meta_pars['group'] == 'energy_reg':
            target_keys = ['true_primary_energy']
        else:
            raise ValueError('Unknown regression type (%s) encountered for dataset %s!'%(meta_pars['group'], dataset_name))
    
    elif dataset_name == 'MuonGun_Level2_139008':
        if meta_pars['group'] == 'direction_reg':
            target_keys = ['true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z']
        else:
            raise ValueError('Unknown regression type (%s) encountered for dataset %s!'%(meta_pars['group'], dataset_name))

    else:
        raise ValueError('Unknown dataset (%s) encountered!'%(dataset_name))
    
    return target_keys

def get_time():
    """Reports the current local time
    
    Returns:
        str -- Current time
    """    
    return strftime("%d/%m %H:%M:", localtime())

def get_train_val_test_indices(n_data, train_frac, val_frac, test_frac, shuffle=True):
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
        transformers = joblib.load(open(model_dir+'/transformers.pickle', "rb"))
    except FileNotFoundError:
        transformers = None
    transformed = {}

    if transformers == None:
        for key in data:
            transformed[key] = data[key]
    
    else:
        for key in data:

            # * If key is not in transformers, it shouldn't be transformed
            if transformers.get(key, None) == None: 
                transformed[key] = data[key]

            # * The reshape is required for scikit to function...
            else: 
                # * Input might be given as a dictionary of lists
                try: 
                    transformed[key] = transformers[key].inverse_transform(data[key].reshape(-1, 1))
                except AttributeError: 
                    transformed[key] = transformers[key].inverse_transform(np.array(data[key]).reshape(-1, 1))

    return transformed

def load_mask(file_path, mask_name):
    """Loads 'allowed' indices in file_path from a mask-file induced by mask_name. If all data is allowed, the mask_name 'all' simply returns np.arange(len(file))
    
    Arguments:
        file_path {str} -- Absolute path to a datafile
        mask_name {str} -- Name of mask (options: 'all', 'dom_interval_min32_max64')
    
    Returns:
        array -- array for allowed indices
    """    
    
    # * Treat 'all'-case in a special way, since no mask is needed.
    if mask_name == 'all':
        with h5.File(file_path, 'r') as f:
            n_events = f['meta/events'][()]
            indices = np.arange(n_events)

    else:
        # * Create path to mask-file
        path = PATH_MASKS+get_dataset_name(file_path)+'/'+mask_name+'.h5'
        try:
            file_name = file_path.stem
        except AttributeError:
            file_name = Path(file_path).stem

        # * Read mask
        with h5.File(path, 'r') as f:
            indices = f['indices/'+file_name][:]
    
    return indices

def load_model_pars(model_dir):
    """Loads and returns hyperparameters, datapatameters, architectureparameters and metaparameters from a model directory
    
    Arguments:
        model_dir {str} -- Full or relative path to the model directory
    
    Returns:
        dicts -- hyper-, data-, architecture- and meta-parameter dictionaries
    """    
     
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

def load_pickle_mask(data_dir, masknames):
    """Given a list of masknames (corresponding to a list of indices), the intersection between all masks are found.
    
    Arguments:
        data_dir {str} -- Absolute path to dataset
        masknames {list} -- Names of masks to apply to dataset.
    
    Returns:
        list -- intersection of masks
    """    

    # * Load masks    
    masks_path = data_dir + '/masks/'
    list_of_masks = []
    for maskname in masknames:
        mask = pickle.load(open(masks_path+maskname+'.pickle', 'rb'))
        list_of_masks.append(mask)
    
    # * Find intersection using sets
    mask = set(list_of_masks[0])
    for i in range(1, len(list_of_masks)):
        mask = mask & set(list_of_masks[i])

    return list(mask)
    

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

    # * Get CubeML/models/ directory
    models_dir = get_project_root()+'/models/'

    # * See if model/dataset exists - if not, make it
    data_name = data_folder_address.split('/')
    if data_name[-1] == '': 
        data_name = data_name[-2]
    else: 
        data_name = data_name[-1]

    if not Path(models_dir+data_name).is_dir(): 
        Path(models_dir+data_name).mkdir()
    data_dir = models_dir+data_name+'/'

    # * Make lr-finder directory if it doesnt exist
    if not Path(data_dir+'lr_finders').is_dir(): 
        Path(data_dir+'lr_finders').mkdir()
    lr_finders_dir = data_dir+'lr_finders/'

    # * Finally! lr-finder dir
    if project.split('_')[-1] == 'test':
        base_name = 'test_BS'+str(batch_size)+'_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())
    else:
        base_name = 'BS'+str(batch_size)+'_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())

    Path(lr_finders_dir+base_name).mkdir()
    lr_dir = lr_finders_dir+base_name
    
    return lr_dir

def make_model_dir(reg_type, data_folder_address, clean_keys, project, particle='any'):
    '''Makes a model folder at CubeML/models/<DATA_TRAINED_ON>/regression/<REGRESSION_TYPE>/<WHEN_TRAINED>/ with subdirectories figures and data.
    '''

    # * Get CubeML/models/ directory
    models_dir = get_project_root()+'/models/'

    # * See if model/dataset exists - if not, make it
    data_name = data_folder_address.split('/')
    if data_name[-1] == '': data_name = data_name[-2]
    else: data_name = data_name[-1]

    if not Path(models_dir+data_name).is_dir(): Path(models_dir+data_name).mkdir()
    data_dir = models_dir+data_name+'/'

    # * Make regression directory if it doesnt exist
    if not Path(data_dir+'regression').is_dir(): Path(data_dir+'regression').mkdir()
    regression_dir = data_dir+'regression/'

    # * Make regression type directory if it doesnt exist
    if not Path(regression_dir+reg_type).is_dir(): Path(regression_dir+reg_type).mkdir()
    reg_type_dir = regression_dir+reg_type
    
    # * Finally! Make model directory
    if project.split('_')[-1] == 'test':
        base_name = 'test_'+strftime("%Y.%m.%d-%H.%M.%S", localtime())
    else:
        base_name = strftime("%Y-%m-%d-%H.%M.%S", localtime())

    Path(reg_type_dir+'/'+base_name).mkdir()
    model_dir = reg_type_dir+'/'+base_name
    
    # * Add subdirectories
    Path(model_dir+'/figures').mkdir()
    Path(model_dir+'/data').mkdir()

    # * Copy transform dicts
    particle_code = get_particle_code(particle)
    transformer_address = get_project_root()+'/data/'+data_name+'/transformers/'+particle_code+'_transform'+str(clean_keys['transform'])+'.pickle'
    try:
        data_dir = shutil.copy(transformer_address, model_dir+'/transformers.pickle')
    except FileNotFoundError:
        pass

    return model_dir

def print_progress(start_time, progress, total):
    """Prints elapsed time in hours and minutes since start of program.
    
    Arguments:
        start_time {float} -- Start time in seconds of program (from time.time())
        progress {int} -- Some measure of progress, like i_file
        total {int} -- Total number of iterations.
    """       
    elapsed_seconds = (time()-start_time)
    speed = progress/elapsed_seconds
    elapsed_hours = elapsed_seconds//3600
    elapsed_minutes = (elapsed_seconds%3600)//60
    remaining_seconds = (total-progress)/speed
    remaining_hours = remaining_seconds//3600
    remaining_minutes = (remaining_seconds%3600)//60
    
    print(strftime("\n%d/%m %H:%M:", localtime()), 'Progress: %d/%d'%(progress, total))
    print('Time elapsed: %.0f hours, %.0f minutes.'%(elapsed_hours, elapsed_minutes))
    print('Time remaining: %.0f hours, %.0f minutes.'%(remaining_hours, remaining_minutes))

def read_h5_dataset(file_address, key, prefix='', from_frac=0, to_frac=1, indices=[]):
    """Reads a dataset from a h5-file induced by key and prefix.
    
    Arguments:
        file_address {str} -- Full address to h5-file
        key {str} -- key/name of variable to read
    
    Keyword Arguments:
        prefix {str} -- a string to ensure correct path in file (path is prefix+'/'+key) (default: {''})
        from_frac {int} -- Used to calculate the index to read from. Index = int(n_data_in_file*from_frac+0.5) (default: {0})
        to_frac {int} -- Index to read to. Calculated as from_frac (default: {1})
        indices {list} -- Optional list of indices to read (overwrites from_frac, to_frac) (default: {[]})
    
    Returns:
        array-like -- Array of desired dataset.
    """    

    with h5.File(file_address, 'r') as f:
        n_events = f['meta/events'][()]
        if len(indices) == 0:
            indices = get_indices_from_fraction(n_events, from_frac, to_frac)
        
        # * If not transformed, it is found under raw/
        try:
            path = prefix+'/'+key
            data = f[path][indices]
        except KeyError:
            path = 'raw/'+key

            data = f[path][indices]
    
    return data

def read_h5_directory(data_dir, keys, prefix=None, from_frac=0, to_frac=1, n_wanted=np.inf, particle='any'):
    """Loops over each h5-file in a directory and reads the datasets induced by keys and prefix. Optionally, only a part of the entire dataset is read.
    
    Arguments:
        data_dir {str} -- path from project root to the data directory.
        keys {list} -- list of keys/variable names to read.
    
    Keyword Arguments:
        prefix {str} -- String to ensure correct path in h5-file (default: {None})
        from_frac {float} -- Used to calculate the index to read from; index = int(N_data_in_file*from_frac+0.5) (default: {0})
        to_frac {float} -- the index to read to. Calculated as from_frac (default: {1})
        n_wanted {int} -- An upper bound on the amount of data to be read. Stops reading from additional files if n_loaded > n_wanted. (default: {np.inf})
        particle {str} -- If a dataset with several particle types is read, the name of the desired particle should be given (default: {None})
    
    Returns:
        dict -- Desired datasets.
    """    
    
    values = {key: {} for key in keys}
    n_loaded = 0

    for file in Path(get_project_root()+data_dir).iterdir():
        
        particle_code = get_particle_code(particle)

        if file.suffix == '.h5' and confirm_particle_type(particle_code, file):
            
            # * Do not readt more than wanted - takes up space aswell...
            if n_loaded >= n_wanted:
                break

            for key in keys:
                values[key][file.stem] = read_h5_dataset(file, key, prefix, from_frac=from_frac, to_frac=to_frac)
            
            n_loaded += values[key][file.stem].shape[0]

    # * Sort wrt file index
    values_sorted = sort_wrt_file_id(str(file), values)

    return values_sorted

# ! Should be deleted - deprecated
def read_predicted_h5_data_old(file_address, keys):
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

    # * Sort wrt file index
    values_sorted = sort_wrt_file_id(file_address, preds) 

    return values_sorted

def read_predicted_h5_data(file_address, keys, data_pars, true_keys):
    """Reads datasets in a predictions-h5-file associated with keys and the matching datasets in the raw data-files associated with true_keys and returns 2 sorted dictionaries such that index_i for any key corresponds to the i'th event.
    
    Arguments:
        file_address {str} -- absolute path to predictions-file.
        keys {list} -- names of datasets to read in predictions-file
        data_pars {dict} -- dictionary containing data-parameters of the model.
        true_keys {list} -- names of datasets to read in raw data-files.
    
    Returns:
        dicts -- predictions_dict, raw_dict
    """    

    data_dir = data_pars['data_dir']
    prefix = 'transform'+str(data_pars['file_keys']['transform'])

    preds = {key: {} for key in keys}
    preds['indices'] = {}
    truths = {key: {} for key in true_keys}

    # * Read the predictions. Each group in the h5-file corresponds to a raw data-file. Each group has same datasets.
    with h5.File(file_address, 'r') as f:
        for dfile in f:
            preds['indices'][str(dfile)] = f[dfile+'/index'][:]
            for key in keys:
                preds[key][str(dfile)] = f[dfile+'/'+key][:]

    # * Now read the matching true values. The group-name from the predictions-file matches the raw data filename
    dummy_key = next(iter(preds))
    filenames = preds[dummy_key].keys()
    for file in filenames:
        # * Fetch the file in question and load the indices of the predictions for the raw file
        path = get_project_root()+data_dir
        file_path = next(iter(Path(path).glob('*'+file+'.h5')))
        indices = preds['indices'][file]
        for key in true_keys:
            truths[key][file] = read_h5_dataset(file_path, key, prefix=prefix, indices=indices)
    
    # * Sort wrt file index
    preds_sorted = sort_wrt_file_id(file_address, preds) 
    truths_sorted = sort_wrt_file_id(file_address, truths) 
    
    return preds_sorted, truths_sorted

def read_pickle_predicted_h5_data(file_address, keys, data_pars, true_keys):
    """Reads datasets in a predictions-h5-file associated with keys and the matching datasets in the raw data-files associated with true_keys and returns 2 sorted dictionaries such that index_i for any key corresponds to the i'th event.
    
    Arguments:
        file_address {str} -- absolute path to predictions-file.
        keys {list} -- names of datasets to read in predictions-file
        data_pars {dict} -- dictionary containing data-parameters of the model.
        true_keys {list} -- names of datasets to read in raw data-files.
    
    Returns:
        dicts -- predictions_dict, raw_dict
    """    

    data_dir = data_pars['data_dir']
    prefix = 'transform'+str(data_pars['file_keys']['transform'])

    preds = {key: [] for key in keys}
    preds['indices'] = []

    # * Read the predictions. Each group in the h5-file corresponds to a raw data-file. Each group has same datasets.
    with h5.File(file_address, 'r') as f:
        preds['indices'] = f['index'][:]
        for key in keys:
            preds[key] = f[key][:]
    
    # * Now read the matching true values
    truths = read_pickle_data(data_dir, preds['indices'], true_keys, prefix=prefix)
    
    return preds, truths

def read_pickle_data(data_dir, indices, keys, prefix='raw', multiprocess=True):
    """Given a dataset, indices to load and which attributes from each event to load, a dictionary is created with wanted attributes in same order as indices is given.
    
    Arguments:
        data_dir {str} -- Relative path to dataset
        indices {list} -- Events to load
        keys {list} -- Desired event properties.
    
    Keyword Arguments:
        prefix {str} -- Which transformation of property wanted (default: {'raw'})
        multiprocess {bool} -- Whether or not to use several workers for loading (default: {True})
    
    Raises:
        ValueError: Only a multiprocessing solution is implemented atm.
    
    Returns:
        dict -- A dictionary with keys corresponding to the desired properties.
    """    
    path = get_project_root() + data_dir
    n_events_per_dir = get_n_events_per_dir(path)
    
    if multiprocess:
        available_cores = cpu_count()
        
        indices_chunked = np.array_split(indices, available_cores)
        keys_list = [keys]*available_cores
        path_list = [path]*available_cores
        prefix_list = [prefix]*available_cores
        n_events_per_dir_list = [n_events_per_dir]*available_cores
        packed = [entry for entry in zip(indices_chunked, keys_list, path_list, prefix_list, n_events_per_dir_list)]
        
        with Pool(available_cores) as p:
            dicts_list = p.map(read_pickle_data_multiprocess, packed)
        
        # * Gather results
        final_dict = {key: [] for key in keys}
        for d in dicts_list:
            for key, items in d.items():
                final_dict[key].extend(items)
    else:
        raise ValueError('read_pickle_data: Only a multiprocessing solution is implemented.')
    
    return final_dict

def read_pickle_data_multiprocess(pack):
    """The multiprocessing function called by read_pickle_data
    
    Arguments:
        pack {tuple} -- indices, keys, path prefix and how many events in each subdir.
    
    Returns:
        dict -- Dictionary with same format as read_pickle_data returns.
    """    
    indices, keys, path, prefix, n_events_per_dir = pack
    data = {key: [0]*len(indices) for key in keys}

    for i_event, true_index in enumerate(indices):
        filename = str(true_index) + '.pickle'
        event_path = path + '/pickles/' + str(true_index//n_events_per_dir) + '/' + str(true_index) + '.pickle'
        event = pickle.load(open(event_path, "rb"))

        for key in data:
            try:
                data[key][i_event] = event[prefix][key]
            except KeyError:
                data[key][i_event] = event['raw'][key]

    return data

def remove_tests_modeldir(directory=get_project_root() + '/models/'):
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
                        
                        # * Attempt to remove its .dvc-file aswell
                        try:
                            dvc_file = str(file)+'.dvc'
                            Path(dvc_file).unlink()
                            print('Deleted', dvc_file)
                        except FileNotFoundError:
                            pass

                if name == 'test': 
                    shutil.rmtree(file)
                    print('Deleted', str(file))

                    # * Attempt to remove its .dvc-file aswell
                    try:
                        dvc_file = str(file)+'.dvc'
                        Path(dvc_file).unlink()
                        print('Deleted', dvc_file)
                    except FileNotFoundError:
                        pass

                continue
            except FileNotFoundError:
                pass
                
            if name == 'test':
                try:
                    shutil.rmtree(file)
                    print('Deleted', str(file))
                except FileNotFoundError:
                    pass
                
                # * Attempt to remove its .dvc-file aswell
                try:
                    dvc_file = str(file)+'.dvc'
                    Path(dvc_file).unlink()
                    print('Deleted', dvc_file)
                except FileNotFoundError:
                    pass
                
            else:
                remove_tests_modeldir(file)

def remove_tests_wandbdir(directory = get_project_root() + '/models/wandb/', rm_all=False):
    '''Deletes all wandb-folder which failed during training or was a test-run.
    '''
    # * Check each run - if test, delete it.
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

            # * Attempt to remove its .dvc-file aswell
            try:
                dvc_file = str(run)+'.dvc'
                Path(dvc_file).unlink()
                print('Deleted', dvc_file)
            except FileNotFoundError:
                pass

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
    
    # * Load pickle-files
    x = pickle.load(open(x_address, "rb"))
    train = pickle.load(open(train_address, "rb"))
    val = pickle.load(open(val_address, "rb"))

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

def split_files_in_dataset(data_dir, train_frac=0.8, val_frac=0.1, test_frac=0.1, particle='any', seed=2912):
    """Given a dataset in several files, it is split into lists containing relative paths to training-files, validation-files and test-files. Assumes files are in hdf5-format as shown on https://github.com/ehrhorn/CubeML. The dataset files are shuffled before being split, preferably in the same way for every model - hence, seed. After shuffling, a random seed is chosen.

    Not used atm.
    
    Arguments:
        data_dir {str} -- Absolute or relative path to dataset
    
    Keyword Arguments:
        train_frac {float} -- fraction of data train on (default: {0.8})
        val_frac {float} -- fraction of data to validate on (default: {0.1})
        test_frac {float} -- fraction of data to test on (default: {0.1})
        particle {str} -- particle name (if the dataset contains data on different particles) (default: {'any'})
    
    Returns:
        [list] -- relative paths to training files
        [list] -- relative paths to validation files
        [list] -- relative paths to test files    
    """    
    return -1
    # * First, get total size of dataset
    n_files, mean, std = get_dataset_size(data_dir, particle=particle)

    # * Calculate amounts wanted
    n_data = n_files*mean
    n_train =n_data*train_frac
    n_val =n_data*val_frac
    n_test =n_data*test_frac

    # * Split files 
    train_files = []
    val_files = []
    test_files = []   
    train_count = 0
    val_count = 0
    test_count = 0

    particle_code = get_particle_code(particle)
    path = get_project_root() + get_path_from_root(data_dir)
    
    # * Shuffle the file-list!
    random.seed(seed)
    files = [file for file in Path(path).iterdir()]
    random.shuffle(files)    
    random.seed(time()*1e7)
    
    for file in files:

        # * If file is not of interest, continue to next file
        if not (file.suffix == '.h5' and confirm_particle_type(particle_code, file)):
            continue
        
        if train_count <= n_train:
            with h5.File(file, 'r') as f:
                train_count += f['meta/events'][()]
                relative_file_path = get_path_from_root(str(file))
                train_files.append(relative_file_path)
        
        elif val_count <= n_val:
            with h5.File(file, 'r') as f:
                val_count += f['meta/events'][()]
                relative_file_path = get_path_from_root(str(file))
                val_files.append(relative_file_path)
        
        elif test_count <= n_test:
            with h5.File(file, 'r') as f:
                test_count += f['meta/events'][()]
                relative_file_path = get_path_from_root(str(file))
                test_files.append(relative_file_path)
    
    return train_files, val_files, test_files

def update_model_pars(new_hyper_pars, new_data_pars, meta_pars):
    '''Updates parameterdictionaries used for continuing training of a pretrained mode. Only certain keys can be updated (for instance, not model architecture.)

    Input:
    new_hyper_pars: Dictionary containing hyperparameters of new model
    new_data_pars: Dictionary containing dataparameters of new model.
    meta_pars: Dictionary containing metaparameters of new model.

    Output:
    Updated hyperparameter-, dataparameter and metaparameter-dictionaries.
    '''
    # * Load hyperparameters of experiment - metapars has to be path from project root
    model_dir = get_project_root() + meta_pars['pretrained_path']    
    hyper_pars, data_pars, arch_pars, _ = load_model_pars(model_dir)
    
    checkpoint_path = model_dir + '/backup.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

    # * Get last epoch number
    hyper_pars['epochs_completed'] = checkpoint['epochs_completed']

    # * Update the old pars - do not update arch-pars: 
    for key, item in new_hyper_pars.items():
        
        # * Let optimizer be fixed
        if key == 'optimizer':
            continue

        hyper_pars[key] = item

    # * LR should be updated though
    hyper_pars['optimizer']['lr'] = new_hyper_pars['optimizer']['lr']    

    # * Update the old pars
    for key, item in new_data_pars.items():
        data_pars[key] = item

    return model_dir, hyper_pars, data_pars, arch_pars
    
    
