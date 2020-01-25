#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess
from multiprocessing import Pool, cpu_count

# from src.modules.classes import *
import src.modules.loss_funcs as lf
import src.modules.helper_functions as hf
from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.constants import *
from src.modules.classes import *
import src.modules.preprocessing as pp


class PickleLoader(data.Dataset):
    '''A Pytorch dataloader for neural nets with sequential and scalar variables. This dataloader does not load data into memory, but opens a h5-file, reads and closes the file again upon every __getitem__.

    Input: Directory to loop over, targetnames, scalar feature names, sequential feature names, type of set (train, val or test), train-, test- and validation-fractions and an optional datapoints_wanted.
    '''
    def __init__(self, directory, seq_features, scalar_features, targets, set_type, train_frac, val_frac, test_frac, prefix=None, mask='all'):

        self.directory = get_project_root() + directory

        self.scalar_features = scalar_features
        self.seq_features = seq_features
        self.targets = targets

        self.n_scalar_features = len(scalar_features)
        self.n_seq_features = len(seq_features)
        self.n_targets = len(targets)

        self.type = set_type
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.prefix = prefix
        self.mask = mask

        self.len = None # * To be determined in get_meta_information

        self.get_meta_information()

    def __getitem__(self, index):

        # * Find right file
        fname_index = np.searchsorted(self.n_events_including, index, side='right')
        fname = self.file_names[fname_index]

        indices_index = index-self.n_events_without[fname_index]
        true_index = self.file_indices[fname][indices_index]

        # * Extract relevant data
        with h5.File(fname, 'r', swmr=True) as f:
            
            # * If key does not exist, it means the key hasn't been transformed - it is therefore located raw/key
             
            # * Sequential data
            for i, key in enumerate(self.seq_features):
                if i == 0:
                    n_doms = f[self.prefix+key][true_index].shape[0]
                    seq_array = np.empty((self.n_seq_features, n_doms))
                try:        
                    seq_array[i, :] = f[self.prefix+key][true_index][:]
                except KeyError:
                    seq_array[i, :] = f['raw/'+key][true_index][:]

            scalar_array = np.empty(self.n_scalar_features)    
            for i, key in enumerate(self.scalar_features):
                try:
                    scalar_array[i] = f[self.prefix+key][true_index]
                except KeyError:
                    scalar_array[i] = f['raw/'+key][true_index]

            targets_array = np.empty(self.n_targets)    
            for i, key in enumerate(self.targets):
                try:
                    targets_array[i] = f[self.prefix+key][true_index]
                except KeyError:
                    targets_array[i] = f['raw/'+key][true_index]
        
        return (seq_array, scalar_array, targets_array)

    def __len__(self):
        return self.len
    
    def get_from_to(self):
        if self.type == 'train':
            from_frac, to_frac = 0.0, self.train_frac
        elif self.type == 'val':
            from_frac, to_frac = self.train_frac, self.train_frac + self.val_frac
        else:
            from_frac, to_frac = self.train_frac + self.val_frac, self.train_frac + self.val_frac + self.test_frac

        return from_frac, to_frac

    def get_meta_information(self):
        '''Extracts filenames, calculates indices induced by train-, val.- and test_frac
        '''
        # * Get mask
        mask = load_pickle_mask(self.directory, self.mask)
        n_events = 0
        from_frac, to_frac = self.get_from_to()
        
        for file in Path(self.directory).iterdir():
            if file.suffix == '.h5':
                with h5.File(file, 'r') as f:

                    n_data_in_file = f['meta/events'][()]
                    indices = get_indices_from_fraction(n_data_in_file, from_frac, to_frac)
                    name = str(file)

                    self.file_names.append(name)
                    self.file_indices[name] = indices
                    self.n_events_without.append(n_events)
                    n_events += len(indices)
                    self.n_events_including.append(n_events)
        
        self.n_events_total = n_events
# tot = []

# for entry in d['dom_charge_over_vertex']:
#     tot.extend(entry)

# tot = sorted(tot)
# tot = np.array(tot)
# print('we done here too')

# tot = (tot - np.median(tot))/calc_iqr(tot)
# path = get_project_root() + '/plots/dom_d_mink_to_prev.png'
# title = r'$d_{Minkowski}$ from $DOM_{t-1}$ to $DOM_{t} $'
# pd = {'data': [tot[:167000]], 'log': [False]}#, 'title': title, 'savefig': path}
# f = rpt.make_plot(pd)
# print(tot.shape)
# tot = []
# for entry in d['dom_t']:
#     tot.extend(entry)

# tot = sorted(tot)
# pd = {'data': [tot], 'log': [False]}#, 'title': title, 'savefig': path}
# f = rpt.make_plot(pd)
