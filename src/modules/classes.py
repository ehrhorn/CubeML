import h5py as h5
import numpy as np
import torch
from torch.utils import data
from pathlib import Path
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from random import shuffle as shuffler
from math import sqrt
# * from pynvml.smi import nvidia_smi

from src.modules.helper_functions import *

# * ======================================================================== 
# * DATALOADERS
# * ========================================================================

class LstmLoader(data.Dataset):
    '''A Pytorch dataloader for LSTM-like NN, which takes sequences and scalar variables as input. 

    Input: Directory to loop over, targetnames, scalar feature names, sequential feature names, type of set (train, val or test), train-, test- and validation-fractions and an optional datapoints_wanted.
    '''
    def __init__(self, directory, file_keys, targets, scalar_features, seq_features, set_type, train_frac, val_frac, test_frac):   
        root = get_project_root()
        directory = root+directory
        # * Retrieve wanted cleaning level and transformation
        data_address = 'transform'+str(file_keys['transform'])+'/'
    
        # * Loop over files in directory - first time to get number of datapoints
        n_data = 0
        for file in Path(directory).iterdir():
            
            if str(file).split('.')[-1] == 'h5':

                with h5.File(file, 'r') as f:
                    n_data_in_file = f['meta/events'][()]
                    n_train, n_val, n_test = get_n_train_val_test(n_data_in_file, train_frac, val_frac, test_frac)
                    if set_type == 'train': n_data += n_train
                    elif set_type == 'val': n_data += n_val
                    else: n_data += n_test

        # * Initialize dataholders
        # * print('%s: n_data: %d'%(set_type, n_data))
        self.scalar_features = {key: np.empty(n_data) for key in scalar_features}
        self.seq_features = {key: [[] for i in range(n_data)] for key in seq_features}
        self.targets = {key: np.empty(n_data) for key in targets}
        
        added = 0
        # * Actually load the data
        for file in Path(directory).iterdir():
            
            if str(file).split('.')[-1] == 'h5':
                
                with h5.File(file, 'r') as f:
                    n_events = f['meta/events'][()]
                    
                    # * Get indices
                    if set_type == 'train':
                        indices, _, _ = get_train_val_test_indices(n_events, train_frac = train_frac, val_frac = val_frac, test_frac = test_frac, shuffle = False)
                    elif set_type == 'val':
                        _, indices, _ = get_train_val_test_indices(n_events, train_frac = train_frac, val_frac = val_frac, test_frac = test_frac, shuffle = False)
                    else:
                        _, _, indices = get_train_val_test_indices(n_events, train_frac = train_frac, val_frac = val_frac, test_frac = test_frac, shuffle = False)
                    
                    # * Append to or make dictionaries of data
                    n_to_add = len(indices)
                    for key in scalar_features:
                        
                        try:
                            to_add = f[data_address+key][indices]
                            self.scalar_features[key][added:added+n_to_add] = to_add
                        
                        # * If key does not exist, it means the key hasn't been transformed - it is therefore located raw/key
                        except KeyError:
                            self.scalar_features[key][added:added+n_to_add] = f['raw/'+key][indices]

                    for key in seq_features:
                        self.seq_features[key][added:added+n_to_add] = f[data_address+key][indices]

                    
                    for key in targets:

                        try:
                            self.targets[key][added:added+n_to_add] = f[data_address+key][indices]
                        
                        # * If key does not exist, it means the key hasn't been transformed - it is therefore located raw/key
                        except KeyError:
                            self.targets[key][added:added+n_to_add] = f['raw/'+key][indices]

                    added += n_to_add

        self.n_scalar_vars = len(scalar_features)
        self.n_seq_vars = len(seq_features)
        self.n_targets = len(targets)

    def __getitem__(self, index):
        if isinstance(self.seq_features[next(iter(self.seq_features))][index], list):
            seq_len = len(self.seq_features[next(iter(self.seq_features))][index])
        else:
            seq_len = self.seq_features[next(iter(self.seq_features))][index].shape[0]

        seq_array = np.empty((self.n_seq_vars, seq_len))
        for i, key in enumerate(self.seq_features):
            seq_array[i, :] = self.seq_features[key][index]
        
        scalar_array = np.empty(self.n_scalar_vars)
        for i, key in enumerate(self.scalar_features):
            scalar_array[i] = self.scalar_features[key][index]
        
        targets_array = np.empty(self.n_targets)
        for i, key in enumerate(self.targets):
            targets_array[i] = self.targets[key][index]

        return (seq_array, scalar_array, targets_array)
    

    def __len__(self):
        return self.scalar_features[next(iter(self.scalar_features))].shape[0]

class CnnLoader(data.Dataset):
    def __init__(
        self,
        directory,
        file_keys,
        targets,
        scalar_features,
        seq_features,
        set_type,
        train_frac,
        val_frac,
        test_frac,
        longest_seq
    ):   
        root = get_project_root()
        directory = root + directory
        #* Retrieve wanted cleaning level and transformation
        data_address = 'transform' + str(file_keys['transform']) + '/'
        self.longest_seq = longest_seq
    
        #* Loop over files in directory - first time to get number of datapoints
        n_data = 0
        for file in Path(directory).iterdir():
            if str(file).split('.')[-1] == 'h5':
                with h5.File(file, 'r') as f:
                    n_data_in_file = f['meta/events'][()]
                    n_train, n_val, n_test = get_n_train_val_test(
                        n_data_in_file,
                        train_frac,
                        val_frac,
                        test_frac
                    )
                    if set_type == 'train':
                        n_data += n_train
                    elif set_type == 'val':
                        n_data += n_val
                    else:
                        n_data += n_test
        #* Initialize dataholders
        self.scalar_features = {
            key: np.empty(n_data) for key in scalar_features
        }
        self.seq_features = {
            key: [
                np.zeros(self.longest_seq) for i in range(n_data)
            ] for key in seq_features
        }
        self.targets = {key: np.empty(n_data) for key in targets}
        added = 0
        # * Actually load the data
        for file in Path(directory).iterdir():
            if str(file).split('.')[-1] == 'h5':
                with h5.File(file, 'r') as f:
                    n_events = f['meta/events'][()]
                    # * Get indices
                    if set_type == 'train':
                        indices, _, _ = get_train_val_test_indices(
                            n_events,
                            train_frac=train_frac,
                            val_frac=val_frac,
                            test_frac=test_frac,
                            shuffle=False
                        )
                    elif set_type == 'val':
                        _, indices, _ = get_train_val_test_indices(
                            n_events,
                            train_frac=train_frac,
                            val_frac=val_frac,
                            test_frac=test_frac,
                            shuffle=False
                        )
                    else:
                        _, _, indices = get_train_val_test_indices(
                            n_events,
                            train_frac=train_frac,
                            val_frac=val_frac,
                            test_frac=test_frac,
                            shuffle=False
                        )
                    # * Append to or make dictionaries of data
                    n_to_add = len(indices)
                    for key in scalar_features:                        
                        try:
                            self.scalar_features[key][added:added + n_to_add] = f[data_address + key][indices]
                        # * If key does not exist, it means the key hasn't been transformed - it is therefore located raw/key
                        except KeyError:
                            self.scalar_features[key][added:added + n_to_add] = f['raw/' + key][indices]
                    for key in seq_features:
                        for no, index in zip(range(added, n_to_add + 1), indices):
                            seq_data = f[data_address + key][index]
                            self.seq_features[key][no][0:len(seq_data)] = seq_data
                    for key in targets:
                        try:
                            self.targets[key][added:added + n_to_add] = f[data_address + key][indices]
                        # * If key does not exist, it means the key hasn't been transformed - it is therefore located raw/key
                        except KeyError:
                            self.targets[key][added:added + n_to_add] = f['raw/' + key][indices]

                    added += n_to_add
        self.n_scalar_vars = len(scalar_features)
        self.n_seq_vars = len(seq_features)
        self.n_targets = len(targets)
    def __getitem__(self, index):
        seq_len = self.longest_seq
        seq_array = np.empty((self.n_seq_vars, seq_len))
        for i, key in enumerate(self.seq_features):
            seq_array[i, :] = self.seq_features[key][index]
        scalar_array = np.empty(self.n_scalar_vars)
        for i, key in enumerate(self.scalar_features):
            scalar_array[i] = self.scalar_features[key][index]
        targets_array = np.empty(self.n_targets)
        for i, key in enumerate(self.targets):
            targets_array[i] = self.targets[key][index]
        if len(self.scalar_features) == 0:
            out = (seq_array, targets_array)
            return out
        else:
            return (seq_array, scalar_array, targets_array)
    def __len__(self):
        return self.targets[next(iter(self.targets))].shape[0]

class LstmPredictLoader(data.Dataset):
    '''Loads a datafile and returns a data.Dataset object for PyTorch's dataloader. The object has the indices of the data from its parent datafile.
    '''
    def __init__(self, file, file_keys, targets, scalar_features, seq_features, set_type, train_frac, val_frac, test_frac):   
        # * Retrieve wanted cleaning level and transformation
        data_address = 'transform'+str(file_keys['transform'])+'/'
        with h5.File(file, 'r') as f:
            n_events = f['meta/events'][()]

            # * Get indices
            if set_type == 'train':
                indices = get_indices_from_fraction(n_events, 0.0, train_frac)
                # * indices, _, _ = get_train_val_test_indices(n_events, train_frac = train_frac, val_frac = val_frac, test_frac = test_frac, shuffle = False)
            elif set_type == 'val':
                indices = get_indices_from_fraction(n_events, train_frac, train_frac+val_frac)

                # * _, indices, _ = get_train_val_test_indices(n_events, train_frac = train_frac, val_frac = val_frac, test_frac = test_frac, shuffle = False)
            else:
                indices = get_indices_from_fraction(n_events, train_frac+val_frac, train_frac+val_frac+test_frac)

                # * _, _, indices = get_train_val_test_indices(n_events, train_frac = train_frac, val_frac = val_frac, test_frac = test_frac, shuffle = False)

            self.indices = indices
            self.scalar_features = {}
            self.seq_features = {}
            self.targets = {} 

            for key in scalar_features:          
                try:
                    self.scalar_features[key] = f[data_address+key][indices]
                
                # * If key does not exist, it means the key hasn't been transformed - it is therefore located raw/key
                except KeyError:
                    self.scalar_features[key] = f['raw/'+key][indices]

            for key in seq_features:
                self.seq_features[key] = f[data_address+key][indices]
            
            for key in targets:

                try:
                    self.targets[key] = f[data_address+key][indices]
                
                # * If key does not exist, it means the key hasn't been transformed - it is therefore located raw/key
                except KeyError:
                    self.targets[key] = f['raw/'+key][indices]

        self.n_scalar_vars = len(scalar_features)
        self.n_seq_vars = len(seq_features)
        self.n_targets = len(targets)

    def __getitem__(self, index):
        if isinstance(self.seq_features[next(iter(self.seq_features))][index], list):
            seq_len = len(self.seq_features[next(iter(self.seq_features))][index])
        else:
            seq_len = self.seq_features[next(iter(self.seq_features))][index].shape[0]

        seq_array = np.empty((self.n_seq_vars, seq_len))
        for i, key in enumerate(self.seq_features):
            seq_array[i, :] = self.seq_features[key][index]
        
        scalar_array = np.empty(self.n_scalar_vars)
        for i, key in enumerate(self.scalar_features):
            scalar_array[i] = self.scalar_features[key][index]
        
        targets_array = np.empty(self.n_targets)
        for i, key in enumerate(self.targets):
            targets_array[i] = self.targets[key][index]

        return (seq_array, scalar_array, targets_array)
    

    def __len__(self):
        return self.scalar_features[next(iter(self.scalar_features))].shape[0]

class SeqScalarTargetLoader(data.Dataset):
    '''A Pytorch dataloader for neural nets with sequential and scalar variables. This dataloader does not load data into memory, but opens a h5-file, reads and closes the file again upon every __getitem__.

    Input: Directory to loop over, targetnames, scalar feature names, sequential feature names, type of set (train, val or test), train-, test- and validation-fractions and an optional datapoints_wanted.
    '''
    def __init__(self, directory, seq_features, scalar_features, targets, set_type, train_frac, val_frac, test_frac, prefix=None):

        self.directory = get_project_root() + directory
        self.scalar_features = scalar_features
        self.n_scalar_features = len(scalar_features)
        self.seq_features = seq_features
        self.n_seq_features = len(seq_features)
        self.targets = targets
        self.n_targets = len(targets)
        self.type = set_type
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.prefix = prefix

        self.file_names = []
        self.file_indices = {}
        self.n_events_without = []
        self.n_events_including = []

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
        return self.n_events_total
    
    def get_from_to(self):
        if self.type == 'train':
            from_frac, to_frac = 0.0, self.train_frac
        elif self.type == 'val':
            from_frac, to_frac = self.train_frac, self.train_frac + self.val_frac
        else:
            from_frac, to_frac = self.train_frac + self.val_frac, self.train_frac + self.val_frac + self.test_frac

        return from_frac, to_frac

    def get_meta_information(self):
        '''Extracts filenames, calculates indices induced by train-, val.- and 
        '''
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
    
class FullBatchLoader(data.Dataset):
    '''A Pytorch dataloader for neural nets with sequential and scalar variables. This dataloader does not load data into memory, but opens a h5-file, reads an entire batch from one file and closes the file again upon every __getitem__. 
    
    REMEMBER TO CALL THE make_batches()-METHOD BEFORE EACH NEW EPOCH!

    Input: Directory to loop over, targetnames, scalar feature names, sequential feature names, type of set (train, val or test), train-, test- and validation-fractions and batch_size.
    '''
    def __init__(self, directory, seq_features, scalar_features, targets, set_type, train_frac, val_frac, test_frac, batch_size, prefix=None):

        self.directory = get_project_root() + directory
        self.scalar_features = scalar_features
        self.n_scalar_features = len(scalar_features)
        self.seq_features = seq_features
        self.n_seq_features = len(seq_features)
        self.targets = targets
        self.n_targets = len(targets)
        self.type = set_type
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.batch_size = batch_size
        self.prefix = prefix

        self.file_path = {}
        self.file_indices = {}
        self.n_batches = {}
        self.file_order = []
        self.batches = []

        self._get_meta_information()
        self.make_batches()
        # * print(self.batches[0])

    def __getitem__(self, index):
        # * Find right file and get sorted indices to load
        fname = self.batches[index]['path']
        indices = self.batches[index]['indices']
        
        # * Extract relevant data from h5-file
        seq_features = {} 
        scalar_features = {}
        targets = {}
        
        with h5.File(fname, 'r', swmr=True) as f:
            
            # * If key does not exist, it means the key hasn't been transformed - it is therefore located at raw/key
            for key in self.seq_features:
                try:
                    path = self.prefix + '/' + key
                    seq_features[key] = f[path][indices]
                except KeyError:
                    path = 'raw/' + key
                    seq_features[key] = f[path][indices]        

            for key in self.scalar_features:
                try:
                    path = self.prefix + '/' + key
                    scalar_features[key] = f[path][indices]
                except KeyError:
                    path = 'raw/' + key
                    scalar_features[key] = f[path][indices]

            for key in self.targets:
                try:
                    path = self.prefix + '/' + key
                    targets[key] = f[path][indices]
                except KeyError:
                    path = 'raw/' + key
                    targets[key] = f[path][indices]
        
        # * Prepare batch for collate_fn
        batch = []
        for i_batch in range(self.batch_size):
            
            seq_array = np.empty((self.n_seq_features, seq_features[self.seq_features[0]][i_batch].shape[0]))
            for i_seq, key in enumerate(seq_features):
                seq_array[i_seq, :] = seq_features[key][i_batch]
            
            scalar_array = np.empty(self.n_scalar_features)
            for i_scalar, key in enumerate(scalar_features):
                scalar_array[i_scalar] = scalar_features[key][i_batch]

            targets_array = np.empty(self.n_targets)
            for i_target, key in enumerate(targets):
                targets_array[i_target] = targets[key][i_batch]

            batch.append((seq_array, scalar_array, targets_array))
        
        return batch

    def __len__(self):
        return self.n_batches_total
    
    def __repr__(self):
        return 'FullBatchLoader'
        
    def _get_from_to(self):
        if self.type == 'train':
            from_frac, to_frac = 0.0, self.train_frac
        elif self.type == 'val':
            from_frac, to_frac = self.train_frac, self.train_frac + self.val_frac
        else:
            from_frac, to_frac = self.train_frac + self.val_frac, self.train_frac + self.val_frac + self.test_frac

        return from_frac, to_frac

    def _get_meta_information(self):
        '''Extracts filenames, calculates indices induced by train-, val.- and 
        '''
        n_batches = 0
        from_frac, to_frac = self._get_from_to()
        ID = 1
        for file in Path(self.directory).iterdir():
            if file.suffix == '.h5':
                with h5.File(file, 'r') as f:

                    n_data_in_file = f['meta/events'][()]
                    indices = get_indices_from_fraction(n_data_in_file, from_frac, to_frac)
                    file_ID = str(ID)
                   
                    # * Ignore last batch if not a full batch
                    self.file_path[file_ID] = str(file)
                    self.file_indices[file_ID] = indices
                    self.n_batches[file_ID] = int(len(indices)/self.batch_size)

                    n_batches += int(len(indices)/self.batch_size)
                    ID += 1
        
        self.n_batches_total = n_batches

    def make_batches(self):
        '''Shuffles the INDICES from each file, the Torch dataloader fetches a batch from. For each file, self.n_batches dictionaries are made as {ID: shuffled_indices_to_load}. They are then extended to a list, which will contain all such dictionaries for all files. The final list is shuffled aswell. 
        '''

        next_epoch_batches = []
        for ID in self.file_indices:
            shuffler(self.file_indices[ID])
            batches = [{'path': self.file_path[ID], 'indices': sorted(self.file_indices[ID][i*self.batch_size:(i+1)*self.batch_size])} for i in range(self.n_batches[ID])]

            next_epoch_batches.extend(batches)
        
        self.batches = next_epoch_batches
        shuffler(self.batches)
        # * Free up some memory
        del batches
        del next_epoch_batches

class PadSequence:
    '''A helper-function for lstm_v2_loader, which zero-pads shortest sequences.
    '''
    def __call__(self, batch):
        
        # * Each element in "batch" is a tuple (sequentialdata, scalardata, label).
        # * Each instance of data is an array of shape (5, *), where 
        # * * is the sequence length
        # * Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
        sequences = [torch.tensor(np.transpose(x[0])) for x in sorted_batch]

        # * Also need to store the length of each sequence
        # * This is later needed in order to unpad the sequences
        lengths = torch.ShortTensor([len(x) for x in sequences])
        
        # * pad_sequence returns a tensor(seqlen, batch, n_features)
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        
        # * Grab the labels and scalarvars of the *sorted* batch
        scalar_vars = torch.Tensor([x[1] for x in sorted_batch])
        targets = torch.Tensor([x[2] for x in sorted_batch])
        return (sequences_padded.float(), lengths, scalar_vars.float()), targets.float()

        # * return PinnedSeqScalarLengthsBatch(sequences_padded.float(), lengths, scalar_vars.float(), targets.float()) #targets.float()

class CnnCollater:
    def __call__(self, batch):
        sequences = torch.Tensor([x[0] for x in batch])
        targets = torch.Tensor([x[1] for x in batch])
        out_seq = (sequences.float(), )
        return out_seq, targets.float()

#*======================================================================== 
#* DATALOADER FUNCTIONS
#*========================================================================

def load_data(hyper_pars, data_pars, architecture_pars, meta_pars, keyword):

    if 'LstmLoader' == data_pars['dataloader']:

        data_dir = data_pars['data_dir'] # * WHere to load data from
        seq_features = data_pars['seq_feat'] # * feature names in sequences (if using LSTM-like network)
        scalar_features = data_pars['scalar_feat'] # * feature names
        targets = data_pars['target'] # * target names
        train_frac = data_pars['train_frac'] # * how much data should be trained on?
        val_frac = data_pars['val_frac'] # * how much data should be used for validation?
        test_frac = data_pars['test_frac'] # * how much data should be used for training
        file_keys = data_pars['file_keys'] # * which cleaning lvl and transform should be applied?

        dataloader = LstmLoader(data_dir, file_keys, targets, scalar_features, seq_features, keyword, train_frac, val_frac, test_frac)
    
    elif 'SeqScalarTargetLoader' == data_pars['dataloader']:

        data_dir = data_pars['data_dir'] # * WHere to load data from
        seq_features = data_pars['seq_feat'] # * feature names in sequences (if using LSTM-like network)
        scalar_features = data_pars['scalar_feat'] # * feature names
        targets = data_pars['target'] # * target names
        train_frac = data_pars['train_frac'] # * how much data should be trained on?
        val_frac = data_pars['val_frac'] # * how much data should be used for validation?
        test_frac = data_pars['test_frac'] # * how much data should be used for training
        file_keys = data_pars['file_keys'] # * which cleaning lvl and transform should be applied?

        prefix = 'transform'+str(file_keys['transform'])+'/'

        dataloader = SeqScalarTargetLoader(data_dir, seq_features, scalar_features, targets, keyword, train_frac, val_frac, test_frac, prefix=prefix)
    
    elif 'FullBatchLoader' == data_pars['dataloader']:

        data_dir = data_pars['data_dir'] # * WHere to load data from
        seq_features = data_pars['seq_feat'] # * feature names in sequences (if using LSTM-like network)
        scalar_features = data_pars['scalar_feat'] # * feature names
        targets = data_pars['target'] # * target names
        train_frac = data_pars['train_frac'] # * how much data should be trained on?
        val_frac = data_pars['val_frac'] # * how much data should be used for validation?
        test_frac = data_pars['test_frac'] # * how much data should be used for training
        file_keys = data_pars['file_keys'] # * which cleaning lvl and transform should be applied?
        if keyword == 'train':
            batch_size = hyper_pars['batch_size']
        elif keyword == 'val':
            batch_size = data_pars['val_batch_size']

        prefix = 'transform'+str(file_keys['transform'])+'/'

        dataloader = FullBatchLoader(data_dir, seq_features, scalar_features, targets, keyword, train_frac, val_frac, test_frac, batch_size, prefix=prefix)

    
    elif 'CnnLoader' == data_pars['dataloader']:

        print('DO SOMETHING MADS PIKFJÆS')
    
    else:
        raise ValueError('Unknown data loader requested!')
    
    return dataloader
    
def load_predictions(data_pars, keyword, file):

    cond1 = 'LstmLoader' == data_pars['dataloader']
    cond2 = 'SeqScalarTargetLoader' == data_pars['dataloader']
    cond3 = 'FullBatchLoader' == data_pars['dataloader']
    if cond1 or cond2 or cond3:
        
        seq_features = data_pars['seq_feat'] # * feature names in sequences (if using LSTM-like network)
        scalar_features = data_pars['scalar_feat'] # * feature names
        targets = data_pars['target'] # * target names
        train_frac = data_pars['train_frac'] # * how much data should be trained on?
        val_frac = data_pars['val_frac'] # * how much data should be used for validation?
        test_frac = data_pars['test_frac'] # * how much data should be used for training
        file_keys = data_pars['file_keys'] # * which cleaning lvl and transform should be applied?

        return LstmPredictLoader(file, file_keys, targets, scalar_features, seq_features, 'val', train_frac, val_frac, test_frac)
    
    else:
        raise ValueError('An unknown prediction loader was requested!')

def get_collate_fn(data_pars):
    '''Returns requested collate-function, if the key 'collate_fn' is in the dictionary data_pars.
    '''

    if 'collate_fn' in data_pars:
        name = data_pars['collate_fn']
        if name == 'CnnCollater': 
            func =  CnnCollater()

        elif name == 'PadSequence':
            func = PadSequence()
        
        else:
            raise ValueError('Unknown collate-function requested!')
    else:
        func = None

    return func

def sort_indices(dataset, data_pars, dataloader_params=None):
    if 'collate_fn' in data_pars:
        collate_fn = data_pars['collate_fn']
    else: 
        collate_fn = None

    if collate_fn == None:
        indices = dataset.indices
    
    # * Since PadSequence sorts each batch wrt the longest sequence, the indices must be sorted aswell!
    elif collate_fn == 'PadSequence':
        batch_size = dataloader_params['batch_size']
        indices = dataset.indices
        n_indices = len(indices)

        for key in dataset.seq_features:
            seq = dataset.seq_features[key]
            break

        index_seq_pairs = [(indices[i], seq[i]) for i in range(n_indices)]
        
        if batch_size > n_indices:
            end = n_indices
        else:
            end = batch_size

        # * While a whole batch is extracted, sort per batch
        while end <= n_indices:
            index_seq_pairs[end-batch_size:end] = sorted(index_seq_pairs[end-batch_size:end], key=lambda x: x[1].shape[0], reverse=True)
            
            end += batch_size
        
        # * Sort remaining aswell
        index_seq_pairs[end-batch_size:-1] = sorted(index_seq_pairs[end-batch_size:-1], key=lambda x: x[1].shape[0], reverse=True)

        indices = [x[0] for x in index_seq_pairs]
    
    else:
        raise ValueError('Unknown sort function requested!')

    return indices

#*======================================================================== 
#* MODELS
#*========================================================================

class LSTM2Linear(nn.Module):
    '''Pytorch LSTM Model - maybe LSTM2Linear
    '''

    def __init__(self, 
                n_seq_vars, 
                n_scalar_vars,
                n_targets,
                n_hidden = 64, 
                n_lstm_layers = 1, 
                dropout = 0.5, 
                batch_first = True):

        super(LSTM2Linear, self).__init__()
        self.lstm = nn.LSTM(input_size=n_seq_vars,
                            hidden_size=n_hidden, 
                            num_layers=n_lstm_layers, 
                            dropout = dropout, 
                            batch_first=batch_first)

        self.hidden2label = nn.Linear(n_hidden + n_scalar_vars, n_targets) 
        
        self.n_seq_vars = n_seq_vars,
        self.n_hidden = n_hidden, 
        self.n_layers = n_lstm_layers, 
        self.dropout = dropout, 

    def forward(self, batch):
    #def forward(self, seq, lengths, scalars):
        # * Unpack batch - it is calculated in 
        # * seq, lengths, scalars, targets = batch
        seq, lengths, scalars = batch
        
        # * Reshape input (batch first), torch wants a certain form..
        batch_size = seq.shape[0]
        n_seq_vars = seq.shape[1]
        
        seq_vars = seq.view(batch_size, -1, n_seq_vars)

        # * Pack it and send to RNN
        packed = PACK(seq_vars, lengths, batch_first=True)
        
        # * Initialize hidden and cell state to random
        hidden = self.init_hidden(batch_size)

        # * Input: (batch, seq_len, n_seq_vars)
        # * hidden: (num_layers * num_directions, batch, hidden_size)
        _, hidden = self.lstm(packed, hidden)

        # * Concatenate (and remove redundant dimension)
        hidden = hidden[0].view((batch_size, self.n_hidden[0]))
        last_out = torch.cat((scalars, hidden), 1)
        
        # * Decode with a fully connected layer
        output = self.hidden2label(last_out)
        
        return output

    def init_hidden(self, batch_size):
        # * Initialize hidden and cell states
        # * (num_layers * num_directions, batch, hidden_size)
        return (torch.randn(self.n_layers[0], batch_size, self.n_hidden[0]),
                torch.randn(self.n_layers[0], batch_size, self.n_hidden[0]))

class MakeModel(nn.Module):
    '''A modular PyTorch model builder
    '''

    def __init__(self, arch_dict, device):
        super(MakeModel, self).__init__()
        self.mods = make_model_architecture(arch_dict)
        self.layer_names = get_layer_names(arch_dict)
        self.arch_dict = arch_dict
        self.device = device
        self.count = 0

    # * Input must be a tuple to be unpacked!
    def forward(self, batch):
        
        # * For linear layers
        if len(batch) == 1: 
            x, = batch

        # * For RNNs with additional scalar values
        if len(batch) == 3: 
            seq, lengths, scalars = batch
            add_scalars = True 
            batch_size = seq.shape[0]
            n_seq_vars = seq.shape[1]
            # * 'Reshape' input (batch first), torch wants a certain form..
            # * seq = seq.view(batch_size, -1, n_seq_vars)
        for layer_name, entry in zip(self.layer_names, self.mods):
            # * Handle different layers in different ways! 
            
            if layer_name == 'LSTM':
                # * A padded sequence is expected
                # * Initialize hidden layer
                h = self.init_hidden(batch_size, entry, self.device)

                # * Send to LSTM!
                seq = pack(seq, lengths, batch_first=True)
                
                seq, h = entry(seq, h)
                x, _ = h
                seq, lengths = unpack(seq, batch_first=True)
                if entry.bidirectional:
                    # * Add hidden states from forward and backward pass to encode information
                    seq = seq[:, :, :entry.hidden_size] + seq[:, :, entry.hidden_size:]
                    x = (x[0,:,:] + x[1,:,:]) 
                else:
                    x = x.view(batch_size, entry.hidden_size)
            
            elif layer_name == 'Linear':
                # * If scalar variables are supplied for concatenation, do it! But make sure to only do it once.
                if 'scalars' in locals(): 
                    if add_scalars: 
                        x, add_scalars = self.concat_scalars(x, scalars)
                
                # * Send through layers!
                x = entry(x)
            
            elif layer_name == 'Conv1d':
                x = entry(x)
            
            elif layer_name == 'Linear_embedder':
                seq = entry(seq)
            
            elif layer_name == 'SelfAttention':
                seq = entry((seq, lengths))

            else:
                raise ValueError('An unknown Module (%s) could not be processed.'%(layer_name))

        return x

    def init_hidden(self, batch_size, layer, device):
        hidden_size = int(layer.weight_ih_l0.shape[0]/4)
        if layer.bidirectional: num_dir = 2
        else: num_dir = 1

        # * Initialize hidden and cell states - to either random nums or 0's
        # * (num_layers * num_directions, batch, hidden_size)
        return (torch.zeros(num_dir, batch_size, hidden_size, device=device),
                torch.zeros(num_dir, batch_size, hidden_size, device=device))
        # * return (torch.randn(num_dir, batch_size, hidden_size, device=device),
        # *         torch.randn(num_dir, batch_size, hidden_size, device=device))
    
    def concat_scalars(self, x, scalars):
        # * x and scalars must be of shape (batch, features)
        return torch.cat((x, scalars), 1), False

class SelfAttention(nn.Module):
    """Implementation of Self Attention as described in 'Attention is All You Need'
    (or https://jalammar.github.io/illustrated-transformer/) - calculates query-, key- and valuevectors, softmaxes and scales the dotproducts and returns weighted sum of values vectors.
    
    Returns:
        nn.Module -- A Self-attention layer 
    """
    def __init__(self, arch_dict, layer_dict, n_in, n_out):
                
        super(SelfAttention, self).__init__()
        self.arch_dict = arch_dict
        self.layer_dict = layer_dict
        self.n_in = n_in
        self.n_out = n_out

        self.Q = nn.Linear(in_features=n_in, out_features=n_out, bias=False)
        self.K = nn.Linear(in_features=n_in, out_features=n_out, bias=False)
        self.V = nn.Linear(in_features=n_in, out_features=n_out, bias=False)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, args):
        seq, lengths = args
        q = self.Q(seq)
        k = self.K(seq)
        v = self.V(seq)
        print(q.shape, k.shape, v.shape)
        print(lengths)
        # * The matrix multiplication is always done with using the last two dimensions
        # * The transpose means swap second to last and last dimension
        dotprods = torch.matmul(q, k.transpose(-2, -1))
        print(dotprods[-1, -1, :])
        print(seq[-1, -1, :])
        softmaxed = self.softmax(dotprods)
        print(softmaxed[5, -1, :], torch.sum(softmaxed[5, -1, :]))
        

# * ======================================================================== 
# * MODEL FUNCTIONS
# * ========================================================================

def add_conv1d(arch_dict, layer_dict):
    n_layers = len(layer_dict['input_sizes']) - 1
    layers = []
    for i_layer in range(n_layers):
        in_channels = layer_dict['input_sizes'][i_layer]
        out_channels = layer_dict['input_sizes'][i_layer + 1]
        layers.append(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=layer_dict['kernel_sizes'][i_layer],
                stride=layer_dict['strides'][i_layer],
                padding=layer_dict['paddings'][i_layer],
                dilation=layer_dict['dilations'][i_layer]
            )
        )
        # * init_weights(arch_dict, arch_dict['non_lin'], layers[-1])
        layers.append(add_non_lin(arch_dict, arch_dict['non_lin']))
        layers.append(add_norm(arch_dict, arch_dict['norm'], in_channels))
        if 'pool' in layer_dict:
            if layer_dict['pool']['on'][i_layer]:
                layers.append(
                    nn.MaxPool1d(
                        kernel_size=layer_dict['pool']['kernel_sizes'][i_layer],
                        stride=layer_dict['pool']['strides'][i_layer],
                        padding=layer_dict['pool']['paddings'][i_layer],
                        dilation=layer_dict['pool']['dilations'][i_layer]
                    )
                )
    return nn.Sequential(*layers)

def add_LSTM_module(arch_dict, layer_dict, modules):
    n_neurons = len(layer_dict['input_sizes'])-1
    
    for i_neurons in range(n_neurons):
        isize = layer_dict['input_sizes'][i_neurons]
        hsize = layer_dict['input_sizes'][i_neurons+1]
        bidir = layer_dict['bidirectional']
        modules.append(nn.LSTM(input_size=isize, hidden_size=hsize, bidirectional=bidir, batch_first=True))
    return modules

def add_linear_embedder(arch_dict, layer_dict):
    n_layers = len(layer_dict['input_sizes'])-1

    layers = []
    for i_layer in range(n_layers):
        isize = layer_dict['input_sizes'][i_layer]
        hsize = layer_dict['input_sizes'][i_layer+1]

        # Add a matrix to linearly 
        layers.append(nn.Linear(in_features=isize, out_features=hsize))
        init_weights(arch_dict, arch_dict['non_lin'], layers[-1])
        # else:
        #    raise ValueError('Unknown nonlinearity in embedding wanted!')
        layers.append(add_non_lin(arch_dict, arch_dict['non_lin']))
    
    return nn.Sequential(*layers)

def add_linear_layers(arch_dict, layer_dict):
    n_layers = len(layer_dict['input_sizes'])-1
    
    # * Add n_layers linear layers with non-linearity and normalization
    layers = []
    for i_layer in range(n_layers):
        isize = layer_dict['input_sizes'][i_layer]
        hsize = layer_dict['input_sizes'][i_layer+1]

        # * Add layer and initialize its weights
        layers.append(nn.Linear(in_features=isize, out_features=hsize))
        init_weights(arch_dict, arch_dict['non_lin'], layers[-1])

        # * If last layer, do not add non-linearities or normalization
        if i_layer+1 == n_layers: continue

        # * If not, add non-linearities and normalization in required order
        else:
            if layer_dict['norm_before_nonlin']:

                # * Only add normalization layer if wanted!
                if arch_dict['norm']['norm'] != None:
                    layers.append(add_norm(arch_dict, arch_dict['norm'], hsize))
                layers.append(add_non_lin(arch_dict, arch_dict['non_lin']))

            else:
                layers.append(add_non_lin(arch_dict, arch_dict['non_lin']))
                if arch_dict['norm']['norm'] != None:
                    layers.append(add_norm(arch_dict, arch_dict['norm'], hsize))

    return nn.Sequential(*layers)

def add_non_lin(arch_dict, layer_dict):
    if layer_dict['func'] == 'ReLU': 
        return nn.ReLU()
    
    elif layer_dict['func'] == 'LeakyReLU':
        if 'negative_slope' in layer_dict: 
            negslope = layer_dict['negative_slope']
        else: 
            negslope = 0.01
        return nn.LeakyReLU(negative_slope = negslope)

    else:
        raise ValueError('An unknown nonlinearity could not be added in model generation.')

def add_norm(arch_dict, layer_dict, n_features):
    
    if layer_dict['norm'] == 'BatchNorm1D':
        
        if 'momentum' in layer_dict: mom = layer_dict['momentum']
        else: mom = 0.1

        if 'eps' in layer_dict: eps = layer_dict['eps']
        else: eps = 1e-05
        
        return nn.BatchNorm1d(n_features, eps=eps, momentum=mom)
    else: 
        raise ValueError('An unknown normalization could not be added in model generation.')

def add_SelfAttention_layer(arch_dict, layer_dict):

    layers = []
    for n_in, n_out in zip(layer_dict['input_sizes'][:-1], layer_dict['input_sizes'][1:]):
        layers.append(SelfAttention(arch_dict, layer_dict, n_in, n_out))

    return nn.Sequential(*layers)

def init_weights(arch_dict, layer_dict, layer):

    if type(layer) == torch.nn.modules.linear.Linear:
        if layer_dict['func'] == 'ReLU':
            
            nn.init.kaiming_normal_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')
            
        elif layer_dict['func'] == 'LeakyReLU':

            if 'negative_slope' in layer_dict: negslope = layer_dict['negative_slope']
            else: negslope = 0.01

            nn.init.kaiming_normal_(layer.weight, a=negslope, mode='fan_in', nonlinearity='leaky_relu')

        else:
            raise ValueError('An unknown initialization was encountered.')
    else:
        raise ValueError('An unknown initialization was encountered.')

def make_model_architecture(arch_dict):

    modules = nn.ModuleList()
    for layer in arch_dict['layers']:
        for key, layer_dict in layer.items():
        
            # * has to split, since identical keys would get overwritten in OrderedDict
            # * key = name.split('_')[-1]

            # * Add modules of LSTMs, since we need to iterate over LSTM layers
            if key == 'LSTM': 
                modules = add_LSTM_module(arch_dict, layer_dict, modules)
            # * Add a Sequential layer consisting of a linear block with normalization and nonlinearities
            elif key == 'Linear': 
                modules.append(add_linear_layers(arch_dict, layer_dict))
            elif key == 'Conv1d':
                modules.append(add_conv1d(arch_dict, layer_dict))
            elif key == 'Linear_embedder':
                modules.append(add_linear_embedder(arch_dict, layer_dict))
            elif key == 'SelfAttention':
                modules.append(add_SelfAttention_layer(arch_dict, layer_dict))
            else: 
                raise ValueError('An unknown module (%s) could not be added in model generation.'%(key))

    return modules 

def get_layer_names(arch_dict):
    '''Extracts layer names from an arch_dict
    '''
    layer_names = []
    for layer in arch_dict['layers']:
        for layer_name in layer:
            layer_names.append(layer_name)
    
    return layer_names