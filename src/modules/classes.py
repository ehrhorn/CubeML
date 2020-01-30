import torch
from torch.utils import data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from math import sqrt
# from pynvml.smi import nvidia_smi

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

class LstmPredictLoader(data.Dataset):
    '''Loads a datafile and returns a data.Dataset object for PyTorch's dataloader. The object has the indices of the data from its parent datafile.
    '''
    def __init__(self, file, file_keys, targets, scalar_features, seq_features, set_type, train_frac, val_frac, test_frac, mask_name='all'):   
        # * Retrieve wanted cleaning level and transformation
        data_address = 'transform'+str(file_keys['transform'])+'/'
        
        # * First, extract indices of all events satisfying the mask
        viable_events = load_mask(file, mask_name)
        n_events = len(viable_events)
        
        with h5.File(file, 'r') as f:
            # n_events = f['meta/events'][()]
            # * Get indices
            if set_type == 'train':
                from_frac = 0.0
                to_frac = train_frac
                indices = get_indices_from_fraction(n_events, from_frac, to_frac, shuffle=True, file_name=Path(file).stem, dataset_path=str(Path(file).parents[0]))
            elif set_type == 'val':
                from_frac = train_frac
                to_frac = train_frac+val_frac
                indices = get_indices_from_fraction(n_events, from_frac, to_frac, shuffle=True, file_name=Path(file).stem, dataset_path=str(Path(file).parents[0]))
            else:
                from_frac = train_frac+val_frac
                to_frac = train_frac+val_frac+test_frac
                indices = get_indices_from_fraction(n_events, from_frac, to_frac, shuffle=True, file_name=Path(file).stem, dataset_path=str(Path(file).parents[0]))
            
            indices = viable_events[indices]
            self.indices = indices
            self.scalar_features = {}
            self.seq_features = {}
            self.targets = {} 

            # * If key does not exist, it means the key hasn't been transformed - it is therefore located raw/key
            for key in scalar_features:          
                try:
                    self.scalar_features[key] = f[data_address+key][indices]
                except KeyError:
                    self.scalar_features[key] = f['raw/'+key][indices]

            for key in seq_features:
                try:
                    self.seq_features[key] = f[data_address+key][indices]
                except KeyError:
                    self.seq_features[key] = f['raw/'+key][indices]

            for key in targets:
                try:
                    self.targets[key] = f[data_address+key][indices]
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
    '''A Pytorch dataloader for neural nets with sequential and scalar variables. This dataloader does not load data into memory, but opens a h5-file, reads an entire batch from one file and closes the file again upon every __getitem__. It also has the option to stop preparing more files, when n_events_wanted have been surpassed.
    
    REMEMBER TO CALL THE make_batches()-METHOD BEFORE EACH NEW EPOCH!

    Input: Directory to loop over, targetnames, scalar feature names, sequential feature names, type of set (train, val or test), train-, test- and validation-fractions and batch_size.
    '''
    def __init__(self, directory, seq_features, scalar_features, targets, set_type, train_frac, val_frac, test_frac, batch_size, prefix=None, n_events_wanted=np.inf, particle_code=None, file_list=None, mask_name='all', drop_last=False, debug_mode=False):

        self.directory = get_project_root() + directory
        self.scalar_features = scalar_features
        self.n_scalar_features = len(scalar_features)
        self.seq_features = seq_features
        self.n_seq_features = len(seq_features)
        self.targets = targets
        self.n_targets = len(targets)
        self.type = set_type
        self.n_events_wanted = n_events_wanted
        self._particle_code = particle_code
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.batch_size = batch_size
        self.prefix = prefix
        self._mask = mask_name
        self._drop_last = drop_last
        self._debug_mode = debug_mode

        self.file_path = {}
        self.file_indices = {}
        self.n_full_batches = {}
        self.file_order = []
        self.batches = [] # * Initiated - is filled later.
        self._file_list = file_list

        self._get_meta_information()
        self.make_batches()
        
        # * If enough datapoints have been prepared, stop loading more
        # ! Actually dont! Instead extract all and change len(.) such that all statistic is used
        n_batches = len(self.batches)
        batches_wanted = np.inf if self.n_events_wanted == np.inf else ceil(self.n_events_wanted/self.batch_size)
        self._len = n_batches if n_batches <= batches_wanted else batches_wanted

    def __getitem__(self, index):
        # * Find right file and get sorted indices to load
        fname = self.batches[index]['path']
        indices = self.batches[index]['indices']
        batch_size = len(indices)
        
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
        for i_batch in range(batch_size):
            
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
        return self._len
    
    def __repr__(self):
        return 'FullBatchLoader'

    def _extract_from_splitted_files(self):
        """Extracts meta-information from files that have been designated as train- test or val-files - therefore, all data in a file is used

        Not used atm.
        """ 
        return 0
        n_batches = 0
        n_events = 0
        file_id = 1

        for rel_path in self._file_list:
            # * If enough datapoints have been prepared, stop loading more
            # ! Actually dont! Instead extract all and change len(.) such that all statistic is used
            # if n_events > self.n_events_wanted:
            #     break
            path = get_project_root()+rel_path
            indices = apply_mask(path, **self._mask_dict)
            with h5.File(path, 'r') as f:

                # n_data_in_file = f['meta/events'][()]
                # indices = get_indices_from_fraction(n_data_in_file, 0, 1)
                file_ID = str(file_id)
                
                # * Ignore last batch if not a full batch
                self.file_path[file_ID] = path
                self.file_indices[file_ID] = indices
                self.n_batches[file_ID] = int(len(indices)/self.batch_size)

                n_events += len(indices)
                n_batches += len(indices)//self.batch_size
                file_id += 1
        
        self._n_batches_total = n_batches
            
    def _get_from_to(self):
        if self.type == 'train':
            from_frac, to_frac = 0.0, self.train_frac
        elif self.type == 'val':
            from_frac, to_frac = self.train_frac, self.train_frac + self.val_frac
        else:
            from_frac, to_frac = self.train_frac + self.val_frac, self.train_frac + self.val_frac + self.test_frac

        return from_frac, to_frac

    def _get_meta_information(self):
        if not self._file_list:
            self._split_in_files()
        else:
            self._extract_from_splitted_files()
        
    def _split_in_files(self):
        '''Extracts filenames, calculates indices induced by train-, val.- and test-fracs. 
        '''

        from_frac, to_frac = self._get_from_to()
        file_id = 1
        for file in Path(self.directory).iterdir():
            
            # * When in debug mode, dont bother loading all files.
            if self._debug_mode and file_id > 2:
                break
            
            # * Only extract events from files with the proper particle type (necessary due to how Icecube-simfiles are named)
            if not confirm_particle_type(self._particle_code, file):
                continue
            
            if file.suffix == '.h5':
                
                # * First, extract indices of all events satisfying the mask
                viable_events = load_mask(file, self._mask)
                with h5.File(file, 'r') as f:
                    
                    # * Now split into test, train and val
                    n_data_in_file = viable_events.shape[0]
                    indices = get_indices_from_fraction(n_data_in_file, from_frac, to_frac, shuffle=True, file_name=file.stem, dataset_path=self.directory)
                    indices = viable_events[indices]
                    file_ID = str(file_id)
                   
                    # * Ignore last batch if not a full batch
                    self.file_path[file_ID] = str(file)
                    self.file_indices[file_ID] = indices
                    self.n_full_batches[file_ID] = len(indices)//self.batch_size
                    file_id += 1

    def make_batches(self):
        '''Shuffles the INDICES from each file, the Torch dataloader fetches a batch from. For each file, self.n_batches dictionaries are made as {file_id: shuffled_indices_to_load}. They are then extended to a list, which will contain all such dictionaries for all files. The final list is shuffled aswell. 
        '''

        next_epoch_batches = []
        for file_id in self.file_indices:
            random.shuffle(self.file_indices[file_id])
            batches = [{'path': self.file_path[file_id], 'indices': sorted(self.file_indices[file_id][i*self.batch_size:(i+1)*self.batch_size])} for i in range(self.n_full_batches[file_id])]

            next_epoch_batches.extend(batches)

            # * Add remaining non-full batch
            if not self._drop_last:
                last_batch = {'path': self.file_path[file_id], 'indices': sorted(self.file_indices[file_id][self.n_full_batches[file_id]*self.batch_size:-1])}
                if len(last_batch['indices'])>0 : 
                    next_epoch_batches.append(last_batch)

        self.batches = next_epoch_batches
        random.shuffle(self.batches)

        # * Free up some memory
        del batches
        del next_epoch_batches

class PadSequence:
    '''A helper-function for lstm_v2_loader, which zero-pads shortest sequences.
    '''
    def __call__(self, batch):
        # * Inference and training is handled differently - therefore a keyword is passed along
        # * During inference, the true index of the event is passed aswell as 4th entry - see what PickleLoader returns 
        keyword = batch[0][3]

        # * Each element in "batch" is a tuple (sequentialdata, scalardata, label).
        # * Each instance of data is an array of shape (5, *), where 
        # * * is the sequence length
        # * Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
        sequences = [torch.tensor(np.transpose(x[0])) for x in sorted_batch]

        # * Also need to store the length of each sequence
        # * This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])
        
        # * pad_sequence returns a tensor(seqlen, batch, n_features)
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        
        # * Grab the labels and scalarvars of the *sorted* batch
        scalar_vars = torch.Tensor([x[1] for x in sorted_batch])
        targets = torch.Tensor([x[2] for x in sorted_batch])
        
        if keyword == 'predict':
            true_indices = [batch[4] for batch in sorted_batch]
            pack = (sequences_padded.float(), lengths, scalar_vars.float(), true_indices)
        else:
            pack = (sequences_padded.float(), lengths, scalar_vars.float())
        return pack, targets.float()

        # * return PinnedSeqScalarLengthsBatch(sequences_padded.float(), lengths, scalar_vars.float(), targets.float()) #targets.float()

class PickleLoader(data.Dataset):
    '''A Pytorch dataloader for neural nets with sequential and scalar variables. This dataloader does not load data into memory, but opens a h5-file, reads and closes the file again upon every __getitem__.

    Input: Directory to loop over, targetnames, scalar feature names, sequential feature names, type of set (train, val or test), train-, test- and validation-fractions and an optional datapoints_wanted.
    '''
    def __init__(self, directory, seq_features, scalar_features, targets, set_type, train_frac, val_frac, test_frac, prefix=None, masks=['all'], n_events_wanted=np.inf):

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
        self.masks = masks
        self.n_events_wanted = n_events_wanted

        self.len = None # * To be determined in get_meta_information
        self.indices = None # * To be determined in get_meta_information
        self._n_events_per_dir = None # * To be determined in get_meta_information

        self._get_meta_information()

    def __getitem__(self, index):
        # * Find path
        true_index = self.indices[index]
        filename = str(true_index) + '.pickle'
        path = self.directory + '/pickles/' + str(true_index//self._n_events_per_dir) + '/' + str(true_index) + '.pickle'
        
        # * Load event
        event = pickle.load(open(path, "rb"))
        
        # * Extract relevant data
        seq_len = event[self.prefix][self.seq_features[0]].shape[0]
        seq_array = np.empty((self.n_seq_features, seq_len))

        # * Sequential data
        for i, key in enumerate(self.seq_features):
            try:        
                seq_array[i, :] = event[self.prefix][key]
            except KeyError:
                seq_array[i, :] = event['raw'][key]

        # * Scalar data
        scalar_array = np.empty(self.n_scalar_features)    
        for i, key in enumerate(self.scalar_features):
            try:
                scalar_array[i] = event[self.prefix][key]
            except KeyError:
                scalar_array[i] = event['raw'][key]

        # * Targets
        targets_array = np.empty(self.n_targets)    
        for i, key in enumerate(self.targets):
            try:
                targets_array[i] = event[self.prefix][key]
            except KeyError:
                targets_array[i] = event['raw'][key]
        
        # * Tuple is now passed to collate_fn - handle training and predicting differently. We need the name of the event for prediction to log which belongs to which
        if self.type == 'predict':
            pack = (seq_array, scalar_array, targets_array, self.type, true_index)
        else:
            pack = (seq_array, scalar_array, targets_array, self.type)
        
        return pack

    def __len__(self):
        return self.len
    
    def _get_from_to(self):
        if self.type == 'train':
            from_frac, to_frac = 0.0, self.train_frac
        elif self.type == 'val' or self.type == 'predict':
            from_frac, to_frac = self.train_frac, self.train_frac + self.val_frac
        else:
            from_frac, to_frac = self.train_frac + self.val_frac, self.train_frac + self.val_frac + self.test_frac

        return from_frac, to_frac

    def _get_meta_information(self):
        '''Extracts filenames, calculates indices induced by train-, val.- and test_frac
        '''
        # * Get mask
        mask_all = np.array(load_pickle_mask(self.directory, self.masks))
        n_events = len(mask_all)

        # * Extract the indices corresponding to the train/val/test part.
        from_frac, to_frac = self._get_from_to()
        indices = get_indices_from_fraction(n_events, from_frac, to_frac)
        
        self.indices = mask_all[indices]
        self.len = min(self.n_events_wanted, len(self.indices))

        # * Now get the number of events per event directory
        self._n_events_per_dir = len([event for event in Path(self.directory+'/pickles/0').iterdir()])
    
    def shuffle_indices(self):
        random.shuffle(self.indices)

#* ======================================================================== 
#* DATALOADER FUNCTIONS
#* ========================================================================

def load_data(hyper_pars, data_pars, architecture_pars, meta_pars, keyword, file_list=None, drop_last=False, debug_mode=False):

    data_dir = data_pars['data_dir'] # * WHere to load data from
    seq_features = data_pars['seq_feat'] # * feature names in sequences (if using LSTM-like network)
    scalar_features = data_pars['scalar_feat'] # * feature names
    targets = get_target_keys(data_pars, meta_pars) # * target names
    particle_code = get_particle_code(data_pars['particle'])
    train_frac = data_pars['train_frac'] # * how much data should be trained on?
    val_frac = data_pars['val_frac'] # * how much data should be used for validation?
    test_frac = data_pars['test_frac'] # * how much data should be used for training
    file_keys = data_pars['file_keys'] # * which cleaning lvl and transform should be applied?
    mask_names = data_pars['masks']
    
    if keyword == 'train':
        drop_last = True
        batch_size = hyper_pars['batch_size']
        n_events_wanted = data_pars.get('n_train_events_wanted', np.inf)
    elif keyword == 'val':
        batch_size = data_pars['val_batch_size']
        n_events_wanted = data_pars.get('n_val_events_wanted', np.inf)
    elif keyword == 'predict':
        batch_size = data_pars['val_batch_size']
        n_events_wanted = data_pars.get('n_predictions_wanted', np.inf)
    
    prefix = 'transform'+str(file_keys['transform'])+'/'
    if file_keys['transform'] == -1:
        prefix = 'raw/'

    if 'LstmLoader' == data_pars['dataloader']:
        dataloader = LstmLoader(data_dir, file_keys, targets, scalar_features, seq_features, keyword, train_frac, val_frac, test_frac)
    elif 'SeqScalarTargetLoader' == data_pars['dataloader']:
        prefix = 'transform'+str(file_keys['transform'])+'/'
        dataloader = SeqScalarTargetLoader(data_dir, seq_features, scalar_features, targets, keyword, train_frac, val_frac, test_frac, prefix=prefix)
    elif 'FullBatchLoader' == data_pars['dataloader']:
        dataloader = FullBatchLoader(data_dir, seq_features, scalar_features, targets, keyword, train_frac, val_frac, test_frac, batch_size, prefix=prefix, n_events_wanted=n_events_wanted, particle_code=particle_code, file_list=file_list, mask_name=mask_name, drop_last=drop_last, debug_mode=debug_mode)
    elif 'PickleLoader' == data_pars['dataloader']:
        prefix = 'transform'+str(file_keys['transform'])
        if file_keys['transform'] == -1:
            prefix = 'raw'
        dataloader = PickleLoader(data_dir, seq_features, scalar_features, targets, keyword, train_frac, val_frac, test_frac, prefix=prefix, n_events_wanted=n_events_wanted, masks=mask_names)
    else:
        raise ValueError('Unknown data loader requested!')
    
    return dataloader
    
def load_predictions(data_pars, meta_pars, keyword, file, use_whole_file=False):

    cond1 = 'LstmLoader' == data_pars['dataloader']
    cond2 = 'SeqScalarTargetLoader' == data_pars['dataloader']
    cond3 = 'FullBatchLoader' == data_pars['dataloader']
    if cond1 or cond2 or cond3:
        
        seq_features = data_pars['seq_feat'] # * feature names in sequences (if using LSTM-like network)
        scalar_features = data_pars['scalar_feat'] # * feature names
        targets = get_target_keys(data_pars, meta_pars) # * target names
        train_frac = data_pars['train_frac'] # * how much data should be trained on?
        val_frac = data_pars['val_frac'] # * how much data should be used for validation?
        test_frac = data_pars['test_frac'] # * how much data should be used for training
        file_keys = data_pars['file_keys'] # * which cleaning lvl and transform should be applied?
        mask_name = data_pars['mask']
        
        if use_whole_file:
            if keyword == 'train':
                train_frac = 1.0
                val_frac = 0.0
                test_frac = 0.0
            elif keyword == 'val':
                train_frac = 0.0
                val_frac = 1.0
                test_frac = 0.0
            elif keyword == 'test':
                train_frac = 0.0
                val_frac = 0.0
                test_frac = 1.0

        return LstmPredictLoader(file, file_keys, targets, scalar_features, seq_features, 'val', train_frac, val_frac, test_frac, mask_name=mask_name)
    
    else:
        raise ValueError('An unknown prediction loader was requested!')

def get_collate_fn(data_pars):
    '''Returns requested collate-function, if the key 'collate_fn' is in the dictionary data_pars.
    '''

    if 'collate_fn' in data_pars:
        name = data_pars['collate_fn']
        if name == 'PadSequence':
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
        # * Get device on each forward-pass to be compatible with training on multiple GPUs. An error is raised if no GPU available --> use except
        try:
            device = get_device(torch.cuda.current_device())
        except AssertionError:
            device = None

        # * For linear layers
        if len(batch) == 1: 
            x, = batch

        # * For RNNs with additional scalar values
        if len(batch) == 3: 
            seq, lengths, scalars = batch
            add_scalars = True 
            batch_size = seq.shape[0]
            longest_seq = seq.shape[1]
            # * 'Reshape' input (batch first), torch wants a certain form..
            # * seq = seq.view(batch_size, -1, n_seq_vars)
        
        for layer_name, entry in zip(self.layer_names, self.mods):
            # * Handle different layers in different ways! 
            if layer_name == 'LSTM':
                # * A padded sequence is expected
                # * Initialize hidden layer
                h = self.init_hidden(batch_size, entry, device)

                # * Send to LSTM!
                seq = pack(seq, lengths, batch_first=True)
                
                # ? No idea why this works, but an error is thrown when using DataParallel and not calling it, see
                # ? https://discuss.pytorch.org/t/rnn-module-weights-are-not-part-of-single-contiguous-chunk-of-memory/6011/13
                entry.flatten_parameters()
                
                seq, h = entry(seq, h)
                x, _ = h
                seq, lengths = unpack(seq, batch_first=True, total_length=longest_seq)
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
            
            elif layer_name == 'Linear_embedder':
                seq = entry(seq)
            
            elif layer_name == 'AttentionEncoder':
                seq = entry(seq, lengths, device=device)
            
            elif layer_name == 'AttentionDecoder':
                x = entry(seq, lengths, device=device)
            
            # * The MaxPool-layer is used after sequences have been treated -> prepare for linear decoding.
            elif layer_name == 'MaxPool':
                x = entry(seq, lengths, device=device)
            
            elif layer_name == 'LstmBlock':
                x = entry(seq, lengths, device=device)

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

class AttentionBlock(nn.Module):
    """Implementation of Self Attention almost as described in 'Attention is All You Need'.
    
    (or https://jalammar.github.io/illustrated-transformer/) - calculates query-, key- and valuevectors, softmaxes a padded sequence and scales the dotproducts and returns weighted sum of values vectors.
    
    Can work both as a seq2seq encoder or as a seq2vec decoder - in this case, the key-matrix produces one key only.
    Returns:
        nn.Module -- A Self-attention layer 
    """
    def __init__(self, arch_dict, layer_dict, n_in, n_out, mode=None, intermediate=None):
                
        super(AttentionBlock, self).__init__()
        self.arch_dict = arch_dict
        self.layer_dict = layer_dict
        self.n_in = n_in
        self.n_out = n_out
        if intermediate:
            self._intermediate = intermediate
        else:
            self._intermediate = n_in
        self.n_out = n_out
        self._batch_first = True

        if mode == 'encoder':
            self.Q = nn.Linear(in_features=n_in, out_features=n_out)
            self.K = nn.Linear(in_features=n_in, out_features=n_out)
            self.V = nn.Linear(in_features=n_in, out_features=n_out)
            init_weights(arch_dict, arch_dict['non_lin'], self.Q)
            init_weights(arch_dict, arch_dict['non_lin'], self.K)
            init_weights(arch_dict, arch_dict['non_lin'], self.V)
        elif mode == 'decoder':
            raise ValueError('AttentionDecoder: Not implemented yet')
            self.Q = nn.Linear(in_features=n_in, out_features=n_out)
            self.K = nn.Linear(in_features=n_in, out_features=1)
            self.V = nn.Linear(in_features=n_in, out_features=n_out)
            init_weights(arch_dict, arch_dict['non_lin'], self.Q)
            init_weights(arch_dict, arch_dict['non_lin'], self.K)
            init_weights(arch_dict, arch_dict['non_lin'], self.V)

        self.softmax = nn.Softmax(dim=-1)
        if self.layer_dict.get('LayerNorm', False):
            self.norm = nn.LayerNorm(n_out)
        self.linear_out = nn.Linear(in_features=n_out, out_features=n_out)
        self.nonlin = add_non_lin(arch_dict, arch_dict['non_lin'])
        if self.layer_dict.get('LayerNorm', False):
            self.norm2 = nn.LayerNorm(n_out)
        if self.layer_dict.get('Residual', False):
            self.residual_connection = True
        
    
    def forward(self, seq, lengths, device=None):
        
        # * The max length is retrieved this way such that dataparallel works
        if self._batch_first:
            max_length = seq.shape[1]
        else:
            max_length = seq.shape[0]

        q = self.Q(seq)
        k = self.K(seq)
        v = self.V(seq)
        
        # * Attention -> potential norm and residual connection
        post_attention = self._calc_self_attention(q, k, v, lengths, max_length, batch_first=self._batch_first, device=device)
        if self.residual_connection and self.n_in == self.n_out:
            post_attention = seq + post_attention
        if self.norm:
            post_attention = self.norm(post_attention)
        
        # * linear layer -> nonlin -> potential norm and residual connection
        output = self.nonlin(self.linear_out(post_attention))
        if self.residual_connection:
            output = output + post_attention
        if self.norm:
            output = self.norm2(output)
        
        return output

    def _calc_self_attention(self, q, k, v, lengths, max_length, batch_first=False, device=None):
        # * The matrix multiplication is always done with using the last two dimensions, i.e. (*, 10, 11).(*, 11, 7) = (*, 10, 7) 
        # * The transpose means swap second to last and last dimension
        # * masked_fill_ is in-place, masked_fill creates a new tensor
        weights = torch.matmul(q, k.transpose(-2, -1)) / sqrt(self.n_out)
        mask = self._get_mask(lengths, max_length, batch_first=batch_first, device=device)
        weights = weights.masked_fill(~mask, float('-inf'))
        weights = self.softmax(weights)
        
        # * Calculate weighted sum of v-vectors.
        output = torch.matmul(weights, v)
        
        return output

    def _get_mask(self, lengths, maxlen, batch_first=False, device=None):
        # * Assumes mask.size[S, B, *] or mask.size[B, S, *]
        if batch_first:
            mask = torch.arange(maxlen, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(1)
        return mask

class LstmBlock(nn.Module):
    
    def __init__(self, n_in, n_out, n_parallel, n_stacks, bidir=False, residual=True, batch_first=True):
                
        super(LstmBlock, self).__init__()

        self._batch_first = batch_first
        self.par_LSTMs = nn.ModuleList()
        self.residual = residual
        
        for i_par in range(n_parallel):
            par_module = nn.ModuleList()
            par_module.append(nn.LSTM(input_size=n_in, hidden_size=n_out, bidirectional=bidir, batch_first=batch_first))
            for i_stack in range(n_stacks-1):
                par_module.append(nn.LSTM(input_size=n_out, hidden_size=n_out, bidirectional=bidir, batch_first=batch_first))

            self.par_LSTMs.append(par_module)

    def forward(self, seq, lengths, device=None):
        
        # * The max length is retrieved this way such that dataparallel works
        if self._batch_first:
            longest_seq = seq.shape[1]
            batch_size = seq.shape[0]
        else:
            longest_seq = seq.shape[0]
            batch_size = seq.shape[1]

        # * Send through LSTMs! Prep for first layer.
        seq = pack(seq, lengths, batch_first=self._batch_first)
        
        # * x is output - concatenate outputs of LSTMs in parallel
        for i_par, stack in enumerate(self.par_LSTMs):
            
            # * Stack section.
            # ? Maybe learn initial state?             
            h_par = self.init_hidden(batch_size, stack[0], device)
            stack[0].flatten_parameters()
            seq_par, h_par = stack[0](seq, h_par)

            # * If residual connection, save the pre-LSTM version
            if self.residual:
                seq_par_pre, lengths = unpack(seq_par, batch_first=True, total_length=longest_seq)

            for i_stack in range(1, len(stack)):
                
                h_par = self.init_hidden(batch_size, stack[i_stack], device)
                stack[i_stack].flatten_parameters()
                seq_par, h_par = stack[i_stack](seq_par, h_par)
                
                # * Residual connection
                # ? Maybe Add + Norm as in Attention?
                if self.residual:
                    seq_par_post, lengths = unpack(seq_par, batch_first=True, total_length=longest_seq)
                    seq_par_pre = seq_par_pre + seq_par_post
                    seq_par = pack(seq_par_pre, lengths, batch_first=self._batch_first)

            # * Define x on first parallel LSTM-module
            if i_par == 0:
                x, _ = h_par

            # * Now keep cat'ing for each parallel stack
            else:
                x = torch.cat((x, h_par[0]), -1)
        
        return x.squeeze(0)
        
    def init_hidden(self, batch_size, layer, device):
        hidden_size = int(layer.weight_ih_l0.shape[0]/4)
        if layer.bidirectional: num_dir = 2
        else: num_dir = 1

        # * Initialize hidden and cell states - to either random nums or 0's
        # * (num_layers * num_directions, batch, hidden_size)
        return (torch.zeros(num_dir, batch_size, hidden_size, device=device),
                torch.zeros(num_dir, batch_size, hidden_size, device=device))

class MaxPool(nn.Module):
    def __init__(self):
                
        super(MaxPool, self).__init__()
        self._batch_first = True
        # self.device = get_device()

    def forward(self, seq, lengths, device=None):
        # * The max length is retrieved this way such that dataparallel works
        if self._batch_first:
            max_length = seq.shape[1]
        else:
            max_length = seq.shape[0]

        # * A tensor of shape (batch_size, longest_seq, *) is expected and a list of len = batch_size and lengths[0] = longest_seq
        # * Maxpooling is done over the second index
        mask = self._get_mask(lengths, max_length, batch_first=True, device=device)
        # * By masking with -inf, it is ensured that DOMs that are actually not there do not have an influence on the max pooling.
        seq = seq.masked_fill(~mask, float('-inf'))
        if self._batch_first:
            seq, _ = torch.max(seq, dim=1)
        else:
            raise ValueError('Not sure when batch not first - MaxPool')
        return seq

    def _get_mask(self, lengths, maxlen, batch_first=False, device=None):
        # * Assumes mask.size[S, B, *] or mask.size[B, S, *]
        if batch_first:
            # * The 'None' is a placeholder so dimensions are matched.
            mask = torch.arange(maxlen, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(2)
        return mask

#* ======================================================================== 
#* MODEL FUNCTIONS
#* ========================================================================

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

        # * Add a matrix to linearly 
        layers.append(nn.Linear(in_features=isize, out_features=hsize))
        init_weights(arch_dict, arch_dict['non_lin'], layers[-1])
        if layer_dict.get('LayerNorm', False):
            layers.append(nn.LayerNorm(hsize))
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

def add_AttentionBlock_modules(arch_dict, layer_dict, modules, mode=None):

    for n_in, n_out in zip(layer_dict['input_sizes'][:-1], layer_dict['input_sizes'][1:]):
        modules.append(AttentionBlock(arch_dict, layer_dict, n_in, n_out, mode=mode))
    
    return modules

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

def load_best_model(save_dir):
    """Loads and prepares the best model for prediction for a given experiment
    
    Arguments:
        save_dir {str} -- Absolute or relative path to the trained model
    
    Returns:
        torch.nn.Module -- A torch NN.
    """     
    
    hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(save_dir)
    particle_code = get_particle_code(data_pars['particle'])
    device = get_device()
    model_dir = save_dir+'/checkpoints'
    best_pars = find_best_model_pars(model_dir)
    n_devices = meta_pars.get('n_devices', 0)
    model = MakeModel(arch_pars, device)
    
    # * If several GPU's have been used during training, wrap it in dataparalelle
    if n_devices > 1:
        model = torch.nn.DataParallel(model, device_ids=None, output_device=None, dim=0)
    model.load_state_dict(torch.load(best_pars, map_location=torch.device(device)))
    model = model.to(device)
    model = model.float()

    return model

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
            elif key == 'AttentionEncoder':
                modules = add_AttentionBlock_modules(arch_dict, layer_dict, modules, mode='encoder')
            elif key == 'AttentionDecoder':
                modules = add_AttentionBlock_modules(arch_dict, layer_dict, modules, mode='decoder')
            elif key == 'MaxPool':
                modules.append(MaxPool())
            elif key == 'LstmBlock':
                modules.append(LstmBlock(**layer_dict))
            else: 
                raise ValueError('An unknown module (%s) could not be added in model generation.'%(key))

    return modules 

def get_layer_names(arch_dict):
    '''Extracts layer names from an arch_dict
    '''
    layer_names = []
    for layer in arch_dict['layers']:
        for layer_name in layer:
            if layer_name == 'AttentionBlock':
                n_attention_modules = len(layer['AttentionBlock']['input_sizes'])-1
                for nth_attention_layer in range(n_attention_modules):
                    layer_names.append(layer_name)
            else:
                layer_names.append(layer_name)
    return layer_names
