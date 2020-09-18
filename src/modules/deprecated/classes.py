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

class PickleLoader(data.Dataset):
    '''A Pytorch dataloader for neural nets with sequential and scalar variables. This dataloader does not load data into memory, but opens a h5-file, reads and closes the file again upon every __getitem__.

    Input: Directory to loop over, targetnames, scalar feature names, sequential feature names, type of set (train, val or test), train-, test- and validation-fractions and an optional datapoints_wanted.
    '''
    def __init__(self, directory, seq_features, scalar_features, targets, 
                set_type, train_frac, val_frac, test_frac, prefix=None, 
                masks=['all'], n_events_wanted=np.inf, weights='None', 
                dom_mask='SplitInIcePulses', max_seq_len=np.inf):

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
        self.max_seq_len = max_seq_len

        # 'SplitInIcePulses' corresponds to all DOMs
        # 'SRTInIcePulses' corresponds to Icecubes cleaned doms
        self.dom_mask = dom_mask

        self.weights = weights # To be determined in get_meta_information
        self.len = None # To be determined in get_meta_information
        self.indices = None # To be determined in get_meta_information
        self._n_events_per_dir = None # To be determined in get_meta_information

        self._get_meta_information()
        
    def __getitem__(self, index):
        # Find path
        true_index = self.indices[index]
        weight = self.weights[index]
        filename = str(true_index) + '.pickle'
        path = self.directory+'/pickles/'+str(true_index//self._n_events_per_dir)\
            +'/'+str(true_index)+'.pickle'
        
        # Load event
        with open(path, 'rb') as f:
            event = pickle.load(f)

        # Extract relevant data.
        dom_indices = event['masks'][self.dom_mask]
        actual_seq_len = event[self.prefix][self.seq_features[0]]\
                        [dom_indices].shape[0]
        seq_len = min(actual_seq_len, self.max_seq_len)
        seq_array = np.empty((self.n_seq_features, seq_len))

        # If a maximum sequence length is given, we overwrite dom_indices with
        # randomly sampled indices from dom_indices without replacement until
        # we have enough.
        if actual_seq_len > self.max_seq_len:
            dom_indices = sorted(np.random.choice(dom_indices, 
                                self.max_seq_len, replace=False))

        # Sequential data
        for i, key in enumerate(self.seq_features):
            try:        
                seq_array[i, :] = event[self.prefix][key][dom_indices]
            except KeyError:
                seq_array[i, :] = event['raw'][key][dom_indices]

        # Scalar data
        scalar_array = np.empty(self.n_scalar_features)    
        for i, key in enumerate(self.scalar_features):
            try:
                scalar_array[i] = event[self.prefix][key]
            except KeyError:
                scalar_array[i] = event['raw'][key]

        # Targets
        targets_array = np.empty(self.n_targets)    
        for i, key in enumerate(self.targets):
            try:
                targets_array[i] = event[self.prefix][key]
            except KeyError:
                targets_array[i] = event['raw'][key]
        
        # Tuple is now passed to collate_fn - handle training and predicting differently. We need the name of the event for prediction to log which belongs to which
        if self.type == 'predict':
            pack = (seq_array, scalar_array, targets_array, weight, self.type, true_index)
        else:
            pack = (seq_array, scalar_array, targets_array, weight, self.type)
        
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
        # Get mask
        mask_all = np.array(load_pickle_mask(self.directory, self.masks))
        n_events = len(mask_all)

        # Extract the indices corresponding to the train/val/test part.
        from_frac, to_frac = self._get_from_to()
        indices = get_indices_from_fraction(n_events, from_frac, to_frac)
        
        # Get weights
        self.indices = mask_all[indices]
        self.weights = load_pickle_weights(self.directory, self.weights)[self.indices]
        self.len = min(self.n_events_wanted, len(self.indices))

        # Now get the number of events per event directory
        self._n_events_per_dir = len([event for event in Path(self.directory+'/pickles/0').iterdir()])
    
    def shuffle_indices(self):
        random.shuffle(self.indices)

class BiLSTM(nn.Module):
    
    def __init__(self, n_in, n_hidden, residual=False, batch_first=True, learn_init=False):
                
        super(BiLSTM, self).__init__()

        self._batch_first = batch_first
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.residual = residual
        self.fwrd = nn.LSTM(input_size=n_in, hidden_size=n_hidden, bidirectional=False, batch_first=batch_first)
        self.bkwrd = nn.LSTM(input_size=n_in, hidden_size=n_hidden, bidirectional=False, batch_first=batch_first)
        self.learn_init = learn_init
        
        if self.learn_init:
            self.hidden_fwrd = nn.Parameter(torch.empty(self.n_hidden).normal_(mean=0,std=1.0), requires_grad=True)
            self.hidden_bkwrd = nn.Parameter(torch.empty(self.n_hidden).normal_(mean=0,std=1.0), requires_grad=True)
            self.state_fwrd = nn.Parameter(torch.empty(self.n_hidden).normal_(mean=0,std=1.0), requires_grad=True)
            self.state_bkwrd = nn.Parameter(torch.empty(self.n_hidden).normal_(mean=0,std=1.0), requires_grad=True)

    def forward(self, seq, lengths, device=None):
       
        # * The max length is retrieved this way such that dataparallel works
        shape = seq.shape
        if self._batch_first:
            longest_seq = shape[1]
            batch_size = shape[0]
            feats = shape[2]
            bk_seq = torch.zeros(batch_size, longest_seq, feats, device=device)
        else:
            longest_seq = shape[0]
            batch_size = shape[1]
        
        # * Make a reversed sequence, but still padded
        for i_batch, length in enumerate(lengths):
            indices = list(reversed(range(length)))
            bk_seq[i_batch,:length,:] = seq[i_batch,indices,:]
        
        # * Prep for lstm
        seq = pack(seq, lengths, batch_first=self._batch_first)
        bk_seq = pack(bk_seq, lengths, batch_first=self._batch_first)
        
        if self.learn_init:
            hidden_fwrd = self.hidden_fwrd.view(1, 1, -1).expand(-1, batch_size, -1)
            state_fwrd = self.state_fwrd.view(1, 1, -1).expand(-1, batch_size, -1)
            hidden_bkwrd = self.hidden_bkwrd.view(1, 1, -1).expand(-1, batch_size, -1)
            state_bkwrd = self.state_bkwrd.view(1, 1, -1).expand(-1, batch_size, -1)

            h_fwrd = (hidden_fwrd, state_fwrd)
            h_bkwrd = (hidden_bkwrd, state_bkwrd)
        else:
            h_fwrd = self.init_hidden(batch_size, self.fwrd, device)
            h_bkwrd = self.init_hidden(batch_size, self.fwrd, device)
        
        self.fwrd.flatten_parameters()
        self.bkwrd.flatten_parameters()

        # * Send through LSTMs!
        seq, h = self.fwrd(seq, h_fwrd)
        bk_seq, bk_h = self.bkwrd(bk_seq, h_bkwrd)

        # * Unpack
        seq, _ = unpack(seq, batch_first=self._batch_first, total_length=longest_seq)
        bk_seq, _ = unpack(bk_seq, batch_first=self._batch_first, total_length=longest_seq)
        
        # * reverse again
        for i_batch, length in enumerate(lengths):
            indices = list(reversed(range(length)))
            bk_seq[i_batch,:length,:] = bk_seq[i_batch,indices,:]
        
        # * Combine. If decoding next, bk_h(t_end) and h(t_end) are catted instead of bk_h(0) and h(t_end)
        # TODO: Implement option to either cat and add.
        combined_seq = torch.cat((seq, bk_seq), dim=-1)
        x = torch.cat((h[0], bk_h[0]), dim=-1).squeeze(0)
        
        return combined_seq, x