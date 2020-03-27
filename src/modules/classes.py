import torch
from torch.utils import data
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from math import sqrt
import sqlite3
# from pynvml.smi import nvidia_smi

from src.modules.helper_functions import *

#* ======================================================================== 
#* DATALOADERS
#* ========================================================================

class PadSequence:
    '''A helper-function for lstm_v2_loader, which zero-pads shortest sequences.
    '''
    def __init__(self, mode='normal', permute_seq_features=None, permute_scalar_features=None):
        self._mode = mode
        self._permute_seq_features = permute_seq_features
        self._permute_scalar_features = permute_scalar_features

    def __call__(self, batch):
        # * Inference and training is handled differently - therefore a keyword is passed along
        # * During inference, the true index of the event is passed aswell as 4th entry - see what PickleLoader returns 
        # * The structure of batch is a list of (seq_array, scalar_array, targets_array, weight, self.type, true_index)
        keyword = batch[0][4]

        # * Each element in "batch" is a tuple (sequentialdata, scalardata, label).
        # * Each instance of data is an array of shape (5, *), where 
        # * * is the sequence length
        # * Sort the batch in the descending order
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)
        
        # * Grab the vars of the *sorted* batch
        sequences = [torch.tensor(np.transpose(x[0])) for x in sorted_batch]
        scalar_vars = torch.Tensor([x[1] for x in sorted_batch])
        targets = torch.Tensor([x[2] for x in sorted_batch])
        weights = torch.Tensor([x[3] for x in sorted_batch])

        # * Also need to store the length of each sequence
        # * This is later needed in order to unpad the sequences
        lengths = torch.LongTensor([len(x) for x in sequences])

        # * permute-mode is used for permutation importance
        if self._mode == 'permute':
            sequences, scalar_vars = self._permute(sequences, 
                                     scalar_vars, [len(x) for x in sequences])

        # * pad_sequence returns a tensor(seqlen, batch, n_features)
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        
        if keyword == 'predict':
            true_indices = [batch[5] for batch in sorted_batch]
            pack = (sequences_padded.float(), lengths, scalar_vars.float(), true_indices)
        else:
            pack = (sequences_padded.float(), lengths, scalar_vars.float())
        
        return pack, (targets.float(), weights.float())

    def _permute(self, seqs, scalars, lengths):
        # * For the sequential features, there are sum(lengths) variables to draw from (if  we only permute in batch)
        # * draw randomly with 
        # * Loop over indices of features to permute
        for index in self._permute_seq_features:

            # * Make a scrambled bag of features
            entries = np.array([])
            for entry in seqs:
                entries = np.append(entries, entry[:, index].numpy()) 
            
            # * Generate a random sample with replacement for each sequence
            for i_seq, length in enumerate(lengths):
                seqs[i_seq][:, index] = torch.tensor(np.random.choice(entries, length))
        
        # * Now do the same for scalar vars
        batch_size = scalars.shape[0]
        for index in self._permute_scalar_features:
            # * Convert to numpy, retrieve a random sample, convert back to tensor
            scalars[:, index] = torch.tensor(np.random.choice(scalars[:, index].numpy(), batch_size))

        return seqs, scalars

class SqliteFetcher:

    def __init__(self, db_path):

        self._path = str(db_path)
        self._tables = None
        self._len = None
        self._event_lengths_key = 'split_in_ice_pulses_event_length' 
        self._max_events_per_query = 50000
        self._fetch_query_seq = 'SELECT {features} FROM {table} WHERE event '\
                'IN ({events})'
        self._fetch_query = 'SELECT {features} FROM {table} WHERE event_no '\
                'IN ({events})'

    def __len__(self):
        return self._len
    
    @property
    def ids(self):
        
        with sqlite3.connect(self._path) as db:
            cursor = db.cursor()
            query = 'SELECT event_no FROM meta'
            cursor.execute(query)

            event_ids = [e[0] for e in cursor.fetchall()]

        return event_ids

    @property
    def length(self):
        
        if not self._len:
        
            with sqlite3.connect(self._path) as db:
                cursor = db.cursor()
                query = 'SELECT event_no FROM meta'
                cursor.execute(query)

                event_nums = [e[0] for e in cursor.fetchall()]
            self._len = len(event_nums)
        
        return self._len
    
    @property
    def tables(self):

        if not self._tables:

            with sqlite3.connect(self._path) as db:
                cursor = db.cursor()

                # * Get table-names
                query = 'SELECT name FROM sqlite_master WHERE type = "table"'
                cursor.execute(query)
                tables_data = {entry[0]: {} for entry in cursor.fetchall()}

                # * Loop over all columns and fetch their info
                for name in tables_data:
                    query = 'PRAGMA TABLE_INFO({tablename})'.format(
                        tablename=name
                    ) 

                    cursor.execute(query)
                    col_data = cursor.fetchall()
                    
                    tables_data[name] = {e[1]: {
                        'type': e[2], 
                        'index': e[0],
                        }
                    for e in col_data
                    }
            
            self._tables = tables_data
        
        return self._tables
    
    def _fetch(
        self, 
        cursor,
        events,
        features,
        table=None
        ):

        fetched = []
        # * Write query for scalar table and fetch all matching rows
        if len(features) > 0:
            if table == 'sequential':
                query = self._fetch_query_seq.format(
                    features=', '.join(features),
                    table=table,
                    events=', '.join(['?'] * len(events))
                )
            else:
                query = self._fetch_query.format(
                    features=', '.join(features),
                    table=table,
                    events=', '.join(['?'] * len(events))
                )

            cursor.execute(query, events)
            fetched = cursor.fetchall()

        return fetched

    def _make_batch(
        self,
        ids,
        scalars,
        seqs, 
        targets,
        mask,
        lengths
        ):

        # * get the from- and to-indices of each event.
        cumsum = np.append([0], np.cumsum([entry[0] for entry in lengths]))
        all_from = cumsum[:-1]
        all_to = cumsum[1:]
        
        # * Append each event to the batch.
        batch = []
        n_events = len(scalars)
        n_seq = len(seqs[0])
        for i_event in range(n_events):
            
            # * Make scalar array
            scalar_arr = np.array([scalar for scalar in scalars[i_event]])
            
            # * Make sequential array
            # * Since each DOM is stored as a row, we first get the to- and 
            # * from-rows that combine to an event
            from_, to_ = all_from[i_event], all_to[i_event]
            # * We then create our mask - a Boolean array saying whether a 
            # * DOM is included or not
            masked_indices = np.array([e[0] for e in mask[from_:to_]], dtype=bool)
            n_doms = np.sum(masked_indices)
            # * Now loop over variables and extract them
            seq_arr = np.zeros((n_seq, n_doms))
            for i_var in range(n_seq):
                seq_arr[i_var, :] = np.array(
                    [
                        e[i_var] for e in seqs[from_:to_]
                    ]
                )[masked_indices]

            # * Get targets
            target_arr = np.array([target for target in targets[i_event]])

            # * Add to list of events
            batch.append(
                (seq_arr, scalar_arr, target_arr)
            )
        
        return batch

    def _make_dict(
        self, 
        events, 
        names_scalar, 
        fetched_scalar,
        names_sequential, 
        fetched_sequential,
        names_meta, 
        fetched_meta,
        event_lengths
        ):
        
        # * get the from- and to-indices of each event.
        cumsum = np.append([0], np.cumsum([entry[0] for entry in event_lengths]))
        all_from = cumsum[:-1]
        all_to = cumsum[1:]

        # * Create dictionary. First level is event
        data_dict = {}
        for i_event, event in enumerate(events):
            
            # * Second level is data
            data_dict[event] = {}

            # * order the data from fetched_scalar
            from_, to_ = all_from[i_event], all_to[i_event]
            for i_name, name in enumerate(names_sequential):
                data = [
                    entry[i_name] for entry in fetched_sequential[from_:to_]
                ]
                data_dict[event][name] = np.array(data)

            # * Do the same for scalar data
            for i_name, name in enumerate(names_scalar):
                data_dict[event][name] = fetched_scalar[i_event][i_name]
            
            # * .. And finally meta
            for i_name, name in enumerate(names_meta):
                data_dict[event][name] = fetched_meta[i_event][i_name]
            
        return data_dict

    def _list_fetched(
        events, 
        fetched
        ):
        pass

    def fetch_features(
        self,
        all_events=[],
        scalar_features=[],
        seq_features=[],
        meta_features=[]
        ):
        
        # * Connect to DB and set cursor
        with sqlite3.connect(self._path) as db:
            cursor = db.cursor()
            n_events = len(all_events)
            # * Ensure some events are passed
            if n_events == 0:
                raise ValueError('NO EVENTS PASSED TO SQLFETCHER')

            # * If events are not strings, convert them
            if not isinstance(all_events[0], str):
                all_events = [str(event) for event in all_events]

            # * If > self._max_events_per_query are wanted, 
            # * load over several rounds
            n_chunks = n_events//self._max_events_per_query
            chunks = np.array_split(all_events, max(1, n_chunks))

            base_query = 'SELECT {features} FROM {table} WHERE event_no '\
                'IN ({events})'
            fetched_scalar, fetched_sequential, fetched_meta = [], [], []
            
            # * Process chunks
            all_dicted_data = {}
            for events in chunks:
                
                # * Write query for scalar table and fetch all matching rows
                if len(scalar_features) > 0:
                    query = base_query.format(
                        features=', '.join(scalar_features),
                        table='scalar',
                        events=', '.join(['?'] * len(events))
                    )

                    cursor.execute(query, events)
                    fetched_scalar = cursor.fetchall()

                # * Write query for sequential table and fetch all matching rows
                if len(seq_features)>0:
                    query = base_query.format(
                        features=', '.join(seq_features),
                        table='sequential',
                        events=', '.join(['?'] * len(events))
                    )

                    cursor.execute(query, events)
                    fetched_sequential = cursor.fetchall()
                
                # * Write query for meta table and fetch all matching rows
                if len(meta_features)>0:
                    query = base_query.format(
                        features=', '.join(meta_features),
                        table='meta',
                        events=', '.join(['?'] * len(events))
                    )
                    cursor.execute(query, events)
                    fetched_meta = cursor.fetchall()

                # * Finally, fetch event lengths as they are needed for making 
                # * sequential dictionary
                query = base_query.format(
                    features=self._event_lengths_key,
                    table='meta',
                    events=', '.join(['?'] * len(events))
                    )
                cursor.execute(query, events)
                event_lengths = cursor.fetchall()

                # * Put in a dictionary and update all_dicted_ata
                dicted_data = self._make_dict(
                    events, scalar_features, fetched_scalar, seq_features,
                    fetched_sequential, meta_features, fetched_meta, event_lengths
                )
                all_dicted_data.update(dicted_data)
            
            return all_dicted_data

    def make_batch(
        self,
        all_events=[],
        scalar_features=[],
        seq_features=[],
        target_features=[],
        mask=[]
        ):
        
        n_events = len(all_events)
        # * Ensure some events are passed
        if n_events == 0:
            raise ValueError('NO EVENTS PASSED TO SQLFETCHER')

        # * If events are not strings, convert them
        if not isinstance(all_events[0], str):
            all_events = [str(event) for event in all_events]

        # * If > self._max_events_per_query are wanted, 
        # * load over several rounds
        n_chunks = n_events//self._max_events_per_query
        chunks = np.array_split(all_events, max(1, n_chunks))
        all_events_list = []
        
        # * Connect to DB and set cursor
        with sqlite3.connect(self._path) as db:
            c = db.cursor()
            lengths_key = ['split_in_ice_pulses_event_length']
            for chunk in chunks:
                scalars = self._fetch(c, chunk, scalar_features, table='scalar')
                seqs = self._fetch(c, chunk, seq_features, table='sequential')
                targets = self._fetch(c, chunk, target_features, table='scalar')
                mask = self._fetch(c, chunk, mask, table='sequential')
                lengths = self._fetch(c, chunk, lengths_key, table='meta')

                events_list = self._make_batch(chunk, scalars, seqs, targets, mask, lengths)
                all_events_list.extend(events_list)
        
        return all_events_list

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

        # * 'SplitInIcePulses' corresponds to all DOMs
        # * 'SRTInIcePulses' corresponds to Icecubes cleaned doms
        self.dom_mask = dom_mask

        self.weights = weights # * To be determined in get_meta_information
        self.len = None # * To be determined in get_meta_information
        self.indices = None # * To be determined in get_meta_information
        self._n_events_per_dir = None # * To be determined in get_meta_information

        self._get_meta_information()
        
    def __getitem__(self, index):
        # * Find path
        true_index = self.indices[index]
        weight = self.weights[index]
        filename = str(true_index) + '.pickle'
        path = self.directory+'/pickles/'+str(true_index//self._n_events_per_dir)\
            +'/'+str(true_index)+'.pickle'
        
        # * Load event
        with open(path, 'rb') as f:
            event = pickle.load(f)

        # * Extract relevant data.
        dom_indices = event['masks'][self.dom_mask]
        actual_seq_len = event[self.prefix][self.seq_features[0]]\
                        [dom_indices].shape[0]
        seq_len = min(actual_seq_len, self.max_seq_len)
        seq_array = np.empty((self.n_seq_features, seq_len))

        # * If a maximum sequence length is given, we overwrite dom_indices with
        # * randomly sampled indices from dom_indices without replacement until
        # * we have enough.
        if actual_seq_len > self.max_seq_len:
            dom_indices = sorted(np.random.choice(dom_indices, 
                                self.max_seq_len, replace=False))

        # * Sequential data
        for i, key in enumerate(self.seq_features):
            try:        
                seq_array[i, :] = event[self.prefix][key][dom_indices]
            except KeyError:
                seq_array[i, :] = event['raw'][key][dom_indices]

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
        # * Get mask
        mask_all = np.array(load_pickle_mask(self.directory, self.masks))
        n_events = len(mask_all)

        # * Extract the indices corresponding to the train/val/test part.
        from_frac, to_frac = self._get_from_to()
        indices = get_indices_from_fraction(n_events, from_frac, to_frac)
        
        # * Get weights
        self.indices = mask_all[indices]
        self.weights = load_pickle_weights(self.directory, self.weights)[self.indices]
        self.len = min(self.n_events_wanted, len(self.indices))

        # * Now get the number of events per event directory
        self._n_events_per_dir = len([event for event in Path(self.directory+'/pickles/0').iterdir()])
    
    def shuffle_indices(self):
        random.shuffle(self.indices)

class SqliteLoader(data.Dataset):
    '''A Pytorch dataloader for neural nets with sequential and scalar variables. This dataloader does not load data into memory, but opens a h5-file, reads and closes the file again upon every __getitem__.

    Input: Directory to loop over, targetnames, scalar feature names, sequential feature names, type of set (train, val or test), train-, test- and validation-fractions and an optional datapoints_wanted.
    '''
    def __init__(self, directory, seq_features, scalar_features, targets, 
                masks=['all'], n_events_wanted=np.inf, weights='None', 
                dom_mask='SplitInIcePulses', max_seq_len=np.inf, keyword=None, batch_size=None):
        if batch_size == None:
            raise ValueError('A batchsize must be specified!')

        self.directory = get_project_root() + directory

        self.scalar_features = scalar_features
        self.seq_features = seq_features
        self.targets = targets

        self.n_scalar_features = len(scalar_features)
        self.n_seq_features = len(seq_features)
        self.n_targets = len(targets)

        self.masks = masks
        self.n_events_wanted = n_events_wanted
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size

        if keyword == 'train':
            self.db = SqliteFetcher(PATH_TRAIN_DB)
        elif keyword == 'val':
            self.db = SqliteFetcher(PATH_VAL_DB)
        elif keyword == 'test':
            self.db = SqliteFetcher(PATH_TEST_DB)
        elif keyword == 'predict':
            self.db = SqliteFetcher(PATH_VAL_DB)
        else:
            raise KeyError('Unknown keyword given (%s) to SqliteLoader'%(keyword))

        # * 'SplitInIcePulses' corresponds to all DOMs
        # * 'SRTInIcePulses' corresponds to Icecubes cleaned doms
        self.dom_mask = [dom_mask]
        self.keyword = keyword

        self.weights = weights # * To be determined in get_meta_information
        self.len = None # * To be determined in get_meta_information
        self.indices = None # * To be determined in get_meta_information

        self._get_meta_information()
        self.shuffle_indices()

    def __getitem__(self, index):

        from_, to_ = index*self.batch_size, (index+1)*self.batch_size
        ids = self.indices[from_:to_]
        weights = [self.weights[str(idx)] for idx in ids]
        
        # * Load batch - gets list back with tuples 
        # * (seq_arr, scalar_arr, target_arr). Add weights afterwards.
        batch = self.db.make_batch(
            all_events=ids,
            seq_features=self.seq_features,
            scalar_features=self.scalar_features,
            targets=self.targets,
            mask=self.dom_mask,
        )
        
        # * Tuple is now passed to collate_fn - handle training and predicting
        # * differently. We need the name of the event for prediction to log
        # * which belongs to which
        if self.type == 'predict':
            pack = [
                e+(weights[i_event], self.keyword, ids[i_event]) for i_event, e in enumerate(batch)
            ]

        else:
            pack = [
                e+(weights[i_event],) for i_event, e in enumerate(batch)
            ]
        
        return pack

    def __len__(self):
        return self.len

    def __repr__(self):
        return 'SqliteLoader'

    def _get_meta_information(self):
        '''Extracts filenames, calculates indices induced by train-, val.- and test_frac
        '''

        # * Get mask
        self.indices = np.array(
            load_sqlite_mask(
                self.directory, self.masks, self.keyword
            )
        )

        # * Get weights
        self.weights = load_sqlite_weights(self.directory, self.weights)
        self.len = min(self.n_events_wanted, len(self.indices))//self.batch_size
    
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
    # * how much data should be trained on?
    train_frac = data_pars.get('train_frac', None) 
    # * how much data should be used for validation?
    val_frac = data_pars.get('val_frac', None) 
    # * how much data should be used for training
    test_frac = data_pars.get('test_frac', None) 
    # * which cleaning lvl and transform should be applied?
    file_keys = data_pars.get('file_keys', None) 
    mask_names = data_pars['masks']
    weights = data_pars.get('weights', 'None')
    dom_mask = data_pars.get('dom_mask', 'SplitInIcePulses')
    max_seq_len = data_pars.get('max_seq_len', np.inf)
    batch_size = hyper_pars.get('batch_size', None)
    
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
    if file_keys:
        prefix = 'transform'+str(file_keys.get('transform', 'None'))+'/'
        if file_keys['transform'] == -1:
            prefix = 'raw/'

    if 'LstmLoader' == data_pars['dataloader']:
        dataloader = LstmLoader(data_dir, file_keys, targets, scalar_features, seq_features, keyword, train_frac, val_frac, test_frac)
    elif 'SeqScalarTargetLoader' == data_pars['dataloader']:
        prefix = 'transform'+str(file_keys['transform'])+'/'
        dataloader = SeqScalarTargetLoader(data_dir, seq_features, 
                        scalar_features, targets, keyword, train_frac, 
                        val_frac, test_frac, prefix=prefix)
    elif 'FullBatchLoader' == data_pars['dataloader']:
        dataloader = FullBatchLoader(data_dir, seq_features, scalar_features, 
                        targets, keyword, train_frac, val_frac, test_frac,
                        batch_size, prefix=prefix, n_events_wanted=n_events_wanted,
                        particle_code=particle_code, file_list=file_list, 
                        mask_name=mask_name, drop_last=drop_last, 
                        debug_mode=debug_mode)
    elif 'PickleLoader' == data_pars['dataloader']:
        prefix = 'transform'+str(file_keys['transform'])
        if file_keys['transform'] == -1:
            prefix = 'raw'
        dataloader = PickleLoader(data_dir, seq_features, scalar_features, 
                                targets, keyword, train_frac, val_frac, 
                                test_frac, prefix=prefix, n_events_wanted=n_events_wanted, 
                                masks=mask_names, weights=weights,
                                dom_mask=dom_mask, max_seq_len=max_seq_len
        )
    elif 'SqliteLoader' == data_pars['dataloader']:
        dataloader = SqliteLoader(data_dir, seq_features, scalar_features, targets, masks=mask_names, n_events_wanted=n_events_wanted, weights=weights, dom_mask=dom_mask, max_seq_len=max_seq_len, keyword=keyword,
        batch_size=batch_size
        )
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

def get_collate_fn(data_pars, mode='normal', permute_seq_features=[], permute_scalar_features=[]):
    '''Returns requested collate-function, if the key 'collate_fn' is in the dictionary data_pars.
    '''

    if 'collate_fn' in data_pars:
        name = data_pars['collate_fn']
        if name == 'PadSequence':
            func = PadSequence(mode=mode, permute_seq_features=permute_seq_features, permute_scalar_features=permute_scalar_features)
        
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

def load_pickle_weights(dataset, weights):
    """Small function to load weights for a dataloader.
    
    Arguments:
        dataset {str} -- path to dataset
        weights {str} -- name of weights.
    
    Returns:
        array -- weights
    """    
    path = get_project_root()+get_path_from_root(dataset)+'/weights/'+weights+'.pickle'
    
    if weights == 'None':
        weights = [1]*get_n_tot_pickles(dataset)
    else:
        with open(path, 'rb') as f:
            weights = pickle.load(f)['weights']
    
    return np.array(weights)

def load_sqlite_weights(dataset, weights):
    """Small function to load weights for a dataloader.

    Weights are loaded as a dictionary with event_id as keys and their 
    corresponding weight
    
    Arguments:
        dataset {str} -- path to dataset
        weights {str} -- name of weights.
    
    Returns:
        dictionary -- dicitonary with weights for train, val. and test-db.
    """    
    path = '/'.join([PATH_DATA_OSCNEXT, 'weights', weights+'.pickle'])
    
    with open(path, 'rb') as f:
        weights = pickle.load(f)['weights']

    return weights
#*======================================================================== 
#* MODELS
#*========================================================================

class AveragePool(nn.Module):
    def __init__(self):
                
        super(AveragePool, self).__init__()
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
        # * By masking with 0, it is ensured that DOMs that are actually not there do not have an influence on the sum. By dividing with sequence length, we get the true mean
        seq = seq.masked_fill(~mask, 0.0)
        if self._batch_first:
            # * (B, L, *) --> (B, *)
            seq = torch.sum(seq, dim=1)
            bs, feats = seq.shape
            # * Some view-acrobatics due to broadcasting semantics.
            seq = (seq.view(feats, bs)/lengths).view(bs, feats)
        else:
            raise ValueError('Not sure when batch not first - AveragePool')

        return seq

    def _get_mask(self, lengths, maxlen, batch_first=False, device=None):
        # * Assumes mask.size[S, B, *] or mask.size[B, S, *]
        if batch_first:
            # * The 'None' is a placeholder so dimensions are matched.
            mask = torch.arange(maxlen, device=device)[None, :] < lengths[:, None]
            mask = mask.unsqueeze(2)
        return mask

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
            init_weights(arch_dict, arch_dict['nonlin'], self.Q)
            init_weights(arch_dict, arch_dict['nonlin'], self.K)
            init_weights(arch_dict, arch_dict['nonlin'], self.V)
        elif mode == 'decoder':
            raise ValueError('AttentionDecoder: Not implemented yet')
            self.Q = nn.Linear(in_features=n_in, out_features=n_out)
            self.K = nn.Linear(in_features=n_in, out_features=1)
            self.V = nn.Linear(in_features=n_in, out_features=n_out)
            init_weights(arch_dict, arch_dict['nonlin'], self.Q)
            init_weights(arch_dict, arch_dict['nonlin'], self.K)
            init_weights(arch_dict, arch_dict['nonlin'], self.V)

        self.softmax = nn.Softmax(dim=-1)
        if self.layer_dict.get('LayerNorm', False):
            self.norm = nn.LayerNorm(n_out)
        self.linear_out = nn.Linear(in_features=n_out, out_features=n_out)
        self.nonlin = add_non_lin(arch_dict, arch_dict['nonlin'])
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

class AttentionBlock2(nn.Module):
    """Implementation of Self Attention almost as described in 'Attention is All You Need'. Uses no value-vectors, but just the sequence itself. Furthermore, experimenting with only normalizing after nonlinearity.
    
    (or https://jalammar.github.io/illustrated-transformer/) - calculates query-, key- and valuevectors, softmaxes a padded sequence and scales the dotproducts and returns weighted sum of values vectors.
    
    Can work both as a seq2seq encoder or as a seq2vec decoder - in this case, the key-matrix produces one key only.
    Returns:
        nn.Module -- A Self-attention layer 
    """
    def __init__(self, arch_dict, layer_dict, n_in, n_out, batch_first=True):
                
        super(AttentionBlock2, self).__init__()
        self.arch_dict = arch_dict
        self.layer_dict = layer_dict
        self.n_in = n_in
        self.n_out = n_out
        self._batch_first = batch_first

        self.Q = nn.Linear(in_features=n_in, out_features=n_out)
        self.K = nn.Linear(in_features=n_in, out_features=n_out)
        init_weights(arch_dict, arch_dict['nonlin'], self.Q)
        init_weights(arch_dict, arch_dict['nonlin'], self.K)

        self.softmax = nn.Softmax(dim=-1)
        self.linear_out = nn.Linear(in_features=n_out, out_features=n_out)
        self.nonlin = add_non_lin(arch_dict, arch_dict['nonlin'])
        if self.layer_dict.get('LayerNorm', False):
            self.norm = nn.LayerNorm(n_out)
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
        
        # * Attention -> potential norm and residual connection
        post_attention = self._calc_self_attention(q, k, seq, lengths, max_length, batch_first=self._batch_first, device=device)
        
        # * linear layer -> nonlin -> potential norm and residual connection
        output = self.nonlin(self.linear_out(post_attention))
        if self.residual_connection:
            output = output + post_attention
        if self.norm:
            output = self.norm(output)

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

class RnnBlock(nn.Module):
    
    def __init__(self, n_in, n_out, n_parallel, num_layers, rnn_type='LSTM', bidir=False, residual=False, batch_first=True, learn_init=False, dropout=0.0):
                
        super(RnnBlock, self).__init__()

        self._batch_first = batch_first
        self.hidden_size = n_out
        self.n_in = n_in
        self.residual = residual
        self.bidir = bidir
        self.n_dirs = 2 if bidir else 1
        self.n_layers = num_layers
        self.learn_init = learn_init
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.par_RNNs = nn.ModuleList()
        if learn_init:
            self.init_hidden_states = nn.ParameterList()
            if rnn_type == 'LSTM':
                self.init_cell_states = nn.ParameterList()
        
        if rnn_type == 'LSTM':
            for i_par in range(n_parallel):
                self.par_RNNs.append(nn.LSTM(input_size=n_in, hidden_size=n_out, bidirectional=bidir, 
                num_layers=num_layers, batch_first=batch_first, dropout=self.dropout))
                if self.learn_init:
                    n_parameters = self.n_dirs*self.hidden_size*self.n_layers
                    self.init_hidden_states.append(nn.Parameter(torch.empty(n_parameters).normal_(mean=0,std=1.0)))
                    self.init_cell_states.append(nn.Parameter(torch.empty(n_parameters).normal_(mean=0,std=1.0)))

        elif rnn_type == 'GRU':
            for i_par in range(n_parallel):
                self.par_RNNs.append(nn.GRU(input_size=n_in, hidden_size=n_out, bidirectional=bidir, 
                num_layers=num_layers, batch_first=batch_first, dropout=self.dropout))
                if self.learn_init:
                    n_parameters = self.n_dirs*self.hidden_size*self.n_layers
                    self.init_hidden_states.append(nn.Parameter(torch.empty(n_parameters).normal_(mean=0,std=1.0)))
                    # self.init_cell_states.append(nn.Parameter(torch.empty(n_parameters).normal_(mean=0,std=1.0)))
        else:
            raise KeyError('UNKNOWN RNN TYPE (%s) PASSED TO MAKEMODEL'%(rnn_type))
    
    def forward(self, seq, lengths, device=None):
        
        # * The max length is retrieved this way such that dataparallel works
        if self._batch_first:
            longest_seq = seq.shape[1]
            batch_size = seq.shape[0]
        else:
            longest_seq = seq.shape[0]
            batch_size = seq.shape[1]

        # * Send through LSTMs! Prep for first layer.
        seq_packed = pack(seq, lengths, batch_first=self._batch_first)
        
        # * x is output - concatenate outputs of LSTMs in parallel
        for i_par in range(len(self.par_RNNs)):
            
            # * Instantiate hidden and cell.
            # ? Maybe learn initial state?
            if self.learn_init:
                
                # * GRUs and LSTMs require different initiations
                if self.rnn_type == 'LSTM':
                    # ? Dont know why, but the .contiguous call is needed, else an error is thrown
                    hidden = self.init_hidden_states[i_par].view(self.n_layers*self.n_dirs, 1, -1).expand(-1, batch_size, -1).contiguous()
                    cell = self.init_cell_states[i_par].view(self.n_layers*self.n_dirs, 1, -1).expand(-1, batch_size, -1).contiguous()
                    h = (hidden, cell)
                elif self.rnn_type == 'GRU':
                    h = self.init_hidden_states[i_par].view(self.n_layers*self.n_dirs, 1, -1).expand(-1, batch_size, -1).contiguous()

            else:
                h = self.init_hidden(batch_size, self.par_RNNs[i_par], device)

            # * Send through LSTM
            self.par_RNNs[i_par].flatten_parameters()
            seq_par, h_par = self.par_RNNs[i_par](seq_packed, h)
            seq_par_post, lengths = unpack(seq_par, batch_first=True, total_length=longest_seq)

            # * when multiple directions and layers, h_out is weird - needs careful treatment
            if self.rnn_type == 'LSTM':
                h_out = h_par[0].view(self.n_layers, self.n_dirs, batch_size, self.hidden_size)
            elif self.rnn_type == 'GRU':
                h_out = h_par.view(self.n_layers, self.n_dirs, batch_size, self.hidden_size)
            
            if self.bidir:
                h_out = torch.cat((h_out[self.n_layers-1, 0, :, :], h_out[self.n_layers-1, 1, :, :]), axis=-1)
            else:
                h_out = h_out[self.n_layers-1, 0, :, :]#torch.cat((, h_out[self.n_layers-1, 1, :, :]), axis=-1)

            if self.residual:
                seq_par_post = seq_par_post + seq
            
            # * Define x on first parallel LSTM-module
            if i_par == 0:
                x = h_out
                seq_out = seq_par_post

            # * Now keep cat'ing for each parallel stack
            else:
                x = torch.cat((x, h_out), -1)
                seq_out = torch.cat((seq_out, seq_par_post), -1)
        
        return seq_out, x.squeeze(0)
        
    def init_hidden(self, batch_size, layer, device):
        hidden_size = int(layer.weight_ih_l0.shape[0]/4)

        # * Initialize hidden and cell states - to either random nums or 0's
        # * (num_layers * num_directions, batch, hidden_size)
        if self.rnn_type == 'LSTM':
            output = (torch.zeros(self.n_dirs*self.n_layers, batch_size, hidden_size, device=device), 
            torch.zeros(self.n_dirs*self.n_layers, batch_size, hidden_size, device=device))
        elif self.rnn_type == 'GRU':
            output = torch.zeros(self.n_dirs*self.n_layers, batch_size, hidden_size, device=device)
        
        return output

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
            if layer_name == 'Linear':

                # * If scalar variables are supplied for concatenation, do it! 
                # * But make sure to only do it once.
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
            
            # * AttentionBlock2 is seq2seq
            elif layer_name == 'AttentionBlock2':
                seq = entry(seq, lengths, device=device)
            
            # * Many to one! Therefore outputs x
            elif layer_name == 'ManyToOneAttention':
                x = entry(seq, lengths, device=device)
            
            # * The MaxPool-layer is used after sequences have been treated 
            # * -> prepare for linear decoding.
            elif layer_name == 'MaxPool':
                x = entry(seq, lengths, device=device)
            
            # * Same goes for average pool.
            elif layer_name == 'AveragePool':
                x = entry(seq, lengths, device=device)
            
            elif layer_name == 'LstmBlock':
                seq, x = entry(seq, lengths, device=device)
            
            elif layer_name == 'RnnBlock':
                seq, x = entry(seq, lengths, device=device)
            
            elif layer_name == 'BiLSTM':
                seq, x = entry(seq, lengths, device=device)
            
            elif layer_name == 'ResBlock':
                # * If scalar variables are supplied for concatenation, do it! 
                # * But make sure to only do it once.
                if 'scalars' in locals(): 
                    if add_scalars: 
                        x, add_scalars = self.concat_scalars(x, scalars)

                x = entry(x)
            
            elif layer_name == 'ResAttention':
                seq = entry(seq, lengths, device=device)
            
            elif layer_name == 'ResBlockSeq':
                seq = entry(seq)

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

class ManyToOneAttention(nn.Module):
    """Implementation of Self Attention almost as described in 'Attention is All You Need'.
    
    (or https://jalammar.github.io/illustrated-transformer/) - calculates query-, key- and valuevectors, softmaxes a padded sequence and scales the dotproducts and returns weighted sum of values vectors.
    
    Can work both as a seq2seq encoder or as a seq2vec decoder - in this case, the key-matrix produces one key only.
    Returns:
        nn.Module -- A Self-attention layer 
    """
    def __init__(self, arch_dict, layer_dict, batch_first=True):
                
        super(ManyToOneAttention, self).__init__()
        self.arch_dict = arch_dict
        self.layer_dict = layer_dict
        self.n_in = layer_dict['n_in']
        self._batch_first = batch_first

        self.Q = nn.Linear(in_features=self.n_in, out_features=self.n_in)
        # * We will only have one keyvector - this is the one we want to learn.
        # * Instantiate with normally distributed numbers. The dotproduct of 2 vectors of dim N with normally distributed numbers will have a mean of 0 and variance of N. 
        self.k = nn.Parameter(torch.empty(self.n_in).normal_(mean=0,std=1.0), requires_grad=True)
        init_weights(arch_dict, arch_dict['nonlin'], self.Q)

        self.softmax = nn.Softmax(dim=-1)
        
    
    def forward(self, seq, lengths, device=None):
        # * The max length is retrieved this way such that dataparallel works
        if self._batch_first:
            max_length = seq.shape[1]
        else:
            max_length = seq.shape[0]

        # TODO: Make Q a nonlinear function i.e. some layers. 
        q = self.Q(seq)
        
        # * Attention -> potential norm and residual connection
        post_attention = self._calc_self_attention(q, seq, lengths, max_length, batch_first=self._batch_first, device=device)
        
        return post_attention.squeeze(1)

    def _calc_self_attention(self, q, v, lengths, max_length, batch_first=False, device=None):
        
        # * The matrix multiplication is always done with using the last two dimensions, i.e. (*, 10, 11).(*, 11, 7) = (*, 10, 7) 
        # * The transpose means swap second to last and last dimension
        # * masked_fill_ is in-place, masked_fill creates a new tensor
        
        # * q: (B, L, F). k: (F, 1)
        weights = torch.matmul(q, self.k.view(self.n_in, -1)) / sqrt(self.n_in)
        mask = self._get_mask(lengths, max_length, batch_first=batch_first, device=device)
        
        # * weights: (B, L, 1)
        weights = weights.squeeze(-1).masked_fill(~mask, float('-inf'))
        weights = self.softmax(weights)
        
        # * Calculate weighted sum of v-vectors.
        shape = weights.shape
        # * output becomes: (B, 1, F)
        output = torch.matmul(weights.view(shape[0], -1, shape[1]), v)
        
        return output

    def _get_mask(self, lengths, maxlen, batch_first=False, device=None):
        
        # * Assumes mask.size[S, B, *] or mask.size[B, S, *]
        if batch_first:
            mask = torch.arange(maxlen, device=device)[None, :] < lengths[:, None]

        return mask

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

class NormNonlinWeight(nn.Module):

    def __init__(self, arch_dict, layer_dict, n_in, n_out, norm=None):
        super(NormNonlinWeight, self).__init__()
        self.norm = norm
        if self.norm:
            self.normalize = add_norm(arch_dict, layer_dict, n_in)
        self.nonlin = add_non_lin(arch_dict, layer_dict)
        self.linear = nn.Linear(in_features=n_in, out_features=n_out)
    
    def forward(self, x, device=None):
        if self.norm:
            output = self.linear(self.nonlin((self.normalize(x))))
        else:
            output = self.linear(self.nonlin(x))
        
        return output

class ResBlock(nn.Module):
    """A Residual block as proposed in 'Identity Mappings in Deep Residual Networks'
    """    
    def __init__(self, arch_dict, layer_dict, n_in, n_out, norm=False):
        super(ResBlock, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if n_in != n_out:
            self.linear0 = nn.Linear(in_features=n_in, out_features=n_out)

        if norm:
            self.norm1 = add_norm(arch_dict, layer_dict, n_out)
        self.non_lin1 = add_non_lin(arch_dict, arch_dict['nonlin'])
        self.linear1 = nn.Linear(in_features=n_out, out_features=n_out)
        init_weights(arch_dict, arch_dict['nonlin'], self.linear1)
        if norm:
            self.norm2 = add_norm(arch_dict, layer_dict, n_out)
        self.non_lin2 = add_non_lin(arch_dict, arch_dict['nonlin'])
        self.linear2 = nn.Linear(in_features=n_out, out_features=n_out)
        init_weights(arch_dict, arch_dict['nonlin'], self.linear2)

    def forward(self, seq, device=None):

        if self.n_in != self.n_out:
            seq = self.linear0(seq)
        
        res = self.linear1(self.non_lin1(self.norm1(seq)))
        res = self.linear2(self.non_lin2(self.norm2(res)))

        return seq+res

class ResAttention(nn.Module):
        
    def __init__(self, arch_dict, layer_dict, n_in, n_out, batch_first=True):
        super(ResAttention, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_layers = layer_dict['n_res_layers']
        self.norm = layer_dict.get('norm', False)

        if self.n_in != self.n_out:
            self.linear0 = nn.Linear(in_features=self.n_in, out_features=self.n_out)
        
        self.attention = SelfAttention(arch_dict, self.n_out, self.n_out, batch_first=batch_first)
        self.post_attntn = nn.Sequential(*[NormNonlinWeight(arch_dict, layer_dict, self.n_out, self.n_out, norm=self.norm) for i in range(self.n_layers)])
    
    def forward(self, seq, lengths, device=None):
        if self.n_in != self.n_out:
            seq = self.linear0(seq)
        post_attention = self.attention(seq, lengths, device=device)
        post_attention = self.post_attntn(post_attention)
        
        return seq+post_attention

class SelfAttention(nn.Module):
    """Implementation of Self Attention almost as described in 'Attention is All You Need'. Uses no value-vectors, but just the sequence itself. Furthermore, experimenting with only normalizing after nonlinearity.
    
    (or https://jalammar.github.io/illustrated-transformer/) - calculates query-, key- and valuevectors, softmaxes a padded sequence and scales the dotproducts and returns weighted sum of values vectors.
    
    Can work both as a seq2seq encoder or as a seq2vec decoder - in this case, the key-matrix produces one key only.
    Returns:
        nn.Module -- A Self-attention layer 
    """
    def __init__(self, arch_dict, n_in, n_out, batch_first=True):
        super(SelfAttention, self).__init__()
        self.arch_dict = arch_dict
        self.n_in = n_in
        self.n_out = n_out
        self._batch_first = batch_first

        self.Q = nn.Linear(in_features=n_in, out_features=n_out)
        self.K = nn.Linear(in_features=n_in, out_features=n_out)
        init_weights(arch_dict, arch_dict['nonlin'], self.Q)
        init_weights(arch_dict, arch_dict['nonlin'], self.K)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, seq, lengths, device=None):
        
        # * The max length is retrieved this way such that dataparallel works
        if self._batch_first:
            max_length = seq.shape[1]
        else:
            max_length = seq.shape[0]

        # * Find queries and keys
        q = self.Q(seq)
        k = self.K(seq)
        
        # * Attention
        output = self._calc_self_attention(q, k, seq, lengths, max_length, batch_first=self._batch_first, device=device)

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

#* ======================================================================== 
#* MODEL FUNCTIONS
#* ========================================================================

def add_LSTM_module(arch_dict, layer_dict, modules):
    n_neurons = len(layer_dict['input_sizes'])-1
    
    for i_neurons in range(n_neurons):
        isize = layer_dict['input_sizes'][i_neurons]
        hsize = layer_dict['input_sizes'][i_neurons+1]
        bidir = layer_dict['bidir']
        modules.append(nn.LSTM(input_size=isize, hidden_size=hsize, bidirectional=bidir, batch_first=True))
    return modules

def add_linear_embedder(arch_dict, layer_dict):
    n_layers = len(layer_dict['input_sizes'])-1

    layers = []
    for i_layer in range(n_layers):
        isize = layer_dict['input_sizes'][i_layer]
        hsize = layer_dict['input_sizes'][i_layer+1]
        
        layers.append(ResBlock(arch_dict, layer_dict, ))
        # * Add a matrix to linearly 
        layers.append(nn.Linear(in_features=isize, out_features=hsize))
        init_weights(arch_dict, arch_dict['nonlin'], layers[-1])
        if layer_dict.get('LayerNorm', False):
            layers.append(nn.LayerNorm(hsize))
        layers.append(add_non_lin(arch_dict, arch_dict['nonlin']))
    
    return nn.Sequential(*layers)

def add_ResBlock(arch_dict, layer_dict):
    n_ins = layer_dict['input_sizes'][:-1]
    n_outs = layer_dict['input_sizes'][1:]

    layers = []
    for n_in, n_out in zip(n_ins, n_outs):
        layers.append(ResBlock(arch_dict, layer_dict, n_in, n_out, layer_dict.get('norm', False)))
    
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
        init_weights(arch_dict, arch_dict['nonlin'], layers[-1])

        # * If last layer, do not add non-linearities or normalization
        if i_layer+1 == n_layers: continue

        # * If not, add non-linearities and normalization in required order
        else:
            if layer_dict['norm_before_nonlin']:

                # * Only add normalization layer if wanted!
                if arch_dict['norm']['norm'] != None:
                    layers.append(add_norm(arch_dict, arch_dict['norm'], hsize))
                layers.append(add_non_lin(arch_dict, arch_dict['nonlin']))

            else:
                layers.append(add_non_lin(arch_dict, arch_dict['nonlin']))
                if arch_dict['norm']['norm'] != None:
                    layers.append(add_norm(arch_dict, arch_dict['norm'], hsize))

    return nn.Sequential(*layers)

def add_non_lin(arch_dict, layer_dict):
    if arch_dict['nonlin']['func'] == 'ReLU': 
        return nn.ReLU()
    
    elif arch_dict['nonlin']['func'] == 'LeakyReLU':
        negslope = arch_dict['nonlin'].get('negslope', 0.01)
        return nn.LeakyReLU(negative_slope=negslope)

    else:
        raise ValueError('An unknown nonlinearity could not be added in model generation.')

def add_norm(arch_dict, layer_dict, n_features):
    
    if layer_dict['norm'] == 'BatchNorm1D':
        
        if 'momentum' in layer_dict: mom = layer_dict['momentum']
        else: mom = 0.1

        if 'eps' in layer_dict: eps = layer_dict['eps']
        else: eps = 1e-05
        
        return nn.BatchNorm1d(n_features, eps=eps, momentum=mom)
    
    elif layer_dict['norm'] == 'LayerNorm':
        return nn.LayerNorm(n_features)

    else: 
        raise ValueError('An unknown normalization could not be added in model generation.')

def add_AttentionBlock_modules(arch_dict, layer_dict, modules, mode=None):

    for n_in, n_out in zip(layer_dict['input_sizes'][:-1], layer_dict['input_sizes'][1:]):
        modules.append(AttentionBlock(arch_dict, layer_dict, n_in, n_out, mode=mode))
    
    return modules

def add_AttentionBlock2_modules(arch_dict, layer_dict, modules):

    for n_in, n_out in zip(layer_dict['input_sizes'][:-1], layer_dict['input_sizes'][1:]):
        modules.append(AttentionBlock2(arch_dict, layer_dict, n_in, n_out))
    
    return modules

def add_ResAttention_modules(arch_dict, layer_dict, modules):

    for n_in, n_out in zip(layer_dict['input_outputs'][:-1], layer_dict['input_outputs'][1:]):
        modules.append(ResAttention(arch_dict, layer_dict, n_in, n_out))

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
    save_dir = get_project_root() + get_path_from_root(save_dir)
    hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(save_dir)
    particle_code = get_particle_code(data_pars['particle'])
    device = get_device(meta_pars['gpu'][0])
    model_dir = save_dir+'/checkpoints'
    best_pars = find_best_model_pars(model_dir)
    n_devices = len(meta_pars['gpu'])
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
            if key == 'ResBlock':
                modules.append(add_ResBlock(arch_dict, layer_dict))
            elif key == 'LSTM': 
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
            elif key == 'AttentionBlock2':
                modules = add_AttentionBlock2_modules(arch_dict, layer_dict, modules)
            elif key == 'ManyToOneAttention':
                modules.append(ManyToOneAttention(arch_dict, layer_dict))
            elif key == 'MaxPool':
                modules.append(MaxPool())
            elif key == 'AveragePool':
                modules.append(AveragePool())
            elif key == 'LstmBlock':
                modules.append(LstmBlock(**layer_dict))
            elif key == 'RnnBlock':
                modules.append(RnnBlock(**layer_dict))
            elif key == 'BiLSTM':
                modules.append(BiLSTM(**layer_dict))
            elif key == 'ResAttention':
                modules = add_ResAttention_modules(arch_dict, layer_dict, modules)
            else: 
                raise ValueError('An unknown module (%s) could not be added in model generation.'%(key))

    return modules 

def get_layer_names(arch_dict):
    '''Extracts layer names from an arch_dict
    '''
    layer_names = []
    for layer in arch_dict['layers']:
        for layer_name, dicts in layer.items():
            
            if layer_name == 'AttentionBlock':
                n_attention_modules = len(layer['AttentionBlock']['input_sizes'])-1
                for nth_attention_layer in range(n_attention_modules):
                    layer_names.append(layer_name)
            
            elif layer_name == 'AttentionBlock2':
                n_attention_modules = len(layer['AttentionBlock2']['input_sizes'])-1
                for nth_attention_layer in range(n_attention_modules):
                    layer_names.append(layer_name)
            
            elif layer_name == 'ResAttention':
                n_attention_modules = len(layer['ResAttention']['input_outputs'])-1
                for nth_attention_layer in range(n_attention_modules):
                    layer_names.append(layer_name)
            
            elif layer_name == 'ResBlock':
                if dicts['type'] == 'seq':
                    layer_names.append('ResBlockSeq')
                elif dicts['type'] == 'x':
                    layer_names.append('ResBlock')
                else:
                    raise KeyError('ResBlock: "type" MUST be supplied!')
            else:
                layer_names.append(layer_name)
    
    return layer_names
