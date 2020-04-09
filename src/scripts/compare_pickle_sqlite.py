from src.modules.classes import *
from pathlib import Path

path_pickles = Path('/'.join([PATH_DATA_OSCNEXT, 'pickles', '757']))
events = [entry.stem for entry in path_pickles.iterdir()]
events = [entry.stem for entry in path_pickles.iterdir()]
events = events[:1000]
db = SqliteFetcher(PATH_TRAIN_DB)


seq_feats = ['dom_charge', 
            'dom_x', 
            'dom_y', 
            'dom_z', 
            'dom_time', 
            'dom_charge_significance',
            'dom_frac_of_n_doms',
            'dom_d_to_prev',
            'dom_v_from_prev',
            'dom_d_minkowski_to_prev',
            'dom_d_closest',
            'dom_d_minkowski_closest',
    ]

targets = ['true_primary_energy', 'true_primary_position_x', 'true_primary_position_y', 'true_primary_position_z', 'true_primary_time', 'true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z']

scalar_feats = ['tot_charge','dom_timelength_fwhm']

np.set_printoptions(precision=3)
sqlite = db.fetch_features(all_events=events, scalar_features=targets+scalar_feats, seq_features=seq_feats)

data = {key: [] for key in seq_feats+targets+scalar_feats}
data_random = {key: [] for key in seq_feats+targets+scalar_feats}
data_pickle = {key: [] for key in seq_feats+targets+scalar_feats}
data_sqlite = {key: [] for key in seq_feats+targets+scalar_feats}

for i_event, event in enumerate(events):
    sqlite_event = sqlite[event]
    pickle_path = str(path_pickles)+'/'+event+'.pickle'
    with open(pickle_path, 'rb') as f:
        pickle_event = pickle.load(f)
    if pickle_event['meta']['particle_code'] != '140000':
        continue
    transformed_pickle = pickle_event['transform1']
    
    for key in seq_feats:
        # for key2 in sqlite_event:
            # print(key2)
        # a+=1
        d1 = sqlite_event[key] 
        d2 = transformed_pickle[key]
        data[key].append(d1-d2)
        random.shuffle(d1)
        data_random[key].append(d1-d2)

    for key in targets+scalar_feats:
        d1 = sqlite_event[key] 
        try:
            d2 = transformed_pickle[key]
        except KeyError:
            d2 = pickle_event['raw'][key]
        data_pickle[key].append(d2)
        data_sqlite[key].append(d1)
        data[key].append(d1-d2)

        
for key in seq_feats:
    data[key] = flatten_list_of_lists(data[key])
    data_random[key] = flatten_list_of_lists(data_random[key])
    d = {'data': [np.clip(data[key], -5, 5), np.clip(data_random[key], -5, 5)]}
    d['savefig'] = get_project_root() +'/reports/sqlite_pickle_comparison/difference_'+key
    _ = make_plot(d)

for key in targets+scalar_feats:
    random.shuffle(data_pickle[key])
    randomized = np.array(data_pickle[key]) - np.array(data_sqlite[key])
    d = {'data': [np.clip(data[key], -5, 5), np.clip(randomized, -5, 5)]}
    d['savefig'] = get_project_root() +'/reports/sqlite_pickle_comparison/difference_'+key
    _ = make_plot(d)

