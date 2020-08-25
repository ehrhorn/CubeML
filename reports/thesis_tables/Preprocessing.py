from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
from matplotlib import pyplot as plt
import sklearn
import os
import pandas as pd

def make_data():
    seq_keys = [
        'dom_charge', 
        'dom_x', 
        'dom_y', 
        'dom_z', 
        'dom_time', 
        'dom_atwd',
        'dom_pulse_width'
    ]
    target_keys = [
        'true_primary_energy', 
        'true_primary_position_x', 
        'true_primary_position_y', 
        'true_primary_position_z', 
        'true_primary_time', 
        'true_primary_direction_x', 
        'true_primary_direction_y', 
        'true_primary_direction_z'
    ]
    db_path = PATH_TRAIN_DB
    key = 'dom_charge'
    transformers = joblib.load(
        open(PATH_DATA_OSCNEXT + '/sqlite_transformers.pickle', 'rb')
    )
    db = SqliteFetcher(db_path)
    # Lets go with 1M ~ approximtely 1M/50 = 20k events
    ids = [str(e) for e in range(1000)]

    all_data = db.fetch_features(
        all_events=ids, 
        seq_features=seq_keys, 
        scalar_features=target_keys
        )
    data_d = {key: [] for key in all_data['0']}
    for key in target_keys:
        data_d[key] = [data[key] for event_id, data in all_data.items()]
    for key in seq_keys:
        data_d[key].extend(
            flatten_list_of_lists(
                [data[key] for event_id, data in all_data.items()]
                )
            )
    # Calculate means and std's before and after transformation
    dicts = {}
    table = np.empty((5, len(seq_keys)+len(target_keys)), dtype=object)
    for i_key, key in enumerate(data_d):
        data = data_d[key]
        d = {}
        if key in transformers:
            if type(transformers[key]) == sklearn.preprocessing._data.QuantileTransformer:
                name = 'ToNormal'
            elif sklearn.preprocessing._data.RobustScaler:
                if key == 'true_primary_energy':
                    name = 'LogRobust'
                else:
                    name = 'Robust'
            table[0, i_key] = name
            table[3, i_key] = r'%.2f'%(np.mean(data))
            table[4, i_key] = r'%.2f'%(np.std(data))
            data_pre = np.squeeze(
                transformers[key].inverse_transform(
                    np.array(data).reshape(-1, 1)
                )
            )
            if key == 'true_primary_energy':
                table[1, i_key] = r'%.2e'%(np.mean(10**data_pre))
                table[2, i_key] = r'%.2e'%(np.std(10**data_pre))
            else:
                table[1, i_key] = r'%.2e'%(np.mean(data_pre))
                table[2, i_key] = r'%.2e'%(np.std(data_pre))
        else:
            table[0, i_key] = 'None'
            table[1, i_key] = r'%.2f'%(np.mean(data))
            table[2, i_key] = r'%.2f'%(np.std(data))
            table[3, i_key] = r'-'
            table[4, i_key] = r'-'

    index = [r'Transformation', r'$\mu$, before', r'$\sigma$, before', r'$\mu$, after', r'$\sigma$, after']
    columns = []
    for col in [key for key in data_d]:
        split = col.split('_')
        new_col = r'\_'.join(split)
        columns.append(new_col)
    table_pd = pd.DataFrame(
        np.transpose(table),                       # values
        index=columns,    # 1st column as index
        columns=index)                # 1st row as the column names

    return table_pd
table_pd = make_data()
print(table_pd.to_latex(escape=False))
