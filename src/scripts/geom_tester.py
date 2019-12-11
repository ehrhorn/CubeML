# %%
from tables import *
import numpy as np
from sklearn.preprocessing import RobustScaler

file = '/home/mads/repos/Powershovel/geom_files/geom.h5'

with File(file, 'r') as f:
    geom = {}
    data = f.root.__getattr__('pmt_geom')
    for colname in data.colnames:
        geom[colname] = data.col(colname)


def transform_lvl_1(events):
    robust_keys = [
        'x',
        'y',
        'z'
    ]
    hists = {}
    data = {}
    for key in events:
        if type(events[key][0]) == np.ndarray:
            for var_len_data in events[key]:
                try:
                    hists[key] = np.append(hists[key], var_len_data[:])
                except KeyError:
                    hists[key] = var_len_data[:]
        else:
            hists[key] = np.array(events[key])
        data[key] = np.array(events[key])
    transformers = {}
    d_transformed = {}
    n_events = data['x'].shape[0]

    hists_transformed = {}

    for key, vals in data.items():
        if key in robust_keys:
            transformers[key] = RobustScaler()
            transformers[key].fit(hists[key].reshape(-1, 1), )
            hists_transformed[key] = transformers[key].transform(
                hists[key].reshape(-1, 1)
            )
            if vals[0].shape:
                d_transformed[key] = [[]] * n_events
                for i_event in range(n_events):
                    d_transformed[key][i_event] = transformers[key].transform(
                        vals[i_event].reshape(-1, 1)
                    ).reshape(vals[i_event].shape[0], )
            else:
                d_transformed[key] = transformers[key].transform(
                    vals.reshape(-1, 1)
                ).reshape(n_events, )
    return d_transformed

# %%
trans = transform_lvl_1(geom)
trans_key = ['x', 'y', 'z']

class Particle(IsDescription):
        pmt = Int64Col()
        om = Int64Col()
        uy = Float64Col()
        string = Int64Col()
        uz = Float64Col()
        area = Float64Col()
        y = Float64Col()
        x = Float64Col()
        z = Float64Col()
        omtype = Int64Col()
        ux = Float64Col()


with File('geom_trans.h5', mode='w') as f:
    table = f.create_table(
        where='/',
        name='pmt_geom',
        description=Particle
    )
    for i in range(len(geom['x'])):
        particle = table.row
        for key in geom.keys():
            if key in trans.keys():
                particle[key] = trans[key][i]
            else:
                particle[key] = geom[key][i]
            particle.append()
# %%
import pandas as pd
# drop_cols = ['pmt', 'om', 'string', 'area', 'omtype', 'ux', 'uy', 'uz']

with File('geom_trans.h5', 'r') as f:
    out = {}
    pmt_geom = f.root.pmt_geom
    col_names = pmt_geom.colnames
    for col_name in col_names:
        out[col_name] = pmt_geom.col(col_name)
    geom = pd.DataFrame.from_dict(out)
# geom_clean = geom.loc[geom.omtype == 20].copy()
# geom_clean.drop(drop_cols, axis=1, inplace=True)


# %%
