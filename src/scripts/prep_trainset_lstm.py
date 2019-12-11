import numpy as np
import time
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import RobustScaler
from pathlib import Path
import pickle

def create_dataset(data_dir, filename):
    '''Transforms unnormalized, uncentered Icecube data in hdf5-format to centered and normalized values in hdf5-format using the Quantile Transformer on charge (due to long tail) and RobustScaler on doms, entry

    input: File-name.
    output: hdf5-file with a prepared sample, weights, pT and eta-histograms.
    ''' 

    # ======================================================================== #
    # LOAD AND EXTRACT DATA
    # ======================================================================== #
    print('Conversion of %s started.'%(data_dir+'/'+filename))
    start_time = time.time()

    hists = {}
    # Convert hdf5 to histograms
    with h5py.File(data_dir+'/'+filename, 'r') as f:
        d_data = f['data']
        data = {}
        for key, vals in d_data.items():
            if vals[0].shape:
                for var_len_data in vals:
                    try:
                        hists[key] = np.append(hists[key], var_len_data[:])
                    except KeyError:
                        hists[key] = var_len_data[:]
                
            else: 
                hists[key] = vals[:]
            
            data[key] = vals[:]

    # Transform data using RobustScaler, Quantile Transformer to a normal distribution
    quantile_keys = ['charge']

    robust_keys = ['dom_x', 'dom_y', 'dom_z', 'time', 'toi_point_on_line_x', 'toi_point_on_line_y', 'toi_point_on_line_z', 'true_muon_energy', 'true_muon_entry_position_x', 'true_muon_entry_position_y', 'true_muon_entry_position_z']

    no_transform = ['toi_direction_x', 'toi_direction_y', 'toi_direction_z', 'toi_evalratio', 'true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z']

    do_not_use = ['linefit_direction_x', 'linefit_direction_y', 'linefit_direction_z', 'linefit_point_on_line_x', 'linefit_point_on_line_y', 'linefit_point_on_line_z']

    transformers = {}
    d_transformed = {}
    n_events = data['charge'].shape[0]

    # ======================================================================== #
    # TRANSFORM RELEVANT FEATURES
    # ======================================================================== #

    hists_transformed = {}

    for key, vals in data.items():
        if key in do_not_use: continue
        if key in no_transform: 
            d_transformed[key] = vals
            hists_transformed[key] = hists[key]                
        if key in quantile_keys:

            # Initialize and fit to a normal distribution
            transformers[key] = QuantileTransformer(output_distribution='normal')
            transformers[key].fit(hists[key].reshape(-1, 1), )
            hists_transformed[key] = transformers[key].transform(hists[key].reshape(-1,1))      

            # If entries are of variable length, transform each each entry individually
            if vals[0].shape:
                d_transformed[key] = [[]]*n_events
                for i_event in range(n_events):
                    d_transformed[key][i_event] = transformers[key].transform(vals[i_event].reshape(-1, 1)).reshape(vals[i_event].shape[0],)
            else:
                d_transformed[key] = transformers[key].transform(vals.reshape(-1, 1)).reshape(n_events,)

        if key in robust_keys:

            # Initialize RobustScaler - docs at https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html
            transformers[key] = RobustScaler()
            transformers[key].fit(hists[key].reshape(-1, 1), )
            hists_transformed[key] = transformers[key].transform(hists[key].reshape(-1,1))

            # If entries are of variable length, transform each each entry individually
            if vals[0].shape:
                d_transformed[key] = [[]]*n_events
                for i_event in range(n_events):
                    d_transformed[key][i_event] = transformers[key].transform(vals[i_event].reshape(-1, 1)).reshape(vals[i_event].shape[0],)
            else:
                d_transformed[key] = transformers[key].transform(vals.reshape(-1, 1)).reshape(n_events,)
    

    # ======================================================================== #
    # SAVE HISTOGRAMS
    # ======================================================================== #

    alpha = 0.5
    plt.style.use('default')
    n_bins = 100

    # Make the directories if they do not exist.
    save_dir = '/groups/hep/bjoernhm/thesis/CubeML/reports/figures/dataset_histograms'
    name_splitted = filename.split('.')
    hist_dir = name_splitted[0]

    if not Path(save_dir+'/'+hist_dir).is_dir():
        Path(save_dir+'/'+hist_dir).mkdir()
    dataset_dir = name_splitted[-2]
    if not Path(save_dir+'/'+hist_dir+'/'+dataset_dir).is_dir():
        Path(save_dir+'/'+hist_dir+'/'+dataset_dir).mkdir()

    address = save_dir+'/'+hist_dir+'/'+dataset_dir

    print('Histograms saved at:')
    for key in hists_transformed:
        h_figure = plt.figure()
        h_subfig = plt.subplot(1, 1, 1)
        plt.hist(hists_transformed[key], bins = n_bins)
        h_subfig.legend(title = key)
        h_subfig.grid(alpha = alpha)
        h_figure.savefig(address+'/'+key+'.pdf')
        print(address+'/'+key+'.pdf')
    
    
     

    # ======================================================================== #
    # SAVE TRANSFORMED DATA AND TRANSFORMERS
    # ======================================================================== #

    # Make directory if it does not exist
    target_dir = name_splitted[0]
    new_filename = name_splitted[-2]
    target_dir = '/groups/hep/bjoernhm/thesis/CubeML/data/processed/' + target_dir

    # Make the directory if it does not exist.
    if not Path(target_dir).is_dir():
        Path(target_dir).mkdir()

    with h5py.File(target_dir+'/'+new_filename+'.h5', 'w') as f:
        grp_data = f.create_group("processed_data")
        
        for key, data in d_transformed.items():
            if data[0].shape: 
                grp_data.create_dataset(key, data=data, dtype=h5py.special_dtype(vlen=data[0][0].dtype))
            else:
                grp_data.create_dataset(key, data=data, dtype=data[0].dtype)
    
    print('Transformed data saved at:')
    print(target_dir+'/'+new_filename+'.h5')

    transformer_save_address = target_dir+'/'+new_filename+'_transformers.sav'
    pickle.dump(transformers, open(transformer_save_address, 'wb'))
    print('Transformers saved at:')
    print(transformer_save_address)

    print("Conversion finished in %.2f seconds.\n" % (time.time() - start_time))  


data_dir = '/groups/hep/bjoernhm/thesis/CubeML/data/interim'
filename = 'MuonGun_Level2_139008.000000.h5'
create_dataset(data_dir, filename)