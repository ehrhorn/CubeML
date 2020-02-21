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
# import src.modules.loss_funcs as lf
# import src.modules.helper_functions as hf
# from src.modules.eval_funcs import *
# import src.modules.reporting as rpt
# from src.modules.constants import *
# from src.modules.classes import *
# import src.modules.preprocessing as pp
from src.modules.main_funcs import *

def calc_permutation_importance(save_dir, wandb_ID=None, seq_features=[]):
    
    # * Load the best model
    hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(save_dir)
    model = load_best_model(save_dir)

    # * Setup dataloader and generator - num_workers choice based on gut feeling - has to be high enough to not be a bottleneck
    n_predictions_wanted = data_pars.get('n_predictions_wanted', np.inf)
    LOG_EVERY = int(meta_pars.get('log_every', 200000)/4) 
    VAL_BATCH_SIZE = data_pars.get('val_batch_size', 256) # ! Predefined size !
    gpus = meta_pars['gpu']
    device = get_device(gpus[0])
    dataloader_params_eval = get_dataloader_params(VAL_BATCH_SIZE, num_workers=8, shuffle=False, dataloader=data_pars['dataloader'])
    val_set = load_data(hyper_pars, data_pars, arch_pars, meta_pars, 'predict')
    # * SET MODE TO PERMUTE IN COLLATE_FN
    collate_fn = get_collate_fn(data_pars, mode='permute', permute_features=seq_features)
    val_generator = data.DataLoader(val_set, **dataloader_params_eval, collate_fn=collate_fn)
    N_VAL = get_set_length(val_set)

    # * Run evaluator!
    predictions, truths, indices = run_pickle_evaluator(model, val_generator, val_set.targets, gpus, 
        LOG_EVERY=LOG_EVERY, VAL_BATCH_SIZE=VAL_BATCH_SIZE, N_VAL=N_VAL)

    # * Run predictions through desired functions - transform back to 'true' values, if transformed
    predictions_transformed = inverse_transform(predictions, save_dir)
    truths_transformed = inverse_transform(truths, save_dir)

    eval_functions = get_eval_functions(meta_pars)
    error_from_preds = {}
    for func in eval_functions:
        error_from_preds[func.__name__] = func(predictions_transformed, truths_transformed)

    # * Save predictions in h5-file.
    pred_full_address = save_dir+'/data/permuted_predictions.h5'
    
    print(get_time(), 'Saving predictions...')
    with h5.File(pred_full_address, 'w') as f:
        f.create_dataset('index', data=np.array(indices))
        for key, pred in predictions.items():
            f.create_dataset(key, data=np.array([x.cpu().numpy() for x in pred]))
        for key, pred in error_from_preds.items():
            f.create_dataset(key, data=np.array([x.cpu().numpy() for x in pred]))
    print(get_time(), 'Predictions saved!')
    # * Permute 
    # # * Read the predictions. Each group in the h5-file corresponds to a raw data-file. Each group has same datasets.
    # file_address = save_dir+'/data/predictions.h5'
    # with h5.File(file_address, 'r') as f:
    #     for key in f:
    #         print(key)
    # print(seq_features)
    # print(scalar_features)

save_dir = '/home/bjoern/Thesis/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/energy_reg/test_2020.02.18-13.40.33'
<<<<<<< HEAD
calc_permutation_importance(save_dir, seq_features=['dom_charge'])
=======
# calc_permutation_importance(save_dir)

path = '/home/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/pickles/0/0'
with open(path+'.pickle', 'rb') as f:
    event = pickle.load(f)

for key in event:
    print(key)
    for key2, items in event[key].items():
        print(key2)
    print('')
>>>>>>> ed615fbe21e10398c52f6fccb96c7bf6d0643dc4
