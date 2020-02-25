import torch
import numpy as np
from matplotlib import pyplot as plt
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
import h5py as h5
from time import time
from scipy.stats import norm
import subprocess
from multiprocessing import Pool, cpu_count

from src.modules.classes import *
import src.modules.loss_funcs as lf
import src.modules.helper_functions as hf
from src.modules.eval_funcs import *
import src.modules.reporting as rpt
from src.modules.constants import *
from src.modules.classes import *
import src.modules.preprocessing as pp
from src.modules.main_funcs import *

# %%
# class FeaturePermutationImportance:
    
#     def __init__(self, save_dir, wandb_ID=None, ):    

#         self.save_dir = get_path_from_root(save_dir)
#         self.wandb_ID = wandb_ID
#         self.feature_importances = {}

#     def calc_feature_importance_from_errors(self, baseline_errors, permuted_errors):
#         # * Use (84th-16th)/2 as metric. Corresponds to sigma. 
#         bl_percentiles, bl_lower, bl_upper = estimate_percentile(baseline_errors, [0.15865, 0.84135], bootstrap=False)
#         permuted_percentiles, permuted_lower, permuted_upper = estimate_percentile(permuted_errors, [0.15865, 0.84135], bootstrap=False)

#         # * Use error propagation to get errors
#         bl_sigmas = (bl_upper-bl_lower)/2
#         bl_metric = [(bl_percentiles[1]-bl_percentiles[0])/2]
#         bl_metric_error = [np.sqrt(np.sum(bl_sigmas*bl_sigmas))/2]

#         permuted_sigmas = (permuted_upper-permuted_lower)/2
#         permuted_metric = [(permuted_percentiles[1]-permuted_percentiles[0])/2]
#         permuted_metric_error = [np.sqrt(np.sum(permuted_sigmas*permuted_sigmas))/2]

#         feature_importance, e_feature_importance = calc_relative_error(bl_metric, permuted_metric, e1=bl_metric_error, e2=permuted_metric_error)
        
#         return feature_importance[0], e_feature_importance[0]

#     def calc_permutation_importance(self, seq_features=[], scalar_features=[]):
        

#         # * Check it hasn't already been calculated
#         if self.check_duplication(seq_features, scalar_features):
#             return
        
#         # * Load the best model
#         hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(self.save_dir)
#         model = load_best_model(self.save_dir)

#         # * Setup dataloader and generator - num_workers choice based on gut feeling - has to be high enough to not be a bottleneck
#         n_predictions_wanted = data_pars.get('n_predictions_wanted', np.inf)
#         LOG_EVERY = int(meta_pars.get('log_every', 200000)/4) 
#         VAL_BATCH_SIZE = data_pars.get('val_batch_size', 256) # ! Predefined size !
#         gpus = meta_pars['gpu']
#         device = get_device(gpus[0])
#         dataloader_params_eval = get_dataloader_params(VAL_BATCH_SIZE, num_workers=8, shuffle=False, dataloader=data_pars['dataloader'])
#         val_set = load_data(hyper_pars, data_pars, arch_pars, meta_pars, 'predict')
        
#         # * Get indices of features to permute - both scalar and sequential
#         all_seq_features = data_pars['seq_feat']
#         all_scalar_features = data_pars['scalar_feat']

#         # * Ensure features actually exist
#         try:    
#             seq_indices = [all_seq_features.index(entry) for entry in seq_features]
#             scalar_indices = [all_scalar_features.index(entry) for entry in scalar_features]
#         except ValueError:
#             print(get_time(), 'ERROR: Atleast one of features (%s, %s) does not exist. Returning.'%(', '.join(seq_features), ', '.join(scalar_features)))
#             return

#         # * SET MODE TO PERMUTE IN COLLATE_FN
#         collate_fn = get_collate_fn(data_pars, mode='permute', permute_seq_features=seq_indices)
#         val_generator = data.DataLoader(val_set, **dataloader_params_eval, collate_fn=collate_fn)
#         N_VAL = get_set_length(val_set)

#         # * Run evaluator!
#         predictions, truths, indices = run_pickle_evaluator(model, val_generator, val_set.targets, gpus, 
#             LOG_EVERY=LOG_EVERY, VAL_BATCH_SIZE=VAL_BATCH_SIZE, N_VAL=N_VAL)

#         # * Run predictions through desired functions - transform back to 'true' values, if transformed
#         predictions_transformed = inverse_transform(predictions, save_dir)
#         truths_transformed = inverse_transform(truths, save_dir)

#         eval_functions = get_eval_functions(meta_pars)
#         error_from_preds = {}
#         baseline = {}
#         pred_full_address = save_dir+'/data/predictions.h5'

#         # * Calculate PFI for all evaluation functions.
#         for func in eval_functions:
#             name = func.__name__
#             # * Calculate new errors.
#             error_from_preds[name] = func(predictions_transformed, truths_transformed)

#             # * load baseline errors
#             with h5.File(pred_full_address, 'r') as f:
#                 baseline[name] = f[name][:]

#             # * Calculate feature importance = (permuted_metric-baseline_metric)/baseline_metric
#             feature_importance, feature_importance_err = self.calc_feature_importance_from_errors(baseline[name], error_from_preds[name])

#             # * Save dictionary as an attribute. Should contain permuted feature-names and FI. Each new permutation importance is saved as an entry in a list.
#             features = seq_features.copy()
#             features.extend(scalar_features)
#             d = {'permuted': features, 'feature_importance': feature_importance, 'error': feature_importance_err}
#             self.save_feature_importance(name, d)
        
#     def save(self):
#         # * Save the results as a class instance
#         save_path = get_project_root()+self.save_dir+'/data/FeaturePermutationImportance.pickle'
#         with open(save_path, 'wb') as f:
#             pickle.dump(self, f)

#     def check_duplication(self, seq_features, scalar_features):
        
#         features = seq_features.copy()
#         features.extend(scalar_features)
        
#         # * Check no duplication
#         try:
#             some_key = [key for key in self.feature_importances][0]
#             for d in self.feature_importances[some_key]:
#                 if features == d['permuted']:
#                     print('')
#                     print(get_time(), 'PFI ALREADY EXISTS OF:', *features, '. SKIPPING RE-CALCULATION')
#                     conclusion = True
#                     break
#             else:
#                 conclusion = False
#         except IndexError:
#             # * Nothing added so far. Continue with calculation
#             conclusion = False

#         if conclusion == False:
#             print('')
#             print(get_time(), 'CALCULATING PFI OF:', *features)

#         return conclusion
    
#     def save_feature_importance(self, name, dfi):
        
#         # * dfi = dict_feature_imporatnce
#         if name not in self.feature_importances:
#             self.feature_importances[name] = [dfi]
#         else:
#             self.feature_importances[name].append(dfi)
    
#     def make_plots(self):

#         # * Loop over each performance function
#         for func, data in self.feature_importances.items():
            
#             # * Loop over all permuted features
#             # * sort wrt importance
#             sorted_data = sorted(data, key=lambda x: x['feature_importance'])
#             names, fi, errors = [], [], []
#             for d in sorted_data:
#                 name = ', '.join(d['permuted'])
#                 names.append(name)
#                 fi.append(d['feature_importance'])
#                 errors.append(d['error'])

#             # * Make barplot
#             bar_pos = np.arange(0, len(names)) + 0.5
#             d = {'keyword': 'barh', 'y': bar_pos, 'width': fi, 'height': 0.7, 'names': names, 'errors': errors}
#             d['title'] = 'Permutation Importance - %s'%(func)
#             d['xlabel'] = 'Feature Importance'
#             d['savefig'] = get_project_root()+self.save_dir+'/figures/PFI_%s.png'%(func)
#             fig = make_plot(d)

save_dir = '/home/bjoern/Thesis/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/energy_reg/test_2020.02.21-12.21.14'
a = rpt.FeaturePermutationImportance(save_dir)
features =  [['dom_x'], ['dom_charge'],['dom_z'],['dom_y'], ['dom_time']]
a.calc_all_seq_importances()


# %%
# print(a.feature_importances)

# def make_barh_plot(d, position=[0.125, 0.11, 0.775, 0.77]):

#     plt.style.use('default')
#     alpha = 0.3
#     h_figure = plt.figure()
#     h_axis = h_figure.add_axes(position)
#     h_axis.barh(d['y'], d['width'], xerr=d.get('errors', None))
#     h_axis.set_yticklabels(d['names'])
#     h_axis.set_yticks(d['y'])
#     h_axis.set_ylim((0, len(d['y'])))
#     h_figure.tight_layout()
#     return h_figure
a.make_plots()


