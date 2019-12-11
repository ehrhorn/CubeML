import torch
from src.modules.helper_functions import *

# ======================================================================== 
# EVALUATION FUNCTIONS
# ======================================================================== 

def relative_logE_error(pred, truth):

    if 'true_muon_energy' in pred:
        logE_pred = torch.tensor(pred['true_muon_energy'])
        logE_truth = torch.tensor(truth['true_muon_energy'], dtype=logE_pred.dtype)

        rel_logE_error = (logE_pred-logE_truth)/logE_truth

        return rel_logE_error

def azi_error(pred, truth, units = 'degrees'):

    # ensure cartesian coordinates are used
    if 'true_muon_direction_x' in pred and 'true_muon_direction_y' in pred and 'true_muon_direction_z' in pred:
        x_key, y_key, z_key = 'true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z'
    elif 'true_neutrino_direction_x' in pred and 'true_neutrino_direction_y' in pred and 'true_neutrino_direction_z' in pred:
        x_key, y_key, z_key = 'true_neutrino_direction_x', 'true_neutrino_direction_y', 'true_neutrino_direction_z'
    else:
        raise KeyError('Unknown predictions given to directional_error.')
    
    # use atan2 to calculate angle 
    # - see https://pytorch.org/docs/stable/torch.html#torch.atan
    pi = 3.14159265359

    xy_pred = torch.tensor([pred[x_key], pred[y_key]])
    xy_truth = torch.tensor([truth[x_key], truth[y_key]], dtype=xy_pred.dtype)


    azi_pred_signed = torch.atan2(xy_pred[1, :], xy_pred[0, :])
    azi_truth_signed = torch.atan2(xy_truth[1, :], xy_truth[0, :])

    diff = azi_pred_signed-azi_truth_signed
    true_diff = torch.where(abs(diff)>pi, -2*torch.sign(diff)*pi+diff, diff)

    if units == 'radians':
        return true_diff 

    elif units == 'degrees':
        return true_diff*(180/3.14159)

def polar_error(pred, truth, units = 'degrees'):
    
    # ensure cartesian coordinates are used
    if 'true_muon_direction_x' in pred and 'true_muon_direction_y' in pred and 'true_muon_direction_z' in pred:
        x_key, y_key, z_key = 'true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z'
    elif 'true_neutrino_direction_x' in pred and 'true_neutrino_direction_y' in pred and 'true_neutrino_direction_z' in pred:
        x_key, y_key, z_key = 'true_neutrino_direction_x', 'true_neutrino_direction_y', 'true_neutrino_direction_z'
    else:
        raise KeyError('Unknown predictions given to directional_error.')
        
    x_pred, y_pred, z_pred = pred[x_key], pred[y_key], pred[z_key]
    x_true, y_true, z_true = truth[x_key], truth[y_key], truth[z_key]

    dir_pred = torch.tensor([x_pred, y_pred, z_pred])
    dir_truth = torch.tensor([x_true, y_true, z_true], dtype=dir_pred.dtype)

    length_preds = torch.sum(dir_pred*dir_pred, dim=0)**0.5
    length_truth = torch.sum(dir_truth*dir_truth, dim=0)**0.5

    polar_preds = torch.acos(dir_pred[2, :]/length_preds)
    polar_truth = torch.acos(dir_truth[2, :]/length_truth)
    
    if units == 'radians':
        diff = polar_preds-polar_truth
    elif units == 'degrees':
        diff = (180/3.14159)*(polar_preds-polar_truth)
    
    return diff

def directional_error(pred, truth, units = 'degrees'):
    
    # ensure cartesian coordinates are used
    if 'true_muon_direction_x' in pred and 'true_muon_direction_y' in pred and 'true_muon_direction_z' in pred:
        x_key, y_key, z_key = 'true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z'
    elif 'true_neutrino_direction_x' in pred and 'true_neutrino_direction_y' in pred and 'true_neutrino_direction_z' in pred:
        x_key, y_key, z_key = 'true_neutrino_direction_x', 'true_neutrino_direction_y', 'true_neutrino_direction_z'
    else:
        raise KeyError('Unknown predictions given to directional_error.')
        
    x_pred, y_pred, z_pred = pred[x_key], pred[y_key], pred[z_key]
    x_true, y_true, z_true = truth[x_key], truth[y_key], truth[z_key]

    dir_pred = torch.tensor([x_pred, y_pred, z_pred])
    dir_truth = torch.tensor([x_true, y_true, z_true], dtype=dir_pred.dtype)

    dot_prods = torch.sum(dir_pred * dir_truth, dim=0)
    length_preds = torch.sum(dir_pred*dir_pred, dim=0)**0.5
    length_truth = torch.sum(dir_truth*dir_truth, dim=0)**0.5

    
    if units == 'radians': 
        angles = torch.acos(dot_prods/(length_truth*length_preds))
    elif units == 'degrees': 
        angles = (180/3.14159)*torch.acos(dot_prods/(length_truth*length_preds))
    
    return angles

def get_eval_functions(meta_pars):
    regression_type = meta_pars['group']

    if regression_type == 'direction_reg':
        eval_funcs = [directional_error, azi_error, polar_error]
    if regression_type == 'full_reg':
        eval_funcs = [relative_logE_error]

    return eval_funcs

def directional_error_from_cartesian(pred, truth, units = 'degrees'):
    '''Calculates the directional difference in degrees or radians. Expects dictionaries with keys x, y, z
    '''

    dir_pred = torch.tensor([pred['x'], pred['y'], pred['z']])
    dir_truth = torch.tensor([truth['x'], truth['y'], truth['z']], dtype=dir_pred.dtype)

    dot_prods = torch.sum(dir_pred * dir_truth, dim=0)
    length_preds = torch.sum(dir_pred*dir_pred, dim=0)**0.5
    length_truth = torch.sum(dir_truth*dir_truth, dim=0)**0.5

    
    if units == 'radians': 
        angles = torch.acos(dot_prods/(length_truth*length_preds))
    elif units == 'degrees': 
        angles = (180/3.14159)*torch.acos(dot_prods/(length_truth*length_preds))
    
    return angles