import torch
from src.modules.helper_functions import *

#*======================================================================== 
#* HELPER FUNCTIONS
#*======================================================================== 

def sqr(x):
    return x*x

def get_eval_functions(meta_pars):
    """Retrieves the relevant evaluation functions for the specific regression type
    
    Arguments:
        meta_pars {dict} -- A dictionary containing the meta-keywords for an experiment
    
    Raises:
        ValueError: If an unknown regression type is encountered
    
    Returns:
        list -- A list with the relevant functions.
    """    

    regression_type = meta_pars['group']

    if regression_type == 'direction_reg' or regression_type == 'angle_reg':
        eval_funcs = [azi_error, polar_error, directional_error]
    elif regression_type == 'full_reg':
        eval_funcs = [relative_E_error, log_frac_E_error, vertex_x_error,
                    vertex_y_error, vertex_z_error, vertex_t_error, len_error, 
                    azi_error, polar_error, directional_error]
    elif regression_type == 'vertex_reg':
        eval_funcs = [vertex_x_error, vertex_y_error, vertex_z_error,
                      vertex_t_error, len_error]
    elif regression_type == 'vertex_reg_no_time':
        eval_funcs = [vertex_x_error, vertex_y_error, vertex_z_error, len_error]
    elif regression_type == 'energy_reg':
        eval_funcs = [relative_E_error, log_frac_E_error]
    elif regression_type in ['nue_numu', 'nue_numu_nutau']:
        eval_funcs = []
    else:
        raise ValueError('eval_funcs: Unknown regression type (%s) encountered!'%(regression_type))

    return eval_funcs    

#*======================================================================== 
#* EVAL FUNCTIONS
#*======================================================================== 

def relative_E_error(pred, truth):
    """Calculates the relative energy error on model predictions.
    
    Arguments:
        pred {array} -- log_10(E)-predictions
        truth {array} -- log_10(E)-truths   
    
    Returns:
        [torch.Tensor] -- relative energy error.
    """    

    if 'true_primary_energy' in pred:
        logE_pred = torch.tensor(pred['true_primary_energy'])
        logE_truth = torch.tensor(truth['true_primary_energy'], dtype=logE_pred.dtype)

        E_pred = 10**logE_pred
        E_truth = 10**logE_truth
        rel_E_error = (E_pred-E_truth)/E_truth

        return rel_E_error

def log_frac_E_error(pred, truth, eps=1e-3):
    r"""Calculates $\log_{10} \left( \frac{E_{pred}}{E_{true}} \right)$
    
    This error measure puts predictions that are either twice as big or small on an equal footing.

    Parameters
    ----------
    pred : array
        log_10(E/GeV)-predictions
    truth : array
        log_10(E/GeV)-truths
    eps : float, optional
        Small number to ensure no log(0) are taken, by default 1e-3
    
    Returns
    -------
    tensor
        log of fractional difference
    """    
    if 'true_primary_energy' in pred:
        logE_pred = torch.tensor(pred['true_primary_energy'])
        logE_truth = torch.tensor(truth['true_primary_energy'], dtype=logE_pred.dtype)
        E_pred = torch.clamp(10**logE_pred, min=eps, max=np.inf)
        E_truth = 10**logE_truth

        log_frac_E = torch.log10(E_pred/E_truth)

        return log_frac_E

def vertex_t_error(pred, truth, reporting=False):
    """Calculates the error on the t-coordinate prediction of the neutrino interaction vertex.
    
    Arguments:
        pred {dict} -- dictionary containing the key 'true_primary_time' or 't' and the predictions.
        truth {dict} -- dictionary containing the true values and the key 'true_primary_time'.   
    
    Raises:
        KeyError: If wrong dictionary given
    
    Returns:
        [torch.tensor] -- Signed error on prediction.
    """    

    # * Ensure we are dealing with the right data
    if 'true_primary_time' in pred:
        t_key = 'true_primary_time'
    elif 't' in pred:
        t_key = 't'
    else:
        raise KeyError('Wrong dictionary given to vertex_t_error!')
    
    t_pred = pred[t_key]
    t_truth = truth[t_key]
    if not reporting:
        t_pred = torch.tensor(t_pred)
        t_truth = torch.tensor(t_truth, dtype=t_pred.dtype)
    else:
        t_pred = np.array(t_pred)
        t_truth = np.array(t_truth)

    diff = t_pred - t_truth
    return diff

def vertex_x_error(pred, truth, reporting=False):
    """Calculates the error on the x-coordinate prediction of the neutrino interaction vertex.
    
    Arguments:
        pred {dict} -- dictionary containing the key 'true_primary_position_x' or 'x' and the predictions.
        truth {dict} -- dictionary containing the true values and the key 'true_primary_position_x'.   
    
    Raises:
        KeyError: If wrong dictionary given
    
    Returns:
        [torch.tensor] -- Signed error on prediction.
    """    

    # * Ensure we are dealing with the right data
    if 'true_primary_position_x' in pred:
        x_key = 'true_primary_position_x'
    elif 'x_vertex' in pred:
        x_key = 'x_vertex'
    else:
        raise KeyError('Wrong dictionary given to vertex_x_error!')
    
    x_pred = pred[x_key]
    x_truth = truth[x_key]
    
    if not reporting:
        x_pred = torch.tensor(x_pred)
        x_truth = torch.tensor(x_truth, dtype=x_pred.dtype)
    else:
        x_pred = np.array(x_pred)
        x_truth = np.array(x_truth)
    
    diff = x_pred - x_truth

    return diff

def vertex_y_error(pred, truth, reporting=False):
    """Calculates the error on the y-coordinate prediction of the neutrino interaction vertex.
    
    Arguments:
        pred {dict} -- dictionary containing the key 'true_primary_position_y' or 'y and the predictions.
        truth {dict} -- dictionary containing the true values and the key 'true_primary_position_y'.   
    
    Raises:
        KeyError: If wrong dictionary given
    
    Returns:
        [torch.tensor] -- Signed error on prediction.
    """    

    # * Ensure we are dealing with the right data
    if 'true_primary_position_y' in pred:
        y_key = 'true_primary_position_y'
    elif 'y_vertex' in pred:
        y_key = 'y_vertex'
    else:
        raise KeyError('Wrong dictionary given to vertex_y_error!')
    
    y_pred = pred[y_key]
    y_truth = truth[y_key]

    if not reporting:
        y_pred = torch.tensor(y_pred)
        y_truth = torch.tensor(y_truth, dtype=y_pred.dtype)
    else:
        y_pred = np.array(y_pred)
        y_truth = np.array(y_truth)

    diff = y_pred - y_truth
    return diff

def vertex_z_error(pred, truth, reporting=False):
    """Calculates the error on the z-coordinate prediction of the neutrino interaction vertex.
    
    Arguments:
        pred {dict} -- dictionary containing the key 'true_primary_position_z' and the predictions.
        truth {dict} -- dictionary containing the true values and the key 'true_primary_position_z'.   
    
    Raises:
        KeyError: If wrong dictionary given
    
    Returns:
        [torch.tensor] -- Signed error on prediction.
    """    

    # * Ensure we are dealing with the right data
    if 'true_primary_position_z' in pred:
        z_key = 'true_primary_position_z'
    elif 'z_vertex' in pred:
        z_key = 'z_vertex'
    else:
        raise KeyError('Wrong dictionary given to vertex_z_error!')
    
    z_pred = pred[z_key]
    z_truth = truth[z_key]

    if not reporting:
        z_pred = torch.tensor(z_pred)
        z_truth = torch.tensor(z_truth, dtype=z_pred.dtype)
    else:
        z_pred = np.array(z_pred)
        z_truth = np.array(z_truth)

    diff = z_pred - z_truth
    return diff

def azi_error(pred, truth, units='degrees'):

    # ensure cartesian coordinates are used
    if 'true_muon_direction_x' in pred and 'true_muon_direction_y' in pred and 'true_muon_direction_z' in pred:
        x_key, y_key, z_key = 'true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z'
    elif 'true_neutrino_direction_x' in pred and 'true_neutrino_direction_y' in pred and 'true_neutrino_direction_z' in pred:
        x_key, y_key, z_key = 'true_neutrino_direction_x', 'true_neutrino_direction_y', 'true_neutrino_direction_z'
    elif 'true_primary_direction_x' in pred and 'true_primary_direction_y' in pred and 'true_primary_direction_z' in pred:
        x_key, y_key, z_key = 'true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z'
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

def polar_error(pred, truth, units='degrees'):
    
    # ensure cartesian coordinates are used
    if 'true_muon_direction_x' in pred and 'true_muon_direction_y' in pred and 'true_muon_direction_z' in pred:
        x_key, y_key, z_key = 'true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z'
    elif 'true_neutrino_direction_x' in pred and 'true_neutrino_direction_y' in pred and 'true_neutrino_direction_z' in pred:
        x_key, y_key, z_key = 'true_neutrino_direction_x', 'true_neutrino_direction_y', 'true_neutrino_direction_z'
    elif 'true_primary_direction_x' in pred and 'true_primary_direction_y' in pred and 'true_primary_direction_z' in pred:
        x_key, y_key, z_key = 'true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z'
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

def directional_error(pred, truth, units='degrees'):
    
    # * ensure cartesian coordinates are used
    if 'true_muon_direction_x' in pred and 'true_muon_direction_y' in pred and 'true_muon_direction_z' in pred:
        x_key, y_key, z_key = 'true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z'
    elif 'true_neutrino_direction_x' in pred and 'true_neutrino_direction_y' in pred and 'true_neutrino_direction_z' in pred:
        x_key, y_key, z_key = 'true_neutrino_direction_x', 'true_neutrino_direction_y', 'true_neutrino_direction_z'
    elif 'true_primary_direction_x' in pred and 'true_primary_direction_y' in pred and 'true_primary_direction_z' in pred:
        x_key, y_key, z_key = 'true_primary_direction_x', 'true_primary_direction_y', 'true_primary_direction_z'
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

def len_error(pred, truth, reporting=False):
    
    x_key = 'true_primary_position_x'
    y_key = 'true_primary_position_y'
    z_key = 'true_primary_position_z'

    xerr = torch.tensor(pred[x_key])-torch.tensor(truth[x_key])
    yerr = torch.tensor(pred[y_key])-torch.tensor(truth[y_key])
    zerr = torch.tensor(pred[z_key])-torch.tensor(truth[z_key])

    direrr = torch.sqrt(sqr(xerr) + sqr(yerr) + sqr(zerr))

    return direrr

def retro_azi_error(retro_dict, true_dict, units='degrees', reporting=False):
    """Calculates the difference in zenith angle between retro_crs_prefit and true values.
    
    Arguments:
        retro_dict {dictionary} -- predictions from retro_crs_prefit with keys as in h5-files. Expects key 'azi'
        true_dict {dictionary} -- true values as unit vectors with keys 'x', 'y'
    
    Keyword Arguments:
        units {str} -- 'degrees' or 'radians' (default: {'degrees'})
    
    Returns:
        torch.tensor -- difference in polar angle between retro and truth
    """     

    # * use atan2 to calculate angle 
    # * - see https://pytorch.org/docs/stable/torch.html#torch.atan
    pi = 3.14159265359

    xy_truth = torch.tensor([true_dict['true_primary_direction_x'], true_dict['true_primary_direction_y']])
    azi_truth_signed = torch.atan2(xy_truth[1, :], xy_truth[0, :])

    # * Convert retro_crs to signed angle
    pred_signed = [entry if entry < pi else entry - 2*pi for entry in retro_dict['retro_crs_prefit_azimuth']]

    #? add 180 degrees - retro_crs appears to predict direction neutrino came from and not neutrino direction..
    pred_signed = torch.tensor([entry-pi if entry > 0 else entry + pi for entry in pred_signed], dtype=azi_truth_signed.dtype)
    diff = pred_signed-azi_truth_signed
    true_diff = torch.where(abs(diff)>pi, -2*torch.sign(diff)*pi+diff, diff)

    if units == 'radians':
        return true_diff 

    elif units == 'degrees':
        return true_diff*(180/pi)

def retro_polar_error(retro_dict, true_dict, units='degrees', reporting=False):
    """Calculates the difference in polar angle between retro_crs_prefit and true values.
    
    Arguments:
        retro_dict {dictionary} -- predictions from retro_crs_prefit with keys as in h5-files. Expects key 'zen'
        true_dict {dictionary} -- true values as unit vectors with keys 'x', 'y', 'z'
    
    Keyword Arguments:
        units {str} -- 'degrees' or 'radians' (default: {'degrees'})
    
    Returns:
        torch.tensor -- difference in polar angle between retro and truth
    """     
    
    pi = 3.14159265359

    x_true, y_true, z_true = true_dict['true_primary_direction_x'], true_dict['true_primary_direction_y'], true_dict['true_primary_direction_z']
    dir_truth = torch.tensor([x_true, y_true, z_true])
    length_truth = torch.sum(dir_truth*dir_truth, dim=0)**0.5
    polar_truth = torch.acos(dir_truth[2, :]/length_truth)

    #? retro_crs seems to predit the direction the neutrino came from and not the neutrinos direction - therefore do a parity.
    polar_preds = pi-torch.tensor(retro_dict['retro_crs_prefit_zenith'], dtype=polar_truth.dtype)
    if units == 'radians':
        diff = polar_preds-polar_truth
    elif units == 'degrees':
        diff = (180/pi)*(polar_preds-polar_truth)
    
    return diff

def retro_relE_error(retro_dict, true_dict, reporting=False):
    """Calculates the relative error (E_pred-E_true)/E_true in energy from Icecube's predictions.
    
    Arguments:
        retro_dict {dict} -- Dictionary with key 'retro_crs_prefit_energy' containing array of energy predictions
        true_dict {dict} -- Dictionary with key 'true_primary_energy' containing array of true log_10 E.
    
    Returns:
        array -- Relative energy error
    """    
    
    E_pred = retro_dict['retro_crs_prefit_energy']
    logE_true = np.array(convert_to_proper_list(true_dict['true_primary_energy']))
    E_true = 10**logE_true
    relE_error = (E_pred-E_true)/E_true

    return relE_error

def retro_log_frac_E_error(retro_dict, true_dict, reporting=False, eps=1e-3):
    """Calculates $\log_{10} \left( \frac{E_{pred}}{E_{true}} \right)$
    
    This error measure puts predictions that are either twice as big or small on an equal footing.
    
    Parameters
    ----------
    retro_dict : dict
        Dictionary containing retro_crs predictions under the key E
    true_dict : dict
        Dictionary containing TRUE logE/GeV
    reporting : bool, optional
        A parameter used by other reporting functions, jsut has to be here for now. Has to be changed, by default False
    eps : float, optional
        small number to ensure no logarithms of 0, by default 1e-3
    
    Returns
    -------
    array
        log of fraction of energies
    """    
    E_pred = np.clip(retro_dict['retro_crs_prefit_energy'], eps, np.inf)
    logE_true = np.array(convert_to_proper_list(true_dict['true_primary_energy']))
    
    log_frac = np.log10(E_pred)-logE_true
    
    return log_frac

def retro_t_error(pred, truth, reporting=False):
    """Calculates the error on the t-coordinate prediction of the neutrino interaction vertex.
    
    Arguments:
        pred {dict} -- dictionary containing the key 'true_primary_time' or 't' and the predictions.
        truth {dict} -- dictionary containing the true values and the key 'true_primary_time'.   
    
    Raises:
        KeyError: If wrong dictionary given
    
    Returns:
        [torch.tensor] -- Signed error on prediction.
    """    

    # * Ensure we are dealing with the right data
    
    t_pred = pred['retro_crs_prefit_time']
    t_truth = truth['true_primary_time']
    if not reporting:
        t_pred = torch.tensor(t_pred)
        t_truth = torch.tensor(t_truth, dtype=t_pred.dtype)
    else:
        t_pred = np.array(t_pred)
        t_truth = np.array(t_truth)

    diff = t_pred - t_truth
    return diff

def retro_x_error(pred, truth, reporting=False):
    """Calculates the error on the x-coordinate prediction of the neutrino interaction vertex.
    
    Arguments:
        pred {dict} -- dictionary containing the key 'true_primary_position_x' or 'x' and the predictions.
        truth {dict} -- dictionary containing the true values and the key 'true_primary_position_x'.   
    
    Raises:
        KeyError: If wrong dictionary given
    
    Returns:
        [torch.tensor] -- Signed error on prediction.
    """    

    x_pred = pred['retro_crs_prefit_x']
    x_truth = truth['true_primary_position_x']
    
    if not reporting:
        x_pred = torch.tensor(x_pred)
        x_truth = torch.tensor(x_truth, dtype=x_pred.dtype)
    else:
        x_pred = np.array(x_pred)
        x_truth = np.array(x_truth)
    
    diff = x_pred - x_truth

    return diff

def retro_y_error(pred, truth, reporting=False):
    """Calculates the error on the y-coordinate prediction of the neutrino interaction vertex.
    
    Arguments:
        pred {dict} -- dictionary containing the key 'true_primary_position_y' or 'y and the predictions.
        truth {dict} -- dictionary containing the true values and the key 'true_primary_position_y'.   
    
    Raises:
        KeyError: If wrong dictionary given
    
    Returns:
        [torch.tensor] -- Signed error on prediction.
    """    

    y_pred = pred['retro_crs_prefit_y']
    y_truth = truth['true_primary_position_y']

    if not reporting:
        y_pred = torch.tensor(y_pred)
        y_truth = torch.tensor(y_truth, dtype=y_pred.dtype)
    else:
        y_pred = np.array(y_pred)
        y_truth = np.array(y_truth)

    diff = y_pred - y_truth
    return diff

def retro_z_error(pred, truth, reporting=False):
    """Calculates the error on the z-coordinate prediction of the neutrino interaction vertex.
    
    Arguments:
        pred {dict} -- dictionary containing the key 'true_primary_position_z' and the predictions.
        truth {dict} -- dictionary containing the true values and the key 'true_primary_position_z'.   
    
    Raises:
        KeyError: If wrong dictionary given
    
    Returns:
        [torch.tensor] -- Signed error on prediction.
    """    

    z_pred = pred['retro_crs_prefit_z']
    z_truth = truth['true_primary_position_z']

    if not reporting:
        z_pred = torch.tensor(z_pred)
        z_truth = torch.tensor(z_truth, dtype=z_pred.dtype)
    else:
        z_pred = np.array(z_pred)
        z_truth = np.array(z_truth)

    diff = z_pred - z_truth
    
    return diff

def retro_len_error(pred, truth, reporting=False):

    retro_x = 'retro_crs_prefit_x'
    retro_y = 'retro_crs_prefit_y'
    retro_z = 'retro_crs_prefit_z'
    
    true_x = 'true_primary_position_x'
    true_y = 'true_primary_position_y'
    true_z = 'true_primary_position_z'

    xerr = torch.tensor(pred[retro_x])-torch.tensor(truth[true_x])
    yerr = torch.tensor(pred[retro_y])-torch.tensor(truth[true_y])
    zerr = torch.tensor(pred[retro_z])-torch.tensor(truth[true_z])

    direrr = torch.sqrt(sqr(xerr)+sqr(yerr)+sqr(zerr))

    return direrr


def retro_directional_error(pred, truth, units='degrees', reporting=False):

    # * Define keys - if these keys are not in pred and truth, 
    # * something is wrong
    retro_zen = 'retro_crs_prefit_zenith'
    retro_azi = 'retro_crs_prefit_azimuth'

    true_x = np.array(truth['true_primary_direction_x'])
    true_y = np.array(truth['true_primary_direction_y'])
    true_z = np.array(truth['true_primary_direction_z'])
    
    # * Convert retros azi + zenith predictions to x, y, z unitvector 
    # * coordinates. Expects azi and zenith in radians. A standard tranform 
    # * from spherical to cartesian.
    retro_x = np.sin(pred[retro_zen])*np.cos(pred[retro_azi])
    retro_y = np.sin(pred[retro_zen])*np.sin(pred[retro_azi])
    retro_z = np.cos(pred[retro_zen])

    # * Now calculate angle between unit vectors.
    len_retro = np.sqrt(sqr(retro_x)+sqr(retro_y)+sqr(retro_z))
    len_true = np.sqrt(sqr(true_x)+sqr(true_y)+sqr(true_z))
    dot_prods = retro_x*true_x + retro_y*true_y +retro_z*true_z
    cos = dot_prods/(len_retro*len_true)
    
    if units == 'radians': 
        angles = np.pi-np.arccos(cos)
    elif units == 'degrees': 
        angles = 180-(180/3.14159)*np.arccos(cos)
    
    return angles
#*======================================================================== 
#* DEPRECATED
#*======================================================================== 

# def directional_error_from_cartesian(pred, truth, units='degrees'):
#     '''Calculates the directional difference in degrees or radians. Expects dictionaries with keys x, y, z
#     '''

#     dir_pred = torch.tensor([pred['x'], pred['y'], pred['z']])
#     dir_truth = torch.tensor([truth['x'], truth['y'], truth['z']], dtype=dir_pred.dtype)

#     dot_prods = torch.sum(dir_pred * dir_truth, dim=0)
#     length_preds = torch.sum(dir_pred*dir_pred, dim=0)**0.5
#     length_truth = torch.sum(dir_truth*dir_truth, dim=0)**0.5

    
#     if units == 'radians': 
#         angles = torch.acos(dot_prods/(length_truth*length_preds))
#     elif units == 'degrees': 
#         angles = (180/3.14159)*torch.acos(dot_prods/(length_truth*length_preds))
    
#     return angles

# def get_retro_crs_prefit_polar_error(retro_dict, true_dict, units='degrees'):
#     """Calculates the difference in polar angle between retro_crs_prefit and true values.
    
#     Arguments:
#         retro_dict {dictionary} -- predictions from retro_crs_prefit with keys as in h5-files. Expects key 'zen'
#         true_dict {dictionary} -- true values as unit vectors with keys 'x', 'y', 'z'
    
#     Keyword Arguments:
#         units {str} -- 'degrees' or 'radians' (default: {'degrees'})
    
#     Returns:
#         torch.tensor -- difference in polar angle between retro and truth
#     """     
    
#     pi = 3.14159265359

#     x_pred, y_pred, z_pred = retro_dict['x'], retro_dict['y'], retro_dict['z']
#     x_true, y_true, z_true = true_dict['x'], true_dict['y'], true_dict['z']
#     dir_truth = torch.tensor([x_true, y_true, z_true])
#     length_truth = torch.sum(dir_truth*dir_truth, dim=0)**0.5
#     polar_truth = torch.acos(dir_truth[2, :]/length_truth)

#     #? retro_crs seems to predit the direction the neutrino came from and not the neutrinos direction - therefore do a parity.
#     polar_preds = pi-torch.tensor(retro_dict['zen'], dtype=polar_truth.dtype)
#     if units == 'radians':
#         diff = polar_preds-polar_truth
#     elif units == 'degrees':
#         diff = (180/pi)*(polar_preds-polar_truth)
    
#     return diff