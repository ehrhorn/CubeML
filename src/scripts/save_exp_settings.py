import json
from pathlib import Path
import argparse

from src.modules.classes import *
from src.modules.loss_funcs import *
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
from src.modules.main_funcs import *

description = 'Saves settings for an experiment to be run.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-r', '--run', action='store_true', help='Runs experiment immediately.')
parser.add_argument('-t', '--test', action='store_true', help='Initiates testmode - logging is turned off.')
parser.add_argument('-d', '--dev', action='store_true', help='Initiates developermode - Logging is done at CubeML_test.')
parser.add_argument('-e', '--explore_lr', action='store_true', help='Performs a learning rate exploration.')
parser.add_argument('-s', '--scan_lr', action='store_true', help='Performs a learning rate scan before training.')
parser.add_argument('--start_lr', default=1e-6, type=float, help='Sets the start learning rate for the learning rate finder.')
parser.add_argument('--end_lr', default=0.1, type=float, help='Sets the end learning rate for the learning rate finder.')
parser.add_argument('--lr_finder_epochs', default=1, type=int, help='Sets the number of epochs the learning rate finder should run.')
parser.add_argument('--regression', default='None', type=str, help='Sets the regression type to run. Options: "full_reg", "direction_reg", "vertex_reg", "vertex_reg_no_time", "energy_reg"')
parser.add_argument('--loss', default='None', type=str, help='Sets the loss function to use. Options: "angle_loss", "L1", "L2", "Huber", "angle_squared_loss"')
parser.add_argument('--masks', nargs='+', default='None', type=str, help='Sets the masks to choose data. Options: "dom_interval_min0_max200", "muon_neutrino", "energy_interval_min0.0_max3.0"')

args = parser.parse_args()

if __name__ == '__main__':

    #* ======================================================================== 
    #* DEFINE SCRIPT OBJECTIVE
    #* ========================================================================

    # * data_dir = '/data/MuonGun_Level2_139008'
    data_dir = '/data/oscnext-genie-level5-v01-01-pass2'
    pretrained_path = '/models/oscnext-genie-level5-v01-01-pass2/regression/direction_reg/2020-01-13-13.58.14' 

    # * Options: 'full_reg', 'direction_reg', 'vertex_reg', 'vertex_reg_no_time', 'energy_reg'
    regression_type = args.regression
    if regression_type == 'None':
        raise KeyError('A regression type must be chosen! Use flag --regression')

    # * Options: 'train_new', 'continue_training', 'explore_lr'
    objective = 'train_new'
    if args.explore_lr:
        objective = 'explore_lr'

    # * Options: 'angle_loss', 'L1', 'L2', 'Huber', 'angle_squared_loss'
    error_func = args.loss
    if error_func == 'None':
        raise KeyError('A loss function must be chosen! Use flag --loss')

    # * Options: 'electron_neutrino', 'muon_neutrino', 'tau_neutrino'
    particle = 'muon_neutrino'

    # * Options: 'all', 'dom_interval_min<VAL>_max<VAL>' (keywords: 'min_doms', 'max_doms')
    mask_names = args.masks
    if mask_names == 'None':
        raise KeyError('Masks must be chosen!')

    # * Set project
    project = 'cubeml_test' if args.dev else 'cubeml'

    dataset = data_dir.split('/')[-1]
    meta_pars = {'tags':                [regression_type, dataset, error_func, particle, *mask_names],
                'group':                regression_type,
                'project':              project,
                'objective':            objective,
                'pretrained_path':      pretrained_path,
                'log_every':            500000 if not args.dev else 50,
                'lr_scan':              args.scan_lr 
                }

    hyper_pars = {'batch_size':        128 if not args.dev else 21,
                'max_epochs':          10 if not args.dev else 2,
                'early_stop_patience': 20,
                'optimizer':           {'optimizer':      'Adam',
                                        'lr':             1e-6,#0.00003,#0.001, 
                                        'betas':          (0.9, 0.998),
                                        'eps':            1.0e-9
                                        },
                'lr_schedule':          {'lr_scheduler':   'CustomOneCycleLR' if not args.dev else None,
                                        'max_lr':          5e-3,
                                        'min_lr':          1e-4,
                                        'frac_up':         0.02,
                                        'frac_down':       1-0.02,
                                        'schedule':        'inverse',
                                        },
                'lr_finder':            {'start_lr':       args.start_lr,
                                        'end_lr':          args.end_lr,
                                        'n_epochs':        args.lr_finder_epochs
                                        }
                                        
                 }


    data_pars = {'data_dir':     data_dir,
                'masks':          mask_names,
                'particle':      particle,
                'seq_feat':    ['dom_charge', 
                                'dom_x', 
                                'dom_y', 
                                'dom_z', 
                                'dom_time'], 
                                # 'dom_charge_significance',
                                # 'dom_frac_of_n_doms',
                                # 'dom_d_to_prev',
                                # 'dom_v_from_prev',
                                # 'dom_d_minkowski_to_prev',
                                # 'dom_d_closest',
                                # 'dom_d_minkowski_closest',
                                # 'dom_d_vertex',
                                # 'dom_d_minkowski_vertex',
                                # 'dom_charge_over_vertex',
                                # 'dom_charge_over_vertex_sqr'], 
                                
                'scalar_feat': ['dom_timelength_fwhm'],
                                # 'tot_charge'],
                                
                'n_val_events_wanted':   100000 if not args.dev else 100,# np.inf,
                'n_train_events_wanted': np.inf if not args.dev else 100,
                'n_predictions_wanted': np.inf if not args.dev else 100,
                'train_frac':  0.80,# if not args.dev else 0.1,
                'val_frac':    0.10,# if not args.dev else 0.05,
                'test_frac':   0.0,
                'file_keys':             {'transform':   1},
                'dataloader':  'PickleLoader',#'LstmLoader',#'LstmLoader',
                'collate_fn': 'PadSequence',
                'val_batch_size':      256 if not args.dev else 21
                }


    n_seq_feat = len(data_pars['seq_feat'])
    n_scalar_feat = len(data_pars['scalar_feat'])
    n_target = len(get_target_keys(data_pars, meta_pars))

    arch_pars =         {'non_lin':             {'func':     'LeakyReLU'},

                        'loss_func':           error_func,#'L2_like_loss','dir_reg_L1_like_loss',

                        'norm':                {'norm':      'BatchNorm1D', #'BatchNorm1D', 'None'
                                                'momentum':  0.9 },

                        'layers':             [ #{'Linear_embedder': {'input_sizes':        [n_seq_feat, 512],
                                                #                     'LayerNorm':          True},},
                                                {'LstmBlock':        {'n_in':               n_seq_feat,
                                                                     'n_out':               256,
                                                                     'n_parallel':          1,
                                                                     'n_stacks':            2,
                                                                     'residual':            False}},
                                                #{'LSTM':            {'input_sizes':        [64, 512],
                                                #                    'dropout':             0.5,
                                                #                    'bidirectional':       False}},
                                                {'Linear':          {'input_sizes':        [256+n_scalar_feat, n_target],
                                                                    'norm_before_nonlin':  True}}]
                        }
                                                

    #* ======================================================================== 
    #* SAVE SETTINGS
    #* ========================================================================

    json_dict = {'hyper_pars': hyper_pars, 'data_pars': data_pars, 'arch_pars': arch_pars, 'meta_pars': meta_pars}
    exp_dir = get_project_root() + '/experiments/'

    # * Finally! Make model directory 
    base_name = strftime("%Y-%m-%d-%H.%M.%S", localtime())
    exp_name = exp_dir+base_name+'.json'
    with open(exp_name, 'w') as name:
        json.dump(json_dict, name)
    
    if args.run:
        if args.test:
            run_experiment(exp_name, log=False)
        else:
            run_experiment(exp_name)

