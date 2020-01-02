import json
from pathlib import Path
import argparse

# from src import modules
from src.modules.classes import *
from src.modules.loss_funcs import *
from src.modules.helper_functions import *
from src.modules.eval_funcs import *
from src.modules.main_funcs import *

description = 'Saves settings for an experiment to be run.'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-r', '--run', action='store_true', help='Runs experiment immediately.')
parser.add_argument('-t', '--test', action='store_true', help='Initiates testmode - logging is turned off.')
parser.add_argument('-e', '--explore_lr', action='store_true', help='Performs a learning rate exploration.')
parser.add_argument('-s', '--scan_lr', action='store_true', help='Performs a learning rate scan before training.')
parser.add_argument('--start_lr', default=1e-6, type=float, help='Sets the start learning rate for the learning rate finder.')
parser.add_argument('--end_lr', default=0.1, type=float, help='Sets the end learning rate for the learning rate finder.')
parser.add_argument('--lr_finder_epochs', default=1, type=int, help='Sets the number of epochs the learning rate finder should run.')

args = parser.parse_args()

if __name__ == '__main__':

    #* ======================================================================== 
    #* DEFINE SCRIPT OBJECTIVE
    #* ========================================================================

    # * data_dir = '/data/MuonGun_Level2_139008'
    data_dir = '/data/oscnext-genie-level5-v01-01-pass2'
    pretrained_path = '/groups/hep/bjoernhm/thesis/CubeML/models/MuonGun_Level2_139008/regression/direction_reg/2019-11-25-04.11.55' 

    # * Options: 'full_reg', 'direction_reg', 'vertex_reg'
    regression_type = 'vertex_reg'

    # * Options: 'train_new', 'continue_training', 'explore_lr'
    objective = 'train_new'
    if args.explore_lr:
        objective = 'explore_lr'

    # * Options: 'angle_loss', 'L1', 'L2', 'Huber'
    error_func = 'L2'

    # * Options: 'electron_neutrino', 'muon_neutrino', 'tau_neutrino'
    particle = 'muon_neutrino'

    dataset = data_dir.split('/')[-1]
    meta_pars = {'tags':                [regression_type, dataset, error_func, particle],
                'group':                regression_type,
                'project':              'cubeml_test',
                'objective':            objective,
                'pretrained_path':      pretrained_path,
                'log_every':            1000,
                'lr_scan':              args.scan_lr 
                }

    hyper_pars = {'batch_size':        128,
                'max_epochs':          1,
                'early_stop_patience': 20,
                'optimizer':           {'optimizer':      'Adam',
                                        'lr':             0.001,#0.00003,#0.001, 
                                        'betas':          (0.9, 0.998),
                                        'eps':            1.0e-9
                                        },
                'lr_schedule':          {'lr_scheduler':   'OneCycleLR',
                                        'max_lr':          0.1,
                                        'min_lr':          1e-6,
                                        'pct_start':       0.3,
                                        },
                'lr_finder':            {'start_lr':       args.start_lr,
                                        'end_lr':          args.end_lr,
                                        'n_epochs':        args.lr_finder_epochs
                                        }
                                        
                 }


    data_pars = {'data_dir':     data_dir,
                'particle':      particle,
                'seq_feat':    ['dom_charge', 'dom_x', 'dom_y', 'dom_z', 'dom_time'], 
                'scalar_feat': ['toi_point_on_line_x', 'toi_point_on_line_y', 'toi_point_on_line_z', 'toi_direction_x', 'toi_direction_y', 'toi_direction_z', 'toi_evalratio', 'dom_timelength_fwhm'],
                'n_val_events_wanted':   1000,# np.inf,
                'n_train_events_wanted': 10000,# np.inf,
                'n_predictions_wanted': 10000,
                'train_frac':  0.80,
                'val_frac':    0.10,
                'test_frac':   0.0,
                'file_keys':             {'transform':   1},
                'dataloader':  'FullBatchLoader',#'LstmLoader',#'LstmLoader',
                'collate_fn': 'PadSequence',
                'val_batch_size':      128
                }


    n_seq_feat = len(data_pars['seq_feat'])
    n_scalar_feat = len(data_pars['scalar_feat'])
    n_target = len(get_target_keys(data_pars, meta_pars))

    arch_pars =         {'non_lin':             {'func':     'LeakyReLU'},

                        'loss_func':           error_func,#'L2_like_loss','dir_reg_L1_like_loss',

                        'norm':                {'norm':      None, #'BatchNorm1D', 'None'
                                                'momentum':  0.9 },

                        'layers':              [{'Linear_embedder': {'input_sizes':        [n_seq_feat, 64],
                                                                     'LayerNorm':          True},},
                                                {'SelfAttention':   {'input_sizes':        [64, 64],
                                                                     'LayerNorm':          True,
                                                                     'Residual':           True,}},
                                                {'LSTM':            {'input_sizes':        [64, 128],
                                                                    'dropout':             0.5,
                                                                    'bidirectional':       True}},
                                                {'Linear':          {'input_sizes':        [128+n_scalar_feat, n_target],
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

