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
parser.add_argument(
        '-r', '--run', action='store_true', help='Runs experiment immediately.'
        )
parser.add_argument(
        '-t', 
        '--test', 
        action='store_true', 
        help='Initiates testmode - logging is turned off.'
        )
parser.add_argument(
        '-d', 
        '--dev', 
        action='store_true', 
        help='Initiates developermode - Logging is done at CubeML_test.'
        )
parser.add_argument('-e', '--explore_lr', action='store_true', 
        help='Performs a learning rate exploration.')
parser.add_argument('-s', '--scan_lr', action='store_true', 
        help='Performs a learning rate scan before training.')
parser.add_argument('--max_lr', default=5e-3, type=float, 
        help='Sets the max learning rate for the OneCycle learning rate schedule.')
parser.add_argument(
        '--min_lr', 
        default=1e-4, 
        type=float, 
        help='Sets the min learning rate for the OneCycle learning rate schedule.'
)
parser.add_argument('--lr', default=1e-3, type=float, 
        help='Sets the learning rate.')
parser.add_argument('--start_lr', default=1e-6, type=float, 
        help='Sets the start learning rate for the learning rate finder.')
parser.add_argument('--end_lr', default=0.1, type=float, 
        help='Sets the end learning rate for the learning rate finder.')
parser.add_argument('--lr_finder_epochs', default=1, type=int, 
        help='Sets the number of epochs the learning rate finder should run.')
parser.add_argument('--max_seq_len', default=np.inf, type=int, 
        help='Sets the maximum sequence length to use. If max_seq_len is' 
        'larger than the actualy sequence length, a random subsample of the'\
        'sequence is chosen')
parser.add_argument('--regression', default='None', type=str, 
        help='Sets the regression type to run. Options: full_reg,' \
        'direction_reg, vertex_reg, vertex_reg_no_time, energy_reg, ' \
        'nue_numu')
parser.add_argument('--loss', default='None', type=str, 
        help='Sets the loss function to use. Options: L2, logcosh')
parser.add_argument('--masks', nargs='+', default='None', type=str, 
        help='Sets the masks to choose data. Options:'\
        'dom_interval_SplitInIcePulses_min0_max200,'\
        'dom_interval_SRTInIcePulses_min0_max200, muon_neutrino,'\
        'energy_interval_min0.0_max3.0, nue_numu')
parser.add_argument('--weights', default='None', type=str, 
        help='Sets the weights to use. Options: geomean_energy_entry,'\
        'inverse_performance_muon_energy, None, nue_numu_balanced')
parser.add_argument('--dom_mask', default='SplitInIcePulses', type=str, 
        help='Sets the DOM mask to use. Options: SplitInIcePulses,'\
        'dom_interval_SRTInIcePulses')
parser.add_argument('--gpu', nargs='+', default='0', type=str, 
        help='Sets the IDs of the GPUs to use')
parser.add_argument('--batch_size', default=128, type=int, 
        help='Sets batchsize.')
parser.add_argument('--tags', nargs='+', default='', type=str, 
        help='Tags a run for easier comparisons on W&B')
parser.add_argument('--shared', action='store_true', 
        help='Saves execution command in shared folder.')
parser.add_argument('--max_epochs', default=17, type=int, 
        help='Sets the max amount of train epochs.')
parser.add_argument('--optimizer', default='Adam', type=str,
        help='Sets which optimizer to use. Options: Adam, SGD')
parser.add_argument('--n_workers', default=8, type=int,
        help='Sets number of workers to use during loading.')
parser.add_argument(
        '--nonlin', 
        default='LeakyReLU', 
        type=str,
        help='Sets which nonlinearity to use.'
)
parser.add_argument(
        '--ensemble', 
        action='store_true', 
        help='Initiates ensemble learning - predictions of other models are loaded.'
        )

args = parser.parse_args()

if __name__ == '__main__':

    #* ======================================================================== 
    #* DEFINE SCRIPT OBJECTIVE
    #* ========================================================================

    # data_dir = '/data/MuonGun_Level2_139008'
    data_dir = '/data/oscnext-genie-level5-v01-01-pass2'

    # Options: 'full_reg', 'direction_reg', 'vertex_reg', 
    # 'vertex_reg_no_time', 'energy_reg', 'angle_reg'
    regression_type = args.regression
    if regression_type == 'None':
        raise KeyError('A regression type must be chosen! Use flag --regression')

    # Options: 'train_new', 'continue_training', 'explore_lr', 
    # 'continue_crashed'
    objective = 'train_new'
    if args.explore_lr:
        objective = 'explore_lr'

    # Options: 'angle_loss', 'L1', 'L2', 'logcosh', 'angle_squared_loss',
    # 'cosine_loss'
    error_func = args.loss
    if error_func == 'None':
        raise KeyError('A loss function must be chosen! Use flag --loss')

    # Options: 'electron_neutrino', 'muon_neutrino', 'tau_neutrino'
    particle = 'muon_neutrino'

    # Options: 'all', 'dom_interval_min<VAL>_max<VAL>' 
    # (keywords: 'min_doms', 'max_doms')
    mask_names = args.masks
    if mask_names == 'None':
        raise KeyError('Masks must be chosen!')

    # Set project
    project = 'cubeml_test' if args.dev else 'cubeml'
    dataset = data_dir.split('/')[-1]

    # Set weights to use by loss-func
    loss_func_weights = [1, 1, 1]

    if args.optimizer == 'Adam':

        optimizer = {'optimizer':      'Adam',
                        'lr':             args.max_lr*0.1,
                        'betas':          (0.9, 0.998),
                        'eps':            1.0e-9
                        }
        lr_schedule = {'lr_scheduler':   'CustomOneCycleLR',
                     'max_lr':          args.max_lr,
                     'min_lr':          args.min_lr,
                     'frac_up':         0.035 if not args.dev else 0.5,
                     'frac_down':       1-0.035 if not args.dev else 0.5,
                     'schedule':        'inverse',
                     }

    elif args.optimizer == 'SGD':

        optimizer = {'optimizer':      'SGD',
                     'lr':             args.lr,#0.00003,#0.001, 
                     'momentum':       0.9,
                     'weight_decay':   0.0,
                     'nesterov':       True
                     }
        lr_schedule = {'lr_scheduler':   'ReduceLROnPlateau',
                       'factor':         0.1,
                       'patience':       15,
                       'cooldown':       1,
                       'min_lr':         5e-4,
                       'verbose':        True
                        }
    else:
        raise KeyError('Unknown optimizer requested!')

    hyper_pars = {'batch_size':        args.batch_size if not args.dev else 21,
                'max_epochs':          args.max_epochs if not args.dev else 2,
                'early_stop_patience': 25,
                'optimizer':           optimizer,
                'lr_schedule':         lr_schedule,
                'lr_finder':           {'start_lr':       args.start_lr,
                                        'end_lr':         args.end_lr,
                                        'n_epochs':       args.lr_finder_epochs
                                        }
                                        
                 }
    
#     optimizer = hyper_pars['optimizer']['optimizer']
    meta_pars = {'tags':                [regression_type, dataset, error_func, 
                                        particle, *mask_names, args.weights, 
                                        args.dom_mask, *args.tags, args.optimizer],
                'group':                regression_type,
                'project':              project,
                'objective':            objective,
                'log_every':            500000 if not args.dev else 50,
                'lr_scan':              args.scan_lr, 
                'gpu':                  args.gpu,
                'n_workers':            args.n_workers 
                }

    data_pars = {'data_dir':     data_dir,
                'masks':         mask_names,
                'particle':      particle,
                'weights':       args.weights,
                'dom_mask':      args.dom_mask,
                'max_seq_len':   args.max_seq_len,
                'seq_feat':    ['dom_charge', 
                                'dom_x', 
                                'dom_y', 
                                'dom_z', 
                                'dom_time', 
                                
                                # 'dom_charge_significance',
                                # 'dom_frac_of_n_doms',
                                # 'dom_d_to_prev',
                                # 'dom_v_from_prev',
                                # 'dom_d_minkowski_to_prev',
                                # 'dom_d_closest',
                                # 'dom_d_minkowski_closest',
                                
                                'dom_atwd',
                                'dom_pulse_width',
                                # 'dom_closest1_x',
                                # 'dom_closest1_y',
                                # 'dom_closest1_z',
                                # 'dom_closest1_time',
                                # 'dom_closest1_charge',
                                # 'dom_closest2_x',
                                # 'dom_closest2_y',
                                # 'dom_closest2_z',
                                # 'dom_closest2_time',
                                # 'dom_closest2_charge'
                                ],
                                # 'dom_d_vertex',
                                # 'dom_d_minkowski_vertex',
                                # 'dom_charge_over_vertex',
                                # 'dom_charge_over_vertex_sqr'], 
                                
                'scalar_feat': [#'tot_charge',
                                #'dom_timelength_fwhm',
                        ],
                                
                'n_val_events_wanted':   100000 if not args.dev else 100,
                'n_train_events_wanted': np.inf if not args.dev else 100,
                'n_predictions_wanted':  np.inf if not args.dev else 100,
                'dataset':               'oscnext',
                # 'train_frac':  0.80,# if not args.dev else 0.1,
                # 'val_frac':    0.10,# if not args.dev else 0.05,
                # 'test_frac':   0.0,
                # 'file_keys':             {'transform':   1},
                'dataloader':  'SqliteLoader',#'LstmLoader',#'LstmLoader',
                'collate_fn': 'PadSequence',
                'val_batch_size':      256 if not args.dev else 21
                }
    
    if (
            'dom_charge_significance' in data_pars['seq_feat'] 
            and 'electron_neutrino' in mask_names
    ):
        raise NameError('Keep in mind that some features are not transformed yet!')
    
    # Load ensemble (if wanted)
    ensemble_preds = (
            [] if not args.ensemble 
            else get_ensemble_predictions(data_pars, meta_pars)
            )
    data_pars['scalar_feat'].extend(ensemble_preds)

    n_seq_feat = len(data_pars['seq_feat'])
    n_scalar_feat = len(data_pars['scalar_feat'])
    n_target = get_n_targets(data_pars, meta_pars, args.loss)
    n1 = 128
    n2 = 2*n1+n_scalar_feat

    layers = [ 
        #{'Linear_embedder': {'input_sizes':        [n_seq_feat, 64],
        #                     'LayerNorm':          True},},
        # {'BiLSTM':          {'n_in':               n_seq_feat,
        #                      'n_hidden':           128,
        #                      'residual':           False,
        #                      'learn_init':         False}},
        # {'ResBlock':        {'input_sizes':       [n_seq_feat, n1//2, n1//2],
        #                      'norm':              'LayerNorm',
        #                      'type':              'seq'}},
        {'RnnBlock':        {'n_in':             n_seq_feat,
                            'n_out':             n1,
                            'rnn_type':          'GRU',
                            'n_parallel':        1,
                            'num_layers':        2,
                            'residual':          False,
                            'bidir':             True,
                            'dropout':           0.0,
                            'learn_init':        True}},
        # {'ResAttention':    {'input_outputs':     [n_seq_feat, n1, n1, n1],
        #                      'n_res_layers':      2,
        #                      'norm':             'LayerNorm'}},
        # # {'ResBlock':        {'input_sizes':       [2*n1, n1],
        # #                      'norm':              'LayerNorm',
        # #                      'type':              'seq'}},
        # {'LstmBlock':       {'n_in':              n1,
        #                      'n_out':             n1,
        #                      'n_parallel':        1,
        #                      'num_layers':        1,
        #                      'residual':          False,
        #                      'bidir':             True,
        #                      'learn_init':        True}},
        # {'AttentionBlock2':  {'input_sizes':        [n_seq_feat, n_seq_feat, n_seq_feat],
        #                       'LayerNorm':          True,
        #                       'Residual':           True},},
        #{'LSTM':            {'input_sizes':        [64, 512],
        #                    'dropout':             0.5,
        #                    'bidirectional':       False}},
        # {'ManyToOneAttention':{'n_in':             n_seq_feat}},
        # {'MaxPool': []},
        {'ResBlock':        {'input_sizes':        [n2, n2, n2, n2],
                             'norm':               'BatchNorm1D',
                             'type':               'x'}},
        {'Linear':          {'input_sizes':        [n2, n_target],
                            'norm_before_nonlin':  True}},
        # {'Linear':          {'input_sizes':        [n_scalar_feat, n1, n_target],
        #                     'norm_before_nonlin':  True}},
                            
        # {'Tanh':           []},
        ]
    if regression_type == 'angle_reg':
        layers.append({'Angle2Unitvector': []})
    elif regression_type == 'nue_numu':
        layers.append({'Tanh': {'scale': 10}})
    elif args.loss == 'logscore':
        layers.append({'SoftPlusSigma': []})

    arch_pars =         {'nonlin':             {'func':     args.nonlin},

                        'loss_func':           error_func,
                        'loss_func_weights':   loss_func_weights,
                        'norm':                {'norm':      'BatchNorm1D',
                                                'momentum':  0.9 },
                        'layers':             layers
                        }


    #* ======================================================================== 
    #* SAVE SETTINGS
    #* ========================================================================

    json_dict = {'hyper_pars': hyper_pars, 'data_pars': data_pars, 
                'arch_pars': arch_pars, 'meta_pars': meta_pars}
    exp_dir = get_project_root() + '/experiments/'

    # Finally! Make model directory 
    base_name = strftime("%Y-%m-%d-%H.%M.%S", localtime())
    exp_name = exp_dir+base_name+'.json'
    with open(exp_name, 'w') as name:
        json.dump(json_dict, name)
    
    if args.shared:
        command = 'python -u ../CubeML/src/script/run_exp_from_shared.py'
        command = 'python -u run_exp_from_shared.py'

        save_shared_exp_folder_command(command)
    
    if args.run:
        if args.test:
            run_experiment(exp_name, log=False)
        else:
            run_experiment(exp_name)

