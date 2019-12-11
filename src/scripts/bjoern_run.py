#%%
from src.modules.main_funcs import *
from src.modules.loss_funcs import *

# ======================================================================== 
# DEFINE SCRIPT OBJECTIVE
# ========================================================================
data_dir = '/data/MuonGun_Level2_139008'
# data_dir = /data/oscnext-genie-level5-v01-01-pass2
pretrained_path = '/groups/hep/bjoernhm/thesis/CubeML/models/MuonGun_Level2_139008/regression/direction_reg/2019-11-25-04.11.55' 

# Options: 'full_reg', 'direction_reg'
regression_type = 'direction_reg'
# Options: 'train_new', 'continue_training'
objective = 'train_new'
error_func = 'angle_loss'
dataset = data_dir.split('/')[-1]
meta_pars = {'tags':                [regression_type, dataset, error_func],
             'group':                regression_type,
             'project':             'cubeml_test',
             'objective':            objective,
             'pretrained_path':      pretrained_path 
             }

hyper_pars = {'batch_size':        128,
            'max_epochs':          200,
            'early_stop_patience': 20,
            'optimizer':           {'optimizer':      'Adam',
                                    'lr':             0.005,#0.00003,#0.001, 
                                    'betas':          (0.9, 0.998),
                                    'eps':            1.0e-9},
            'lr_schedule':          {'lr_scheduler':   'ReduceLROnPlateau',
                                    'factor':         5,
                                    'patience':       2,
                                    }
        }


data_pars = {'data_dir':     data_dir,
            'seq_feat':    ['charge', 'dom_x', 'dom_y', 'dom_z', 'time'], 
            'scalar_feat': ['toi_point_on_line_x', 'toi_point_on_line_y', 'toi_point_on_line_z', 'toi_direction_x', 'toi_direction_y', 'toi_direction_z', 'toi_evalratio'],
            'target':      ['true_muon_direction_x', 'true_muon_direction_y', 'true_muon_direction_z'],#, 'true_muon_entry_position_x', 'true_muon_entry_position_y', 'true_muon_entry_position_z', 'true_muon_energy'],
            'train_frac':  0.1000,
            'val_frac':    0.100,
            'test_frac':   0.0,
            'file_keys':             {'transform':   0},
            'dataloader':  'FullBatchLoader',#'LstmLoader',#'LstmLoader',
            'collate_fn': 'PadSequence',
            'val_batch_size':      256
            }


n_seq_feat = len(data_pars['seq_feat'])
n_scalar_feat = len(data_pars['scalar_feat'])
n_target = len(data_pars['target'])

arch_pars =         {'non_lin':             {'func':     'LeakyReLU'},

                    'loss_func':           error_func,#'L2_like_loss','dir_reg_L1_like_loss',

                    'norm':                {'norm':      None, #'BatchNorm1D',
                                            'momentum':  0.9 },

                    'layers':              [{'Linear_embedder': {'input_sizes':        [n_seq_feat, 32]}},
                                            {'LSTM':            {'input_sizes':        [32, 128],
                                                                'dropout':             0.5,
                                                                'bidirectional':       True}},
                                            {'Linear':          {'input_sizes':        [128+n_scalar_feat, n_target],
                                                                'norm_before_nonlin':  True}}]
                    }
                                    
# model_dir = train(hyper_pars, data_pars, arch_pars, meta_pars, save=True, scan_lr_before_train = True)
# evaluate_model(model_dir)
model_dir, wandb_ID = train_model(hyper_pars, data_pars, arch_pars, meta_pars)
# model_dir = '/home/bjoern/Thesis/CubeML/models/MuonGun_Level2_139008/regression/direction_reg/test_2019.12.02-12.26.17'
evaluate_model(model_dir, wandb_ID=wandb_ID)
# explore_lr(hyper_pars, data_pars, arch_pars, meta_pars, start_lr = 1e-6, end_lr = 1e-2)
