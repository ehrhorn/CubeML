#%%
import sys 
# sys.path.append('..')
from tables import *
from pathlib import Path
import numpy as np
from src.modules.main_funcs import *
from src.modules.loss_funcs import *
from collections import OrderedDict

# ======================================================================== 
# DEFINE SCRIPT OBJECTIVE
# ======================================================================== 
hyper_pars = {
    'batch_size': 64, 
    'max_epochs': 1,
    'early_stop_patience': 1,
    'optimizer': {
        'optimizer': 'Adam',
        'lr': 0.001,
        'betas': (
            0.9,
            0.999
        ),
        'eps': 1.0e-8
    },
    'lr_schedule': {
        'lr_scheduler': None
    } 
}

data_pars = {
    'data_dir': '/data/MuonGun_Level2_139008',
    'seq_feat': [
        'charge',
        'dom_x',
        'dom_y',
        'dom_z',
        'time'
    ],
    'scalar_feat': [],
    'target': [
        'true_muon_direction_x',
        'true_muon_direction_y',
        'true_muon_direction_z',
    ],
    'train_frac': 0.00025,
    'val_frac': 0.00025,
    'test_frac': 0.0,
    'file_keys': {
        'transform': 2
    }
}

n_seq_feat = len(data_pars['seq_feat'])
n_scalar_feat = len(data_pars['scalar_feat'])
n_target = len(data_pars['target'])

arch_pars = {
    'non_lin': {
        'func': 'LeakyReLU'
    },
    'loss_func': 'dir_reg_L1_like_loss',
    'norm': {
        'norm': 'BatchNorm1D',
        'momentum': 0.9
    },
    'layers': OrderedDict(
        [
            (
                'Conv1d',
                {
                    'input_sizes': [
                        5,
                        32,
                        64,
                        128
                    ],
                    'kernel_sizes': [
                        5,
                        5,
                        5
                    ],
                    'strides': [
                        2,
                        2,
                        2
                    ],
                    'paddings': [ 
                        0,
                        0,
                        0
                    ],
                    'dilations': [
                        1,
                        1,
                        1
                    ],
                    'pools': {
                        'on': [
                            False,
                            True,
                            True
                        ],
                        'kernel_sizes': [
                            None,
                            1,
                            1
                        ],
                        'strides': [
                            None,
                            2,
                            2
                        ],
                        'paddings': [
                            None,
                            0,
                            0
                        ],
                        'dilations': [
                            None,
                            1,
                            1
                        ]
                    }
                }
            ),
            (
                'Linear',
                {
                    'input_sizes': [
                        None,
                        n_target
                    ],
                    'norm_before_nonlin': True
                }
            )
        ]
    )
}


def longest_sequence(directory, transform):
    file_list = [
        f for f in directory.iterdir() if f.is_file() and f.suffix == '.h5'
    ]
    size = []
    for file in file_list:
        with File(file, 'r') as f:
            group = f.root._f_get_child(transform)
            size.append(max(group._f_get_child('no_of_doms')))
    return max(size)


def linear_layer_size(arch_pars, longest_seq):
    L_out = longest_seq
    arch_pars = arch_pars['layers']['Conv1d']
    n_layers = len(arch_pars['input_sizes']) - 1
    for i_layer in range(n_layers):
        L_in = L_out
        padding = arch_pars['paddings'][i_layer]
        dilation = arch_pars['dilations'][i_layer]
        kernel_size = arch_pars['kernel_sizes'][i_layer]
        stride = arch_pars['strides'][i_layer]
        L_out = np.floor((
            L_in + 2 * padding - dilation * (kernel_size - 1) - 1
        ) / stride + 1)
        if arch_pars['pools']['on'][i_layer]:
            L_in = L_out
            padding = arch_pars['pools']['paddings'][i_layer]
            dilation = arch_pars['pools']['dilations'][i_layer]
            kernel_size = arch_pars['pools']['kernel_sizes'][i_layer]
            stride = arch_pars['pools']['strides'][i_layer]
            L_out = np.floor((
                L_in + 2 * padding - dilation * (kernel_size - 1) - 1
            ) / stride + 1)
    return int(L_out)


directory = Path('data/MuonGun_Level2_139008')
longest_seq = longest_sequence(directory, 'transform1')

# lin_lay_size = linear_layer_size(arch_pars, longest_seq)
lin_lay_size = 3274

arch_pars['layers']['Linear']['input_sizes'][0] = lin_lay_size

dataset = data_pars['data_dir'].split('/')[-1]

# Options: 'full_reg', 'direction_reg'
regression_type = 'direction_reg'
meta_pars = {
    'tags': [
        regression_type,
        dataset
    ],
    'group': regression_type,
    'project': 'cubeml_test'
}

train(hyper_pars, data_pars, arch_pars, meta_pars, save=False, longest_seq=longest_seq)


# %%
