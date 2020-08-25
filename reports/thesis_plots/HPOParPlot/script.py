from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
from src.modules.thesis_plotting import *
import os

setup_pgf_plotting()

# ============================================================================
# IMPORT/MAKE DATA     
# ============================================================================
def make_data_old():
    models = {
        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-13-12.34.45
        '2020-07-13-12.34.45': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 4,
            'Width': 256,
            'enctype': 2,
        },
        
        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-13-12.35.23?workspace=user-bjoernmoelvig
        '2020-07-13-12.35.23': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 4,
            'Width': 512,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-13-22.03.25?workspace=user-bjoernmoelvig
        '2020-07-13-22.03.25': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 4,
            'Width': 64,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-13-22.03.33?workspace=user-bjoernmoelvig
        '2020-07-13-22.03.33': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 4,
            'Width': 128,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-04.11.36?workspace=user-bjoernmoelvig
        '2020-07-14-04.11.36': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 2,
            'Width': 256,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-04.33.22?workspace=user-bjoernmoelvig
        '2020-07-14-04.33.22': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 1,
            'Width': 256,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-09.07.30?workspace=user-bjoernmoelvig
        '2020-07-14-09.07.30': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 5,
            'Width': 256,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-09.15.21?workspace=user-bjoernmoelvig
        '2020-07-14-09.15.21': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 3,
            'Width': 256,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-03-02-19.02.57/overview?workspace=user-bjoernmoelvig
        '2020-03-02-19.02.57': {
            'error': -1,
            'N_pre': 1,
            'N_enc': 2,
            'N_dec': 4,
            'Width': 256,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-13.57.47/overview?workspace=user-bjoernmoelvig
        '2020-07-14-13.57.47': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 6,
            'Width': 256,
            'enctype': 2
        },

         # https://app.wandb.ai/cubeml/cubeml/runs/2020-03-20-11.41.40?workspace=user-bjoernmoelvig
        '2020-03-20-11.41.40': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 2,
            'N_dec': 4,
            'Width': 256,
            'enctype': 1
        },

         # https://app.wandb.ai/cubeml/cubeml/runs/2020-02-28-12.16.24?workspace=user-bjoernmoelvig
        '2020-02-28-12.16.24': {
            'error': -1,
            'N_pre': 4,
            'N_enc': 1,
            'N_dec': 4,
            'Width': 128,
            'enctype': 2
        },

         # https://app.wandb.ai/cubeml/cubeml/runs/2020-02-29-14.10.52/overview?workspace=user-bjoernmoelvig
        '2020-02-29-14.10.52': {
            'error': -1,
            'N_pre': 5,
            'N_enc': 1,
            'N_dec': 4,
            'Width': 128,
            'enctype': 2
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-02-27-09.31.21?workspace=user-bjoernmoelvig
        '2020-02-27-09.31.21': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 6,
            'N_dec': 4,
            'Width': 64,
            'enctype': 0
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-02-26-17.12.56?workspace=user-bjoernmoelvig
        '2020-02-26-17.12.56': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 5,
            'N_dec': 4,
            'Width': 32,
            'enctype': 0
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-02-27-01.01.07?workspace=user-bjoernmoelvig
        '2020-02-27-01.01.07': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 5,
            'N_dec': 4,
            'Width': 64,
            'enctype': 0
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-02-25-18.50.49/overview?workspace=user-bjoernmoelvig
        '2020-02-25-18.50.49': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 6,
            'N_dec': 4,
            'Width': 128,
            'enctype': 0
        },

        # https://app.wandb.ai/cubeml/cubeml/runs/2020-02-24-02.47.22/overview?workspace=user-bjoernmoelvig
        '2020-02-24-02.47.22': {
            'error': -1,
            'N_pre': 0,
            'N_enc': 3,
            'N_dec': 4,
            'Width': 128,
            'enctype': 0
        },
    }

    for i_model, model in enumerate(models):
        model_path = locate_model(model)

        # for each model, load min error
        models[model]['error'] = min(
            pickle.load(open(
                model_path + '/data/val_error.pickle', 'rb'
                )
            )
        )
        
    return models

def make_data():
    data = {
        'Preproc. Depth': [],
        'Decode Depth': [],
        'Width': [],
        'Encode Depth': [],
        'Error': [],
    }
    models = [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-13-12.34.45?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-13-12.35.23?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-13-22.03.25?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-13-22.03.33?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-04.11.36?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-04.33.22?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-09.07.30?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-09.15.21?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-14-13.57.47?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-15-16.36.14?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-15-16.36.20?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-15-21.32.25?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-15-22.08.04?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-16-02.48.49?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-16-03.35.09?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-16-08.11.08?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-16-08.11.32?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-16-12.21.44?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-16-17.13.16?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-16-22.59.26?workspace=user-bjoernmoelvig',
        # 'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-01.52.16?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-04.51.03?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-07.23.17?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-16.47.07?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-16.47.12?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-22.25.30?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-18-00.28.56?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-18-05.35.10?workspace=user-bjoernmoelvig'
    ]

    for i_model, model_full in enumerate(models):
        model = model_full.split('/')[-1].split('?')[0]
        model_path = locate_model(model)

        # for each model, load min error
        data['Error'].append(
            min(
                pickle.load(open(
                    model_path + '/data/val_error.pickle', 'rb'
                    )
                )
            )
        )

        arch = json.load(open(
            model_path + '/architecture_pars.json', 'r'
            )
        )['layers']

        if 'RnnBlock' in arch[0]:
            i_dec = 0
            data['Preproc. Depth'].append(0)
        elif 'ResBlock' in arch[0]:
            i_dec = 1
            data['Preproc. Depth'].append(len(arch[0]['ResBlock']['input_sizes'])-1)
        else:
            raise ValueError('Unknown first module')
        data['Width'].append(arch[i_dec]['RnnBlock']['n_out'])
        data['Encode Depth'].append(arch[i_dec]['RnnBlock']['num_layers'])
        
        if 'ResBlock' in arch[i_dec+1]:
            data['Decode Depth'].append(
                len(
                    arch[i_dec+1]['ResBlock']['input_sizes']
                )
            )
        elif 'Linear' in arch[i_dec+1]:
            data['Decode Depth'].append(1)
        else:
            raise ValueError('Unknown decode module')
        
        
    return data

data_d = make_data()
n_models = len(data_d[next(iter(data_d))])
n_pars = len(data_d)
par_names = [key for key in data_d]

data = np.empty(shape=(n_models, n_pars))

for i_var, varname in enumerate(data_d):
    data[:, i_var] = data_d[varname]

# a+=1
# for i_model, model in enumerate(data_d):
#     for i_par, par in enumerate(par_names):

#         data[i_model, i_par] = data_d[model][par]

enctype = ['Attention', 'GRU', 'LSTM', 'Hybrid']
d = {
    'parallel_plot': data,
    'names': par_names,
    'color_index': 0,
    # 'labels': [None, None, None, None, None, enctype],
    'grid': False
}
f = make_plot(d, for_thesis=True)
ax = f.gca()
# update_ylabels(ax)

# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# broad_figure: FOTW = 2.
# single_fig, 2subfigs
FOTW = get_frac_of_textwidth(keyword='broad_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
w2 = get_figure_width(
    get_frac_of_textwidth(keyword='single_fig')
)
height = get_figure_height(width=w2)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
