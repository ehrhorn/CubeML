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

def make_data():
    
    models = [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-03-03-13.40.34?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-02-28-12.16.24?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-04.51.03?workspace=user-bjoernmoelvig',
    ]
    val_error = []
    train_error = []
    epochs = []


    for i_model, model_full in enumerate(models):
        model = model_full.split('/')[-1].split('?')[0]
        model_path = locate_model(model)

        # for each model, load min error
        val_error.append(
            np.array(
                pickle.load(
                    open(
                        model_path + '/data/val_error.pickle', 'rb'
                    )
                )
            )
        )
        
        train_error.append(
            np.array(
                pickle.load(
                    open(
                        model_path + '/data/train_error.pickle', 'rb'
                    )
                )
            )
        )

        epochs.append(
            np.array(
                pickle.load(
                    open(
                        model_path + '/data/epochs.pickle', 'rb'
                    )
                )
            )
        )

        
    return val_error, train_error, epochs

val_error, train_error, epochs = make_data()


labels = [
    r'Best Hybrid',
    r'Best Attention-based',
    r'Best RNN-based',
]
# colors = [
#     'orange',
#     'blue',
#     'red',
# ]

# x = np.linspace(0.0, 3.0, 200)
d = {
    'x': [],
    'y': [],
    # 'linestyle': [],
    # 'color': [],
    'xlabel': r'Events seen',
    'ylabel': r'Loss',
    # 'xscale': 'log',
    # 'yscale': 'log',
    'label': [],
    'title': 'Validation loss',
    'yrange': [0.049, 0.061]
}
for i_model in range(len(val_error)):
    # d['x'].append(epochs[i_model])
    # d['y'].append(train_error[i_model])
    # d['linestyle'].append('dotted')
    # d['color'].append(colors[i_model])
    # d['label'].append(labels[i_model]+'- training loss')
    # ids = (val_error[i_model] > 0.05) & (val_error[i_model] < 0.062)

    d['x'].append(epochs[i_model])
    d['y'].append(val_error[i_model])
    # d['linestyle'].append('solid')
    # d['color'].append(colors[i_model])
    d['label'].append(labels[i_model])



f = make_plot(d, for_thesis=True)
ax = f.gca()
ax.set_yticks(np.linspace(0.05, 0.06, num=6))
def func(x):
    y = r'%d M'%(x//1000000)
    return y
update_xlabels(ax, keyword='func', func=func)

def func(x):
    y = r'%.1f $\cdot 10^{-2}'%(x*100)
    return y
update_ylabels(ax, keyword='func', func=func, minor=False)

# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# broad_figure: FOTW = 2.
# single_fig, 2subfigs
FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
