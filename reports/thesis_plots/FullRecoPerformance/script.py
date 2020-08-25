from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
from src.modules.thesis_plotting import *
import os

setup_pgf_plotting()

model_full = 'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-21-15.26.17?workspace=user-bjoernmoelvig'

model = model_full.split('/')[-1].split('?')[0]
model_path = locate_model(model)
data_path = model_path + '/figures/pickle_'
wanted_plots = [
    'directional_error_performance',
    'polar_error_performance',
    'log_frac_E_error_performance',
    'relative_E_error_performance',
    'len_error_performance',
    'vertex_t_error_performance',   
    'polar_error_2DPerformance',
    'relative_E_error_2DPerformance' 
]

plotnames = [
    '_directional',
    '_polar',
    '_log_frac_E',
    '_rel_E',
    '_vert_dist',
    '_vert_t',
    '_polar_2D',
    '_rel_E_2D'
]

save_path = get_project_root() + '/reports/thesis_plots/all_pgf/FullRecoPerformance'
for wanted_plot, name in zip(wanted_plots, plotnames):
    path = data_path + wanted_plot + '.pickle'
    with open(path, 'rb') as f:
        fig = pickle.load(f)

        # print(type(fig), fig)
    savefig_path = save_path + name + '.png'
    fig.savefig(savefig_path, bbox_inches='tight')

# 
