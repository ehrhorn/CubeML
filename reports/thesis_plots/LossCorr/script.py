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
data_path = model_path + '/figures/pickle_CorrCoeff.pickle'

with open(data_path, 'rb') as f:
    f = pickle.load(f)

FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=1.2*FOTW)
height = get_figure_height(width=width*1.1)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)

