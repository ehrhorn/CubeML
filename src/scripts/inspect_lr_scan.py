import argparse
from pathlib import Path
import pickle
import numpy as np

from src.modules.helper_functions import find_files
from src.modules.reporting import make_plot

description = 'Given a learning rate directory name, a certain range of the learning rate is inspected'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-p', '--path', metavar='', type=str, help='Path to model directory')
parser.add_argument('--max_lr', default=1e0, type=float, help='Sets the maximum learning rate for scan inspection.')
parser.add_argument('--min_lr', default=1e-6, type=float, help='Sets the minimum learning rate for scan inspection.')
args = parser.parse_args()

if __name__ == '__main__':
    if args.path != None:
        model_dir = args.path
    if 'model_dir' not in locals():
        raise ValueError('No path supplied!')

    # * Locate the model directory
    paths = find_files(model_dir)
    for path in paths:
        if path.split('/')[-1] == model_dir:
            model = path
            break

    from_lr, to_lr = args.min_lr, args.max_lr
    lrs = pickle.load(open(model+'/lr.pickle', 'rb'))
    losses = pickle.load(open(model+'/loss_vals.pickle', 'rb'))
    indices = [index for index in range(len(lrs)) if from_lr <= lrs[index] <= to_lr]

    chosen_lrs = np.array(lrs)[indices]
    chosen_losses = np.array(losses)[indices]
    d = {'x': [chosen_lrs], 'y': [chosen_losses], 'xscale': 'log', 'savefig': model+'/lr_vs_loss.png', 'xlabel': 'Learning Rate', 'ylabel': 'Loss'}
    fig = make_plot(d)

