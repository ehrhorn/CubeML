import src.modules.helper_functions as hf
import src.modules.reporting as rprt
import argparse
from pathlib import Path

description = 'Compares the performance graphs of two models'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-i', '--inputs', metavar='', type=str, help='Model', action='append')
parser.add_argument('-s', '--save', help='Saves figure(s) in root directory', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':

    # * First create plot dictionaries
    plot_dicts = []
    for model in args.inputs:
        
        #* Locate the model directory
        paths = hf.find_files(model)
        for path in paths:
            if path.split('/')[-1] == model:
                break
        
        # * Make a plotting dictionary with the datasets from the different models
        plot_dicts = rprt.get_performance_plot_dicts(path, plot_dicts)
    
    # * Now display (or save) desired performance plots
    for i, plot_dict in enumerate(plot_dicts):
        if args.save:
            plot_dict['savefig'] = hf.get_project_root() + '/comparisons/' + plot_dict['title'] + '.png' 

        try:
            fig = rprt.make_plot(plot_dict)
        except FileNotFoundError:
            Path(hf.get_project_root() + '/comparisons/').mkdir()
            fig = rprt.make_plot(plot_dict)
    
    