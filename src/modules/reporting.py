import wandb
import PIL
import pickle
import torch
from matplotlib import pyplot as plt
from time import localtime, strftime
from pathlib import Path

from src.modules.helper_functions import *
from src.modules.eval_funcs import *

# ======================================================================== 
# PERFORMANCE CLASSES
# ========================================================================

class AziPolarHists:
    '''A class to create azimuthal and polar error plots - one 2D-histogram and two performance plots.
    '''

    def __init__(self, model_dir, wandb_ID=None):
        _, data_pars, _, meta_pars = load_model_pars(model_dir)

        self.model_dir = get_path_from_root(model_dir)
        self.data_dir = data_pars['data_dir']
        self.wandb_ID = wandb_ID
        self.meta_pars = meta_pars
        self.data_dict = self._get_data_dict()

    def _get_data_dict(self):
        full_pred_address = self._get_pred_path()
        keys = self._get_keys()
        data_dict = read_predicted_h5_data(full_pred_address, keys)
        return data_dict
    
    def _get_pred_path(self):
        path_to_data = get_project_root() + self.model_dir + '/data'
        for file in Path(path_to_data).iterdir():
            if file.suffix == '.h5':
                path = str(file)
        return path
    
    def _get_keys(self):
        funcs = get_eval_functions(self.meta_pars)
        keys = []

        for func in funcs:
            keys.append(func.__name__)
        
        return keys

    def _exclude_azi_polar(self):
        azi_max_dev = 15
        polar_max_dev = 15
        
        # exclude errors outside of first interval
        azi_sorted, polar_sorted = sort_pairs(self.data_dict['azi_error'], self.data_dict['polar_error'])
        i_azi = np.searchsorted(azi_sorted, [-azi_max_dev, azi_max_dev])
        azi_sorted = azi_sorted[i_azi[0]:i_azi[1]]
        polar_sorted = polar_sorted[i_azi[0]:i_azi[1]]

        # exclude errors outside of second interval
        polar_sorted, azi_sorted = sort_pairs(polar_sorted, azi_sorted)
        i_polar = np.searchsorted(polar_sorted, [-polar_max_dev, polar_max_dev])
        azi_sorted = azi_sorted[i_polar[0]:i_polar[1]]
        polar_sorted = polar_sorted[i_polar[0]:i_polar[1]]

        return azi_sorted, polar_sorted

    def save(self):
        
        # Save standard histograms first
        for key, pred in self.data_dict.items():
            img_address = get_project_root() + self.model_dir+'/figures/'+str(key)+'.png'
            figure = make_plot({'data': [pred], 'xlabel': str(key), 'savefig': img_address})

            # Load img with PIL - png format can be logged
            if self.wandb_ID is not None:
                im = PIL.Image.open(img_address)
                wandb.log({str(key): wandb.Image(im, caption=key)}, commit = False)
        
        # Save 2D-histogram
        img_address = get_project_root() + self.model_dir+'/figures/azi_vs_polar.png'
        azi, polar = self._exclude_azi_polar()

        plot_dict = {'hexbin': [azi, polar],
                    'xlabel': 'Azimuthal error [deg]',
                    'ylabel': 'Polar error [deg]',
                    'savefig': img_address}
        fig = make_plot(plot_dict)

        if fig != -1:
            if self.wandb_ID is not None:
                im = PIL.Image.open(img_address)
                wandb.log({'azi_vs_polar': wandb.Image(im, caption='azi_vs_polar')}, commit = False)

class AziPolarPerformance:
    '''A class to create azimuthal and polar error plots - one 2D-histogram and two performance plots.
    '''

    def __init__(self, model_dir, wandb_ID=None):
        _, data_pars, _, meta_pars = load_model_pars(model_dir)
        prefix = 'transform'+str(data_pars['file_keys']['transform'])
        from_frac = data_pars['train_frac']
        to_frac = data_pars['train_frac'] + data_pars['val_frac']

        self.model_dir = get_path_from_root(model_dir)
        self.data_dir = data_pars['data_dir']
        self.meta_pars = meta_pars
        self.prefix = prefix
        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        self.data_dict = self._get_data_dict()
        self.calculate()

    def _get_data_dict(self):
        full_pred_address = self._get_pred_path()
        keys = self._get_keys()
        data_dict = read_predicted_h5_data(full_pred_address, keys)
        return data_dict
    
    def _get_pred_path(self):
        path_to_data = get_project_root() + self.model_dir + '/data'
        for file in Path(path_to_data).iterdir():
            if file.suffix == '.h5':
                path = str(file)
        return path
    
    def _get_keys(self):
        funcs = get_eval_functions(self.meta_pars)
        keys = []

        for func in funcs:
            keys.append(func.__name__)
        return keys

    def calculate(self):
        # Read data
        try:
            energy = read_h5_directory(self.data_dir, ['true_neutrino_energy'], self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)
        except KeyError:
            energy = read_h5_directory(self.data_dir, ['true_muon_energy'], self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)

        # Transform back and extract values into list
        energy = inverse_transform(energy, get_project_root() + self.model_dir)
        energy = [y for _, y in energy.items()]
        energy = [x[0] for x in energy[0]]
        self.bin_edges = np.linspace(min(energy), max(energy), 13)

        # Calculate performance as a fn of energy for polar and azi errors
        polar_error = self.data_dict['polar_error']
        print('\nCalculating polar performance...')
        self.polar_sigmas, self.polar_errors = calc_perf2_as_fn_of_energy(energy, polar_error, self.bin_edges)
        print('Calculation finished!')

        azi_error = self.data_dict['azi_error']
        print('\nCalculating azimuthal performance...')
        self.azi_sigmas, self.azi_errors = calc_perf2_as_fn_of_energy(energy, azi_error, self.bin_edges)
        print('Calculation finished!')

    def get_polar_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.polar_sigmas], 'yerr': [self.polar_errors], 'xlabel': r'log(E) [GeV]', 'ylabel': 'Error [Deg]'}

    def get_azi_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.azi_sigmas], 'yerr': [self.azi_errors], 'xlabel': r'log(E) [GeV]', 'ylabel': 'Error [Deg]'}

    def save(self):

        # Save Azi first
        perf_savepath = get_project_root()+self.model_dir+'/data/AziErrorPerformance.pickle'
        img_address = get_project_root()+self.model_dir+'/figures/AziErrorPerformance.png'
        d = self.get_azi_dict()
        d['savefig'] = img_address
        _ = make_plot(d)

        # Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'AziErrorPerformance': wandb.Image(im, caption='AziErrorPerformance')}, commit = False)

        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)
        
        # Save polar next
        perf_savepath = get_project_root()+self.model_dir+'/data/PolarErrorPerformance.pickle'
        img_address = get_project_root()+self.model_dir+'/figures/PolarErrorPerformance.png'
        d = self.get_polar_dict()
        d['savefig'] = img_address
        _ = make_plot(d)

        # Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'PolarErrorPerformance': wandb.Image(im, caption='PolarErrorPerformance')}, commit = False)

        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)

class DirErrorPerformance:
    '''A class to calculate and save performance wrt directional error

    Saves calculated histograms as self.edges, self.maes, self.errors.
    '''

    def __init__(self, model_dir, n_bins=15, wandb_ID=None):

        _, data_pars, arch_pars, meta_pars = load_model_pars(model_dir)
        prefix = 'transform'+str(data_pars['file_keys']['transform'])
        from_frac = data_pars['train_frac']
        to_frac = data_pars['train_frac'] + data_pars['val_frac']

        self.model_dir = get_path_from_root(model_dir)
        self.data_dir = data_pars['data_dir']
        self.predictor_keys = ['directional_error']
        self.prefix = prefix
        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        self.calculate()

    def calculate(self):
        # Read data
        try:
            energy = read_h5_directory(self.data_dir, ['true_neutrino_energy'], self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)
        except KeyError:
            energy = read_h5_directory(self.data_dir, ['true_muon_energy'], self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)
        
        # Transform back and extract values into list
        energy = inverse_transform(energy, get_project_root() + self.model_dir)
        energy = [y for _, y in energy.items()]
        energy = [x[0] for x in energy[0]]

        # Get path to predictions
        path = self.get_predictions_path()

        # Read prediction
        best_pred = read_predicted_h5_data(path, self.predictor_keys)

        # Convert to list
        best_pred = [y for _, y in best_pred.items()][0]
        self.median = np.nanpercentile(best_pred, 50)

        # Make performance histogram
        self.edges, self.maes, self.errors = calc_perf_as_fn_of_energy(energy, best_pred)
        self.n_events = len(energy)

    def get_predictions_path(self):
        path = get_project_root()+self.model_dir+'/data/'
        for obj in Path(path).iterdir():
            if obj.suffix == '.h5':
                return str(obj)

    def get_dict(self):
        return {'edges': [self.edges], 'y': [self.maes], 'yerr': [self.errors], 'xlabel': r'log(E) [GeV]', 'ylabel': 'Error [Deg]'}
    
    def save(self):
        perf_savepath = get_project_root()+self.model_dir+'/data/DirErrorPerformance.pickle'
        img_address = get_project_root()+self.model_dir+'/figures/DirErrorPerformance.png'
        figure = make_plot(self.get_dict())
        figure.savefig(img_address)

        # Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'DirErrorPerformance': wandb.Image(im, caption='DirErrorPerformance')}, commit = False)

        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)

def log_operation_plots(model_dir, wandb_ID=None):
    '''Default functionality: Searches in model_dir for pickle-files containing epochs, lr, val. error and training error and saves plots to wandb and locally.
    '''
    
    epochs = model_dir+'/data/epochs.pickle'
    with open(epochs, 'rb') as f:
        epochs = pickle.load(f)

    train_error = model_dir+'/data/train_error.pickle'
    with open(train_error, 'rb') as f:
        train_error = pickle.load(f)

    val_error = model_dir+'/data/val_error.pickle'
    with open(val_error, 'rb') as f:
        val_error = pickle.load(f)

    lr_list = model_dir+'/data/lr.pickle'
    with open(lr_list, 'rb') as f:
        lr_list = pickle.load(f)
    
    img_address = model_dir+'/figures/train_val_error.png'
    _ = make_plot({'x': [epochs, epochs], 'y': [train_error, val_error], 'label': ['train error', 'val. error'], 'xlabel': 'Epoch', 'ylabel': 'Loss', 'savefig': img_address})
    
    if wandb_ID is not None:
        im = PIL.Image.open(img_address)
        wandb.log({'Train and val. error': wandb.Image(im, caption='Train and val. error')}, commit = False)
    
    img_address = model_dir+'/figures/lr_vs_epoch.png'
    _ = make_plot({'x': [epochs], 'y': [lr_list], 'xlabel': 'Epoch', 'ylabel': 'Learning rate', 'savefig': img_address})
    
    if wandb_ID is not None:
        im = PIL.Image.open(img_address)
        wandb.log({'Learning rate vs epoch': wandb.Image(im, caption='Learning rate vs epoch')}, commit = False)

def log_performance_plots(model_dir, wandb_ID=None):
    '''Creates and logs performance plots relevant to the regression model
    '''
    _, _, _, meta_pars = load_model_pars(model_dir)
    

    print(strftime("%d/%m %H:%M", localtime()), ': Logging plots...')
    if meta_pars['group'] == 'direction_reg':

        # Plot and save azi- and polarplots
        azi_polar = AziPolarHists(model_dir, wandb_ID=wandb_ID)
        azi_polar.save()

        azi_polar_perf = AziPolarPerformance(model_dir, wandb_ID=wandb_ID)
        azi_polar_perf.save()

        # Plot and save directional error performance
        perf = DirErrorPerformance(model_dir, wandb_ID=wandb_ID)
        perf.save()

    else:
        print('Unknown regression type - no plots have been produced.')
    print(strftime("%d/%m %H:%M", localtime()), ': Logging finished!')

def make_plot(plot_dict):
    '''A custom plot function using PyPlot. If 'x' AND 'y' are in plot_dict, a xy-graph is returned, if 'data' is given, a histogram is returned. 

    Example dictionary: 
    plot_dict = {'data': [set1, set2], 'xlabel': '<LABEL_NAME>', 'ylabel': '<LABEL_NAME>', 'label':['<PLOT1_NAME>', '<PLOT2_NAME>']}

    Input: Figure dictionary
    Output: Figure handle
    '''
    # Make a xy-plot
    if 'x' in plot_dict and 'y' in plot_dict:
        alpha = 0.5
        plt.style.use('default')

        h_figure = plt.figure()
        h_subfig = plt.subplot(1, 1, 1)
        if 'xlabel' in plot_dict: h_subfig.set_xlabel(plot_dict['xlabel'])
        if 'ylabel' in plot_dict: h_subfig.set_ylabel(plot_dict['ylabel'])
        
        for i_set, dataset in enumerate(plot_dict['y']):
            plot_keys = ['label']
            # Set baseline
            d = {'linewidth': 1.5}
            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            plt.plot(plot_dict['x'][i_set], dataset, **d)
            
            if 'xscale' in plot_dict: h_subfig.set_xscale(plot_dict['xscale'])
            if 'yscale' in plot_dict: h_subfig.set_yscale(plot_dict['yscale'])
        
        # Plot vertical lines if wanted
        if 'axvline' in plot_dict:
            for vline in plot_dict['axvline']:
                h_subfig.axvline(x=vline, color = 'k', ls = ':')
            
        if 'label' in plot_dict: h_subfig.legend()
        h_subfig.grid(alpha = alpha)
        
    elif 'data' in plot_dict:
        
        alpha = 0.5
        plt.style.use('default')

        h_figure = plt.figure()
        h_subfig = plt.subplot(1, 1, 1)
        h_subfig.grid(alpha = alpha)
        if 'xlabel' in plot_dict: h_subfig.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_subfig.set_ylabel(plot_dict['ylabel'])
        else: h_subfig.set_ylabel('Density')


        for i_set, data in enumerate(plot_dict['data']):
            
            plot_keys = ['label', 'alpha', 'density']
            
            # Set baseline
            if len(plot_dict['data']) > 1:
                d = {'alpha': 0.6, 'density': True, 'bins': 'fd'}
            else:
                d = {'alpha': 1.0, 'density': True, 'bins': 'fd'}
            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            plt.hist(data, **d)

            if 'label' in plot_dict: h_subfig.legend()
        
        if 'yscale' in plot_dict: h_subfig.set_yscale(plot_dict['yscale'])
        if 'xscale' in plot_dict: h_subfig.set_xscale(plot_dict['xscale'])

    elif 'hist2d' in plot_dict:

        plt.style.use('default')

        h_figure = plt.figure()
        h_subfig = plt.subplot(1, 1, 1)
        if 'xlabel' in plot_dict: h_subfig.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_subfig.set_ylabel(plot_dict['ylabel'])

        set1 = plot_dict['hist2d'][0]
        set2 = plot_dict['hist2d'][1]

        if type(set1) == torch.Tensor: set1 = set1.cpu().numpy()
        if type(set2) == torch.Tensor: set2 = set2.cpu().numpy()

        # Get bin-widths
        _, widths1 = np.histogram(set1, bins='fd')
        _, widths2 = np.histogram(set2, bins='fd')

        # Rescale 
        widths1 = np.linspace(min(widths1), max(widths1), int(0.5 + widths1.shape[0]/4.0))
        widths2 = np.linspace(min(widths2), max(widths2), int(0.5 + widths2.shape[0]/4.0))
        plt.hist2d(set1, set2, bins = [widths1, widths2])
        plt.colorbar()

    elif 'hexbin' in plot_dict:

        plt.style.use('default')

        h_figure = plt.figure()
        h_subfig = plt.subplot(1, 1, 1)
        if 'xlabel' in plot_dict: h_subfig.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_subfig.set_ylabel(plot_dict['ylabel'])

        set1 = plot_dict['hexbin'][0]
        set2 = plot_dict['hexbin'][1]

        if type(set1) == torch.Tensor: set1 = set1.cpu().numpy()
        if type(set2) == torch.Tensor: set2 = set2.cpu().numpy()

        # Get bin-widths - my attempt to modularize generation of 2d-histograms
        _, widths1 = np.histogram(set1, bins='fd')
        _, widths2 = np.histogram(set2, bins='fd')

        if 'range' in plot_dict:
            xmin = plot_dict['range'][0][0]
            xmax = plot_dict['range'][0][1]
            ymin = plot_dict['range'][1][0]
            ymax = plot_dict['range'][1][1]
        else:
            try:
                if set1.shape[0] == 0:
                    print('ERROR: Empty list was given to make_plot (hexbin). No plot created.')
                    return -1
                xmin = set1.min()
                xmax = set1.max()
                ymin = set2.min()
                ymax = set2.max()
            except AttributeError:
                if len(set1) == 0:
                    print('ERROR: Empty list was given to make_plot (hexbin). No plot created.')
                    return -1
                xmin = min(set1)
                xmax = max(set1)
                ymin = min(set2)
                ymax = max(set2)
            
        # Rescale. Factor of 4 comes form trial and error
        n1 = int(0.5 + widths1.shape[0]/4.0)
        n2 = int(0.5 + widths2.shape[0]/4.0)
        plt.hexbin(set1, set2, gridsize = (n1, n2))
        plt.axis([xmin, xmax, ymin, ymax])
        plt.colorbar()
        
    elif 'edges' in plot_dict:
        alpha = 0.5
        plt.style.use('default')

        h_figure = plt.figure()
        h_subfig = plt.subplot(1, 1, 1)
        h_subfig.grid(alpha = alpha)
        if 'xlabel' in plot_dict: h_subfig.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_subfig.set_ylabel(plot_dict['ylabel'])

        # Calculate bin centers and 'x-error'
        centers = []
        xerrs = []
        for edges in plot_dict['edges']:
            centers.append(calc_bin_centers(edges))
            xerrs.append(calc_dists_to_binedges(edges))
        
        for i_set in range(len(centers)):
            x = centers[i_set]
            xerr = xerrs[i_set]
            y = plot_dict['y'][i_set]
            yerr = plot_dict['yerr'][i_set]
            edges = plot_dict['edges'][i_set]

            plot_keys = ['label']
            # Set baseline
            d = {'linewidth': 1.5}
            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            
            plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='.', **d)
            
        if 'xscale' in plot_dict: h_subfig.set_xscale(plot_dict['xscale'])
        if 'yscale' in plot_dict: h_subfig.set_yscale(plot_dict['yscale'])

        # Plot vertical lines if wanted
        if 'axvline' in plot_dict:
            for vline in plot_dict['axvline']:
                h_subfig.axvline(x=vline, color = 'k', ls = ':')
            
        if 'label' in plot_dict: h_subfig.legend()
        h_subfig.grid(alpha = alpha)

    else:
        raise ValueError('Unknown plot wanted!')

    if 'text' in plot_dict:
        plt.text(*plot_dict['text'], transform=h_subfig.transAxes)
    
    if 'savefig' in plot_dict: 
            h_figure.savefig(plot_dict['savefig'])
            print('\nFigure saved at:')
            print(plot_dict['savefig'])

    return h_figure

def summarize_model_performance(model_dir, wandb_ID=None):
    '''Summarizes a model's performance with a single number.

    Input
    model_dir: Full path to the model directory
    wandb_ID: Optionally, a string with the wandb ID to log to aswell

    Output
    onenum_performance: A single number encapsulating the model's performance
    '''
    
    _, _, _, meta_pars = load_model_pars(model_dir)

    if meta_pars['group'] == 'direction_reg':
        direrr_path = model_dir + '/data/DirErrorPerformance.pickle'
        direrrperf_class = pickle.load( open( direrr_path, "rb" ) )

        onenum_performance = direrrperf_class.median
    
    else:
        print('\nERROR: No one-number performance measure defined. Returning -1\n')
        onenum_performance = -1
    
    if wandb_ID is not None:
        wandb.config.update({'Performance': onenum_performance})

    meta_pars['performance'] = onenum_performance
    with open(model_dir+'/meta_pars.json', 'w') as fp:
        json.dump(meta_pars, fp)
