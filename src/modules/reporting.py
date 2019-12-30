import wandb
import PIL
import pickle
import torch
from matplotlib import pyplot as plt
from time import localtime, strftime

from src.modules.helper_functions import *
from src.modules.eval_funcs import *

#* ======================================================================== 
#* PERFORMANCE CLASSES
#* ========================================================================

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
    """A class to create and save performance plots for azimuthal and polar predictions. If available, the relative improvement compared to Icecubes reconstruction is plotted aswell     
    
    Raises:
        KeyError: If an unknown dataset is encountered.
    
    Returns:
        [type] -- Instance of class.
    """    

    def __init__(self, model_dir, wandb_ID=None):
        _, data_pars, _, meta_pars = load_model_pars(model_dir)
        prefix = 'transform'+str(data_pars['file_keys']['transform'])
        from_frac = data_pars['train_frac']
        to_frac = data_pars['train_frac'] + data_pars['val_frac']

        self.model_dir = get_path_from_root(model_dir)
        self.data_dir = data_pars['data_dir']
        self.meta_pars = meta_pars
        self.prefix = prefix
        self._get_energy_key()
        self._get_reco_keys()

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

    def _get_energy_key(self):
        dataset_name = get_dataset_name(self.data_dir)

        if dataset_name == 'MuonGun_Level2_139008':
            self.energy_key = ['true_muon_energy']
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            self.energy_key = ['true_neutrino_energy']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        

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

    def _get_reco_keys(self):
        dataset_name = get_dataset_name(self.data_dir)

        if dataset_name == 'MuonGun_Level2_139008':
            self._reco_keys = None
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            self._reco_keys = ['retro_crs_prefit_azi', 'retro_crs_prefit_zen']
            self._true_xyz = ['true_neutrino_direction_x', 'true_neutrino_direction_y',  'true_neutrino_direction_z']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))


    def calculate(self):
        energy = read_h5_directory(self.data_dir, self.energy_key, self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)

        #* Transform back and extract values into list
        # * Our predictions have already been converted in predict()
        energy = inverse_transform(energy, get_project_root() + self.model_dir)
        energy = [y for _, y in energy.items()]
        energy = [x[0] for x in energy[0]]
        self.counts, self.bin_edges = np.histogram(energy, bins=12)
        
        polar_error = self.data_dict['polar_error']
        print('\nCalculating polar performance...')
        self.polar_sigmas, self.polar_errors = calc_perf2_as_fn_of_energy(energy, polar_error, self.bin_edges)
        print('Calculation finished!')

        azi_error = self.data_dict['azi_error']
        print('\nCalculating azimuthal performance...')
        self.azi_sigmas, self.azi_errors = calc_perf2_as_fn_of_energy(energy, azi_error, self.bin_edges)
        print('Calculation finished!')

        # * If an I3-reconstruction exists, get it
        if self._reco_keys:
            azi_crs = read_h5_directory(self.data_dir, self._reco_keys, prefix=self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)
            true = read_h5_directory(self.data_dir, self._true_xyz, prefix=self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)

            # * Ensure keys are proper so the angle calculations work
            # * Our predictions have already been converted in predict()
            true = inverse_transform(true, get_project_root() + model_dir)
            azi_crs = convert_keys(azi_crs, [key for key in azi_crs], ['azi', 'zen'])
            true = convert_keys(true, [key for key in true], ['x', 'y', 'z'])

            azi_crs_error = get_retro_crs_prefit_azi_error(azi_crs, true)
            polar_crs_error = get_retro_crs_prefit_polar_error(azi_crs, true)

            print('\nCalculating crs polar performance...')
            self.polar_crs_sigmas, self.polar_crs_errors = calc_perf2_as_fn_of_energy(energy, polar_crs_error, self.bin_edges)
            print('Calculation finished!')

            print('\nCalculating crs azimuthal performance...')
            self.azi_crs_sigmas, self.azi_crs_errors = calc_perf2_as_fn_of_energy(energy, azi_crs_error, self.bin_edges)
            print('Calculation finished!')            

            #* Calculate the relative improvement - e_diff/I3_error. Report decrease in error as a positive result
            a, b = calc_relative_error(self.polar_crs_sigmas, self.polar_sigmas, self.polar_crs_errors, self.polar_errors)
            self.polar_relative_improvements, self.polar_sigma_improvements = -a, b
            a, b = calc_relative_error(self.azi_crs_sigmas, self.azi_sigmas, self.azi_crs_errors, self.azi_errors)
            self.azi_relative_improvements, self.azi_sigma_improvements = -a, b
        else:
            self.polar_relative_improvements = None
            self.polar_sigma_improvements = None
            self.azi_relative_improvements = None
            self.azi_sigma_improvements = None

    def get_azi_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.azi_sigmas], 'yerr': [self.azi_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [Deg]', 'grid': False}
    
    def get_energy_dict(self):
        return {'data': [self.bin_edges[:-1]], 'bins': [self.bin_edges], 'weights': [self.counts], 'histtype': ['step'], 'log': [True], 'color': ['lightgray'], 'twinx': True, 'grid': False, 'ylabel': 'Events'}

    def get_polar_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.polar_sigmas], 'yerr': [self.polar_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [Deg]', 'grid': False}

    def get_rel_azi_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.azi_relative_improvements], 'yerr': [self.azi_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}

    def get_rel_polar_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.polar_relative_improvements], 'yerr': [self.polar_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}

    def save(self):

        #* Save Azi first
        perf_savepath = get_project_root()+self.model_dir+'/data/AziErrorPerformance.pickle'
        img_address = get_project_root()+self.model_dir+'/figures/AziErrorPerformance.png'
        d = self.get_azi_dict()
        h_fig = make_plot(d)
        
        if self._reco_keys:
            h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
            d = self.get_rel_azi_dict()
            d['subplot'] = True
            d['axhline'] = [0.0]
            h_fig = make_plot(d, h_figure=h_fig, position=[0.125, 0.11, 0.775, 0.15])
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)
        else:
            h_fig = make_plot(d)
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)

        #* Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'AziErrorPerformance': wandb.Image(im, caption='AziErrorPerformance')}, commit = False)
        
        #* Save polar next
        perf_savepath = get_project_root() + self.model_dir + '/data/PolarErrorPerformance.pickle'
        img_address = get_project_root() + self.model_dir + '/figures/PolarErrorPerformance.png'
        d = self.get_polar_dict()
        h_fig = make_plot(d)
        
        if self._reco_keys:
            h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
            d = self.get_rel_polar_dict()
            d['subplot'] = True
            d['axhline'] = [0.0]
            h_fig = make_plot(d, h_figure=h_fig, position=[0.125, 0.11, 0.775, 0.15])
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)
        else:
            h_fig = make_plot(d)
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)

        #* Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'PolarErrorPerformance': wandb.Image(im, caption='PolarErrorPerformance')}, commit = False)

        perf_savepath = get_project_root() + self.model_dir + '/data/AziPolarPerformance.pickle'
        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)

class IceCubePerformance:

    def __init__(self, dataset_name, n_data=50000):
        """A class to create and hold histogram-data for the resolution of Icecubes own algorithms. The error-histograms can be plotted by passing dictionaries (IceCubePerformance.get_x_dict etc.) to make_graph.
        
        Arguments:
            dataset_name {str} -- the name of the directory containing the dataset-files.
        
        Keyword Arguments:
            n_data {int} -- How many reconstructions to calculate the Icecube performance from (default: {50000})
        """        

        self.dataset_name = '/data/'+dataset_name
        self.n_data = n_data
        self._calculate_histograms()

    
    def _calculate_histograms(self):
        
        if self.dataset_name == '/data/oscnext-genie-level5-v01-01-pass2':
            vertex_reg_keys = get_target_keys({'data_dir': self.dataset_name}, {'group': 'vertex_reg'})
            dir_reg_keys = get_target_keys({'data_dir': self.dataset_name}, {'group': 'direction_reg'})
            i3_reco = get_retro_crs_prefit_vertex_keys()
            self._save_vertex_histograms(vertex_reg_keys, i3_reco)
            
        else:
            raise ValueError('Unknown dataset (%s) given!'%(self.dataset_name))
    
    def _save_vertex_histograms(self, keys, i3_reco_keys):
        path_to_data = get_project_root() + self.dataset_name
        n_events = 0
        true = {key: np.array([]) for key in keys}
        pred = {key: np.array([]) for key in i3_reco_keys}

        # * Read the data
        while n_events < self.n_data:
            for file in Path(path_to_data).iterdir():
                if file.suffix == '.h5':
                    
                    for key in keys:
                        true[key] = np.append(true[key], read_h5_dataset(str(file), key))
                    for key in i3_reco_keys:
                        pred[key] = np.append(pred[key], read_h5_dataset(str(file), key))

                    n_events = pred[key].shape[0]

            # * Exit when all files read
            break
        
        # * Calculate error
        errors = {'x_error': np.array([]), 'y_error': np.array([]), 'z_error': np.array([]), }
        for true_key, pred_key, error_key in zip(true, pred, errors):
            errors[error_key] = pred[pred_key]-true[true_key]
        
        # * Bin and save
        self.x_count, self.x_edges = np.histogram(errors['x_error'], bins='fd')
        self.y_count, self.y_edges = np.histogram(errors['y_error'], bins='fd')
        self.z_count, self.z_edges = np.histogram(errors['z_error'], bins='fd')
    
    def get_x_dict(self):
        d = {'data': [self.x_edges[:-1]], 'bins': [self.x_edges], 'weights': [self.x_count]}
        return d
    
    def get_y_dict(self):
        d = {'data': [self.y_edges[:-1]], 'bins': [self.y_edges], 'weights': [self.y_count]}
        return d
    
    def get_z_dict(self):
        d = {'data': [self.z_edges[:-1]], 'bins': [self.z_edges], 'weights': [self.z_count]}
        return d
    
    def save(self):
        print('NOT MADE YET')
        pass

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

class VertexPerformance:
    """A class to create and save performance plots for interaction vertex predictions. If available, the relative improvement compared to Icecubes reconstruction is plotted aswell. A one-number performance summary is saved as the median of the total vertex distance error.     
    
    Raises:
        KeyError: If an unknown dataset is encountered.
    
    Returns:
        [type] -- Instance of class.
    """    

    def __init__(self, model_dir, wandb_ID=None):
        _, data_pars, _, meta_pars = load_model_pars(model_dir)
        prefix = 'transform'+str(data_pars['file_keys']['transform'])
        from_frac = data_pars['train_frac']
        to_frac = data_pars['train_frac'] + data_pars['val_frac']

        self.model_dir = get_path_from_root(model_dir)
        self.data_pars = data_pars
        self.meta_pars = meta_pars
        self.prefix = prefix
        self.energy_key = self._get_energy_key()
        self._reco_keys = self._get_reco_keys()
        self._true_xyz_keys = self._get_true_xyz_keys()

        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        data_dict = self._get_data_dict()
        self._create_performance_plots(data_dict)
        self._calculate_onenum_performance(data_dict)

    def _get_data_dict(self):
        full_pred_address = self._get_pred_path()
        keys = self._get_keys()

        data_dict = read_predicted_h5_data(full_pred_address, keys)
        return data_dict

    def _get_energy_key(self):
        dataset_name = get_dataset_name(self.data_pars['data_dir'])

        if dataset_name == 'MuonGun_Level2_139008':
            energy_key = ['true_muon_energy']
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            energy_key = ['true_primary_energy']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return energy_key
    
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

    def _get_reco_keys(self):
        dataset_name = get_dataset_name(self.data_pars['data_dir'])

        if dataset_name == 'MuonGun_Level2_139008':
            reco_keys = None
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            reco_keys = ['retro_crs_prefit_x', 'retro_crs_prefit_y', 'retro_crs_prefit_z']
            self._true_xyz = ['true_primary_position_x', 'true_primary_position_y',  'true_primary_position_z']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return reco_keys

    def _get_true_xyz_keys(self):
        dataset_name = get_dataset_name(self.data_pars['data_dir'])

        if dataset_name == 'MuonGun_Level2_139008':
            true_xyz = None
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            true_xyz = ['true_primary_position_x', 'true_primary_position_y',  'true_primary_position_z']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return true_xyz
    
    def _create_performance_plots(self, data_dict):
        energy = read_h5_directory(self.data_pars['data_dir'], self.energy_key, self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)

        #* Transform back and extract values into list
        energy = inverse_transform(energy, get_project_root() + self.model_dir)
        energy = [y for _, y in energy.items()][0]
        self.counts, self.bin_edges = np.histogram(energy, bins=12)
        
        x_error = data_dict['vertex_x_error']
        print('\nCalculating x performance...')
        self.x_sigmas, self.x_errors = calc_perf2_as_fn_of_energy(energy, x_error, self.bin_edges)
        print('Calculation finished!')

        y_error = data_dict['vertex_y_error']
        print('\nCalculating y performance...')
        self.y_sigmas, self.y_errors = calc_perf2_as_fn_of_energy(energy, y_error, self.bin_edges)
        print('Calculation finished!')

        z_error = data_dict['vertex_z_error']
        print('\nCalculating z performance...')
        self.z_sigmas, self.z_errors = calc_perf2_as_fn_of_energy(energy, z_error, self.bin_edges)
        print('Calculation finished!')

        # * Calculate one-number performance

        # * If an I3-reconstruction exists, get it
        if self._reco_keys:
            pred_crs = read_h5_directory(self.data_pars['data_dir'], self._reco_keys, prefix=self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)
            true = read_h5_directory(self.data_pars['data_dir'], self._true_xyz, prefix=self.prefix, from_frac=self.from_frac, to_frac=self.to_frac)

            # * Ensure keys are proper so the angle calculations work
            pred_crs = inverse_transform(pred_crs, get_project_root() + self.model_dir)
            true = inverse_transform(true, get_project_root() + self.model_dir)

            pred_crs = convert_keys(pred_crs, [key for key in pred_crs], ['x', 'y', 'z'])
            true = convert_keys(true, [key for key in true], ['x', 'y', 'z'])

            x_crs_error = vertex_x_error(pred_crs, true)
            y_crs_error = vertex_y_error(pred_crs, true)
            z_crs_error = vertex_z_error(pred_crs, true)

            print('\nCalculating crs x performance...')
            self.x_crs_sigmas, self.x_crs_errors = calc_perf2_as_fn_of_energy(energy, x_crs_error, self.bin_edges)
            print('Calculation finished!')

            print('\nCalculating crs y performance...')
            self.y_crs_sigmas, self.y_crs_errors = calc_perf2_as_fn_of_energy(energy, y_crs_error, self.bin_edges)
            print('Calculation finished!')

            print('\nCalculating crs z performance...')
            self.z_crs_sigmas, self.z_crs_errors = calc_perf2_as_fn_of_energy(energy, z_crs_error, self.bin_edges)
            print('Calculation finished!')

            #* Calculate the relative improvement - e_diff/I3_error. Report decrease in error as a positive result
            a, b = calc_relative_error(self.x_crs_sigmas, self.x_sigmas, self.x_crs_errors, self.x_errors)
            self.x_relative_improvements, self.x_sigma_improvements = -a, b

            a, b = calc_relative_error(self.y_crs_sigmas, self.y_sigmas, self.y_crs_errors, self.y_errors)
            self.y_relative_improvements, self.y_sigma_improvements = -a, b

            a, b = calc_relative_error(self.z_crs_sigmas, self.z_sigmas, self.z_crs_errors, self.z_errors)
            self.z_relative_improvements, self.z_sigma_improvements = -a, b
        
        else:
            self.x_relative_improvements = None
            self.x_sigma_improvements = None
            self.y_relative_improvements = None
            self.y_sigma_improvements = None
            self.z_relative_improvements = None
            self.z_sigma_improvements = None
    
    def _calculate_onenum_performance(self, data_dict):

        x_error = data_dict['vertex_x_error']
        y_error = data_dict['vertex_y_error']
        z_error = data_dict['vertex_z_error']

        len_error = np.sqrt(x_error**2 + y_error**2 + z_error**2)
        self.median_len_error = np.nanpercentile(len_error, 50)

    def get_energy_dict(self):
        return {'data': [self.bin_edges[:-1]], 'bins': [self.bin_edges], 'weights': [self.counts], 'histtype': ['step'], 'log': [True], 'color': ['lightgray'], 'twinx': True, 'grid': False, 'ylabel': 'Events'}

    def get_x_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.x_sigmas], 'yerr': [self.x_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [m]', 'grid': False}
    def get_y_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.y_sigmas], 'yerr': [self.y_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [m]', 'grid': False}
    def get_z_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.z_sigmas], 'yerr': [self.z_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [m]', 'grid': False}

    def get_rel_x_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.x_relative_improvements], 'yerr': [self.x_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}

    def get_rel_y_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.y_relative_improvements], 'yerr': [self.y_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}
    
    def get_rel_z_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.z_relative_improvements], 'yerr': [self.z_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}

    def save(self):

        #* Save x first
        img_address = get_project_root()+self.model_dir+'/figures/xVertexPerformance.png'
        d = self.get_x_dict()
        h_fig = make_plot(d)
        
        if self._reco_keys:
            h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
            d = self.get_rel_x_dict()
            d['subplot'] = True
            d['axhline'] = [0.0]
            h_fig = make_plot(d, h_figure=h_fig, position=[0.125, 0.11, 0.775, 0.15])
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)
        else:
            h_fig = make_plot(d)
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)

        #* Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'xVertexPerformance': wandb.Image(im, caption='xVertexPerformance')}, commit = False)

    
        #* Save y next
        img_address = get_project_root()+self.model_dir+'/figures/yVertexPerformance.png'
        d = self.get_y_dict()
        h_fig = make_plot(d)
        
        if self._reco_keys:
            h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
            d = self.get_rel_y_dict()
            d['subplot'] = True
            d['axhline'] = [0.0]
            h_fig = make_plot(d, h_figure=h_fig, position=[0.125, 0.11, 0.775, 0.15])
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)
        else:
            h_fig = make_plot(d)
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)

        #* Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'yVertexPerformance': wandb.Image(im, caption='yVertexPerformance')}, commit = False)
        

        #* Save z last
        img_address = get_project_root()+self.model_dir+'/figures/zVertexPerformance.png'
        d = self.get_z_dict()
        h_fig = make_plot(d)
        
        if self._reco_keys:
            h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
            d = self.get_rel_z_dict()
            d['subplot'] = True
            d['axhline'] = [0.0]
            h_fig = make_plot(d, h_figure=h_fig, position=[0.125, 0.11, 0.775, 0.15])
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)
        else:
            h_fig = make_plot(d)
            d_energy = self.get_energy_dict()
            d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)

        #* Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'zVertexPerformance': wandb.Image(im, caption='zVertexPerformance')}, commit = False)

        perf_savepath = get_project_root() + self.model_dir + '/data/VertexPerformance.pickle'
        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)

#* ======================================================================== 
#* PERFORMANCE FUNCTIONS
#* ========================================================================

def get_performance_plot_dicts(model_dir, plot_dicts):
    """Loads the performance classes associated with a certain regression and returns their plot dictionaries to plot and compare their performances.
    
    Arguments:
        model_dir {str} -- full or relative path to a model directory
        plot_dicts {list} -- empty or filled list with a plot dictionary for each performance plot
    
    Returns:
        list -- list with updated plot dictionaries
    """  

    _, _, _, meta_pars = load_model_pars(model_dir)

    if meta_pars['group'] == 'direction_reg':
        azipolar_path = get_project_root() + get_path_from_root(model_dir) + '/data/AziPolarPerformance.pickle'
        azipolar_class = pickle.load( open( azipolar_path, "rb" ) )
        
        azi_dict = azipolar_class.get_azi_dict()
        azi_dict['label'] = [Path(model_dir).stem]

        polar_dict = azipolar_class.get_polar_dict()
        polar_dict['label'] = [Path(model_dir).stem]

        if len(plot_dicts) == 0:
            azi_dict['title'] = 'Azimuthal Performance'
            azi_dict['grid'] = True
            plot_dicts.append(azi_dict)
            
            polar_dict['title'] = 'Polar Performance'
            polar_dict['grid'] = True
            plot_dicts.append(polar_dict)
        
        else:
            
            for key, item in plot_dicts[0].items():
                # * Only add to the lists - labels remain the same - this is what the try/except catches
                try:
                    plot_dicts[0][key].append(azi_dict[key][0])
                except AttributeError:
                    pass

            for key, item in plot_dicts[1].items():
                # * Only add to the lists - labels remain the same - this is what the try/except catches
                try:
                    plot_dicts[1][key].append(polar_dict[key][0])
                except AttributeError:
                    pass

    return plot_dicts
        
def log_operation_plots(model_dir, wandb_ID=None):
    """Searches in the model_dir for pickle-files containing epochs, lr, validation- and training error and saves plots the aforementioned quantities as a function of epoch to W&B (if a unique ID is supplied) and locally.
    
    Arguments:
        model_dir {str} -- Absolute or relative path to the model directory.
    
    Keyword Arguments:
        wandb_ID {str} -- If wanted, the unique wandb-ID can be supplied to log to W&B (default: {None})
    """    
    
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
    """Creates and logs performance plots relevant to the regression model by calling special classes
    
    Arguments:
        model_dir {str} -- Absolute or relative path to the model directory.
    
    Keyword Arguments:
        wandb_ID {str} -- If wanted, the unique wandb-ID can be supplied to log to W&B (default: {None})
    """    
    
    _, _, _, meta_pars = load_model_pars(model_dir)
    

    print(strftime("%d/%m %H:%M", localtime()), ': Logging plots...')
    if meta_pars['group'] == 'direction_reg':

        #* Plot and save azi- and polarplots
        azi_polar = AziPolarHists(model_dir, wandb_ID=wandb_ID)
        azi_polar.save()

        azi_polar_perf = AziPolarPerformance(model_dir, wandb_ID=wandb_ID)
        azi_polar_perf.save()

        #* Plot and save directional error performance
        perf = DirErrorPerformance(model_dir, wandb_ID=wandb_ID)
        perf.save()

    elif meta_pars['group'] == 'vertex_reg':
        vertex_perf = VertexPerformance(model_dir, wandb_ID=wandb_ID)
        vertex_perf.save()
    else:
        print('Unknown regression type - no plots have been produced.')
    print(strftime("%d/%m %H:%M", localtime()), ': Logging finished!')

def make_plot(plot_dict, h_figure=None, axes_index=None, position=[0.125, 0.11, 0.775, 0.77]):
    """A custom plot function using PyPlot. If 'x' AND 'y' are in plot_dict, a xy-graph is returned, if 'data' is given, a histogram is returned.
    
    Arguments:
        plot_dict {dictionary} -- a dictionary with custom keywords, see each plotting function.
    
    Keyword Arguments:
        h_figure {plt.figure_handle} -- a figure handle to plot more stuff to (default: {None})
        axes_index {int} -- which axis to plot on (in case of several axes on one figure) (default: {None})
        position {list} -- the position of the plot as [x_left_lower_corner, y_left_lower_corner, width, height] (default: {[0.125, 0.11, 0.775, 0.77]})
    
    Raises:
        ValueError: If unknown plot wanted
    
    Returns:
        [plt.figure_handle] -- handle to figure.
    """    
    
    plt.style.use('default')
    alpha = 0.25
    if 'grid' in plot_dict:
        grid_on = plot_dict['grid']
    else:
        grid_on = True

    if h_figure == None:

        h_figure = plt.figure()
        h_axis = h_figure.add_axes(position)

    if 'twinx' in plot_dict and h_figure != None:
        if plot_dict['twinx']:
            if axes_index == None:
                h_axis = h_figure.axes[0].twinx()
            else:
                h_axis = h_figure.axes[axes_index].twinx()
    
    if 'subplot' in plot_dict:
        h_axis = h_figure.add_axes(position)
    
    #* Make a xy-plot
    if 'x' in plot_dict and 'y' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])
        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])
        
        for i_set, dataset in enumerate(plot_dict['y']):
            plot_keys = ['label']
            #* Set baseline
            d = {'linewidth': 1.5}
            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            plt.plot(plot_dict['x'][i_set], dataset, **d)
            
            if 'xscale' in plot_dict: h_axis.set_xscale(plot_dict['xscale'])
            if 'yscale' in plot_dict: h_axis.set_yscale(plot_dict['yscale'])
            
        if 'label' in plot_dict: h_axis.legend()
        
    elif 'data' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])
        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])

        for i_set, data in enumerate(plot_dict['data']):
            
            plot_keys = ['label', 'alpha', 'density', 'bins', 'weights', 'histtype', 'log', 'color']
            
            #* Set baseline
            if len(plot_dict['data']) > 1:
                d = {'alpha': 0.6, 'density': False, 'bins': 'fd'}
            else:
                d = {'alpha': 1.0, 'density': False, 'bins': 'fd'}

            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            
            plt.hist(data, **d)

            if 'label' in plot_dict: h_axis.legend()
        
        if 'yscale' in plot_dict: h_axis.set_yscale(plot_dict['yscale'])
        if 'xscale' in plot_dict: h_axis.set_xscale(plot_dict['xscale'])

    elif 'hist2d' in plot_dict:

        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])

        set1 = plot_dict['hist2d'][0]
        set2 = plot_dict['hist2d'][1]

        if type(set1) == torch.Tensor: set1 = set1.cpu().numpy()
        if type(set2) == torch.Tensor: set2 = set2.cpu().numpy()

        #* Get bin-widths
        _, widths1 = np.histogram(set1, bins='fd')
        _, widths2 = np.histogram(set2, bins='fd')

        #* Rescale 
        widths1 = np.linspace(min(widths1), max(widths1), int(0.5 + widths1.shape[0]/4.0))
        widths2 = np.linspace(min(widths2), max(widths2), int(0.5 + widths2.shape[0]/4.0))
        plt.hist2d(set1, set2, bins = [widths1, widths2])
        plt.colorbar()

    elif 'hexbin' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])

        set1 = plot_dict['hexbin'][0]
        set2 = plot_dict['hexbin'][1]

        if type(set1) == torch.Tensor: set1 = set1.cpu().numpy()
        if type(set2) == torch.Tensor: set2 = set2.cpu().numpy()

        #* Get bin-widths - my attempt to modularize generation of 2d-histograms
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
            
        #* Rescale. Factor of 4 comes form trial and error
        n1 = int(0.5 + widths1.shape[0]/4.0)
        n2 = int(0.5 + widths2.shape[0]/4.0)
        plt.hexbin(set1, set2, gridsize = (n1, n2))
        plt.axis([xmin, xmax, ymin, ymax])
        plt.colorbar()
        
    elif 'edges' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])

        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])

        #* Calculate bin centers and 'x-error'
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
            #* Set baseline
            d = {'linewidth': 1.5}
            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            
            plt.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='.', **d)
            
        if 'xscale' in plot_dict: h_axis.set_xscale(plot_dict['xscale'])
        if 'yscale' in plot_dict: h_axis.set_yscale(plot_dict['yscale'])

        if 'label' in plot_dict: h_axis.legend()

    else:
        raise ValueError('Unknown plot wanted!')
    
    # * Plot vertical lines if wanted
    if 'axvline' in plot_dict:
        for vline in plot_dict['axvline']:
            h_axis.axvline(x=vline, color = 'k', ls = ':')
    
    # * ... And horizontal lines
    if 'axhline' in plot_dict:
        for hline in plot_dict['axhline']:
            h_axis.axhline(y=hline, color = 'k', ls = '--')

    if grid_on:
        h_axis.grid(alpha=alpha)

    if 'text' in plot_dict:
        plt.text(*plot_dict['text'], transform=h_axis.transAxes)
    
    if 'title' in plot_dict:
        plt.title(plot_dict['title'])

    if 'savefig' in plot_dict: 
        h_figure.savefig(plot_dict['savefig'])
        print('\nFigure saved at:')
        print(plot_dict['savefig'])
    

    return h_figure

def summarize_model_performance(model_dir, wandb_ID=None):
    """Summarizes a model's performance with a single number by updating the meta_pars-dictionary of the experiment.
    
    Arguments:
        model_dir {str} -- full or relative path to the model directory
    
    Keyword Arguments:
        wandb_ID {str} -- The wandb-ID for the specific experimented. Supplied if logging to wandb.com is wanted. (default: {None})
    """    
    
    _, _, _, meta_pars = load_model_pars(model_dir)

    if meta_pars['group'] == 'direction_reg':
        direrr_path = model_dir + '/data/DirErrorPerformance.pickle'
        direrrperf_class = pickle.load( open( direrr_path, "rb" ) )

        onenum_performance = direrrperf_class.median
    
    elif meta_pars['group'] == 'vertex_reg':
        vertex_err_path = model_dir + '/data/VertexPerformance.pickle'
        vertex_err_perf_class = pickle.load( open( vertex_err_path, "rb" ) )

        onenum_performance = vertex_err_perf_class.median_len_error
        
    else:
        print('\nNO ONE-NUMBER PERFORMANCE MEASURE DEFINED. RETURNING -1\n')
        onenum_performance = -1
    
    if wandb_ID is not None:
        wandb.config.update({'Performance': onenum_performance}, allow_val_change=True)

    meta_pars['performance'] = onenum_performance
    with open(model_dir+'/meta_pars.json', 'w') as fp:
        json.dump(meta_pars, fp)
