
class AziPolarPerformance:
    """A class to create and save performance plots for azimuthal and polar predictions. If available, the relative improvement compared to Icecubes reconstruction is plotted aswell. Furthermore, the median of the directional error is logged as a one-number performance measure.  
    
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
        self.data_pars = data_pars
        self.meta_pars = meta_pars
        self.prefix = prefix
        self._energy_key = self._get_energy_key()
        self._reco_keys, self._true_xyz = self._get_reco_keys()
        self._pred_keys = self._get_pred_keys()

        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        pred_dict, true_dict = self._get_data_dicts()
        self._calculate(pred_dict, true_dict)
        self._save_onenum_perf(pred_dict)

    def _get_data_dicts(self):
        full_pred_address = self._get_pred_path()
        true_keys = self._energy_key + self._reco_keys + self._true_xyz
        pred_dict, true_dict = read_predicted_h5_data(full_pred_address, self._pred_keys, self.data_pars, true_keys)
        return pred_dict, true_dict

    def _get_energy_key(self):
        dataset_name = get_dataset_name(self.data_dir)

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
    
    def _get_pred_keys(self):
        funcs = get_eval_functions(self.meta_pars)
        keys = []

        for func in funcs:
            keys.append(func.__name__)
        return keys

    def _get_reco_keys(self):
        dataset_name = get_dataset_name(self.data_dir)

        if dataset_name == 'MuonGun_Level2_139008':
            reco_keys = None
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            reco_keys = ['retro_crs_prefit_azimuth', 'retro_crs_prefit_zenith']
            true_xyz = ['true_primary_direction_x', 'true_primary_direction_y',  'true_primary_direction_z']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return reco_keys, true_xyz

    def _calculate(self, pred_dict, true_dict):
        #* Transform back and extract values into list
        # * Our predictions have already been converted in predict()
        true_transformed = inverse_transform(true_dict, get_project_root() + self.model_dir)
        energy = convert_to_proper_list(true_transformed[self._energy_key[0]])
        self.counts, self.bin_edges = np.histogram(energy, bins=N_BINS_PERF_PLOTS)
        
        polar_error = pred_dict['polar_error']
        print('\nCalculating polar performance...')
        self.polar_sigmas, self.polar_errors = calc_perf2_as_fn_of_energy(energy, polar_error, self.bin_edges)
        print('Calculation finished!')

        azi_error = pred_dict['azi_error']
        print('\nCalculating azimuthal performance...')
        self.azi_sigmas, self.azi_errors = calc_perf2_as_fn_of_energy(energy, azi_error, self.bin_edges)
        print('Calculation finished!')

        # * If an I3-reconstruction exists, get it
        if self._reco_keys:
            # * Ensure keys are proper so the angle calculations work
            # * Our predictions have already been converted in predict()
            pred_crs = {key: true_transformed[key] for key in self._reco_keys}
            pred_crs = convert_keys(pred_crs, [key for key in pred_crs], ['azi', 'zen'])
            
            true = {key: true_transformed[key] for key in self._true_xyz}
            true = convert_keys(true, [key for key in true], ['x', 'y', 'z'])

            azi_crs_error = get_retro_crs_prefit_azi_error(pred_crs, true)
            polar_crs_error = get_retro_crs_prefit_polar_error(pred_crs, true)

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

    def _save_onenum_perf(self, pred_dict):

        self.median_direrr = np.nanpercentile(pred_dict['directional_error'], 50)

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

class EnergyPerformance:
    """A class to create and save performance plots for energy predictions. If available, the relative improvement compared to Icecubes reconstruction is plotted aswell. Furthermore, the median of the relative error is logged as a one-number performance measure.  
    
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
        self.data_pars = data_pars
        self.meta_pars = meta_pars
        self.prefix = prefix
        self._energy_key = self._get_energy_key()
        self._reco_keys= self._get_reco_keys()
        self._pred_keys = self._get_pred_keys()

        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        pred_dict, true_dict = self._get_data_dicts()
        self._calculate(pred_dict, true_dict)
        self._save_onenum_perf(pred_dict)

    def _get_data_dicts(self):
        full_pred_address = self._get_pred_path()
        true_keys = self._energy_key + self._reco_keys
        if self.data_pars['dataloader'] == 'PickleLoader':
            pred_dict, true_dict = read_pickle_predicted_h5_data(full_pred_address, self._pred_keys, self.data_pars, true_keys)
        else:
            pred_dict, true_dict = read_predicted_h5_data(full_pred_address, self._pred_keys, self.data_pars, true_keys)
        return pred_dict, true_dict

    def _get_energy_key(self):
        dataset_name = get_dataset_name(self.data_dir)

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
    
    def _get_pred_keys(self):
        funcs = get_eval_functions(self.meta_pars)
        keys = []

        for func in funcs:
            keys.append(func.__name__)
        return keys

    def _get_reco_keys(self):
        dataset_name = get_dataset_name(self.data_dir)

        if dataset_name == 'MuonGun_Level2_139008':
            reco_keys = None
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            reco_keys = ['retro_crs_prefit_energy']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return reco_keys

    def _calculate(self, pred_dict, true_dict):

        #* Transform back and extract values into list
        # * Our predictions have already been converted in predict()
        true_transformed = inverse_transform(true_dict, get_project_root() + self.model_dir)
        energy = convert_to_proper_list(true_transformed[self._energy_key[0]])
        self.counts, self.bin_edges = np.histogram(energy, bins=N_BINS_PERF_PLOTS)
        
        relE_error = pred_dict['relative_E_error']
        print('\nCalculating energy performance...')
        self.relE_sigmas, self.relE_errors = calc_perf2_as_fn_of_energy(energy, relE_error, self.bin_edges)
        print('Calculation finished!')

        # * If an I3-reconstruction exists, get it
        if self._reco_keys:
            # * Ensure keys are proper so the angle calculations work
            # * Our predictions have already been converted in predict()
            pred_crs = {key: true_transformed[key] for key in self._reco_keys}
            pred_crs = convert_keys(pred_crs, [key for key in pred_crs], ['E'])
            
            true = {key: true_transformed[key] for key in self._energy_key}
            true = convert_keys(true, [key for key in true], ['logE'])

            relE_crs_error = get_retro_crs_prefit_relE_error(pred_crs, true)

            print('\nCalculating crs energy performance...')
            self.relE_crs_sigmas, self.relE_crs_errors = calc_perf2_as_fn_of_energy(energy, relE_crs_error, self.bin_edges)
            print('Calculation finished!')

            #* Calculate the relative improvement - e_diff/retro_error. Report decrease in error as a positive result
            a, b = calc_relative_error(self.relE_crs_sigmas, self.relE_sigmas, self.relE_crs_errors, self.relE_errors)
            self.relE_relative_improvements, self.relE_sigma_improvements = -a, b
        else:
            self.relE_relative_improvements = None
            self.relE_sigma_improvements = None

    def _save_onenum_perf(self, pred_dict):
        # * Report the average sigma.
        avg = np.nansum(self.counts*self.relE_sigmas)/np.nansum(self.counts)
        self.onenum_performance = avg

    def get_energy_dict(self):
        return {'data': [self.bin_edges[:-1]], 'bins': [self.bin_edges], 'weights': [self.counts], 'histtype': ['step'], 'log': [True], 'color': ['lightgray'], 'twinx': True, 'grid': False, 'ylabel': 'Events'}
    
    def get_relE_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.relE_sigmas], 'yerr': [self.relE_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Relative Error', 'grid': False}
    
    def get_rel_relE_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.relE_relative_improvements], 'yerr': [self.relE_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': False}

    def save(self):

        #* Save Azi first
        perf_savepath = get_project_root()+self.model_dir+'/data/EnergyPerformance.pickle'
        img_address = get_project_root()+self.model_dir+'/figures/EnergyPerformance.png'
        d = self.get_relE_dict()
        h_fig = make_plot(d)
        
        if self._reco_keys:
            h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
            d = self.get_rel_relE_dict()
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
            wandb.log({'EnergyPerformance': wandb.Image(im, caption='EnergyPerformance')}, commit=False)
        
        # * Save the class instance
        perf_savepath = get_project_root() + self.model_dir + '/data/EnergyPerformance.pickle'
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
        errors = {'x_error': np.array([]), 'y_error': np.array([]), 'z_error': np.array([]), 't_error': np.array([])}
        for true_key, pred_key, error_key in zip(true, pred, errors):
            errors[error_key] = pred[pred_key]-true[true_key]
        
        # * Bin and save
        self.x_count, self.x_edges = np.histogram(errors['x_error'], bins='fd')
        self.y_count, self.y_edges = np.histogram(errors['y_error'], bins='fd')
        self.z_count, self.z_edges = np.histogram(errors['z_error'], bins='fd')
        self.t_count, self.t_edges = np.histogram(errors['t_error'], bins='fd')

    
    def get_x_dict(self):
        d = {'data': [self.x_edges[:-1]], 'bins': [self.x_edges], 'weights': [self.x_count]}
        return d
    
    def get_y_dict(self):
        d = {'data': [self.y_edges[:-1]], 'bins': [self.y_edges], 'weights': [self.y_count]}
        return d
    
    def get_z_dict(self):
        d = {'data': [self.z_edges[:-1]], 'bins': [self.z_edges], 'weights': [self.z_count]}
        return d
    
    def get_t_dict(self):
        d = {'data': [self.t_edges[:-1]], 'bins': [self.t_edges], 'weights': [self.t_count]}
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
            energy = read_h5_directory(self.data_dir, ['true_neutrino_energy'], self.prefix, from_frac=self.from_frac, to_frac=self.to_frac, n_wanted=self.data_pars.get('n_predictions_wanted', np.inf))
        except KeyError:
            energy = read_h5_directory(self.data_dir, ['true_muon_energy'], self.prefix, from_frac=self.from_frac, to_frac=self.to_frac, n_wanted=self.data_pars.get('n_predictions_wanted', np.inf))
        
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

        # * Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'DirErrorPerformance': wandb.Image(im, caption='DirErrorPerformance')}, commit = False)

        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)

class VertexPerformance_old:
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
        
        self._energy_key = self._get_energy_key()
        self._pred_keys = self._get_prediction_keys()
        self._reco_keys = self._get_reco_keys()
        self._true_xyzt_keys = get_target_keys(data_pars, meta_pars)

        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        pred_dict, true_dict = self._get_data_dicts()
        self._create_performance_plots(pred_dict, true_dict)
        self._calculate_onenum_performance(pred_dict)

    def _get_data_dicts(self):
        full_pred_address = self._get_pred_path()
        true_keys = self._energy_key + self._reco_keys + self._true_xyzt_keys
        if self.data_pars['dataloader'] == 'PickleLoader':
            pred_dict, true_dict = read_pickle_predicted_h5_data(full_pred_address, self._pred_keys, self.data_pars, true_keys)
        else:
            pred_dict, true_dict = read_predicted_h5_data(full_pred_address, self._pred_keys, self.data_pars, true_keys)
        return pred_dict, true_dict

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
    
    def _get_prediction_keys(self):
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
            if self.meta_pars['group'] == 'vertex_reg':
                reco_keys = ['retro_crs_prefit_x', 'retro_crs_prefit_y', 'retro_crs_prefit_z', 'retro_crs_prefit_time']
            elif self.meta_pars['group'] == 'vertex_reg_no_time':
                reco_keys = ['retro_crs_prefit_x', 'retro_crs_prefit_y', 'retro_crs_prefit_z']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return reco_keys
    
    def _create_performance_plots(self, pred_dict, true_dict):
        # * Transform back and extract values into list
        true_transformed = inverse_transform(true_dict, get_project_root() + self.model_dir)
        energy = convert_to_proper_list(true_transformed[self._energy_key[0]])
    
        self.counts, self.bin_edges = np.histogram(energy, bins=N_BINS_PERF_PLOTS)
        
        x_error = pred_dict['vertex_x_error']
        print('\nCalculating x performance...')
        self.x_sigmas, self.x_errors = calc_perf2_as_fn_of_energy(energy, x_error, self.bin_edges)
        print('Calculation finished!')

        y_error = pred_dict['vertex_y_error']
        print('\nCalculating y performance...')
        self.y_sigmas, self.y_errors = calc_perf2_as_fn_of_energy(energy, y_error, self.bin_edges)
        print('Calculation finished!')

        z_error = pred_dict['vertex_z_error']
        print('\nCalculating z performance...')
        self.z_sigmas, self.z_errors = calc_perf2_as_fn_of_energy(energy, z_error, self.bin_edges)
        print('Calculation finished!')

        try:
            t_error = pred_dict['vertex_t_error']
            print('\nCalculating time performance...')
            self.t_sigmas, self.t_errors = calc_perf2_as_fn_of_energy(energy, t_error, self.bin_edges)
            print('Calculation finished!')
        except KeyError:
            pass

        # * If an I3-reconstruction exists, get it
        if self._reco_keys:
            # * Ensure keys are proper so the angle calculations work
            pred_crs = {key: true_transformed[key] for key in self._reco_keys}
            if self.meta_pars['group'] == 'vertex_reg':
                vertex_keys = ['x', 'y', 'z', 't']
            elif self.meta_pars['group'] == 'vertex_reg_no_time':
                vertex_keys = ['x', 'y', 'z']

            pred_crs = convert_keys(pred_crs, self._reco_keys, vertex_keys)

            true = {key: true_transformed[key] for key in self._true_xyzt_keys}
            true = convert_keys(true, self._true_xyzt_keys, vertex_keys)
            true = { key: convert_to_proper_list(item) for key, item in true.items() }

            x_crs_error = vertex_x_error(pred_crs, true)
            y_crs_error = vertex_y_error(pred_crs, true)
            z_crs_error = vertex_z_error(pred_crs, true)
            try:
                t_crs_error = vertex_t_error(pred_crs, true)
            except KeyError:
                pass

            print('\nCalculating crs x performance...')
            self.x_crs_sigmas, self.x_crs_errors = calc_perf2_as_fn_of_energy(energy, x_crs_error, self.bin_edges)
            print('Calculation finished!')

            print('\nCalculating crs y performance...')
            self.y_crs_sigmas, self.y_crs_errors = calc_perf2_as_fn_of_energy(energy, y_crs_error, self.bin_edges)
            print('Calculation finished!')

            print('\nCalculating crs z performance...')
            self.z_crs_sigmas, self.z_crs_errors = calc_perf2_as_fn_of_energy(energy, z_crs_error, self.bin_edges)
            print('Calculation finished!')
            
            try:
                print('\nCalculating crs time performance...')
                self.t_crs_sigmas, self.t_crs_errors = calc_perf2_as_fn_of_energy(energy, t_crs_error, self.bin_edges)
                print('Calculation finished!')
            except UnboundLocalError:
                pass

            # * Calculate the relative improvement - e_diff/I3_error. Report decrease in error as a positive result
            rel_e, sigma_rel = calc_relative_error(self.x_crs_sigmas, self.x_sigmas, self.x_crs_errors, self.x_errors)
            self.x_relative_improvements, self.x_sigma_improvements = -rel_e, sigma_rel

            rel_e, sigma_rel = calc_relative_error(self.y_crs_sigmas, self.y_sigmas, self.y_crs_errors, self.y_errors)
            self.y_relative_improvements, self.y_sigma_improvements = -rel_e, sigma_rel

            rel_e, sigma_rel = calc_relative_error(self.z_crs_sigmas, self.z_sigmas, self.z_crs_errors, self.z_errors)
            self.z_relative_improvements, self.z_sigma_improvements = -rel_e, sigma_rel

            try:
                rel_e, sigma_rel = calc_relative_error(self.t_crs_sigmas, self.t_sigmas, self.t_crs_errors, self.t_errors)
                self.t_relative_improvements, self.t_sigma_improvements = -rel_e, sigma_rel
            except AttributeError:
                pass
        
        else:
            self.x_relative_improvements = None
            self.x_sigma_improvements = None
            self.y_relative_improvements = None
            self.y_sigma_improvements = None
            self.z_relative_improvements = None
            self.z_sigma_improvements = None
            self.t_relative_improvements = None
            self.t_sigma_improvements = None
    
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
    def get_t_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.t_sigmas], 'yerr': [self.t_errors], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Error [ns]', 'grid': False}

    def get_rel_x_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.x_relative_improvements], 'yerr': [self.x_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': True, 'y_minor_ticks_multiple': 0.2}
    def get_rel_y_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.y_relative_improvements], 'yerr': [self.y_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': True, 'y_minor_ticks_multiple': 0.2}
    def get_rel_z_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.z_relative_improvements], 'yerr': [self.z_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': True, 'y_minor_ticks_multiple': 0.2}
    def get_rel_t_dict(self):
        return {'edges': [self.bin_edges], 'y': [self.t_relative_improvements], 'yerr': [self.t_sigma_improvements], 'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': True, 'y_minor_ticks_multiple': 0.2}

    def save(self):

        # * Save x first
        img_address = get_project_root()+self.model_dir+'/figures/xVertexPerformance.png'
        d = self.get_x_dict()
        
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
            # d_energy['savefig'] = img_address
            _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)

        #* Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({'xVertexPerformance': wandb.Image(im, caption='xVertexPerformance')}, commit = False)

    
        # * Save y next
        img_address = get_project_root()+self.model_dir+'/figures/yVertexPerformance.png'
        d = self.get_y_dict()
        
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
        
        # * Save z
        img_address = get_project_root()+self.model_dir+'/figures/zVertexPerformance.png'
        d = self.get_z_dict()
        
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
        

        # * Save time last
        try:
            img_address = get_project_root()+self.model_dir+'/figures/tVertexPerformance.png'
            d = self.get_t_dict()
            
            if self._reco_keys:
                h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
                d = self.get_rel_t_dict()
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
                wandb.log({'tVertexPerformance': wandb.Image(im, caption='tVertexPerformance')}, commit = False)
        except AttributeError:
            pass

        perf_savepath = get_project_root() + self.model_dir + '/data/VertexPerformance.pickle'
        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)
