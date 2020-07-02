
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

class Performance_pickle:
    """A class to create and save performance plots for interaction vertex 
    predictions. 
    
    If available, the relative improvement compared to Icecubes 
    reconstruction is plotted aswell. A one-number performance summary is saved 
    as the median of the total vertex distance error.     
    
    Raises:
        KeyError: If an unknown dataset is encountered.
    
    Returns:
        [type] -- Instance of class.
    """    

    def __init__(self, model_dir, wandb_ID=None):
        hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(model_dir)
        # prefix = 'transform'+str(data_pars['file_keys']['transform'])
        # from_frac = data_pars['train_frac']
        # to_frac = data_pars['train_frac'] + data_pars['val_frac']

        self.model_dir = get_path_from_root(model_dir)
        self.data_pars = data_pars
        self.meta_pars = meta_pars
        self.keyword = meta_pars['group']
        # self.prefix = prefix
        self.dom_mask = data_pars['dom_mask']
        self.loss_func = arch_pars['loss_func']
        
        self._energy_key = self._get_energy_key()
        self._pred_keys = self._get_prediction_keys()
        self._reco_keys = self._get_reco_keys()
        # There should be equally many performance keys and icecube perf keys
        self._performance_keys = self._get_performance_keys()
        self._icecube_perf_keys = self._get_icecube_perf_keys()
        self._true_keys = get_target_keys(data_pars, meta_pars)
        self._conf_interval_keys = self._get_conf_interval_keys()

        # self.from_frac = from_frac
        # self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        energy_dict, pred_dict, crs_dict, true_dict, n_doms, loss = self._get_data_dicts()
        self._calculate_performance(energy_dict, pred_dict, crs_dict, true_dict, n_doms)
        self.onenumber_performance = self._calculate_onenum_performance(pred_dict)
        self.loss_error_correlations = self._calculate_loss_error_corrs(pred_dict, loss)

    def _calculate_loss_error_corrs(self, pred_d, loss):

        # Correlation between loss and predicted values is calculated to 
        # hopefully see if loss function is focused too heavily on some of the variables
        # We look at ABSOLUTE value of error, since we are using symmetric
        # loss functions.
        corrs = {}
        loss_clip = self._get_loss_clip_vals()
        loss_clipped = np.clip(loss, loss_clip[0], loss_clip[1])
        for key, data in pred_d.items():
            # Clip values, since it's concentrated at a certain range.
            _, clip_vals = self._get_I3_2D_clip_and_d(key)
            
            # Strip NaNs
            n_nans, data, loss_stripped = strip_nans(data, loss)

            # corrcoef returns correlation matrix - we are only interested 
            # in one of the numbers
            abs_data = np.abs(np.clip(data, clip_vals[0], clip_vals[1]))
            
            print('WARNING: %d NAN(S) FOUND IN LOSS-ERROR CORRELATIONS!'%(n_nans)) if n_nans>0 else None
            corrs[key] = np.corrcoef(abs_data, loss_stripped)[1, 0]

            # Make 2d-scatterplot or HEATMAP! like i3-performance
            d = {}
            title = r'%s - loss correlation'%(key)
            savepath = get_project_root()+self.model_dir+'/figures/'+key+'_LossCorr.png'
            d['hist2d'] = [loss_stripped, abs_data]
            d['xlabel'] = 'Loss value'
            d['ylabel'] = 'Abs(%s)'%(key)
            d['savefig'] = savepath
            d['title'] = title
            fig_h = make_plot(d)

        # Make barh-plots
        bar_pos = np.arange(0, len(corrs)) + 0.5
        names = [key for key in corrs]
        data = [data for key, data in corrs.items()]
        d = {'keyword': 'barh', 'y': bar_pos, 'width': data, 'height': 0.7, 'names': names}
        d['title'] = 'Correlation coefficients w.r.t. %s-loss'%(self.loss_func)
        d['xlabel'] = 'Correlation'
        d['savefig'] = get_project_root()+self.model_dir+'/figures/CorrCoeff.png'
        fig = make_plot(d)

        # Save a histogram of loss values aswell
        d = {'data': [loss]}
        d['xlabel'] = 'Loss value (%s)'%(self.loss_func)
        d['ylabel'] = 'Count'
        d['log'] = [True]
        d['savefig'] = get_project_root()+self.model_dir+'/figures/loss_vals.png'
        _ = make_plot(d)
        plt.close('all')

        return corrs

    def _calculate_onenum_performance(self, data_dict=None):
        # Performance for each variable in names is calculated as the 
        # geometric mean of the bin-values. Finally, they are combined as
        # a geometric mean of all the geometric means. This has the
        # advantage that a fractional decrease of x percent in any of 
        # them contributes equally to the final performance
        
        if self.meta_pars['group'] == 'vertex_reg':
            names = ['len_error_68th', 'vertex_t_error_sigma']
        if self.meta_pars['group'] == 'vertex_reg_no_time':
            names = ['len_error_68th']
        elif self.meta_pars['group'] == 'direction_reg':
            names = ['directional_error_68th']
        elif self.meta_pars['group'] == 'energy_reg':
            names = ['log_frac_E_error_sigma']
        elif self.meta_pars['group'] == 'full_reg':
            names = ['len_error_68th', 'directional_error_68th',
                    'vertex_t_error_sigma', 'log_frac_E_error_sigma']
        
        errors = []
        for name in names:
            data = getattr(self, name)
            errors.append(calc_geometric_mean(data))
        one_num = calc_geometric_mean(errors)

        return one_num

    def _calculate_performance(self, energy_dict, pred_dict, crs_dict, true_dict, n_doms):
        
        # Transform back and extract values into list
        true_transformed = inverse_transform(true_dict, 
                                get_project_root()+self.model_dir)
        energy_transformed = inverse_transform(energy_dict, 
                                get_project_root()+self.model_dir)
        # We want energy as array
        energy = np.array(convert_to_proper_list(energy_transformed[self._energy_key[0]]))

        self.counts, self.bin_edges = np.histogram(energy, 
                                                bins=N_BINS_PERF_PLOTS)
        self.bin_centers = calc_bin_centers(self.bin_edges)
        
        self.dom_counts, self.dom_bin_edges = np.histogram(n_doms, 
                                                bins=N_BINS_PERF_PLOTS)
        self.dom_bin_centers = calc_bin_centers(self.dom_bin_edges)
        
        # Calculate how well Icecube does.
        for name, func, model_key in zip(self._icecube_perf_keys, self._get_error_funcs(), self._performance_keys):
            
            # We calculate 2 different forms of performance measures: One
            # kind as the width of the distribution (for unbounded
            # distributions), another as the upper 68th percentile 
            # of the distribution (for distributions with a lower bound of 0)
            print(get_time(), 'Calculating %s performance...'%(name))

            # Performance as a function of energy
            error = func(crs_dict, true_transformed, reporting=True)

            # # Make Wilcoxon signed-rank test
            # print('')
            # print(get_time(), 'Making Wilcoxon signed-rank test')
            # print(type(error), type(pred_dict[model_key]))
            # diff = np.array(error)-pred_dict[model_key]
            # test_statistic, p_val = wilcoxon(diff)
            # print(p_val) 
            # print('')
            if name not in self._conf_interval_keys:
                
                sigma, sigmaerr, median, upper_perc, lower_perc =\
                    calc_width_as_fn_of_data(energy, error, self.bin_edges)
               
                setattr(self, name+'_sigma', sigma)
                setattr(self, name+'_sigmaerr', sigmaerr)
                setattr(self, name+'_50th', median)
                setattr(self, name+'_84th', upper_perc)
                setattr(self, name+'_16th', lower_perc)

                # Performance as a function of number of doms
                sigma, sigmaerr, median, upper_perc, lower_perc =\
                    calc_width_as_fn_of_data(n_doms, error, self.dom_bin_edges)
                setattr(self, 'dom_'+name+'_sigma', sigma)
                setattr(self, 'dom_'+name+'_sigmaerr', sigmaerr)
                setattr(self, 'dom_'+name+'_50th', median)
                setattr(self, 'dom_'+name+'_84th', upper_perc)
                setattr(self, 'dom_'+name+'_16th', lower_perc)
            
            else:
                percentiles = [68, 50, 32]
                
                # As a function of energy
                vals, errs = calc_percentiles_as_fn_of_data(
                    energy, error, self.bin_edges, percentiles)
                
                for perc, val, err, in zip(percentiles, vals, errs):
                    setattr(self, '%s_%sth'%(name, str(perc)), val)
                    setattr(self, '%s_err%sth'%(name, str(perc)), err)

                # As a function of DOMs
                vals, errs = calc_percentiles_as_fn_of_data(
                    n_doms, error, self.dom_bin_edges, percentiles)
                
                for perc, val, err, in zip(percentiles, vals, errs):
                    setattr(self, 'dom_%s_%sth'%(name, str(perc)), val)
                    setattr(self, 'dom_%s_err%sth'%(name, str(perc)), err)
            
            print(get_time(), 'Calculation finished!')
            print('')
                
        # Calculate performance for our predictions
        for (key, data), i3_key in zip(pred_dict.items(), self._icecube_perf_keys):
            
            print(get_time(), 'Calculating %s performance...'%(key))
            
            # Same split as above
            if key not in self._conf_interval_keys:
            
                # Performance as a function of energy
                sigma, sigmaerr, median, upper_perc, lower_perc = calc_width_as_fn_of_data(energy, data, self.bin_edges)
                setattr(self, key+'_sigma', sigma)
                setattr(self, key+'_sigmaerr', sigmaerr)
                setattr(self, key+'_50th', median)
                setattr(self, key+'_84th', upper_perc)
                setattr(self, key+'_16th', lower_perc)

                # We make the I3-plot here so we do not have to save all 
                # the data. First we retrieve the corresponding Icecube data
                i3_med = getattr(self, i3_key+'_50th')
                i3_upper = getattr(self, i3_key+'_84th')
                i3_lower = getattr(self, i3_key+'_16th')
                self._make_I3_perf_plot(key, energy, data, median, upper_perc, lower_perc, i3_med=i3_med, i3_upper=i3_upper, i3_lower=i3_lower)

                # Performance as a function of number of doms
                sigma, sigmaerr, median, upper_perc, lower_perc = calc_width_as_fn_of_data(n_doms, data, self.dom_bin_edges)
                setattr(self, 'dom_'+key+'_sigma', sigma)
                setattr(self, 'dom_'+key+'_sigmaerr', sigmaerr)
                setattr(self, 'dom_'+key+'_50th', median)
                setattr(self, 'dom_'+key+'_84th', upper_perc)
                setattr(self, 'dom_'+key+'_16th', lower_perc)
                print('')
            
            else:
                percentiles = [68, 50, 32]
                
                # As a function of energy
                vals, errs = calc_percentiles_as_fn_of_data(
                    energy, data, self.bin_edges, percentiles)
                for perc, val, err, in zip(percentiles, vals, errs):
                    setattr(self, '%s_%sth'%(key, str(perc)), val)
                    setattr(self, '%s_err%sth'%(key, str(perc)), err)
                
                # As a function of DOMs
                vals, errs = calc_percentiles_as_fn_of_data(
                    n_doms, data, self.dom_bin_edges, percentiles)
                
                for perc, val, err, in zip(percentiles, vals, errs):
                    setattr(self, 'dom_%s_%sth'%(key, str(perc)), val)
                    setattr(self, 'dom_%s_err%sth'%(key, str(perc)), err)
            
            print(get_time(), 'Calculation finished!')

        # Calculate the relative improvement - e_diff/I3_error. Report decrease in error as a positive 
        for model_key, retro_key in zip(self._performance_keys, self._icecube_perf_keys):
            
            # Remember to split in conf bounds and widths
            if (model_key not in self._conf_interval_keys):
                retro_sigma = getattr(self, retro_key+'_sigma')
                model_sigma = getattr(self, model_key+'_sigma')
                retro_sigmaerr = getattr(self, retro_key+'_sigmaerr')
                model_sigmaerr = getattr(self, model_key+'_sigmaerr')

                rel_e, sigma_rel = calc_relative_error(retro_sigma, model_sigma, retro_sigmaerr, model_sigmaerr)
                setattr(self, model_key+'_RI', -rel_e)
                setattr(self, model_key+'_RIerr', sigma_rel)
            else:
                retro_sigma = getattr(self, retro_key+'_68th')
                model_sigma = getattr(self, model_key+'_68th')
                retro_sigmaerr = getattr(self, retro_key+'_err68th')
                model_sigmaerr = getattr(self, model_key+'_err68th')

                rel_e, sigma_rel = calc_relative_error(retro_sigma, model_sigma, retro_sigmaerr, model_sigmaerr)
                setattr(self, model_key+'_RI', -rel_e)
                setattr(self, model_key+'_RIerr', sigma_rel)

    def _get_conversion_keys_crs(self):
        
        if self.meta_pars['group'] == 'vertex_reg':
            keys = ['x_vertex', 'y_vertex', 'z_vertex', 't']
        
        elif self.meta_pars['group'] == 'vertex_reg_no_time':
            keys = ['x_vertex', 'y_vertex', 'z_vertex']
        
        elif self.meta_pars['group'] == 'direction_reg':
            keys = ['azi', 'zen']
        
        elif self.meta_pars['group'] == 'energy_reg':
            keys = ['E']
        
        elif self.meta_pars['group'] == 'full_reg':
            keys = ['E', 'x_vertex', 'y_vertex', 'z_vertex', 't', 'azi', 'zen']

        else:
            raise KeyError('PerformanceClass: Unknown regression type encountered!')
        
        return keys
    
    def _get_conversion_keys_true(self):
        
        if self.meta_pars['group'] == 'vertex_reg':
            keys = ['x_vertex', 'y_vertex', 'z_vertex', 't']
        
        elif self.meta_pars['group'] == 'vertex_reg_no_time':
            keys = ['x_vertex', 'y_vertex', 'z_vertex']
        
        elif self.meta_pars['group'] == 'direction_reg':
            keys = ['x_dir', 'y_dir', 'z_dir']
        
        elif self.meta_pars['group'] == 'energy_reg':
            keys = ['logE']
        
        elif self.meta_pars['group'] == 'full_reg':
            keys = ['logE', 'x_vertex', 'y_vertex', 'z_vertex', 't', 'x_dir', 'y_dir', 'z_dir']
        
        else:
            raise KeyError('PerformanceClass: Unknown regression type encountered!')
        
        return keys
    
    def _get_data_dicts(self):
        full_pred_address = self._get_pred_path()
        data_dir = self.data_pars['data_dir']
        prefix = 'transform'+str(self.data_pars['file_keys']['transform'])

        # Load loss aswell
        keys = self._pred_keys+['loss']
        
        print(get_time(), 'Loading predictions...')
        pred_dict = read_pickle_predicted_h5_data_v2(full_pred_address, keys)
        energy_dict = read_pickle_data(data_dir, pred_dict['indices'], self._energy_key, prefix=prefix)
        crs_dict = read_pickle_data(data_dir, pred_dict['indices'], self._reco_keys, prefix=prefix)
        true_dict = read_pickle_data(data_dir, pred_dict['indices'], self._true_keys, prefix=prefix)
        print(get_time(), 'Predictions loaded!')

        # Pop loss from dict
        loss = pred_dict['loss']
        del pred_dict['loss']

        print(get_time(), 'Finding number of DOMs in events')
        n_doms = get_n_doms(pred_dict['indices'], self.dom_mask, data_dir)
        # Indices have done, what we wanted them to do - delete them
        print(get_time(), 'Number of DOMs found!')
        print('')
        del pred_dict['indices']

        return energy_dict, pred_dict, crs_dict, true_dict, n_doms, loss
    
    def _get_dom_dict(self):
        d = {'data': [self.dom_bin_edges[:-1]], 'bins': [self.dom_bin_edges], 
        'weights': [self.dom_counts], 'histtype': ['step'], 'log': [True], 
        'color': ['gray'], 'twinx': True, 'grid': False, 'ylabel': 'Events'}
        return d

    def _get_energy_dict(self):
        d = {'data': [self.bin_edges[:-1]], 'bins': [self.bin_edges], 
        'weights': [self.counts], 'histtype': ['step'], 'log': [True], 
        'color': ['gray'], 'twinx': True, 'grid': False, 'ylabel': 'Events'}
        return d
    
    def _get_energy_key(self):
        dataset_name = get_dataset_name(self.data_pars['data_dir'])

        if dataset_name == 'MuonGun_Level2_139008':
            energy_key = ['true_muon_energy']
        elif dataset_name == 'oscnext-genie-level5-v01-01-pass2':
            energy_key = ['true_primary_energy']
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return energy_key
    
    def _get_error_funcs(self):

        if self.meta_pars['group'] == 'vertex_reg':
            funcs = [retro_x_error, retro_y_error, retro_z_error,
                    retro_t_error, retro_len_error]
        
        elif self.meta_pars['group'] == 'vertex_reg_no_time':
            funcs = [retro_x_error, retro_y_error, retro_z_error,
                    retro_len_error]
        
        elif self.meta_pars['group'] == 'direction_reg':
            funcs = [retro_azi_error, retro_polar_error, 
                    retro_directional_error]
        
        elif self.meta_pars['group'] == 'energy_reg':
            funcs = [retro_relE_error, retro_log_frac_E_error]

        elif self.meta_pars['group'] == 'full_reg':
            funcs = [retro_relE_error, retro_log_frac_E_error, retro_x_error, 
                    retro_y_error, retro_z_error, retro_t_error,
                    retro_len_error, retro_azi_error, retro_polar_error,
                    retro_directional_error]
        
        else:
            raise KeyError('PerformanceClass: Unknown regression type encountered!')
        
        return funcs

    def _get_I3_2D_clip_and_d(self, key):
        d2, clip_vals = {}, [-np.inf, np.inf]

        if key == 'relative_E_error':
            clip_vals = [-4.0, 4.0]
            d2['ylabel'] = r'$(\mathrm{E}_{reco}-\mathrm{E}_{true})/\mathrm{E}_{true}$ [%]'
            d2['title'] = 'Model Energy reco. results'

        elif key == 'log_frac_E_error':
            clip_vals = [-1.0, 1.0]
            d2['ylabel'] = r'$\log_{10} \left( \frac{E_{pred}}{E_{true}} \right)$'
            # d2['ylabel'] = r'$\text{Width}\left( \log_{10} E_{pred} - \log_{10} E_{true} \right)$'

            d2['title'] = 'Model Energy reco. results'

        elif key == 'vertex_x_error':
            clip_vals = [-80.0, 80.0]
            d2['ylabel'] = 'Error [m]'
            d2['title'] = 'Model vertex x reco. results'

        elif key == 'vertex_y_error':
            clip_vals = [-80.0, 80.0]
            d2['ylabel'] = 'Error [m]'
            d2['title'] = 'Model vertex y reco. results'

        elif key == 'vertex_z_error':
            clip_vals = [-80.0, 80.0]
            d2['ylabel'] = 'Error [m]'
            d2['title'] = 'Model vertex z reco. results'
        
        elif key == 'vertex_t_error':
            clip_vals = [-200.0, 200.0]
            d2['ylabel'] = 'Error [ns]'
            d2['title'] = 'Model interaction time reco. results'
        
        elif key == 'polar_error':
            clip_vals = [-80.0, 80.0]
            d2['ylabel'] = 'Error [deg]'
            d2['title'] = 'Model polar angle reco. results'
        
        elif key == 'azi_error':
            clip_vals = [-150.0, 150.0]
            d2['ylabel'] = 'Error [deg]'
            d2['title'] = 'Model azimuthal angle reco. results'
        
        elif key == 'directional_error':
            clip_vals = [-150.0, 150.0]
            d2['ylabel'] = 'Directional error'
            d2['title'] = 'Model directional error reco. results'
        
        elif key == 'len_error':
            clip_vals = [0.0, 100.0]
            d2['ylabel'] = 'Distance to vertex error [m]'
            d2['title'] = 'Model distance to vertex reco. results'
        
        return d2, clip_vals
    
    def _get_conf_interval_keys(self):

        # A combination of I3-keys and our predictions
        if self.meta_pars['group'] == 'vertex_reg':
            keys = ['len_error', 'retro_len_error']
        elif self.meta_pars['group'] == 'vertex_reg_no_time':
            keys = ['len_error', 'retro_len_error']
        elif self.meta_pars['group'] == 'direction_reg':
            keys = ['directional_error', 'retro_directional_error']
        elif self.meta_pars['group'] == 'energy_reg':
            keys = []
        elif self.meta_pars['group'] == 'full_reg':
            keys = ['len_error', 'directional_error', 'retro_len_error', 'retro_directional_error']
        else:
            raise ValueError('Performance: Unknown regression encountered!')

        return keys

    def _get_len_error(self, data_dict):
        x_error = data_dict['vertex_x_error']
        y_error = data_dict['vertex_y_error']
        z_error = data_dict['vertex_z_error']

        len_error = np.sqrt(x_error**2 + y_error**2 + z_error**2)
        return len_error
    
    def _get_loss_clip_vals(self):

        if self.loss_func == 'logcosh':
            clip_vals = [0.0, 0.2]
        else:
            raise KeyError('Performance._get_loss_clip_vals: Undefined loss'\
                 'function (%s) given!'%(self.loss_func))
    
        return clip_vals
    
    def _get_perf_dict(self, model_key, reco_key):
        
        if model_key not in self._conf_interval_keys:
            metric = getattr(self, model_key+'_sigma')
            reco_metric = getattr(self, reco_key+'_sigma')
            metricerr = getattr(self, model_key+'_sigmaerr')
            reco_metricerr = getattr(self, reco_key+'_sigmaerr')
        else:
            metric = getattr(self, model_key+'_68th')
            reco_metric = getattr(self, reco_key+'_68th')
            metricerr = getattr(self, model_key+'_err68th')
            reco_metricerr = getattr(self, reco_key+'_err68th')
        
        label = self._get_ylabel(model_key)
        title = self._get_perf_plot_title(model_key)

        d = {'edges': [self.bin_edges, self.bin_edges], 'y': [metric, reco_metric], 
        'yerr': [metricerr, reco_metricerr], 'xlabel': r'log(E) [E/GeV]', 
        'ylabel': label, 'grid': False, 'label': ['Model', 'Icecube'], 
        'yrange': {'bottom': 0.001}, 'title': title}

        return d
    
    def _get_DOMperf_dict(self, model_key, reco_key):
        
        if model_key not in self._conf_interval_keys:
            metric = getattr(self, model_key+'_sigma')
            reco_metric = getattr(self, reco_key+'_sigma')
            metricerr = getattr(self, model_key+'_sigmaerr')
            reco_metricerr = getattr(self, reco_key+'_sigmaerr')
        else:
            metric = getattr(self, 'dom_%s_68th'%(model_key))
            reco_metric = getattr(self, 'dom_%s_68th'%(reco_key))
            metricerr = getattr(self, 'dom_%s_err68th'%(model_key))
            reco_metricerr = getattr(self, 'dom_%s_err68th'%(reco_key))

        label = self._get_ylabel(model_key)
        title = self._get_perf_plot_title(model_key)

        d = {'edges': [self.dom_bin_edges, self.dom_bin_edges], 
        'y': [metric, reco_metric], 'yerr': [metricerr, reco_metricerr], 
        'xlabel': r'$N_{DOMs}$', 'ylabel': label, 'grid': False, 
        'label': ['Model', 'Icecube'], 
        'yrange': {'bottom': 0.001}, 'title': title}

        return d

    def _get_perf_plot_title(self, key):

        if key == 'relative_E_error':
            title = 'Model energy reco. performance'

        elif key == 'vertex_x_error':
            title = 'Model x-vertex reco. performance'

        elif key == 'vertex_y_error':
            title = 'Model y-vertex reco. performance'

        elif key == 'vertex_z_error':
            title = 'Model z-vertex reco. performance'
        
        elif key == 'vertex_t_error':
            title = 'Model interaction time reco. performance'

        elif key == 'polar_error':
            title = 'Model polar angle reco. performance'
        
        elif key == 'azi_error':
            title = 'Model azimuthal angle reco. performance'
        
        elif key == 'log_frac_E_error':
            title = 'Model energy reco. width'
        
        elif key == 'len_error':
            title = 'Model distance to vertex 68th percentile'
        
        elif key == 'directional_error':
            title = 'Model directional error 68th percentile'

        return title

    def _get_performance_keys(self):
        
        if self.meta_pars['group'] == 'vertex_reg':
            keys = self._get_prediction_keys()
        
        elif self.meta_pars['group'] == 'vertex_reg_no_time':
            keys = self._get_prediction_keys()
        
        elif self.meta_pars['group'] == 'direction_reg':
            keys = ['azi_error', 'polar_error']
        
        elif self.meta_pars['group'] == 'energy_reg':
            keys = self._get_prediction_keys()
        
        elif self.meta_pars['group'] == 'full_reg':
            keys = self._get_prediction_keys()

        else:
            raise KeyError('PerformanceClass: Unknown regression type encountered!')
        
        return keys

    def _get_icecube_perf_keys(self):
        funcs = self._get_error_funcs()
        keys = [func.__name__ for func in funcs]
        return keys

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
                reco_keys = ['retro_crs_prefit_x', 'retro_crs_prefit_y', 
                'retro_crs_prefit_z', 'retro_crs_prefit_time']
            
            elif self.meta_pars['group'] == 'vertex_reg_no_time':
                reco_keys = ['retro_crs_prefit_x', 'retro_crs_prefit_y', 
                'retro_crs_prefit_z']
            
            elif self.meta_pars['group'] == 'direction_reg':
                reco_keys = ['retro_crs_prefit_azimuth', 'retro_crs_prefit_zenith']
            
            elif self.meta_pars['group'] == 'energy_reg':
                reco_keys = ['retro_crs_prefit_energy']
            
            elif self.meta_pars['group'] == 'full_reg':
                reco_keys = ['retro_crs_prefit_energy', 'retro_crs_prefit_x', 
                'retro_crs_prefit_y', 'retro_crs_prefit_z', 
                'retro_crs_prefit_time', 'retro_crs_prefit_azimuth', 
                'retro_crs_prefit_zenith']

            else:
                raise KeyError('Unknown reco_keys requested in Performance!')
        else:
            raise KeyError('Unknown dataset encountered (%s)'%(dataset_name))
        
        return reco_keys
    
    def _get_rel_perf_dict(self, key):
        rel_imp = getattr(self, key+'_RI')
        rel_imp_err = getattr(self, key+'_RIerr')

        d = {'edges': [self.bin_edges], 'y': [rel_imp], 'yerr': [rel_imp_err], 
            'xlabel': r'log(E) [E/GeV]', 'ylabel': 'Rel. Imp.', 'grid': True, 
            'y_minor_ticks_multiple': 0.2}
        
        yrange_d = {}
        if max(-0.5, min(rel_imp)) == -0.5:
            yrange_d['bottom'] = -0.5
            yrange_d['top'] = 0.5
            d['yrange'] = yrange_d

        return d

    def _get_ylabel(self, key):
        
        if key == 'vertex_t_error':
            label = 'Resolution [ns]'
        elif key == 'vertex_x_error' or key == 'vertex_y_error' or key == 'vertex_z_error':
            label = 'Resolution [m]'
        elif key == 'azi_error' or key == 'polar_error':
            label = 'Resolution [deg]'
        elif key == 'relative_E_error':
            label = 'Resolution [%]'
        elif key == 'log_frac_E_error':
            label = r'$\log_{10} \left( \frac{E_{pred}}{E_{true}} \right)$'
        elif key == 'len_error':
            label = 'Distance to vertex [m]'
        elif key == 'directional_error':
            label = 'Angle error [deg]'
        else:
            raise KeyError('PerformanceClass._get_ylabel: Unknown key (%s)given!'%(key))

        return label

    def _make_I3_perf_plot(self, key, energy, data, median, upper_perc, 
                           lower_perc, i3_med=None, i3_upper=None, i3_lower=None):
        
        d2, clip_vals = self._get_I3_2D_clip_and_d(key)

        # # Strip potential NaNs: subtract arrays --> find indices, since a number - nan is also nan.
        # # Notify if NaNs are found.
        # bad_indices = np.isnan(energy-data)
        # good_indices = ~bad_indices
        # n_nans = np.sum(bad_indices)
        n_nans, energy, data = strip_nans(energy, data)
        print('WARNING: %d NAN(S) FOUND IN I3 PERFORMANCE PLOT!'%(n_nans)) if n_nans>0 else None
        
        # energy = energy[good_indices]
        # data = data[good_indices]
        d2['hist2d'] = [energy, np.clip(data, clip_vals[0], clip_vals[1])]
        d2['zorder'] = 0
        d2['xlabel'] = r'log(E) [E/GeV]' 
        f2 = make_plot(d2)
        
        d = {}
        # If an Icecube reconstruction is available, plot it aswell
        if not i3_med:
            d['x'] = [self.bin_centers, self.bin_centers, self.bin_centers]
            d['y'] = [upper_perc, median, lower_perc]
            d['drawstyle'] = ['steps-mid', 'steps-mid', 'steps-mid']
            d['linestyle'] = [':', '-', '--']
            d['label'] = ['84th percentile', '50th percentile', '16th percentile']
            d['color'] = ['red','red', 'red']
            d['zorder'] = [1, 1, 1]
        else:
            d['x'] = [self.bin_centers, self.bin_centers, self.bin_centers, 
                      self.bin_centers, self.bin_centers, self.bin_centers]
            d['y'] = [upper_perc, median, lower_perc, i3_upper, i3_med, i3_lower]
            d['drawstyle'] = ['steps-mid', 'steps-mid', 'steps-mid', 
                              'steps-mid', 'steps-mid', 'steps-mid']
            d['linestyle'] = [':', '-', '--', ':', '-', '--']
            d['label'] = ['Model 84th perc.', 'Model 50th perc.', 'Model 16th perc.',
                          'I3 84th perc.', 'I3 50th perc.', 'I3 16th perc.']
            d['color'] = ['red','red', 'red', 'forestgreen', 'forestgreen', 
                          'forestgreen']
            d['zorder'] = [1, 1, 1, 1, 1, 1]
        img_address = get_project_root()+self.model_dir+'/figures/'+key+'_2DPerformance.png'
        d['savefig'] = img_address
        f3 = make_plot(d, h_figure=f2)
        plt.close('all')
        
        # Load img with PIL - this format can be logged
        if self.wandb_ID is not None:
            im = PIL.Image.open(img_address)
            wandb.log({key+'_2Dperformance': wandb.Image(im, caption=key+'_2Dperformance')}, commit=False)
            im.close()

    def update_onenumber_performance(self):
        energy_dict, pred_dict, crs_dict, true_dict, n_doms, loss = self._get_data_dicts()

        self.onenumber_performance = self._calculate_onenum_performance(pred_dict)

    def save(self):
        
        perf_savepath = get_project_root() + self.model_dir + '/data/Performance.pickle'
        with open(perf_savepath, 'wb') as f:
            pickle.dump(self, f)

        for pred_key, reco_key in zip(self._performance_keys, self._icecube_perf_keys):
            img_address = get_project_root()+self.model_dir+'/figures/'+pred_key+'_performance.png'
            d = self._get_perf_dict(pred_key, reco_key)
        
            if self._reco_keys:
                h_fig = make_plot(d, position=[0.125, 0.26, 0.775, 0.62])
                d = self._get_rel_perf_dict(pred_key)
                d['subplot'] = True
                d['axhline'] = [0.0]
                h_fig = make_plot(d, h_figure=h_fig, position=[0.125, 0.11, 0.775, 0.15])
                d_energy = self._get_energy_dict()
                d_energy['savefig'] = img_address
                _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)
            else:
                h_fig = make_plot(d)
                d_energy = self._get_energy_dict()
                d_energy['savefig'] = img_address
                _ = make_plot(d_energy, h_figure=h_fig, axes_index=0)
            
            plt.close('all')

            if self.wandb_ID is not None:
                # Load img with PIL - this format can be logged. Remmeber to close it again
                im = PIL.Image.open(img_address)
                wandb.log({pred_key+'_performance': wandb.Image(im, caption=pred_key+'_performance')}, commit=False)
                im.close()
                
                # Log the data for nice plotting on W&B
                if pred_key not in self._conf_interval_keys:
                    
                    for num1, num2 in zip(getattr(self, pred_key+'_sigma'), self.bin_centers):
                        wandb.log({pred_key+'_sigma': num1, pred_key+'_bincenter2': num2})
                else:
                    for num1, num2 in zip(getattr(self, pred_key+'_68th'), self.bin_centers):
                        wandb.log({pred_key+'_68th': num1, pred_key+'_bincenter2': num2})
                
                # Save the performance class-instance for easy transfers between local and cloud
                wandb.save(perf_savepath)

            img_address = get_project_root()+self.model_dir+'/figures/'+pred_key+'_DOMperformance.png'
            d = self._get_DOMperf_dict(pred_key, reco_key)

            h_fig = make_plot(d)
            d_dom = self._get_dom_dict()
            d_dom['savefig'] = img_address
            _ = make_plot(d_dom, h_figure=h_fig, axes_index=0)

            # Log to W&B to compare
            if self.wandb_ID is not None:
                
                # Log the data for nice plotting on W&B
                if pred_key not in self._conf_interval_keys:

                    for num1, num2 in zip(getattr(self, 'dom_'+pred_key+'_sigma'), self.dom_bin_centers):
                        wandb.log({'dom_'+pred_key+'_sigma': num1, pred_key+'dom_bincenter': num2})
                else:

                    for num1, num2 in zip(getattr(self, 'dom_'+pred_key+'_68th'), self.dom_bin_centers):
                        wandb.log({'dom_'+pred_key+'_68th': num1, pred_key+'dom_bincenter': num2})
