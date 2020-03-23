import wandb
import PIL
import pickle
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from time import localtime, strftime

from torch.utils import data
from scipy.stats import wilcoxon
from src.modules.main_funcs import *
from src.modules.helper_functions import strip_nans

#* ======================================================================== 
#* PERFORMANCE CLASSES
#* ========================================================================


class AziPolarHists:
    '''A class to create azimuthal and polar error plots - one 2D-histogram and two performance plots.
    '''

    def __init__(self, model_dir, wandb_ID=None):
        _, data_pars, _, meta_pars = load_model_pars(model_dir)

        self.model_dir = get_path_from_root(model_dir)
        self.data_pars = data_pars
        self.data_dir = data_pars['data_dir']
        self.wandb_ID = wandb_ID
        self.meta_pars = meta_pars
        self.pred_dict, self.true_dict = self._get_data_dict()

    def _get_data_dict(self):
        full_pred_address = self._get_pred_path()
        keys = self._get_keys()
        pred_dict, true_dict = read_predicted_h5_data(full_pred_address, keys, self.data_pars, [])
        return pred_dict, true_dict
    
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
        
        # * exclude errors outside of first interval
        azi_sorted, polar_sorted = sort_pairs(self.pred_dict['azi_error'], self.pred_dict['polar_error'])
        i_azi = np.searchsorted(azi_sorted, [-azi_max_dev, azi_max_dev])
        azi_sorted = azi_sorted[i_azi[0]:i_azi[1]]
        polar_sorted = polar_sorted[i_azi[0]:i_azi[1]]

        # * exclude errors outside of second interval
        polar_sorted, azi_sorted = sort_pairs(polar_sorted, azi_sorted)
        i_polar = np.searchsorted(polar_sorted, [-polar_max_dev, polar_max_dev])
        azi_sorted = azi_sorted[i_polar[0]:i_polar[1]]
        polar_sorted = polar_sorted[i_polar[0]:i_polar[1]]

        return azi_sorted, polar_sorted

    def save(self):
        
        # * Save standard histograms first
        for key, pred in self.pred_dict.items():
            img_address = get_project_root() + self.model_dir+'/figures/'+str(key)+'.png'
            figure = make_plot({'data': [pred], 'xlabel': str(key), 'savefig': img_address})

            # * Load img with PIL - png format can be logged
            if self.wandb_ID is not None:
                im = PIL.Image.open(img_address)
                wandb.log({str(key): wandb.Image(im, caption=key)}, commit=False)
                im.close()

        # * Save 2D-histogram
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
                wandb.log({'azi_vs_polar': wandb.Image(im, caption='azi_vs_polar')}, commit=False)
                im.close()

class Performance:
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
        prefix = 'transform'+str(data_pars['file_keys']['transform'])
        from_frac = data_pars['train_frac']
        to_frac = data_pars['train_frac'] + data_pars['val_frac']

        self.model_dir = get_path_from_root(model_dir)
        self.data_pars = data_pars
        self.meta_pars = meta_pars
        self.keyword = meta_pars['group']
        self.prefix = prefix
        self.dom_mask = data_pars['dom_mask']
        self.loss_func = arch_pars['loss_func']
        
        self._energy_key = self._get_energy_key()
        self._pred_keys = self._get_prediction_keys()
        self._reco_keys = self._get_reco_keys()
        # * There should be equally many performance keys and icecube perf keys
        self._performance_keys = self._get_performance_keys()
        self._icecube_perf_keys = self._get_icecube_perf_keys()
        self._true_keys = get_target_keys(data_pars, meta_pars)
        self._conf_interval_keys = self._get_conf_interval_keys()

        self.from_frac = from_frac
        self.to_frac = to_frac
        self.wandb_ID = wandb_ID

        energy_dict, pred_dict, crs_dict, true_dict, n_doms, loss = self._get_data_dicts()
        self._calculate_performance(energy_dict, pred_dict, crs_dict, true_dict, n_doms)
        self.onenumber_performance = self._calculate_onenum_performance(pred_dict)
        self.loss_error_correlations = self._calculate_loss_error_corrs(pred_dict, loss)

    def _calculate_loss_error_corrs(self, pred_d, loss):

        # * Correlation between loss and predicted values is calculated to 
        # * hopefully see if loss function is focused too heavily on some of the variables
        # * We look at ABSOLUTE value of error, since we are using symmetric
        # * loss functions.
        corrs = {}
        loss_clip = self._get_loss_clip_vals()
        loss_clipped = np.clip(loss, loss_clip[0], loss_clip[1])
        for key, data in pred_d.items():
            # * Clip values, since it's concentrated at a certain range.
            _, clip_vals = self._get_I3_2D_clip_and_d(key)
            
            # * Strip NaNs
            n_nans, data, loss_stripped = strip_nans(data, loss)

            # * corrcoef returns correlation matrix - we are only interested 
            # * in one of the numbers
            abs_data = np.abs(np.clip(data, clip_vals[0], clip_vals[1]))
            
            print('WARNING: %d NAN(S) FOUND IN LOSS-ERROR CORRELATIONS!'%(n_nans)) if n_nans>0 else None
            corrs[key] = np.corrcoef(abs_data, loss_stripped)[1, 0]

            # * Make 2d-scatterplot or HEATMAP! like i3-performance
            d = {}
            title = r'%s - loss correlation'%(key)
            savepath = get_project_root()+self.model_dir+'/figures/'+key+'_LossCorr.png'
            d['hist2d'] = [loss_stripped, abs_data]
            d['xlabel'] = 'Loss value'
            d['ylabel'] = 'Abs(%s)'%(key)
            d['savefig'] = savepath
            d['title'] = title
            fig_h = make_plot(d)

        # * Make barh-plots
        bar_pos = np.arange(0, len(corrs)) + 0.5
        names = [key for key in corrs]
        data = [data for key, data in corrs.items()]
        d = {'keyword': 'barh', 'y': bar_pos, 'width': data, 'height': 0.7, 'names': names}
        d['title'] = 'Correlation coefficients w.r.t. %s-loss'%(self.loss_func)
        d['xlabel'] = 'Correlation'
        d['savefig'] = get_project_root()+self.model_dir+'/figures/CorrCoeff.png'
        fig = make_plot(d)

        # * Save a histogram of loss values aswell
        d = {'data': [loss]}
        d['xlabel'] = 'Loss value (%s)'%(self.loss_func)
        d['ylabel'] = 'Count'
        d['log'] = [True]
        d['savefig'] = get_project_root()+self.model_dir+'/figures/loss_vals.png'
        _ = make_plot(d)
        plt.close('all')

        return corrs

    def _calculate_onenum_performance(self, data_dict=None):
        # * Performance for each variable in names is calculated as the 
        # * geometric mean of the bin-values. Finally, they are combined as
        # * a geometric mean of all the geometric means. This has the
        # * advantage that a fractional decrease of x percent in any of 
        # * them contributes equally to the final performance
        
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
        
        # * Transform back and extract values into list
        true_transformed = inverse_transform(true_dict, 
                                get_project_root()+self.model_dir)
        energy_transformed = inverse_transform(energy_dict, 
                                get_project_root()+self.model_dir)
        # * We want energy as array
        energy = np.array(convert_to_proper_list(energy_transformed[self._energy_key[0]]))

        self.counts, self.bin_edges = np.histogram(energy, 
                                                bins=N_BINS_PERF_PLOTS)
        self.bin_centers = calc_bin_centers(self.bin_edges)
        
        self.dom_counts, self.dom_bin_edges = np.histogram(n_doms, 
                                                bins=N_BINS_PERF_PLOTS)
        self.dom_bin_centers = calc_bin_centers(self.dom_bin_edges)
        
        # * Calculate how well Icecube does.
        for name, func, model_key in zip(self._icecube_perf_keys, self._get_error_funcs(), self._performance_keys):
            
            # * We calculate 2 different forms of performance measures: One
            # * kind as the width of the distribution (for unbounded
            # * distributions), another as the upper 68th percentile 
            # * of the distribution (for distributions with a lower bound of 0)
            print(get_time(), 'Calculating %s performance...'%(name))

            # * Performance as a function of energy
            error = func(crs_dict, true_transformed, reporting=True)
            print(name, model_key)

            # # * Make Wilcoxon signed-rank test
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

                # * Performance as a function of number of doms
                sigma, sigmaerr, median, upper_perc, lower_perc =\
                    calc_width_as_fn_of_data(n_doms, error, self.dom_bin_edges)
                setattr(self, 'dom_'+name+'_sigma', sigma)
                setattr(self, 'dom_'+name+'_sigmaerr', sigmaerr)
                setattr(self, 'dom_'+name+'_50th', median)
                setattr(self, 'dom_'+name+'_84th', upper_perc)
                setattr(self, 'dom_'+name+'_16th', lower_perc)
            
            else:
                percentiles = [68, 50, 32]
                
                # * As a function of energy
                vals, errs = calc_percentiles_as_fn_of_data(
                    energy, error, self.bin_edges, percentiles)
                
                for perc, val, err, in zip(percentiles, vals, errs):
                    setattr(self, '%s_%sth'%(name, str(perc)), val)
                    setattr(self, '%s_err%sth'%(name, str(perc)), err)

                # * As a function of DOMs
                vals, errs = calc_percentiles_as_fn_of_data(
                    n_doms, error, self.dom_bin_edges, percentiles)
                
                for perc, val, err, in zip(percentiles, vals, errs):
                    setattr(self, 'dom_%s_%sth'%(name, str(perc)), val)
                    setattr(self, 'dom_%s_err%sth'%(name, str(perc)), err)
            
            print(get_time(), 'Calculation finished!')
            print('')
                
        # * Calculate performance for our predictions
        for (key, data), i3_key in zip(pred_dict.items(), self._icecube_perf_keys):
            
            print(get_time(), 'Calculating %s performance...'%(key))
            
            # * Same split as above
            if key not in self._conf_interval_keys:
            
                # * Performance as a function of energy
                sigma, sigmaerr, median, upper_perc, lower_perc = calc_width_as_fn_of_data(energy, data, self.bin_edges)
                setattr(self, key+'_sigma', sigma)
                setattr(self, key+'_sigmaerr', sigmaerr)
                setattr(self, key+'_50th', median)
                setattr(self, key+'_84th', upper_perc)
                setattr(self, key+'_16th', lower_perc)

                # * We make the I3-plot here so we do not have to save all 
                # * the data. First we retrieve the corresponding Icecube data
                i3_med = getattr(self, i3_key+'_50th')
                i3_upper = getattr(self, i3_key+'_84th')
                i3_lower = getattr(self, i3_key+'_16th')
                self._make_I3_perf_plot(key, energy, data, median, upper_perc, lower_perc, i3_med=i3_med, i3_upper=i3_upper, i3_lower=i3_lower)

                # * Performance as a function of number of doms
                sigma, sigmaerr, median, upper_perc, lower_perc = calc_width_as_fn_of_data(n_doms, data, self.dom_bin_edges)
                setattr(self, 'dom_'+key+'_sigma', sigma)
                setattr(self, 'dom_'+key+'_sigmaerr', sigmaerr)
                setattr(self, 'dom_'+key+'_50th', median)
                setattr(self, 'dom_'+key+'_84th', upper_perc)
                setattr(self, 'dom_'+key+'_16th', lower_perc)
                print('')
            
            else:
                percentiles = [68, 50, 32]
                
                # * As a function of energy
                vals, errs = calc_percentiles_as_fn_of_data(
                    energy, data, self.bin_edges, percentiles)
                for perc, val, err, in zip(percentiles, vals, errs):
                    setattr(self, '%s_%sth'%(key, str(perc)), val)
                    setattr(self, '%s_err%sth'%(key, str(perc)), err)
                
                # * As a function of DOMs
                vals, errs = calc_percentiles_as_fn_of_data(
                    n_doms, data, self.dom_bin_edges, percentiles)
                
                for perc, val, err, in zip(percentiles, vals, errs):
                    setattr(self, 'dom_%s_%sth'%(key, str(perc)), val)
                    setattr(self, 'dom_%s_err%sth'%(key, str(perc)), err)
            
            print(get_time(), 'Calculation finished!')

        # * Calculate the relative improvement - e_diff/I3_error. Report decrease in error as a positive 
        for model_key, retro_key in zip(self._performance_keys, self._icecube_perf_keys):
            
            # * Remember to split in conf bounds and widths
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

        # * Load loss aswell
        keys = self._pred_keys+['loss']
        
        print(get_time(), 'Loading predictions...')
        pred_dict = read_pickle_predicted_h5_data_v2(full_pred_address, keys)
        energy_dict = read_pickle_data(data_dir, pred_dict['indices'], self._energy_key, prefix=prefix)
        crs_dict = read_pickle_data(data_dir, pred_dict['indices'], self._reco_keys, prefix=prefix)
        true_dict = read_pickle_data(data_dir, pred_dict['indices'], self._true_keys, prefix=prefix)
        print(get_time(), 'Predictions loaded!')

        # * Pop loss from dict
        loss = pred_dict['loss']
        del pred_dict['loss']

        print(get_time(), 'Finding number of DOMs in events')
        n_doms = get_n_doms(pred_dict['indices'], self.dom_mask, data_dir)
        # * Indices have done, what we wanted them to do - delete them
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

        # * A combination of I3-keys and our predictions
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

        # # * Strip potential NaNs: subtract arrays --> find indices, since a number - nan is also nan.
        # # * Notify if NaNs are found.
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
        # * If an Icecube reconstruction is available, plot it aswell
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
        
        # * Load img with PIL - this format can be logged
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
                # * Load img with PIL - this format can be logged. Remmeber to close it again
                im = PIL.Image.open(img_address)
                wandb.log({pred_key+'_performance': wandb.Image(im, caption=pred_key+'_performance')}, commit=False)
                im.close()
                
                # * Log the data for nice plotting on W&B
                if pred_key not in self._conf_interval_keys:
                    
                    for num1, num2 in zip(getattr(self, pred_key+'_sigma'), self.bin_centers):
                        wandb.log({pred_key+'_sigma': num1, pred_key+'_bincenter2': num2})
                else:
                    for num1, num2 in zip(getattr(self, pred_key+'_68th'), self.bin_centers):
                        wandb.log({pred_key+'_68th': num1, pred_key+'_bincenter2': num2})
                
                # * Save the performance class-instance for easy transfers between local and cloud
                wandb.save(perf_savepath)

            img_address = get_project_root()+self.model_dir+'/figures/'+pred_key+'_DOMperformance.png'
            d = self._get_DOMperf_dict(pred_key, reco_key)

            h_fig = make_plot(d)
            d_dom = self._get_dom_dict()
            d_dom['savefig'] = img_address
            _ = make_plot(d_dom, h_figure=h_fig, axes_index=0)

            # * Log to W&B to compare
            if self.wandb_ID is not None:
                
                # * Log the data for nice plotting on W&B
                if pred_key not in self._conf_interval_keys:

                    for num1, num2 in zip(getattr(self, 'dom_'+pred_key+'_sigma'), self.dom_bin_centers):
                        wandb.log({'dom_'+pred_key+'_sigma': num1, pred_key+'dom_bincenter': num2})
                else:

                    for num1, num2 in zip(getattr(self, 'dom_'+pred_key+'_68th'), self.dom_bin_centers):
                        wandb.log({'dom_'+pred_key+'_68th': num1, pred_key+'dom_bincenter': num2})

class FeaturePermutationImportance:
    
    def __init__(self, save_dir, wandb_ID=None, ):    

        self.save_dir = get_path_from_root(save_dir)
        self.wandb_ID = wandb_ID
        self.feature_importances = {}

    def calc_feature_importance_from_errors(self, baseline_errors, permuted_errors):
        # * Use (84th-16th)/2 as metric. Corresponds to sigma. 
        bl_percentiles, bl_lower, bl_upper = estimate_percentile(baseline_errors, [0.15865, 0.84135], bootstrap=False)
        permuted_percentiles, permuted_lower, permuted_upper = estimate_percentile(permuted_errors, [0.15865, 0.84135], bootstrap=False)

        # * Use error propagation to get errors
        bl_sigmas = (bl_upper-bl_lower)/2
        bl_metric = [(bl_percentiles[1]-bl_percentiles[0])/2]
        bl_metric_error = [np.sqrt(np.sum(bl_sigmas*bl_sigmas))/2]

        permuted_sigmas = (permuted_upper-permuted_lower)/2
        permuted_metric = [(permuted_percentiles[1]-permuted_percentiles[0])/2]
        permuted_metric_error = [np.sqrt(np.sum(permuted_sigmas*permuted_sigmas))/2]

        feature_importance, e_feature_importance = calc_relative_error(bl_metric, permuted_metric, e1=bl_metric_error, e2=permuted_metric_error)
        
        return feature_importance[0], e_feature_importance[0]

    def calc_all_seq_importances(self, n_predictions_wanted=np.inf):
        # * Option given to just calculate all
        hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(self.save_dir)
        
        # * Get indices of features to permute - both scalar and sequential
        all_seq_features = data_pars['seq_feat']
        for feature in all_seq_features:
            self.calc_permutation_importance(seq_features=[feature], n_predictions_wanted=n_predictions_wanted)
        
    def calc_permutation_importance(self, seq_features=[], scalar_features=[], n_predictions_wanted=np.inf):
        hyper_pars, data_pars, arch_pars, meta_pars = load_model_pars(self.save_dir)
        
        # * Get indices of features to permute - both scalar and sequential
        all_seq_features = data_pars['seq_feat']
        all_scalar_features = data_pars['scalar_feat']
        
        # * Ensure features actually exist
        try:    
            seq_indices = [all_seq_features.index(entry) for entry in seq_features]
            scalar_indices = [all_scalar_features.index(entry) for entry in scalar_features]
        except ValueError:
            print(get_time(), 'ERROR: Atleast one of features (%s, %s) does not exist. Returning.'%(', '.join(seq_features), ', '.join(scalar_features)))
            return
        
        # * Check it hasn't already been calculated
        if self.check_duplication(seq_features, scalar_features):
            return
        
        # * Load the best model
        model = load_best_model(self.save_dir)

        # * Setup dataloader and generator - num_workers choice based on gut feeling - has to be high enough to not be a bottleneck
        data_pars['n_predictions_wanted'] = n_predictions_wanted
        LOG_EVERY = int(meta_pars.get('log_every', 200000)/4) 
        VAL_BATCH_SIZE = data_pars.get('val_batch_size', 256) # ! Predefined size !
        gpus = meta_pars['gpu']
        device = get_device(gpus[0])
        dataloader_params_eval = get_dataloader_params(VAL_BATCH_SIZE, num_workers=8, shuffle=False, dataloader=data_pars['dataloader'])
        val_set = load_data(hyper_pars, data_pars, arch_pars, meta_pars, 'predict')
        
        # * SET MODE TO PERMUTE IN COLLATE_FN
        collate_fn = get_collate_fn(data_pars, mode='permute', permute_seq_features=seq_indices)
        val_generator = data.DataLoader(val_set, **dataloader_params_eval, collate_fn=collate_fn)
        N_VAL = get_set_length(val_set)

        # * Run evaluator!
        predictions, truths, indices = run_pickle_evaluator(model, val_generator, val_set.targets, gpus, 
            LOG_EVERY=LOG_EVERY, VAL_BATCH_SIZE=VAL_BATCH_SIZE, N_VAL=N_VAL)

        # * Run predictions through desired functions - transform back to 'true' values, if transformed
        predictions_transformed = inverse_transform(predictions, self.save_dir)
        truths_transformed = inverse_transform(truths, self.save_dir)

        eval_functions = get_eval_functions(meta_pars)
        error_from_preds = {}
        baseline = {}
        pred_full_address = get_project_root()+self.save_dir+'/data/predictions.h5'

        # * Calculate PFI for all evaluation functions.
        print(get_time(), 'Calculating PFI for evaluation functions...')
        for func in eval_functions:
            name = func.__name__
           
            # * Calculate new errors.
            error_from_preds[name] = func(predictions_transformed, truths_transformed)

            # * load baseline errors
            with h5.File(pred_full_address, 'r') as f:
                baseline[name] = f[name][:]

            # * Calculate feature importance = (permuted_metric-baseline_metric)/baseline_metric
            feature_importance, feature_importance_err = self.calc_feature_importance_from_errors(baseline[name], error_from_preds[name])

            # * Save dictionary as an attribute. Should contain permuted feature-names and FI. Each new permutation importance is saved as an entry in a list.
            features = seq_features.copy()
            features.extend(scalar_features)
            d = {'permuted': features, 'feature_importance': feature_importance, 'error': feature_importance_err}
            self.save_feature_importance(name, d)
        print(get_time(), 'PFI Calculation finished!')
        
    def save(self):
        # * Save the results as a class instance
        save_path = get_project_root()+self.save_dir+'/data/FeaturePermutationImportance.pickle'
        self.make_plots()
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
            print(get_time(), 'Saved FeaturePermutationImportance at %s'%(get_path_from_root(save_path)))

    def check_duplication(self, seq_features, scalar_features):
        
        features = seq_features.copy()
        features.extend(scalar_features)
        
        # * Check no duplication
        try:
            some_key = [key for key in self.feature_importances][0]
            for d in self.feature_importances[some_key]:
                if features == d['permuted']:
                    print('')
                    print(get_time(), 'PFI ALREADY EXISTS OF:', *features, '. SKIPPING RE-CALCULATION')
                    conclusion = True
                    break
            else:
                conclusion = False
        except IndexError:
            # * Nothing added so far. Continue with calculation
            conclusion = False

        if conclusion == False:
            print('')
            print(get_time(), 'CALCULATING PFI OF:', *features)

        return conclusion
    
    def save_feature_importance(self, name, dfi):
        
        # * dfi = dict_feature_imporatnce
        if name not in self.feature_importances:
            self.feature_importances[name] = [dfi]
        else:
            self.feature_importances[name].append(dfi)
    
    def make_plots(self):

        # * Loop over each performance function
        for func, data in self.feature_importances.items():
            
            # * Loop over all permuted features
            # * sort wrt importance
            sorted_data = sorted(data, key=lambda x: x['feature_importance'])
            names, fi, errors = [], [], []
            for d in sorted_data:
                name = ', '.join(d['permuted'])
                names.append(name)
                fi.append(d['feature_importance'])
                errors.append(d['error'])

            # * Make barplot
            bar_pos = np.arange(0, len(names)) + 0.5
            d = {'keyword': 'barh', 'y': bar_pos, 'width': fi, 'height': 0.7, 'names': names, 'errors': errors}
            d['title'] = 'Permutation Importance - %s'%(func)
            d['xlabel'] = 'Feature Importance'
            d['savefig'] = get_project_root()+self.save_dir+'/figures/PFI_%s.png'%(func)
            fig = make_plot(d)                

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
    
    if meta_pars['group'] == 'vertex_reg' or meta_pars['group'] == 'vertex_reg_no_time':
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
    
    # ! FOR NOW DONT PLOT - IT IS ALL ON WANDB!
    # img_address = model_dir+'/figures/train_val_error.png'
    # _ = make_plot({'x': [epochs, epochs], 'y': [train_error, val_error], 'label': ['train error', 'val. error'], 'xlabel': 'Events processed', 'ylabel': 'Loss', 'savefig': img_address})
    
    # if wandb_ID is not None:
    #     im = PIL.Image.open(img_address)
    #     wandb.log({'Train and val. error': wandb.Image(im, caption='Train and val. error')}, commit = False)
    
    # img_address = model_dir+'/figures/lr_vs_epoch.png'
    # _ = make_plot({'x': [epochs], 'y': [lr_list], 'xlabel': 'Events processed', 'ylabel': 'Learning rate', 'savefig': img_address})
    
    # if wandb_ID is not None:
    #     im = PIL.Image.open(img_address)
    #     wandb.log({'Learning rate vs epoch': wandb.Image(im, caption='Learning rate vs epoch')}, commit = False)

def log_performance_plots(model_dir, wandb_ID=None):
    """Creates and logs performance plots relevant to the regression model by calling special classes
    
    Arguments:
        model_dir {str} -- Absolute or relative path to the model directory.
    
    Keyword Arguments:
        wandb_ID {str} -- If wanted, the unique wandb-ID can be supplied to log to W&B (default: {None})
    """    
    
    _, _, _, meta_pars = load_model_pars(model_dir)
    
    print('')
    print(get_time(), 'Evaluation of model performance initiated.')
    performance = Performance(model_dir, wandb_ID=wandb_ID)
    performance.save()
    print(get_time(), 'Evaluation of performance finished!')

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
    alpha = 0.3
    if 'grid' in plot_dict:
        grid_on = plot_dict['grid']
    else:
        grid_on = True

    if h_figure == None:

        h_figure = plt.figure()
        h_axis = h_figure.add_axes(position)
    else:
        h_axis = h_figure.gca()

    if 'twinx' in plot_dict and h_figure != None:
        if plot_dict['twinx']:
            if axes_index == None:
                h_axis = h_figure.axes[0].twinx()
            else:
                h_axis = h_figure.axes[axes_index].twinx()
    
    if 'subplot' in plot_dict:
        # * By default, x-axis is shared
        h_axis = h_figure.add_axes(position, sharex=h_figure.axes[0])
    
    if 'x' in plot_dict and 'y' in plot_dict:
        if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])
        if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])
        
        for i_set, dataset in enumerate(plot_dict['y']):
            # * Drawstyle can be 'default', 'steps-mid', 'steps-pre' etc.
            plot_keys = ['label', 'drawstyle', 'color', 'zorder', 'linestyle']
            #* Set baseline
            d = {'linewidth': 1.5}
            for key in plot_dict:
                if key in plot_keys: d[key] = plot_dict[key][i_set] 
            h_axis.plot(plot_dict['x'][i_set], dataset, **d)
            
        if 'label' in plot_dict: 
            h_axis.legend()
        
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
            
            h_axis.hist(data, **d)

            if 'label' in plot_dict: h_axis.legend()
        
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
        plt.hist2d(set1, set2, bins=[widths1, widths2], zorder=plot_dict.get('zorder', 0), cmap='Oranges')
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
            
        if 'label' in plot_dict: h_axis.legend()

    elif plot_dict.get('keyword', 'None') == 'barh':
        h_axis.barh(plot_dict['y'], plot_dict['width'], xerr=plot_dict.get('errors', None))
        h_axis.set_yticklabels(plot_dict['names'])
        h_axis.set_yticks(plot_dict['y'])
        h_axis.set_ylim((0, len(plot_dict['y'])))
        # plt.tight_layout()
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
        if 'y_major_ticks_multiple' in plot_dict:
            multiple = plot_dict['y_major_ticks_multiple']
            h_axis.yaxis.set_major_locator(MultipleLocator(multiple))
        if 'y_minor_ticks_multiple' in plot_dict:
            multiple = plot_dict['y_minor_ticks_multiple']
            major_ticks = h_axis.get_yticks()
            major_size = abs(major_ticks[0]-major_ticks[1])
            h_axis.yaxis.set_minor_locator(MultipleLocator(multiple*major_size))
        h_axis.grid(alpha=alpha)
        h_axis.grid(True, which='minor', alpha=alpha, linestyle=':')

    if 'text' in plot_dict:
        plt.text(*plot_dict['text'], transform=h_axis.transAxes)
    
    if 'title' in plot_dict:
        plt.title(plot_dict['title'])

    if 'yrange' in plot_dict:
        h_axis.set_ylim(**plot_dict['yrange'])

    if 'xlabel' in plot_dict: h_axis.set_xlabel(plot_dict['xlabel'])
    if 'ylabel' in plot_dict: h_axis.set_ylabel(plot_dict['ylabel'])
    if 'xscale' in plot_dict: h_axis.set_xscale(plot_dict['xscale'])
    if 'yscale' in plot_dict: h_axis.set_yscale(plot_dict['yscale'])

    if 'savefig' in plot_dict: 
        h_figure.savefig(plot_dict['savefig'], bbox_inches='tight')
        fig_name = Path(plot_dict['savefig']).name
        print(get_time(), 'Saved figure: %s'%(fig_name))
        plt.close(fig=h_figure)

    return h_figure

def summarize_model_performance(model_dir, wandb_ID=None):
    """Summarizes a model's performance with a single number by updating the meta_pars-dictionary of the experiment.
    
    Arguments:
        model_dir {str} -- full or relative path to the model directory
    
    Keyword Arguments:
        wandb_ID {str} -- The wandb-ID for the specific experimented. Supplied if logging to wandb.com is wanted. (default: {None})
    """    
    
    _, _, _, meta_pars = load_model_pars(model_dir)
    path = model_dir + '/data/Performance.pickle'
    perf_class = pickle.load(open(path,"rb"))

    try:
        onenum_performance = perf_class.onenumber_performance
    except AttributeError:
        print('\nNO ONE-NUMBER PERFORMANCE MEASURE DEFINED. RETURNING -1\n')
        onenum_performance = -1
    
    if wandb_ID is not None:
        wandb.config.update({'Performance': onenum_performance}, allow_val_change=True)

    meta_pars['performance'] = onenum_performance
    
    with open(model_dir+'/meta_pars.json', 'w') as fp:
        json.dump(meta_pars, fp)
