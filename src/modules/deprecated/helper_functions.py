# def calc_perf_as_fn_of_energy(energy, predictor_vals, n_bins=15):
#     '''Calculates error histograms as a function of energy.

#     Input
#     true: list of true values
#     predictor_vals: list of predictions

#     returns: lists of edges, Median Absolute errors and interquartile errors.
#     '''

#     energy_sorted, predictor_vals_sorted = sort_pairs(energy, predictor_vals)

#     entries, edges = calc_histogram(energy_sorted, n_bins=n_bins)
#     maes = calc_MAEs(predictor_vals_sorted, entries)
#     widths_lower, widths_upper = calc_widths(predictor_vals_sorted, entries)

#     return edges, maes, [widths_lower, widths_upper]

# def calc_perf2_as_fn_of_energy_old(energy, predictor_vals, bin_edges):
#     '''Calculates error histograms as a function of energy.

#     Input
#     true: list of true values
#     predictor_vals: list of predictions

#     returns: lists of edges, Median Absolute errors and interquartile errors.
#     '''
#     energy_sorted, predictor_vals_sorted = sort_pairs(energy, predictor_vals)
#     _, predictor_bins = bin_data(energy_sorted, predictor_vals_sorted, bin_edges)
#     sigmas, e_sigmas = [], []
#     for entry in predictor_bins:
#         means, plussigmas, minussigmas = estimate_percentile(entry, [0.25, 0.75])
#         e_quartiles = []
#         e_quartiles.append((plussigmas[0]-minussigmas[0])/2)
#         e_quartiles.append((plussigmas[1]-minussigmas[1])/2)

#         # * Assume errors are symmetric - which they look to be (quick inspection)
#         # * Look at plussigma[0]-mean[0], mean[0]-minussigma[0] for instance
#         sigma, e_sigma = convert_iqr_to_sigma(means, e_quartiles)
        
#         # * Ignore nans - it is due to too little statistics in a bin
#         if e_sigma != e_sigma:
#             sigma = np.nan
            
#         sigmas.append(sigma)
#         e_sigmas.append(e_sigma)
#     # return predictor_bins, energy_sorted

#     return sigmas, e_sigmas

def read_pickle_predicted_h5_data(file_address, keys, data_pars, true_keys):
    """Reads datasets in a predictions-h5-file associated with keys and the matching datasets in the raw data-files associated with true_keys and returns 2 sorted dictionaries such that index_i for any key corresponds to the i'th event.
    
    Arguments:
        file_address {str} -- absolute path to predictions-file.
        keys {list} -- names of datasets to read in predictions-file
        data_pars {dict} -- dictionary containing data-parameters of the model.
        true_keys {list} -- names of datasets to read in raw data-files.
    
    Returns:
        dicts -- predictions_dict, raw_dict
    """    

    data_dir = data_pars['data_dir']
    prefix = 'transform'+str(data_pars['file_keys']['transform'])

    preds = {key: [] for key in keys}
    preds['indices'] = []

    # * Read the predictions. Each group in the h5-file corresponds to a raw data-file. Each group has same datasets.
    with h5.File(file_address, 'r') as f:
        preds['indices'] = f['index'][:]
        for key in keys:
            preds[key] = f[key][:]
    
    # * Now read the matching true values
    truths = read_pickle_data(data_dir, preds['indices'], true_keys, prefix=prefix)
    
    return preds, truths