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