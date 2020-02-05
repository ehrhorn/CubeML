import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
import pickle
from src.modules.helper_functions import get_project_root
from src.modules.reporting import make_plot

# * Just load an energy regression model, the amount of data and Icecubes reco are the same for them all
perf_path = '/home/bjoernhm/CubeML/models/oscnext-genie-level5-v01-01-pass2/regression/energy_reg/2020-02-03-22.22.30/data/Performance.pickle'
performance = pickle.load(open(perf_path, "rb"))
d = {'x': [], 'y': [], 'label': []}
x_vals = np.array(performance.bin_centers)

# * The values we want to create weights from
counts = np.array(performance.counts)
energy_sigmas = np.array(performance.retro_crs_prefit_energy_sigma)

# * Transform to weights as 1/val
counts_weights = 1/counts
counts_weights_normed = counts_weights/np.mean(counts_weights)
energy_weights = 1/energy_sigmas
energy_weights_normed = energy_weights/np.mean(energy_weights)

# * Now combine via different methods
# * Geometric mean
gmeans = np.sqrt(counts_weights*energy_weights)
gmeans_normed = gmeans/np.mean(gmeans)
d['x'].append(x_vals)
d['y'].append(gmeans_normed)
d['label'].append('Geo-mean')

# * Product
weights_prods = energy_weights*counts_weights
weights_prods_normed = weights_prods/np.mean(weights_prods)
d['x'].append(x_vals)
d['y'].append(weights_prods_normed)
d['label'].append('Product')

# * Standard mean
weights_meaned = (energy_weights_normed+counts_weights_normed)/2
d['x'].append(x_vals)
d['y'].append(weights_meaned)
d['label'].append('Std. mean')

# * Greatest fractional difference.
# * Calculated by requiring (maxprod/minprd)**n = 10
greatest_frac_diff = (max(weights_prods)/min(weights_prods))
new_exp = np.log(10.0)/np.log(greatest_frac_diff)
gfd_10 = weights_prods**new_exp
gfd_10_normed = gfd_10/np.mean(gfd_10)
d['x'].append(x_vals)
d['y'].append(gfd_10_normed)
d['label'].append('GFD=10')


# * Make a spline
interpolator_linear = interpolate.interp1d(x_vals, gfd_10_normed, fill_value="extrapolate", kind='quadratic')
x_extrapolate = np.linspace(0.0, 3.0, 200)
gfd_10_extrapolate = interpolator_linear(x_extrapolate)
d['x'].append(x_extrapolate)
d['y'].append(gfd_10_extrapolate)
d['label'].append('GFD=10, quadratic interp.')
# # * Print values
# for count, e_sigma, gmean, prod, mean, gfd in zip(counts_weights_normed, energy_weights_normed, gmeans_normed, weights_prods_normed, weights_meaned, gfd_10_normed):
#     print('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f'%(count, e_sigma, gmean, prod, mean, gfd))

d['yscale'] = 'log'
d['savefig'] = get_project_root() + '/reports/plots/energyreg_weight_propositions.png' 
d['title'] = 'Combination of entries in each range + Icecube performance'
d['xlabel'] = r'log(E) [E/GeV]'
d['ylabel'] = r'Weight value'

fig = make_plot(d)