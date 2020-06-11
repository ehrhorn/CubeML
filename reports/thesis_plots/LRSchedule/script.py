from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
from matplotlib import pyplot as plt
import os

setup_pgf_plotting()


f, _ = plt.subplots(1, 2)

# ============================================================================
# PLOT 1  
# ============================================================================
lr = pickle.load(open('lr.pickle', 'rb'))
epoch = pickle.load(open('epochs.pickle', 'rb'))
trainsetsize = 5751590
epoch = np.array(epoch)/trainsetsize

d1 = {
    'x': [epoch],
    'y': [lr],
    'xlabel': r'Epoch',
    'ylabel': r'Learning Rate',
    'title': r'$1/T$ Learning Rate Schedule',
    'grid': False
}
f = make_plot(d1, h_figure=f, axes_index=0, for_thesis=True)
ax = f.get_axes()[0]
def converter(label):
    converted = '%.1f'%(label * 10**3)+r'$\cdot 10^{-3}$'
    return converted
update_ylabels(ax, keyword='func', func=converter)
# mean1t = r'$\hat{\mu} = %.2f$'%(mean1)
# std1t = r'$\hat{\sigma} = %.2f$'%(std1)

maxeta = r'$\eta_{\mathrm{Max}} = 1.0\cdot10^{-3}'
mineta = r'$\eta_{\mathrm{Min}} = 5.0\cdot10^{-5}'
warmup = r'$T_{\mathrm{Warmup}} = 3.5$ \% of $T_{\mathrm{Max}}$'


xpos = 0.2
ypos = 0.65
x_disp = 0.47
y_disp = -0.06

f.text(xpos, ypos, maxeta)
f.text(xpos, ypos+y_disp, mineta)
f.text(xpos, ypos+2*y_disp, warmup)

# ============================================================================
# PLOT 2  
# ============================================================================
lr256 = pickle.load(open('256lr.pickle', 'rb'))
scan_256 = pickle.load(open('256loss.pickle', 'rb'))
# epoch = np.array(epoch)/trainsetsize

d2 = {
    'x': [lr256],
    'y': [scan_256],
    'xlabel': r'Learning Rate',
    'ylabel': r'Loss value',
    'title': r'Learning Rate Scan',
    'xrange': [8e-5, 6e-3],
    'yrange': [0.0, 0.4],
    'xscale': 'log',
    'grid': True
}
f = make_plot(d2, h_figure=f, axes_index=1, for_thesis=True)
# ax = f.get_axes()[0]
# def converter(label):
#     converted = '%.1f'%(label * 10**3)+r'$\cdot 10^{-3}$'
#     return converted
# update_ylabels(ax, keyword='func', func=converter)
# # mean1t = r'$\hat{\mu} = %.2f$'%(mean1)
# # std1t = r'$\hat{\sigma} = %.2f$'%(std1)

# maxeta = r'$\eta_{\mathrm{Max}} = 1.0\cdot10^{-3}'
# mineta = r'$\eta_{\mathrm{Min}} = 5.0\cdot10^{-5}'
# warmup = r'$T_{\mathrm{Warmup}} = 3.5$ \% of $T_{\mathrm{Max}}$'


# xpos = 0.2
# ypos = 0.65
# x_disp = 0.47
# y_disp = -0.06

# f.text(xpos, ypos, maxeta)
# f.text(xpos, ypos+y_disp, mineta)
# f.text(xpos, ypos+2*y_disp, warmup)

# Standard ratio of width to height it 6.4/4.8
# Standard figure: FOTW = 1.0
# Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right
# single_fig, 2subfigs
FOTW = get_frac_of_textwidth(keyword='subplots', rows=1, cols=2)
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width, rows=1, cols=2)
f.set_size_inches(width, height)

# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
