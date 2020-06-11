import matplotlib as mpl
from src.modules.constants import *
from src.modules.helper_functions import get_time

def update_ylabels(ax, keyword='num2k', func=None):
    if keyword == 'num2k':
        ylabels = [str(int(label/1000))+'k' for label in ax.get_yticks()]
    elif keyword == 'func':
        ylabels = [func(label) for label in ax.get_yticks()]
    ax.set_yticklabels(ylabels)


def save_thesis_pgf(path, f, save_pgf=False):
    all_figs_path = str(path.parent.parent) + '/all_pgf/' + str(path.parent.stem) + '.pgf'
    f.savefig(str(path.parent) + '/fig.png', bbox_inches='tight')
    print(get_time(), 'Saved .png')
    if save_pgf:
        f.savefig(all_figs_path, bbox_inches='tight')
        print(get_time(), str(path.parent.stem) + ' saved.')

def setup_pgf_plotting(fontsize=10):
    # Setup saving of pgf-plots for thesis
    mpl.use("pgf")
    mpl.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': fontsize,
        'text.usetex': True,
        'pgf.rcfonts': False,
        'text.latex.preamble': [
            r'\usepackage{amsmath}',
            r'\usepackage[pdfspacing]{classicthesis}'
        ],
    })

def get_figure_width(frac_of_textwidth=1.0, rows=None, cols=None):
    textwidth = 4.65015 # inches
    return textwidth*frac_of_textwidth

def get_figure_height(width=None, rows=None, cols=None):
    ratio = 4.8/6.4

    if rows and cols:
        ratio = ratio * (rows/cols)**0.8
    return width * ratio

def get_frac_of_textwidth(keyword='single_fig', rows=None, cols=None):
    
    if keyword == 'single_fig':
        frac = 1.0
    elif keyword == '2subfigs':
        frac = 0.65
    elif keyword == 'subplots':
        frac = 1.3
    else:
        raise KeyError('Unknown keyword (%s) given to get_frac_of_textwidth!'%(keyword))
    return frac