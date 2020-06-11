from src.modules.thesis_plotting import *
from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
import os

setup_pgf_plotting()

def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim() 
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([]) # labels 
    plt.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute 
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin) 
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width 
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, ymin, xmax-xmin, 0., fc='k', ec='k', lw = lw, 
             head_width=hw, head_length=hl, overhang = ohg, 
             length_includes_head= True, clip_on = False) 

    ax.arrow(xmin, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw, 
             head_width=yhw, head_length=yhl, overhang = ohg, 
             length_includes_head= True, clip_on = False)

def main(): 
    # ============================================================================
    # IMPORT/MAKE DATA     
    # ============================================================================

    x = np.linspace(0.2, 2)
    a = 1.5
    overfitting = a*x + 1.0/x
    train = 1.00/x


    # ============================================================================
    # SET LABELS      
    # ============================================================================

    # x and y
    xlabel = r'Complexity'
    ylabel = r'Error'

    overfit_label = r'Loss on $\mathcal{S}_{\mathrm{val}}$'
    train_label = r'Loss on $\mathcal{S}_{\mathrm{train}}$'

    d = {
        'x': [x, x],
        'y': [train, overfitting],
        'label': [train_label, overfit_label],
        'ylabel': ylabel,
        'xlabel': xlabel,
        'axvline': [1/np.sqrt(a)],
        'grid': False
    }
    f = make_plot(d, for_thesis=True)

    # ============================================================================
    # ADD TEXT
    # ============================================================================
    text = r'Optimal complexity'
    xpos = 0.42
    ypos = 0.6
    # f.text(xpos, ypos, text)

    # ============================================================================
    # FINAL EDITS     
    # ============================================================================

    ax = plt.gca()
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    arrowed_spines(f, ax)

    # Standard ratio of width to height it 6.4/4.8
    # Standard figure: FOTW = 1.0
    # Subfigure 1/2: FOTW = 0.65. Remember to use a .5 cm of left and 0 cm of right

    FOTW = get_frac_of_textwidth(keyword='2subfigs')
    width = get_figure_width(frac_of_textwidth=FOTW)
    height = get_figure_height(width=width)
    f.set_size_inches(width, height)

    # ============================================================================
    # SAVE PGF AND PNG FOR VIEWING    
    # ============================================================================

    path = Path(os.path.realpath(__file__))
    save_thesis_pgf(path, f, save_pgf=True)

if __name__ == '__main__':
    main()