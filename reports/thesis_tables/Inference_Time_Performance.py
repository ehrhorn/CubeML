from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
# from matplotlib import pyplot as plt
# import sklearn
import os
import pandas as pd

def make_data():
    # data = {
    #     'Preproc. Depth': [],
    #     'Decode Depth': [],
    #     'Width': [],
    #     'Encode Depth': [],
    #     'Error': [],
    # }
    models = [
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-16.47.07?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-17-04.51.03?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-21-15.26.17?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-21-14.05.05?workspace=user-bjoernmoelvig',
        'https://app.wandb.ai/cubeml/cubeml/runs/2020-07-12-13.19.47?workspace=user-bjoernmoelvig'
    ]

    names = [
        r'$RNN^{SRT}(4, 128, 2)$',
        r'$RNN^{SRT}(3, 128, 2)$',
        r'$RNN(3, 256, 2)$',
        r'$RNN(3, 128, 2)$',
        # r'$RNN(2, 256, 4)$',
    ]

    inf_speed = [
        7233,
        8047,
        6338,
        7325,
      #  10445,
    ]

    parameters = np.array([
        1183112,
        886408,
        3402120,
        886408,
       # 3176968, 
    ])
    prec_byte = 32/8
    memory = parameters*prec_byte/10**6

    path = Path(os.path.realpath(__file__))

    savepath = str(path.parent) + '/' + path.stem + '.tex'
    with open(savepath, 'w') as f:
        f.write(
            r'\begin{tabular}{lccccc}' + '\n' + r'\toprule' + '\n'
        )
        _names = [' '] + names
        f.write(
            ' & '.join(_names) + ''.join([r'\\', '\n', r'\midrule', '\n'])
        )
        
        # ADD INFERENCE TIME
        data = ' & '.join(
            ['Inference Speed'] + [str(e) for e in inf_speed]
        )
        f.write(
            data + '\\\\ \n'
        )
        
        # ADD PARAMETERS
        data = ' & '.join(
            ['Parameters'] + [str('%.2f M'%(e/1e6)) for e in parameters]
        )
        f.write(
            data + '\\\\ \n'
        )

        # ADD SIZE
        data = ' & '.join(
            ['Size'] + [str('%.2f MB'%(e)) for e in memory]
        )
        f.write(
            data + '\\\\ \n'
        )
        # # ADD SUBHEADER 2
        # subheader = r'$\log_{10} E \in [1, 2)$, $\left[\frac{E}{GeV}\right]$'
        # f.write(
        #     '\\midrule \\midrule \n' + '& & %s & & \\\\'%(subheader) + '\n' + r'\midrule'+ '\n'
        # )
        
        # # MID E
        # min_vals = np.min(mid, axis=0)
        # for i in range(len(names)):
        #     data = ' & '.join(
        #         [names[i]] + [str(e) if e not in min_vals else '\\textbf{%s}'%(str(e)) for e in list(mid[i, :]) ]
        #     )
        #     f.write(
        #         data + '\\\\ \n'
        #     )
        # # ADD SUBHEADER 3
        # subheader = r'$\log_{10} E \in [2, 3)$, $\left[\frac{E}{GeV}\right]$'
        # f.write(
        #     '\\midrule \\midrule \n' + '& & %s & & \\\\'%(subheader) + '\n' + r'\midrule'+ '\n'
        # )
        
        # # HIGH E
        # min_vals = np.min(high, axis=0)
        # for i in range(len(names)):
        #     data = ' & '.join(
        #         [names[i]] + [str(e) if e not in min_vals else '\\textbf{%s}'%(str(e)) for e in list(high[i, :]) ]
        #     )
        #     f.write(
        #         data + '\\\\ \n'
        #     )
        f.write(
            '\\bottomrule \n\\end{tabular}'
        )


make_data()

# \begin{tabular}{lcccc}
# \toprule
# {} &  $W\left(\log_{10} \left[ \frac{E_{pred}}{E_{true}} \right]\right)$ &  $U(|\vec{x}_{reco}-\vec{x}_{true}|)$ [m] &  $U\left(\Delta\Psi\right)$ [deg] &  $W(\Delta t)$ [ns] \\
# \midrule
# $RNN^{SRT}(4, 128, 2)$ &                                             0.2646 &                                     36.05 &                             25.36 &               54.90 \\
# $RNN^{SRT}(3, 128, 2)$ &                                             0.2658 &                                     36.80 &                             25.49 &               54.14 \\
# $RNN(3, 256, 2)$       &                                             0.2462 &                                     35.31 &                             24.56 &               52.22 \\
# $RNN(3, 128, 2)$       &                                             0.2496 &                                     35.87 &                             25.28 &               55.95 \\
# $RNN(2, 256, 4)$       &                                             0.2518 &                                     34.84 &                             24.69 &               54.93 \\
# \bottomrule
# \end{tabular}
# for df, extension in zip([low, mid, high], ['_low.tex', '_mid.tex', '_high.tex']):
#     savepath = str(path.parent) + '/' + path.stem + extension
#     with open(savepath, 'w') as f:
#         table = df.to_latex(escape=False)
#         table = table.replace(r'{lrrrr}', r'{lcccc}' )
#         f.write(table)