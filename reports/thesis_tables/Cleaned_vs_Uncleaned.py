from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
# from matplotlib import pyplot as plt
# import sklearn
import os
import pandas as pd

def make_data():
    data = {
        'Preproc. Depth': [],
        'Decode Depth': [],
        'Width': [],
        'Encode Depth': [],
        'Error': [],
    }
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
        r'$RNN(2, 256, 4)$',
    ]

    perf_keys = {
        'log_frac_E_error_sigma': r'$W\left(\log_{10} \left[ \frac{E_{pred}}{E_{true}} \right]\right)$',
        'len_error_68th': r'$U(|\vec{x}_{reco}-\vec{x}_{true}|)$ [m]',
        'directional_error_68th': r'$U\left(\Delta\Psi\right)$ [deg]',
        'vertex_t_error_sigma': r'$W(\Delta t)$ [ns]'
    }

    low = np.empty((len(names), len(perf_keys)))
    mid = np.empty((len(names), len(perf_keys)))
    high = np.empty((len(names), len(perf_keys)))

    for i_model, model_full in enumerate(models):
        model = model_full.split('/')[-1].split('?')[0]
        model_path = locate_model(model)
        data_path = model_path + '/data/Performance.pickle'
        performance = vars(pickle.load(open(data_path, 'rb')))
        for i_key, key in enumerate(perf_keys):
            low[i_model, i_key] = num2str(np.mean(performance[key][:6]))
            mid[i_model, i_key] = num2str(np.mean(performance[key][6:12]))
            high[i_model, i_key] = num2str(np.mean(performance[key][12:]))

            if len(performance[key][12:]) != 6 or len(performance[key][6:12]) != 6 or len(performance[key][:6]) != 6:
                raise ValueError('Wrong length!')

        # for each model, load min error
        
    
    df_cols = [data for name, data in perf_keys.items()]
    # print(df_cols)
    # df_low = pd.DataFrame(low)
    # print(df_low)
    # a+=1     
    df_low = pd.DataFrame(low, index=names, columns=df_cols)
    df_mid = pd.DataFrame(mid, index=names, columns=df_cols)
    df_high = pd.DataFrame(high, index=names, columns=df_cols)

    # table_low = df_low.to_latex(header=df_cols, escape=False, column_format='r')

    # print(table_low)
    path = Path(os.path.realpath(__file__))

    savepath = str(path.parent) + '/' + path.stem + '.tex'
    with open(savepath, 'w') as f:
        f.write(
            r'\begin{tabular}{lcccc}' + '\n' + r'\toprule' + '\n'
        )
        _names = ['Model'] + df_cols 
        f.write(
            ' & '.join(_names) + ''.join([r'\\', '\n', r'\midrule\midrule', '\n'])
        )

        # ADD FIRST SUBHEADER
        subheader = r'$\log_{10} E \in [0, 1)$, $\left[\frac{E}{GeV}\right]$'
        f.write(
            '& & %s & & \\\\'%(subheader) + '\n' + r'\midrule'+ '\n'
        )

        # ADD DATA - LOW E
        min_vals = np.min(low, axis=0)
        for i in range(len(names)):
            data = ' & '.join(
                [names[i]] + [str(e) if e not in min_vals else '\\textbf{%s}'%(str(e)) for e in list(low[i, :]) ]
            )
            f.write(
                data + '\\\\ \n'
            )
        
        # ADD SUBHEADER 2
        subheader = r'$\log_{10} E \in [1, 2)$, $\left[\frac{E}{GeV}\right]$'
        f.write(
            '\\midrule \\midrule \n' + '& & %s & & \\\\'%(subheader) + '\n' + r'\midrule'+ '\n'
        )
        
        # MID E
        min_vals = np.min(mid, axis=0)
        for i in range(len(names)):
            data = ' & '.join(
                [names[i]] + [str(e) if e not in min_vals else '\\textbf{%s}'%(str(e)) for e in list(mid[i, :]) ]
            )
            f.write(
                data + '\\\\ \n'
            )
        # ADD SUBHEADER 3
        subheader = r'$\log_{10} E \in [2, 3)$, $\left[\frac{E}{GeV}\right]$'
        f.write(
            '\\midrule \\midrule \n' + '& & %s & & \\\\'%(subheader) + '\n' + r'\midrule'+ '\n'
        )
        
        # HIGH E
        min_vals = np.min(high, axis=0)
        for i in range(len(names)):
            data = ' & '.join(
                [names[i]] + [str(e) if e not in min_vals else '\\textbf{%s}'%(str(e)) for e in list(high[i, :]) ]
            )
            f.write(
                data + '\\\\ \n'
            )
        f.write(
            '\\bottomrule \n \\end{tabular}'
        )

    return df_low, df_mid, df_high

low, mid, high = make_data()

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