from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
from matplotlib import pyplot as plt
import sklearn
import os
import pandas as pd

def make_data():
    table_pd = {}
    table_pd['Batchsize'] = r'32, 64, 128, 256, 512'
    table_pd['Optimizer'] = r'SGD, Adam, NAG'
    table_pd['LR schedule'] = r'Inverse decay w. warmup'
    table_pd['Layer Widths'] = r'64, 128, 256, 512, 1028'
    table_pd['Decoding ResBlocks'] = r'0, 1, 2, 3, 4, 5, 6'
    table_pd['Encoding Att. Blocks'] = r'0, 1, 2, 3, 4, 5, 6, 7'
    table_pd['Encoding RNN layers'] = r'0, 1, 2, 3, 4'
    table_pd['Encoding RNN type'] = r'Vanilla, GRU, LSTM, BiGRU, BiLSTM'
    table_pd['Nonlinearity'] = r'LeakyReLU, Mish'
    table_pd['Encoding norm.'] = r'None, LayerNorm'
    table_pd['Decoding norm.'] = r'None, BatchNorm'
    table_pd['Regularization'] = r'None, EarlyStopping, Dropout($p\in [30\%,\,50\%,\,80\%])$'
    table_pd['Regression loss'] = r'L1, L2, logcosh'
    table_pd['Classification loss'] = r'CrossEntropy'
    table_pd['Many-to-One'] = r'MaxPool, AvePool, KeepLast'
    table_pd['Weight init.'] = r'Kaiming'

    return table_pd
table = make_data()
names = [key for key in table]
spacing = [' ' for i in range(len(names))]
data = [data for _, data in table.items()]
table_pd = pd.DataFrame(np.array([names, spacing, data]).reshape((3,-1)).transpose())


latex_table = table_pd.to_latex(
    escape=False,
    header=['Hyperparameter', r'$\,$', 'Searchspace'],
    index=False,
    # column_format='rc'
    )
latex_table = latex_table.replace(r'\toprule', r'\toprule\toprule' )
# latex_table = latex_table.replace(r'\end{tabular}', r'\end{tabular}'+'\n%s'%(caption) )
# with open('mytable.tex','w') as tf:
    # tf.write(df.to_latex())
path = Path(os.path.realpath(__file__))
savepath = str(path.parent) + '/' + path.stem + '.tex'
with open(savepath, 'w') as f:
    f.write(latex_table)