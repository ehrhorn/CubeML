# %%
from tables import *
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def histogram_reader(file, group, variable):
    with File(file, 'r') as f:
        array = f.root._f_get_child(group)._f_get_child(variable).read()
    return array


file = Path('/datadrive/home/mads/osc_test/oscNext_genie_level5_v01.01_pass2.120000.000000__retro_crs_prefit.h5')
array = np.array(histogram_reader(file, 'raw', 'dom_charge'))
array = np.concatenate(array).ravel()
array = array[(array < 1)]
fig, ax = plt.subplots()
ax.hist(array, histtype='step', bins='fd')
fig.show()
# %%
array_around = array[(array < -2) & (array > -2.1)]
print(array_around)

# %%
