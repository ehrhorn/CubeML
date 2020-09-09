
from src.modules.helper_functions import *
from src.modules.reporting import *
from src.modules.classes import *
import numpy as np
import pickle

from src.modules.reporting import *
from src.modules.constants import *
from matplotlib import pyplot as plt
from src.modules.classes import SqliteFetcher
from src.modules.thesis_plotting import *
import os

# ! INSERT MASKNAMES HERE
masks = [
    'tau_neutrino'
]
ids = [str(e) for e in load_sqlite_mask(PATH_DATA_OSCNEXT, masks, 'val')]


# ! INSERT VARIABLENAME FOR INSPECTION HERE
scalar_var = ['energy_balanced_alpha70', 'true_primary_energy']
db = SqliteFetcher(PATH_VAL_DB)
events = db.fetch_features(
    all_events=ids[:10], scalar_features=scalar_var
    )

for event, data in events.items():
    print(event)
    print(data)
    print('')












