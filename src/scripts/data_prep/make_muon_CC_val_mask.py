import pickle
from src.modules.constants import *

with open(PATH_DATA_OSCNEXT + '/matched_val.pickle', 'rb') as f:
    found = pickle.load(f)

muons_path = PATH_DATA_OSCNEXT + '/masks/muon_neutrino_val.pickle'
with open(muons_path, 'rb') as f:
    muons = [str(e) for e in pickle.load(f)]

new_mask = []
for e in muons:
    if e in found:
        if found[e]['interaction_type'] == 1: # CC
            new_mask.append(int(e))

print(len(new_mask)/len(muons))
muons_CC_val_path = PATH_DATA_OSCNEXT + '/masks/muon_CC_neutrino_val.pickle'
with open(muons_CC_val_path, 'wb') as f:
    pickle.dump(new_mask, f)