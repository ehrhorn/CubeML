import src.modules.helper_functions as hf
import src.modules.reporting as rpt
from src.modules.classes import *
import numpy as np


# dbs = {
#     'train': PATH_TRAIN_DB, 
#     'val': PATH_VAL_DB
# }
particle_masks = {
    'electron': PATH_DATA_OSCNEXT+'/masks/electron_neutrino',
    'muon': PATH_DATA_OSCNEXT+'/masks/muon_neutrino',
    'tau': PATH_DATA_OSCNEXT+'/masks/tau_neutrino'
}
masks = {
    'SRT': PATH_DATA_OSCNEXT+'/masks/dom_interval_SRTInIcePulses_min0_max200',
    'energy': PATH_DATA_OSCNEXT+'/masks/energy_interval_min0.0_max3.0'
}

for setname, suffix in zip(['train', 'val'], ['_train.pickle', '_val.pickle']):
    srt = pickle.load(open(masks['SRT']+suffix, 'rb'))
    energy = pickle.load(open(masks['energy']+suffix, 'rb'))

    for particle, particle_mask in particle_masks.items():
        pmask = pickle.load(open(particle_mask+suffix, 'rb'))
        print('No cut:', setname, particle, len(pmask))

        cut = set(pmask) & set(energy) & set(srt)
        print('With cut:', setname, particle, len(list(cut)))
    print('')
