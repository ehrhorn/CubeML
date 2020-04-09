import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path
import argparse
import sys
import sqlite3

from src.modules.helper_functions import get_project_root, get_time, get_path_from_root, make_multiprocess_pack, convert_keys
from src.modules.reporting import make_plot
from src.modules.constants import *
# from src.modules.preprocessing import *

sequential_keys = [
    'dom_x',
    'dom_y',
    'dom_z',
    'dom_charge',
    'dom_time',
    'dom_atwd',
    'dom_pulse_width',
    'dom_charge_significance',
    'dom_frac_of_n_doms',
    'dom_d_to_prev',
    'dom_v_from_prev',
    'dom_d_minkowski_to_prev',
    'dom_d_closest',
    'dom_d_minkowski_closest',

    'event',
    'dom_key',
    'pulse_no',
    'SplitInIcePulses',
    'SRTInIcePulses'
]

scalar_keys = [
    'event_no',
    'dom_timelength_fwhm',
    'true_primary_direction_x',
    'true_primary_direction_y',
    'true_primary_direction_z',
    'true_primary_position_x',
    'true_primary_position_y',
    'true_primary_position_z',
    'true_primary_time',
    'true_primary_energy',
    'linefit_direction_x',
    'linefit_direction_y',
    'linefit_direction_z',
    'linefit_point_on_line_x',
    'linefit_point_on_line_y',
    'linefit_point_on_line_z',
    'toi_direction_x',
    'toi_direction_y',
    'toi_direction_z',
    'toi_point_on_line_x',
    'toi_point_on_line_y',
    'toi_point_on_line_z',
    'toi_eval_ratio',
    'retro_crs_prefit_x',
    'retro_crs_prefit_y',
    'retro_crs_prefit_z',
    'retro_crs_prefit_time',
    'retro_crs_prefit_energy',
    'retro_crs_prefit_azimuth',
    'retro_crs_prefit_zenith',
]

meta_keys = [
    'event_no',
    'file',
    'idx',
    'particle_code',
    'level',
    'split_in_ice_pulses_event_length',
    'srt_in_ice_pulses_event_length'
]
# Open one picklefile, extract names
pickle_path = '/'.join([PATH_DATA_OSCNEXT, 'pickles', '757', '7570001.pickle'])
pickle_event = pickle.load(open(pickle_path, 'rb'))
for key in pickle_event:
    print(key)
    for key2 in pickle_event[key]:
        print(key2)
    print('')
# Make db's

# loop over pickle directories

