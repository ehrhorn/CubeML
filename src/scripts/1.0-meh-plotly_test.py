# %%
import h5py
from sklearn.model_selection import train_test_split
from src.data.shovel_funcs import space_time_cleanup, transform_events
import numpy as np
import torch
from torch.utils import data
from scipy.spatial.distance import cdist

# Get event data
in_file = '/Users/mads/Downloads/CubeML/data/processed/' \
      'MuonGun_Level2_139008/000000.h5'
with h5py.File(in_file, 'r') as f:
      events = {}
      for key in f['processed_data']:
            events[key] = f['processed_data'][key][:]

# Split test/training/val sets
event_ids = list(range(len(events['true_muon_energy'][:])))
train_ids, test_ids = train_test_split(
    event_ids,
    test_size=0.2,
    random_state=29897070
)
train_ids, val_ids = train_test_split(
    event_ids,
    test_size=0.2,
    random_state=29897070
)

# Misc
targets = events['true_muon_energy'][:]
longest_event = max([len(events['dom_x'][i]) for i in event_ids])
# %%
# Cleanup function
def space_time_cleanup(activations, clean_distance):
    activations = activations[:, 0:4]
    dom_spacetime_distance_table = cdist(activations, activations)
    print(dom_spacetime_distance_table)
    np.fill_diagonal(dom_spacetime_distance_table, np.nan)
    good_activation_mins = np.nanmin(dom_spacetime_distance_table, axis=0)
    print(good_activation_mins)
    good_activation_doms = np.where(
          good_activation_mins < int(clean_distance)
      )
    activations = activations[
          np.isin(
                activations[:, 0],
                good_activation_doms
            )
      ]
    return activations

# Test
my_keys = ['dom_x', 'dom_y', 'dom_z', 'time', 'charge']
temp_np = np.zeros((44, len(my_keys)))
event_length = len(events['dom_x'][train_ids[0]])
for i, key in enumerate(my_keys):
      temp_np[0:event_length, i] = events[key][train_ids[0]]
test = space_time_cleanup(temp_np, 300)
# %%
# Dataloader
class Dataset(data.Dataset):
      def __init__(self, list_IDs, labels, keys, longest_event):
            self.labels = labels
            self.list_IDs = list_IDs
            self.keys = keys
            self.longest_event = longest_event
      def __len__(self):
            return len(self.list_IDs)
      def __getitem__(self, index):
            # Select sample
            ID = self.list_IDs[index]
            # Load data and get label        
            event_length = len(events['dom_x'][ID])
            X = np.zeros((event_length, len(self.keys)))
            for i, key in enumerate(self.keys):
                  X[0:event_length, i] = events[key][ID]
            y = self.labels[ID]
            return X, y
