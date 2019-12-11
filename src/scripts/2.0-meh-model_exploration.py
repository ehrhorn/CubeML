# %%
import h5py
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
from ignite.engine import Events
from ignite.engine import create_supervised_trainer
from ignite.engine import create_supervised_evaluator
from ignite import metrics as mtr
from sklearn.model_selection import train_test_split
from src.shovel_funcs import space_time_cleanup, transform_events

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def collate_fn_padd(batch):
#     '''
#     Padds batch of variable length

#     note: it converts things ToTensor manually here since the ToTensor transform
#     assume it takes in images rather than arbitrary tensors.
#     '''
#     ## get sequence lengths
#     print(type(batch[0]))
#     lengths = torch.tensor([ t.shape[0] for t in batch ]).to(device)
#     ## padd
#     batch = [ torch.Tensor(t).to(device) for t in batch ]
#     batch = torch.nn.utils.rnn.pad_sequence(batch)
#     ## compute mask
#     mask = (batch != 0).to(device)
#     return batch, lengths, mask


class SimpleNet(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=1, stride=2)
        )
        self.fc = nn.Linear(589, num_classes)
    def forward(self, x):
        # print('Input:', x.shape)
        out = self.layer1(x)
        # print('Layer 1:', out.shape)
        out = self.layer2(out)
        # print('Layer 2:', out.shape)
        out = self.layer3(out)
        # print('Layer 3:', out.shape)
        out = torch.sum(out, dim=1)
        # print('Sum layer:', out.shape)
        out = out.reshape(out.size(0), -1)
        # print('Flattening:', out.shape)
        out = self.fc(out)
        # print('Linear layer:', out.shape)
        return out


class Dataset(data.Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        ID = self.ids[index]
        X = np.load(save_root_dir + interim_dir + str(ID) + '.npy')
        X = torch.from_numpy(X)
        y = self.labels[ID]
        y = np.array(y)
        y = y.reshape(1)
        y = torch.from_numpy(y)

        return X, y


# %%
ids = np.arange(0, no_of_events)

in_file = 'data/interim/MuonGun_Level2_139008.000000.h5'
_, transformed_events = transform_events(in_file)
events = list(range(len(transformed_events['true_muon_energy'][:])))
targets = transformed_events['true_muon_energy'][:]
longest_event = 

train_id, test_id = train_test_split(
    events,
    test_size=0.2,
    random_state=29897070
)
train_id, val_id = train_test_split(
    events,
    test_size=0.2,
    random_state=29897070
)

training_set = Dataset(train_id, targets)
val_set = Dataset(val_id, targets)
test_set = Dataset(test_id, targets)

batch_size = 16
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}
# %%
model = SimpleNet(num_classes=1)
model.double()

train_loader = data.DataLoader(training_set, **params,
    collate_fn=collate_fn_padd)
val_loader = data.DataLoader(val_set, **params,
    collate_fn=collate_fn_padd)
test_loader = data.DataLoader(test_set, **params,
    collate_fn=collate_fn_padd)

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
loss = nn.L1Loss()

trainer = create_supervised_trainer(model, optimizer, loss)
evaluator = create_supervised_evaluator(
    model,
    metrics={
        'nll': mtr.Loss(loss)
    }
)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print("Training Results - Epoch: {}  Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['nll']))


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print("Validation Results - Epoch: {}  Avg loss: {:.2f}"
          .format(trainer.state.epoch, metrics['nll']))


trainer.run(train_loader, max_epochs=100)
# %%
