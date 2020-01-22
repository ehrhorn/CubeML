#!/bin/bash -e
dvc add data/MuonGun_Level2_139008
dvc add data/oscnext-genie-level5-v01-01-pass2
dvc push -r myremote data/MuonGun_Level2_139008.dvc data/oscnext-genie-level5-v01-01-pass2.dvc
