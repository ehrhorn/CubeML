#!/bin/bash
python shelve2shelve.py --path /groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/shelve/train_set --new_name transformed_train_set --fit_transformers
python shelve2shelve.py --path /groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/shelve/val_set --new_name transformed_val_set
python shelve2shelve.py --path /groups/hep/bjoernhm/CubeML/data/oscnext-genie-level5-v01-01-pass2/shelve/test_set --new_name transformed_test_set
