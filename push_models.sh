#!/bin/bash -e
dvc push -r models -R models/
dvc push -r models data/masks.dvc
