#!/bin/bash -e
dvc pull -r models -R models/
dvc pull -r models data/masks.dvc
