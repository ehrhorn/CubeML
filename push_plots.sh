#!/bin/bash -e
dvc add plots
dvc push -r models plots.dvc
