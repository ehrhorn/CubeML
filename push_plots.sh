#!/bin/bash -e
dvc add bak_plots
dvc push -r models bak_plots.dvc
