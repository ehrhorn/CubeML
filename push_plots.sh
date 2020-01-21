#!/bin/bash -e
dvc add reports
dvc push -r models reports.dvc
