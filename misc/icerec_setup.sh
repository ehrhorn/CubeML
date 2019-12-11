#! /usr/bin/env bash
eval $(/cvmfs/icecube.opensciencegrid.org/py3-v4/setup.sh)
ICE_PATH=$(find $HOME -path "*/icerec/build/env-shell.sh")
$ICE_PATH
