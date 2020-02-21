#!/bin/bash
python make_pickle_mask.py --name dom_interval_SRTInIcePulses --min_doms 0 --max_doms 40
python make_pickle_mask.py --name dom_interval_SRTInIcePulses --min_doms 40 --max_doms 100
python make_pickle_mask.py --name dom_interval_SRTInIcePulses --min_doms 100 --max_doms 200
