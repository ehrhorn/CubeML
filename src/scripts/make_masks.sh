python data_prep/make_sqlite_mask.py --name all
python data_prep/make_sqlite_mask.py --name muon_neutrino
python data_prep/make_sqlite_mask.py --name electron_neutrino
python data_prep/make_sqlite_mask.py --name tau_neutrino
python data_prep/make_sqlite_mask.py --name dom_interval_SRTInIcePulses --min_doms 0 --max_doms 200
python data_prep/make_sqlite_mask.py --name energy_interval --min_energy 0.0 --max_energy 3.0

python data_prep/combine_masks.py --masks muon_neutrino electron_neutrino --name nue_numu
python data_prep/combine_masks.py --masks muon_neutrino electron_neutrino tau_neutrino --name nue_numu_nutau

python data_prep/sqlite_weight_calc.py --name nue_numu_balanced --masks nue_numu
python data_prep/sqlite_weight_calc.py --name nue_numu_nutau_balanced --masks nue_numu_nutau
python data_prep/sqlite_weight_calc.py --name inverse_performance_muon_energy --masks muon_neutrino
python data_prep/sqlite_weight_calc.py --masks energy_interval_min0.0_max3.0 --name energy_balanced --make_plot