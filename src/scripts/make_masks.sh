python data_prep/make_sqlite_mask.py --name all
python data_prep/make_sqlite_mask.py --name muon_neutrino
python data_prep/make_sqlite_mask.py --name electron_neutrino
python data_prep/make_sqlite_mask.py --name tau_neutrino
python data_prep/make_sqlite_mask.py --name dom_interval_SRTInIcePulses --min_doms 0 --max_doms 200
python data_prep/make_sqlite_mask.py --name energy_interval --min_energy 0.0 --max_energy 3.0

python data_prep/combine_masks.py --masks muon_neutrino electron_neutrino --name nue_numu
python data_prep/combine_masks.py --masks tau_neutrino electron_neutrino --name nue_nutau
python data_prep/combine_masks.py --masks muon_neutrino electron_neutrino tau_neutrino --name nue_numu_nutau

python data_prep/sqlite_weight_calc.py --name nue_numu_balanced --masks nue_numu
python data_prep/sqlite_weight_calc.py --name nue_numu_nutau_balanced --masks nue_numu_nutau
python data_prep/sqlite_weight_calc.py --name inverse_performance_muon_energy --masks muon_neutrino
python data_prep/sqlite_weight_calc.py --masks energy_interval_min0.0_max3.0 --name energy_balanced --make_plot --save_interpolator
python data_prep/sqlite_weight_calc.py --masks energy_interval_min0.0_max3.0 --name energy_balanced --make_plot --save_interpolator --alpha 0.7
python data_prep/sqlite_weight_calc.py --masks energy_interval_min0.0_max3.0 --name uniform_direction_weights --make_plot
python data_prep/sqlite_weight_calc.py --masks muon_neutrino --name inverse_performance_muon_energy --make_plot --save_interpolator

python data_prep/make_low_E_high_E_interpolators.py
python data_prep/sqlite_weight_calc.py --masks energy_interval_min0.0_max3.0 --name inverse_low_E --make_plot --interpolator
python data_prep/sqlite_weight_calc.py --masks energy_interval_min0.0_max3.0 --name inverse_high_E --make_plot --interpolator

# If wanted, do ensemble predictions
python -u make_ensemble_predictions.py --predefined full_reg