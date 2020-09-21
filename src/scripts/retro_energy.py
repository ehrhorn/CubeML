# '''
# Functions related to computations we makes on
# Retro Reco frame objects

# Etienne Bourbeau, Kayla Leonard, Tom Stuttard
# '''

# import numpy as np
# from six import string_types

# def convert_EM_to_hadronic_cascade_energy(E_em):
#     '''
#     Convert EM cascade energy to hadronic equivalent (e.g. the energy of hadronic cascade
#     it would required to produce the same light as an EM cascade of the energy provided).
#     '''

#     # Redone hadronic factor
#     E_o = 0.18791678
#     m   = 0.16267529
#     f0  = 0.30974123

#     max_possible_Hd_cascade_energy = 10000.

#     assert np.all(E_em >= 0.), "Negative EM cascade energy found"
#     #TODO check EM energy is not out of max range for interpolation

#     HD_cascade_range = np.linspace(0.0,max_possible_Hd_cascade_energy,500001)
    
#     E_threshold = 0.2 #2.71828183

#     y = (HD_cascade_range/E_o)*(HD_cascade_range>E_threshold) + (E_threshold/E_o)*(HD_cascade_range<=E_threshold)

#     F_em = 1-y**(-m)

#     EM_cascade_energy= HD_cascade_range*(F_em + (1-F_em)*f0)
#     HD_casc_interpolated = np.interp(x=E_em,xp=EM_cascade_energy,fp=HD_cascade_range)

#     assert np.all(HD_casc_interpolated <= max_possible_Hd_cascade_energy), "Hadronic cascade energy out or range"

#     return HD_casc_interpolated

# def convert_retro_reco_energy_to_neutrino_energy(em_cascade_energy, track_length, GMS_LEN2EN=None) :
#     '''
#     Function to convert from the RetroReco fitted variables:
#       a) EM cascade energy
#       b) Track length
#     To the underlying neutrino properties:
#       a) Cascade energy (energy of all particles EXCEPT the outgoing muon)
#       b) Outgoing muon energy
#       c) Initial neutrino energy

#     We use:
#       - `convert_EM_to_hadronic_cascade_energy` to convert from EM to hadronic cascade energy
#       - `GMS_LEN2EN` to convert track length to energy
#       - Simple multiplicative fudge factors to correct the track length and cascade energy to 
#         best match the neutrino properties, based on the oscNext MC (weighted to Honda flux 
#         and nufit 2.0)

#     We see good agreement in total energy for nue/mu CC, and also for nutau CC and NC events 
#     but with an energy bias that matches the expectation due to the missing energy from final 
#     state neutrinos (~25% missing energyfor nutau CC, 50% for NC).

#     Agreement is worse for:
#       - Very low energy (<5 GeV), where there seems to be a floor in reco energy
#       - High energy (>100 GeV), where we seem to underestimate energy, although stats are bad 
#         here so hard to compute percentiles.

#     We also get good agreement for the track <-> muon length.

#     Good data-MC agreement is observed in all cases.
#     '''

#     # from retro.i3processing.retro_recos_to_i3files import GMS_LEN2EN
#     if GMS_LEN2EN == None:
#         _, GMS_LEN2EN, _ = generate_gms_table_converters(losses="all")
#     cascade_hadronic_energy = convert_EM_to_hadronic_cascade_energy(em_cascade_energy)
    
#     # Apply a fudge factor for overall cascade energy
#     cascade_energy = 1.7 * cascade_hadronic_energy

#     # Apply a fudge factor to the length
#     track_length = 1.45 * track_length

#     # Recompute track energy from fudged length, using GMS tables
#     track_energy = GMS_LEN2EN(track_length)

#     # Combine into a total energy
#     total_energy = cascade_energy + track_energy

#     return cascade_energy, track_energy, total_energy, track_length

# def generate_gms_table_converters(losses="all"):
#     """Generate converters for expected values of muon length <--> muon energy based on
#     the tabulated muon energy loss model [1], spline-interpolated for smooth behavior
#     within the range of tabulated energies / lengths.
#     Note that "gms" in the name comes from the names of the authors of the table used.
#     Parameters
#     ----------
#     losses : comma-separated str or iterable of strs
#         Valid sub-values are {"all", "ionization", "brems", "photonucl", "pair_prod"}
#         where if any in the list is specified to be "all" or if all of {"ionization",
#         "brems", "photonucl", and "pair_prod"} are specified, this supercedes all
#         other choices and the CSDA range values from the table are used..
#     Returns
#     -------
#     muon_energy_to_length : callable
#         Call with a muon energy to return its expected length
#     muon_length_to_energy : callable
#         Call with a muon length to return its expected energy
#     energy_bounds : tuple of 2 floats
#         (lower, upper) energy limits of table; below the lower limit, lengths are
#         estimated to be 0 and above the upper limit, a ValueError is raised;
#         corresponding behavior is enforced for lengths passed to `muon_length_to_energy`
#         as well.
#     References
#     ----------
#     [1] D. E. Groom, N. V. Mokhov, and S. I. Striganov, Atomic Data and Nuclear Data
#         Tables, Vol. 78, No. 2, July 2001, p. 312. Table II-28.
#     """
#     if isinstance(losses, string_types):
#         losses = tuple(x.strip().lower() for x in losses.split(","))

#     VALID_MECHANISMS = ("ionization", "brems", "pair_prod", "photonucl", "all")
#     for mechanism in losses:
#         assert mechanism in VALID_MECHANISMS

#     if "all" in losses or set(losses) == set(m for m in VALID_MECHANISMS if m != "all"):
#         losses = ("all",)

#     fpath = join(RETRO_DIR, "retro_data", "muon_stopping_power_and_range_table_II-28.csv")
#     table = np.loadtxt(fpath, delimiter=",")

#     kinetic_energy = table[:, 0] # (GeV)
#     total_energy = kinetic_energy + MUON_REST_MASS

#     mev_per_gev = 1e-3
#     cm_per_m = 1e2

#     if "all" in losses:
#         # Continuous-slowing-down-approximation (CSDA) range (cm * g / cm^3)
#         csda_range = table[:, 7]
#         mask = np.isfinite(csda_range)
#         csda_range = csda_range[mask]
#         ice_csda_range_m = csda_range / NOMINAL_ICE_DENSITY / cm_per_m # (m)
#         energy_bounds = (np.min(total_energy[mask]), np.max(total_energy[mask]))
#         _, muon_energy_to_length = generate_lerp(
#             x=total_energy[mask],
#             y=ice_csda_range_m,
#             low_behavior="constant",
#             high_behavior="extrapolate",
#             low_val=0,
#         )
#         _, muon_length_to_energy = generate_lerp(
#             x=ice_csda_range_m,
#             y=total_energy[mask],
#             low_behavior="constant",
#             high_behavior="extrapolate",
#             low_val=0,
#         )
#     else:
#         from scipy.interpolate import UnivariateSpline

#         # All stopping powers given in (MeV / cm * cm^3 / g)
#         stopping_power_by_mechanism = dict(
#             ionization=table[:, 2],
#             brems=table[:, 3],
#             pair_prod=table[:, 4],
#             photonucl=table[:, 5],
#         )

#         stopping_powers = []
#         mask = np.zeros(shape=table.shape[0], dtype=bool)
#         for mechanism in losses:
#             addl_stopping_power = stopping_power_by_mechanism[mechanism]
#             mask |= np.isfinite(addl_stopping_power)
#             stopping_powers.append(addl_stopping_power)
#         stopping_power = np.nansum(stopping_powers, axis=0)[mask]
#         stopping_power *= cm_per_m * mev_per_gev * NOMINAL_ICE_DENSITY

#         valid_energies = total_energy[mask]
#         energy_bounds = (valid_energies.min(), valid_energies.max())
#         sample_energies = np.logspace(
#             start=np.log10(valid_energies.min()),
#             stop=np.log10(valid_energies.max()),
#             num=1000,
#         )
#         spl = UnivariateSpline(x=valid_energies, y=1/stopping_power, s=0, k=3)
#         ice_range = np.array(
#             [spl.integral(valid_energies.min(), e) for e in sample_energies]
#         )
#         _, muon_energy_to_length = generate_lerp(
#             x=sample_energies,
#             y=ice_range,
#             low_behavior="constant",
#             high_behavior="extrapolate",
#             low_val=0,
#         )
#         _, muon_length_to_energy = generate_lerp(
#             x=ice_range,
#             y=sample_energies,
#             low_behavior="constant",
#             high_behavior="extrapolate",
#             low_val=0,
#         )

#     return muon_energy_to_length, muon_length_to_energy, energy_bounds