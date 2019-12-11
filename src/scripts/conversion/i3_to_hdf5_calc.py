"""Reading some atmopsheric muon sim files and grabbing some basic information.

How to run: `python i3_to_hdf5_calc.py`

Note that an IceTray environment _must_ be loaded.

Tom Stuttard
Ammendments by Mads Ehrhorn
"""
import os
import numpy as np
from pathlib import Path
from I3Tray import I3Tray
from icecube import icetray
from icecube import dataclasses
from icecube import tensor_of_inertia
from i3_download import download_i3_gcd_file
from i3_download import download_i3_files
from i3_download import enumerate_i3_files
from argparse import ArgumentParser
from tables import *
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument(
    '-i',
    '--i3_type',
    dest='i3_type',
    help='What i3 type to download'
)
parser.add_argument(
    '-n',
    '--number_of_files_to_download',
    dest='number_of_files_to_download',
    help='How many files to download'
)


args = parser.parse_args()
i3_type = args.i3_type
number_of_files_to_download = int(args.number_of_files_to_download)

if __name__ == '__main__':
    data_dir = Path('./temp')
    data_dir.mkdir(parents=True, exist_ok=True)
    # Geometry/Calibration/DOMs file
    gcd_file_name = download_i3_gcd_file(data_dir)
    gcd_file = data_dir.joinpath(gcd_file_name)

    i3_files_blob_names = enumerate_i3_files(
        number_of_files_to_download,
        i3_type
    )
    i3_files_downloaded = download_i3_files(
        i3_files_blob_names,
        data_dir,
        i3_type
    )
    # Muon simulation files
    i3_files = sorted(i3_files_downloaded)
    # i3_files = i3_files[:1]
    column_names = ['true_muon_energy',
        'true_muon_direction_x',
        'true_muon_direction_y',
        'true_muon_direction_z',
        'true_muon_entry_position_x',
        'true_muon_entry_position_y',
        'true_muon_entry_position_z',
        'linefit_direction_x',
        'linefit_direction_y',
        'linefit_direction_z',
        'linefit_point_on_line_x',
        'linefit_point_on_line_y',
        'linefit_point_on_line_z',
        'toi_direction_x',
        'toi_direction_y',
        'toi_direction_z',
        'toi_point_on_line_x',
        'toi_point_on_line_y',
        'toi_point_on_line_z',
        'toi_evalratio',
        'dom_x',
        'dom_y',
        'dom_z',
        'time',
        'charge'
    ]

    def do_things(frame):
        """Retrieve and calculate frames from an i3 file.

        Args:
            frame (i3 frame): i3 frame from i3 file.

        Returns:
            i3 object: i3 object with requested frames.

        """
        cond1 = frame['I3EventHeader'].sub_event_stream != 'InIceSplit'
        cond2 = 'LineFit' not in frame or 'ToI' not in frame
        cond3 = 'SRTInIcePulses' not in frame
        final_cond = cond1 or cond2 or cond3

        if final_cond:
            return False
        else:
            linefit_result = frame['LineFit']
            linefit_fit_status = linefit_result.fit_status
            toi_result = frame['ToI']
            toi_fit_status = toi_result.fit_status
            if linefit_fit_status != 0 or toi_fit_status != 0:
                return False

        # TODO Could we maybe do this in a smarter, programmatic way?
        # Get the true muon particle
        # Note that the muon is really produced in the atmosphere, but for the
        # simulation it is generated at the surface of a cylinder
        # surrounding IceCube
        # <icecube.dataclasses.I3Particle>
        true_muon = dataclasses.get_most_energetic_muon(frame['I3MCTree'])

        # Add the muon energy (defined at the generation cylinder) to the
        # data dicitonary
        data['true_muon_energy'].append(np.log(true_muon.energy))

        # Direction of the muon
        # <icecube.dataclasses.I3Particle>
        true_muon_direction = true_muon.dir
        # Add the muon direction to the data dictionary
        data['true_muon_direction_x'].append(true_muon_direction.x)
        data['true_muon_direction_y'].append(true_muon_direction.y)
        data['true_muon_direction_z'].append(true_muon_direction.z)

        # Point on the generation cylinder at which the muon is produced
        # <icecube.dataclasses.I3Particle>
        true_muon_entry_position = true_muon.pos
        # Add the entry position to the data dictionary
        data['true_muon_entry_position_x'].append(true_muon_entry_position.x)
        data['true_muon_entry_position_y'].append(true_muon_entry_position.y)
        data['true_muon_entry_position_z'].append(true_muon_entry_position.z)

        # Get the uncleaned pulses
        # <icecube.dataclasses.I3RecoPulseSeriesMap>
        uncleaned_pulses = frame['SplitInIcePulses'].apply(frame)

        dom_geom = frame['I3Geometry'].omgeo

        # Create empty lists for holding the pulse information
        # TODO Turn into Numpy arrays from the start
        dom_x_temp = []
        dom_y_temp = []
        dom_z_temp = []
        time_temp = []
        charge_temp = []

        # Go through all pulses, get OM key and pair with time and charge info
        for entry in uncleaned_pulses.items():
            this_om_key = entry[0]
            # This grabs you an object containing the geometry for this
            # particular OM
            this_om_geom = dom_geom[this_om_key]
            # This has x,y,z members
            this_om_position = this_om_geom.position
            for pulse in entry[1]:
                dom_x_temp.append(this_om_position.x)
                dom_y_temp.append(this_om_position.y)
                dom_z_temp.append(this_om_position.z)
                time_temp.append(pulse.time)
                charge_temp.append(pulse.charge)

        # Turn lists into Numpy array, for h5py's sake
        dom_x_temp = np.array(dom_x_temp)
        dom_y_temp = np.array(dom_y_temp)
        dom_z_temp = np.array(dom_z_temp)
        time_temp = np.array(time_temp)
        charge_temp = np.array(charge_temp)

        # Add Numpy arrays to data dictionary
        data['dom_x'].append(dom_x_temp)
        data['dom_y'].append(dom_y_temp)
        data['dom_z'].append(dom_z_temp)
        data['time'].append(time_temp)
        data['charge'].append(charge_temp)

        # Get the cleaned pulses (if available)
        # <icecube.dataclasses.I3RecoPulseSeriesMap>
        # cleaned_pulses = frame['SRTInIcePulses'].apply(frame)

        # Get the LineFit reconstruction
        # The direction of the straight line
        # <icecube.dataclasses.I3Particle>
        linefit_direction = linefit_result.dir
        # Add the direction of the linefit to the data dictionary
        data['linefit_direction_x'].append(linefit_direction.x)
        data['linefit_direction_y'].append(linefit_direction.y)
        data['linefit_direction_z'].append(linefit_direction.z)

        # An arbitrary point along the line
        # <icecube.dataclasses.I3Particle>
        linefit_point_on_line = linefit_result.pos
        # Add the arbitrary point along the line to the data dictionary
        data['linefit_point_on_line_x'].append(linefit_point_on_line.x)
        data['linefit_point_on_line_y'].append(linefit_point_on_line.y)
        data['linefit_point_on_line_z'].append(linefit_point_on_line.z)

        # Some additional params
        # <icecube.recclasses.I3LineFitParams>
        # linefit_params = frame['LineFitParams']

        # Get the tensor of inertia
        # The direction of the ToI
        # <icecube.dataclasses.I3Particle>
        toi_direction = toi_result.dir
        # Add the direction to the data dictionary
        data['toi_direction_x'].append(toi_direction.x)
        data['toi_direction_y'].append(toi_direction.y)
        data['toi_direction_z'].append(toi_direction.z)

        # An arbitrary point along the line
        # <icecube.dataclasses.I3Particle>
        toi_point_on_line = toi_result.pos
        # Add arbitrary point along the line to data dictionary
        data['toi_point_on_line_x'].append(toi_point_on_line.x)
        data['toi_point_on_line_y'].append(toi_point_on_line.y)
        data['toi_point_on_line_z'].append(toi_point_on_line.z)

        # Some additional params
        # <icecube.recclasses.I3TensorOfInertiaFitParams>
        toi_params = frame['ToIParams']
        # This is the ratio of the smallest component of the ToI to the sum of
        # them all. A value close to 0. means a track-like event.
        # Addd it to the data dictionary
        data['toi_evalratio'].append(toi_params.evalratio)
    
    out_folder_name = Path('/home/mads/new/data')
    out_folder_name = out_folder_name.joinpath(i3_type)
    out_folder_name.mkdir(parents=True, exist_ok=True)
    print('Converting i3 files:')
    for file in tqdm(i3_files):
        data = {}
        for name in column_names:
            data['{0}'.format(name)] = []
        # Create the tray
        tray = I3Tray()

        # Read input file(s)
        tray.AddModule('I3Reader', 'reader', FilenameList=[str(gcd_file)]
            + [str(file)])

        # Calculate tensor of interatia
        # TODO use clean or unclean pulses for InputReadout?
        tray.AddModule('I3TensorOfInertia', 'tensor_of_interia',
            AmplitudeOption=1, AmplitudeWeight=1, InputReadout='SRTInIcePulses',
            InputSelection='', MinHits=3, Name='ToI')

        # Add our own module
        tray.Add(do_things, 'do_things')
        # Actually run the tray
        tray.Execute()
        tray.Finish()

        # Where to save the hdf5 file; a path relative to the folder of this
        # actual file, no matter from where it is run
        out_name = file.stem
        out_name = out_name.replace('.i3', '')
        out_name = out_name.split('.')
        out_file_name = out_name[-1]
        out_path = out_folder_name.joinpath(out_file_name + '.h5')
        array_keys = [
            'dom_x',
            'dom_y',
            'dom_z',
            'time',
            'charge',
            'toi_point_on_line_x',
            'toi_point_on_line_y',
            'toi_point_on_line_z',
            'true_muon_energy',
            'true_muon_entry_position_x',
            'true_muon_entry_position_y',
            'true_muon_entry_position_z',
            'no_of_doms',
            'linefit_direction_x',
            'linefit_direction_y',
            'linefit_direction_z',
            'linefit_point_on_line_x',
            'linefit_point_on_line_y',
            'linefit_point_on_line_z',
            'toi_direction_x',
            'toi_direction_y',
            'toi_direction_z',
            'toi_evalratio',
            'true_muon_direction_x',
            'true_muon_direction_y',
            'true_muon_direction_z'
        ]
        with open_file(str(out_path), mode='w') as f:
            meta_data = {}
            meta_data['events'] = len(data['true_muon_energy'])
            data['no_of_doms'] = [
                len(data['dom_x'][i]) for i in range(
                    len(data['true_muon_energy'])
                )
            ]
            group = f.create_group(
                where='/',
                name='raw',
            )
            meta_group = f.create_group(
                where='/',
                name='meta'
            )
            hist_group = f.create_group(
                where='/histograms',
                name='raw',
                createparents=True
            )
            f.create_array(
                where=meta_group,
                name='events',
                obj=meta_data['events']
            )
            for key in array_keys:
                if type(data[key][0]) == np.ndarray:
                    variable = np.array(data[key])
                    vlarray = f.create_vlarray(
                        where=group,
                        name=key,
                        atom=Float64Atom(shape=())
                    )
                    for i in range(len(variable)):
                        vlarray.append(variable[i])
                    f.create_array(
                        where=hist_group,
                        name=key,
                        obj=np.hstack(variable)
                    )
                else:
                    f.create_array(
                        where=group,
                        name=key,
                        obj=data[key]
                    )
                    f.create_array(
                        where=hist_group,
                        name=key,
                        obj=data[key]
                    )
