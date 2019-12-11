"""Read i3 files, and save them as hdf5 files.

How to run: `python [args] i3_to_hdf5_no_calc.py`
in the command line, where [args] represents the input arguments.

Output file gets a random file name.

Help is found by running `python i3_to_hdf5_no_calc.py --help` from the command 
line.
Note that a valid IceCube environment must be loaded.

Created by Mads Ehrhorn on 2019-10-06.
"""
from icecube import icetray, dataio
from I3Tray import I3Tray
from icecube.hdfwriter import I3HDFWriter
import argparse
import os
import tempfile
from pathlib import Path
import h5py
import pandas as pd
import shutil

def i3_read(in_file, keys, sub_event_streams):
    """Read i3 file, book keys and convert to hdf5.

    Args:
        in_file (str): Input file in i3 format. <
        out_dir (str): Output directory.
        keys (list): List of i3 file keys.
        sub_event_streams (list): List of i3 sub event streams.

    """
    output = temp_path.joinpath(in_file.stem + '.hdf5')
    tray = I3Tray()
    tray.AddModule('I3Reader', filename=str(in_file))
    tray.AddSegment(I3HDFWriter,
        output=str(output), Keys=keys, SubEventStreams=sub_event_streams)
    tray.Execute()


def read_hdf5(hdf5_file):
    """Read an hdf5 file, and save content as Pandas dataframes in a dictionary.

    Args:
        hdf5_file (str): Filename

    Returns:
        dict: Dictionary of Pandas dataframes

    """
    output_dict = {}
    hdf5_data = h5py.File(str(hdf5_file), 'r')
    keys = list(hdf5_data.keys())
    if '__I3Index__' in keys:
        keys.remove('__I3Index__')
    for key in keys:
        output_dict[key] = pd.DataFrame(hdf5_data[key][:])
    return output_dict


def append_hdf5(in_files, out_file):
    f = h5py.File(str(out_file), 'w')
    for file in in_files:
        data = h5py.File(str(file), 'r')
        f.create_group(file.stem)
        keys = list(data.keys())
        if '__I3Index__' in keys:
            keys.remove('__I3Index__')
        for key in keys:
            f.create_dataset('/' + file.stem + '/' + key, data=data[key])
    f.close()


default_keys = ['I3MCTree', 'InIcePulses', 'LineFit', 'LineFit_DC',
    'MCNeutrino']
default_sub_event_streams = ['InIceSplit']

parser = argparse.ArgumentParser(description='Convert i3 files to hdf5 files.',
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('-i', '--in_dir', type=str, help='Input directory.\n'
    'No default, required.')
parser.add_argument('-k', '--keys', type=list, help='List of i3 file keys.\n'
    'Defaults to [\'ToI_DC\', \'InIcePulses\']', nargs='?',
    default=default_keys)
parser.add_argument('-s', '--sub_event_streams', type=list, nargs='?',
    help='List of i3 sub event stream names.\n'
    'Defaults to [\'InIceSplit\']', default=default_sub_event_streams)
parser.add_argument('-o', '--out_dir', type=str, help='Output directory.\n'
    'Defaults to the current directory.', nargs='?',
    default=os.getcwd())

args = parser.parse_args()
in_dir = Path(args.in_dir)
out_dir = Path(args.out_dir)
keys = args.keys
sub_event_streams = args.sub_event_streams

temp_path = Path('temp/')
temp_path.mkdir(parents=True, exist_ok=False)

in_folders = [folder for folder in in_dir.iterdir() if folder.is_dir()]
in_files = []
for folder in in_folders:
    for file in folder.iterdir():
        if file.is_file() and file.suffix == '.i3':
            in_files.append(file)

for file in in_files:
    i3_read(file, keys, sub_event_streams)

hdf5_files = [file for file in temp_path.iterdir()
    if file.is_file() and file.suffix == '.hdf5']

out_file = out_dir.joinpath('total.hdf5')

append_hdf5(hdf5_files, out_file)

shutil.rmtree(temp_path)
