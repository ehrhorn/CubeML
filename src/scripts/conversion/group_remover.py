import h5py as h5
from pathlib import Path
import time
import datetime

INPUT_DIR = Path('/groups/hep/ehrhorn/bak_14_oscNext')
OUTPUT_DIR = Path('/groups/hep/ehrhorn/oscnext-genie-level5-v01-01-pass2_new')
DATA_FILES = sorted([file for file in INPUT_DIR.glob('*.h5')])
DATA_FILES = [DATA_FILES[0]]
print(DATA_FILES)
GROUPS_TO_REMOVE = ['transform1']

for i, file in enumerate(DATA_FILES):
    if i % 20 == 0:
        ts = time.time()
        st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
        print(
            'At time {} I handled file no. {} of {}'
            .format(
                st,
                i + 1,
                len(DATA_FILES)
            )
        )
    with h5.File(file, 'a') as old:
        for group in GROUPS_TO_REMOVE:
            del old[group]
        group_iter = old['/'].__iter__()
        with h5.File(OUTPUT_DIR.joinpath(file.name), 'w') as new:
            for group in group_iter:
                new.copy(old[group], new)
