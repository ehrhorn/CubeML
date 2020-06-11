from src.modules.reporting import *
from src.modules.constants import *
from src.modules.thesis_plotting import *
from src.modules.classes import SqliteFetcher
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
import healpy as hp
import numpy as np
import argparse
import os

def unitvec2cat(x, y, z):
    # Signed azimuthal angle - exactly what we want for healoy
    phi = np.arctan2(y, x)*180/np.pi
    # Not what we want - healpy takes signed angle
    theta = np.arccos(z)*180/np.pi
    theta_signed = theta - 90

    return phi, theta_signed

def cat2hpx(lon, lat, nside, radec=True):
    """
    Convert a catalogue to a HEALPix map of number counts per resolution
    element.

    Parameters
    ----------
    lon, lat : (ndarray, ndarray)
        Coordinates of the sources in degree. If radec=True, assume input is in the icrs
        coordinate system. Otherwise assume input is glon, glat

    nside : int
        HEALPix nside of the target map

    radec : bool
        Switch between R.A./Dec and glon/glat as input coordinate system.

    Return
    ------
    hpx_map : ndarray
        HEALPix map of the catalogue number counts in Galactic coordinates

    """

    npix = hp.nside2npix(nside)

    if radec:
        eq = SkyCoord(lon, lat, 'icrs', unit='deg')
        l, b = eq.galactic.l.value, eq.galactic.b.value
    else:
        l, b = lon, lat

    # conver to theta, phi
    theta = np.radians(90. - b)
    phi = np.radians(l)

    # convert to HEALPix indices
    indices = hp.ang2pix(nside, theta, phi)

    idx, counts = np.unique(indices, return_counts=True)

    # fill the fullsky map
    hpx_map = np.zeros(npix, dtype=int)
    hpx_map[idx] = counts

    return hpx_map

def make_data():
    all_x, all_y, all_z = [], [], []
    for path in [PATH_TRAIN_DB, PATH_VAL_DB, PATH_TEST_DB]:
        db = SqliteFetcher(path)
        xyz = [
            'true_primary_direction_x',
            'true_primary_direction_y',
            'true_primary_direction_z'
        ]
        dd = db.fetch_features(all_events=db.ids, scalar_features=xyz)
        x = [data[xyz[0]] for index, data in dd.items()]
        y = [data[xyz[1]] for index, data in dd.items()]
        z = [data[xyz[2]] for index, data in dd.items()]
        all_x.extend(x)
        all_y.extend(y)
        all_z.extend(z)
    
    l, b = unitvec2cat(np.array(all_x), np.array(all_y), np.array(all_z))
    hpx_map = cat2hpx(l, b, nside=64, radec=False)
    path = Path(os.path.realpath(__file__))

    # Save data
    with open(str(path.parent) + '/data.pickle', 'wb') as f:
        pickle.dump(hpx_map, f)

setup_pgf_plotting()

# make_data()

path = Path(os.path.realpath(__file__))
hpx_map = pickle.load(open(str(path.parent) + '/data.pickle', 'rb'))

hp.mollview(hpx_map, cbar=False, cmap='Oranges', title=None)
hp.graticule() # Adds meridians and parallels

# Do stuff to the figure created
f = plt.gcf()
ax = plt.gca()
image = ax.get_images()[0]
cmap = f.colorbar(image, 
    ax=ax, 
    orientation='horizontal',
    pad=0.05,
    shrink=0.7,
    aspect=40
    )
plt.title(r'OscNext Lvl5', loc='left')
plt.draw()
# ============================================================================
# SAVE PGF AND PNG FOR VIEWING    
# ============================================================================

# single_fig, 2subfigs
FOTW = get_frac_of_textwidth(keyword='single_fig')
width = get_figure_width(frac_of_textwidth=FOTW)
height = get_figure_height(width=width)
f.set_size_inches(width, height)

path = Path(os.path.realpath(__file__))
save_thesis_pgf(path, f, save_pgf=True)
