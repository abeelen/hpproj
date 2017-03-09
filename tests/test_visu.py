

import pytest
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.io import fits

from hpproj import mollview, carview, orthview, \
    merview, coeview, bonview, pcoview, tscview

import matplotlib
matplotlib.use('Agg')


@pytest.fixture(scope='session')
def generate_hpmap():
    """Generate an uniform healpix map"""

    nside = 2**6
    hp_map = np.arange(hp.nside2npix(nside))
    hp_header = {'NSIDE': nside,
                 'ORDERING': 'RING',
                 'COORDSYS': 'G'}
    hp_hdu = fits.ImageHDU(hp_map, fits.Header(hp_header))

    return (hp_hdu)


@pytest.mark.mpl_image_compare
def test_orthview(generate_hpmap):
    hp_hdu = generate_hpmap

    _ = orthview(hp_hdu)
    fig = plt.figure()
    fig.suptitle('ORTH')
    for index, __ in enumerate(_):
        ax = fig.add_subplot(1, 2, index + 1, projection=WCS(__.header))
        ax.imshow(__.data, interpolation='none', origin='lower')
        ax.grid()

    return fig


@pytest.mark.parametrize("title, view", [
    ('MOLL', mollview),
    ('CAR', carview),
    ('MER', merview),
    ('COE', coeview),
    ('BON', bonview),
    ('PCO', pcoview),
    ('TSC', tscview), ])
@pytest.mark.mpl_image_compare
def test_view(generate_hpmap, title, view):
    hp_hdu = generate_hpmap

    fig = plt.figure()
    fig.suptitle(title)
    _ = view(hp_hdu)
    ax = fig.add_subplot(1, 1, 1, projection=WCS(_.header))
    ax.imshow(_.data, interpolation='none', origin='lower')
    ax.grid()
    return fig


@pytest.mark.mpl_image_compare
def test_frame(generate_hpmap):
    hp_hdu = generate_hpmap

    fig = plt.figure()
    _ = mollview(hp_hdu, proj_sys='eq')
    ax = fig.add_subplot(1, 1, 1, projection=WCS(_.header))
    ax.imshow(_.data, interpolation='none', origin='lower')
    ax.grid()
    return fig


@pytest.mark.mpl_image_compare
def test_npix(generate_hpmap):
    hp_hdu = generate_hpmap

    fig = plt.figure()
    _ = mollview(hp_hdu, npix=360 * 2)
    ax = fig.add_subplot(1, 1, 1, projection=WCS(_.header))
    ax.imshow(_.data, interpolation='none', origin='lower')
    ax.grid()
    return fig


@pytest.mark.mpl_image_compare
def test_coord(generate_hpmap):
    hp_hdu = generate_hpmap

    np.random.seed(0)
    lon, lat = np.random.uniform(size=2) * [360, 180] - [0, 90]
    coord = SkyCoord(lon, lat, unit='deg', frame='fk5')

    fig = plt.figure()
    _ = mollview(hp_hdu, coord=coord)
    ax = fig.add_subplot(1, 1, 1, projection=WCS(_.header))
    ax.imshow(_.data, interpolation='none', origin='lower')
    ax.grid()
    return fig
