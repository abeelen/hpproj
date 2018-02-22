#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""Series of full sky visualization function, with proper wcs header"""

from __future__ import print_function, division
from functools import partial, update_wrapper

import numpy as np

from astropy import units as u
from astropy.io.fits import ImageHDU
from astropy.coordinates import SkyCoord

from .hp_helper import build_wcs, hp_to_wcs, equiv_celestial, _hpmap

__all__ = ['view', 'mollview', 'orthview', 'carview', 'merview',
           'coeview', 'bonview', 'pcoview', 'tscview']

# pv matrix for some of the projection
PV = {'COE': [(1, 1, -20), (2, 1, -70)],
      'BON': [(1, 1, 0), (2, 1, 45)], }


@_hpmap
def view(hp_hdu, coord=None, npix=360, proj_sys='GALACTIC', proj_type='TAN', aspect=1.):
    """projection of the full sky

    Parameters
    ----------
    hp_hdu : :class:`~astropy.io.fits.ImageHDU`
        a pseudo ImageHDU with the healpix map and the associated header to be projected
    coord : :class:`astropy.coordinate.SkyCoord`
        the sky coordinate of the center of the projection
    npix : int
        number of pixels in the latitude direction
    proj_sys : str, ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the projection
    proj_type: str ('TAN', 'SIN', 'GSL', ...)
        any projection system supported by WCS
    aspect : float
        the resulting figure aspect ratio 1:aspect_ratio

    Returns
    -------
    :class:`astropy.io.fits.ImageHDU`
        2D images with header
    """

    if not coord:
        coord = SkyCoord(0, 0, unit='deg', frame=equiv_celestial(proj_sys))

    shape = (np.asarray([1., aspect]) * npix).astype(np.int)
    _wcs = build_wcs._coord(coord, 360. / npix, shape, proj_sys=proj_sys, proj_type=proj_type)

    pv = PV.get(proj_type, None)
    if pv:
        _wcs.wcs.set_pv(pv)
    _data = hp_to_wcs._hphdu(hp_hdu, _wcs, shape[::-1])

    return ImageHDU(_data, _wcs.to_header())


mollview = partial(view, proj_type='MOL', aspect=0.5)
update_wrapper(mollview, view)
mollview.__name__ = "mollview"
mollview.__doc__ = "Mollweide " + mollview.__doc__

carview = partial(view, proj_type='CAR', aspect=0.5)
update_wrapper(carview, view)
carview.__name__ = "carview"
carview.__doc__ = "Plate carrée " + carview.__doc__

merview = partial(view, proj_type='MER', aspect=0.5)
update_wrapper(merview, view)
merview.__name__ = "merview"
merview.__doc__ = "Mercator " + merview.__doc__

coeview = partial(view, proj_type='COE')
update_wrapper(coeview, view)
coeview.__name__ = "coeview"
coeview.__doc__ = "Conic Equal Area " + coeview.__doc__

# TODO: Check ratio
bonview = partial(view, proj_type='BON', aspect=4. / 5)
update_wrapper(bonview, view)
bonview.__name__ = "bonview"
bonview.__doc__ = "Bonne's Equal Area " + bonview.__doc__

# TODO: Check ratio
pcoview = partial(view, proj_type='PCO', aspect=7. / 9)
update_wrapper(pcoview, view)
pcoview.__name__ = "pcoview"
pcoview.__doc__ = "Hassler's polyconic " + pcoview.__doc__

# TODO: Check ratio
tscview = partial(view, proj_type='TSC', aspect=3. / 4)
update_wrapper(tscview, view)
tscview.__name__ = "tscview"
tscview.__doc__ = "Tangential spherical cube " + tscview.__doc__


@_hpmap
def orthview(hp_hdu, coord=None, npix=360, proj_sys='GALACTIC'):
    """Slant orthographic projection of the full sky

    Parameters
    ----------
    hp_hdu : `:class:astropy.io.fits.ImageHDU`
        a pseudo ImageHDU with the healpix map and the associated header to be projected
    coord : :class:`astropy.coordinate.SkyCoord`
        the sky coordinate of the center of the projection
    npix : int
        number of pixels in the latitude direction
    proj_sys : str, ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the projection

    Returns
    -------
    :class:`astropy.io.fits.ImageHDU`
        2D images with header
    """
    if not coord:
        coord = SkyCoord(0, 0, unit='deg', frame=equiv_celestial(proj_sys))

    shape = (npix, npix)

    coord_opposite = SkyCoord(
        coord.data.lon + 180 * u.deg, -1 * coord.data.lat, frame=coord.frame)

    wcs1, wcs2 = [build_wcs._coord(coord, 360. / np.pi / (npix - 1), shape, proj_sys=proj_sys, proj_type='SIN') for coord in [coord, coord_opposite]]

    orth_1 = hp_to_wcs._hphdu(hp_hdu, wcs1, shape[::-1])
    orth_2 = hp_to_wcs._hphdu(hp_hdu, wcs2, shape[::-1])

    return (ImageHDU(orth_1, wcs1.to_header()),
            ImageHDU(orth_2, wcs2.to_header()))
