#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""Series of helper function to deal with healpix maps"""

from __future__ import print_function, division

import logging
import numpy as np
import healpy as hp

from astropy.io import fits
from astropy.wcs import utils as wcs_utils
from astropy.coordinates import UnitSphericalRepresentation
from astropy import units as u

from .wcs_helper import equiv_celestial, build_wcs
from .wcs_helper import DEFAULT_SHAPE_OUT

from .decorator import _hpmap

logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s', level=logging.DEBUG)

__all__ = ['hp_is_nest', 'hp_celestial', 'hp_to_wcs', 'hp_to_wcs_ipx',
           'hp_project', 'gen_hpmap', 'build_hpmap', 'hpmap_key']


def hp_celestial(hp_header):
    """Retrieve the celestial system used in healpix maps. From Healpix documentation this can have 3 forms :

    - 'EQ', 'C' or 'Q' : Celestial2000 = eQuatorial,
    - 'G' : Galactic
    - 'E' : Ecliptic,

    only Celestial and Galactic are supported right now as the Ecliptic coordinate system was just recently pulled to astropy

    Similar to :class:`~astropty.wcs.utils.wcs_to_celestial_frame` but for header from healpix maps

    Parameters
    ----------
    hp_header : :class:`~astropy.io.fits.header.Header`
        the header of the healpix map

    Returns
    -------
    frame : :class:`~astropy.coordinates.baseframe.BaseCoordinateFrame` subclass instance
        An instance of a :class:`~astropy.coordinates.baseframe.BaseCoordinateFrame`
        subclass instance that best matches the specified WCS.

    """
    coordsys = hp_header.get('COORDSYS')

    if coordsys:
        return equiv_celestial(coordsys)
    else:
        raise ValueError("No COORDSYS in header")


def hp_is_nest(hp_header):
    """Return True if the healpix header is in nested

    Parameters
    ----------
    hp_header : :class:`~astropy.io.fits.header.Header`
        the header 100
    -------
    bool :
        True if the header is nested
    """

    ordering = hp_header.get('ORDERING')

    if ordering:
        if ordering.lower() == 'nested' or ordering.lower() == 'nest':
            return True
        elif ordering.lower() == 'ring':
            return False
        else:
            raise ValueError("Unknown ordering in healpix header")
    else:
        raise ValueError("Np ordering in healpix header")


def rotate_frame(alon, alat, hp_header, wcs):
    """Change frame if needed

    Parameters
    ----------
    alon, alat : array_like
        longitudes and latitudes describe by the
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header
    wcs : :class:`~astropy.wcs.WCS`
        An corresponding wcs object

    Returns
    -------
    tuple of array_like
        longitude and latitude arrays rotated if neeed
    """

    # Determine if we need a rotation for different coordinate systems
    frame_in = hp_celestial(hp_header)
    frame_out = wcs_utils.wcs_to_celestial_frame(wcs)

    if not frame_in.is_equivalent_frame(frame_out):
        logging.debug('... converting coordinate system')
        coords = frame_out.realize_frame(
            UnitSphericalRepresentation(alon * u.deg, alat * u.deg)).transform_to(frame_in)
        alon = coords.data.lon.deg
        alat = coords.data.lat.deg

    return alon, alat


@_hpmap
def hp_to_wcs(hp_hdu, wcs, shape_out=DEFAULT_SHAPE_OUT, npix=None, order=0):
    """Project an Healpix map on a wcs header, using nearest neighbors.

    Parameters
    ----------
    hp_hdu : `:class:astropy.io.fits.ImageHDU`
        a pseudo ImageHDU with the healpix map and the associated header
    wcs : :class:`astropy.wcs.WCS`
        wcs object to project with
    shape_out : tuple
        shape of the output map (n_y, n_x)
    npix : int
        number of pixels in the final square map, superseed shape_out
    order : int (0|1)
        order of the interpolation 0: nearest-neighbor, 1: bi-linear interpolation

    Returns
    -------
    array_like
        the projected map in a 2D array of shape shape_out
    """

    if npix:
        shape_out = (npix, npix)

    # Array of pixels centers -- from the astropy.wcs documentation :
    #
    # Here, *origin* is the coordinate in the upper left corner of the
    # image.In FITS and Fortran standards, this is 1.  In Numpy and C
    # standards this is 0.
    #
    # This as we are using the C convention, the center of the first pixel is 0, 0
    y_tab, x_tab = np.indices(shape_out)

    alon, alat = wcs.wcs_pix2world(x_tab, y_tab, 0)
    # mask for pixels lying outside of projected area
    mask = ~np.logical_or(np.isnan(alon), np.isnan(alat))
    proj_map = np.ma.array(np.zeros(shape_out), mask=~mask, fill_value=np.nan)

    # Determine if we need a rotation for different coordinate systems
    alon, alat = rotate_frame(alon, alat, hp_hdu.header, wcs)

    # Healpix conventions...
    phi = np.radians(alon)
    theta = np.radians(90 - alat)

    if order == 0:
        ipix = hp.ang2pix(hp.npix2nside(len(hp_hdu.data)),
                          theta[mask], phi[mask],
                          nest=hp_is_nest(hp_hdu.header))
        proj_map[mask] = hp_hdu.data[ipix]
    elif order == 1:
        proj_map[mask] = hp.get_interp_val(
            hp_hdu.data, theta[mask], phi[mask], nest=hp_is_nest(hp_hdu.header))
    else:
        raise ValueError("Unsupported order for the interpolation")

    return proj_map.filled()


def hp_to_wcs_ipx(hp_header, wcs, shape_out=DEFAULT_SHAPE_OUT, npix=None):
    """Return the indexes of pixels of a given wcs and shape_out,
    within a nside healpix map.

    Parameters
    ----------
    hp_header : :class:`astropy.fits.header.Header`
        header of the healpix map, should contain nside and coordsys and ordering
    wcs : :class:`astropy.wcs.WCS`
        wcs object to project with
    shape_out : tuple
        shape of the output map (n_y, n_x)
    npix : int
        number of pixels in the final square map, superseed shape_out

    Returns
    -------
    2D array_like
        mask for the given map
    array_like
        corresponding pixel indexes

    Notes
    -----
    The map could then easily be constructed using

    proj_map = np.ma.array(np.zeros(shape_out), mask=~mask, fill_value=np.nan)
    proj_map[mask] = healpix_map[ipix]
    """

    if npix:
        shape_out = (npix, npix)

    # Array of pixels centers -- from the astropy.wcs documentation :
    #
    # Here, *origin* is the coordinate in the upper left corner of the
    # image.In FITS and Fortran standards, this is 1.  In Numpy and C
    # standards this is 0.
    #
    # This as we are using the C convention, the center of the first pixel is 0, 0
    y_tab, x_tab = np.indices(shape_out)

    alon, alat = wcs.wcs_pix2world(x_tab, y_tab, 0)
    # mask for pixels lying outside of projected area
    mask = ~np.logical_or(np.isnan(alon), np.isnan(alat))

    nside = hp_header['NSIDE']

    # Determine if we need a rotation for different coordinate systems
    alon, alat = rotate_frame(alon, alat, hp_header, wcs)

    # Healpix conventions...
    phi = np.radians(alon)
    theta = np.radians(90 - alat)

    ipix = hp.ang2pix(nside, theta[mask], phi[mask], nest=hp_is_nest(hp_header))

    return mask, ipix


@_hpmap
def hp_project(hp_hdu, coord, pixsize=0.01, npix=512, order=0, projection=('GALACTIC', 'TAN')):
    """Project an healpix map at a single given position

    Parameters
    ----------
    hp_hdu : `:class:astropy.io.fits.ImageHDU`
        a pseudo ImageHDU with the healpix map and the associated header to be projected
    coord : :class:`astropy.coordinate.SkyCoord`
        the sky coordinate of the center of the projection
    pixsize : float
        size of the pixel (in degree)
    npix : int
        number of pixels in the final map, the reference pixel will be at the center
    order : int (0|1)
        order of the interpolation 0: nearest-neighbor, 1: bi-linear interpolation
    projection : tuple of str
        the coordinate ('GALACTIC', 'EQUATORIAL') and projection ('TAN', 'SIN', 'GSL', ...) system
    hdu : bool
        return a :class:`astropy.io.fits.PrimaryHDU` instead of just a ndarray

    Returns
    -------
    :class:`astropy.io.fits.PrimaryHDU`
        containing the array and the corresponding header
    """

    proj_sys, proj_type = projection

    wcs = build_wcs(coord, pixsize, npix=npix, proj_sys=proj_sys, proj_type=proj_type)
    proj_map = hp_to_wcs(hp_hdu, wcs, npix=npix, order=order)

    return fits.PrimaryHDU(proj_map, wcs.to_header(relax=0x20000))


def gen_hpmap(maps):
    """Generator function for large maps and low memory system

    Parameters
    ----------
    maps : list
        A list of Nmap tuples with either:
            * (filename, path_to_localfilename, healpix header)
            * (filename, healpix vector, healpix header)

    Returns
    -------
    tuple
        Return a tuple (filename, healpix map, healpix header) corresponding to the inputed list
    """
    for filename, i_map, i_header in maps:
        if isinstance(i_map, str):
            i_map = hp.read_map(i_map, verbose=False, nest=None)
            i_map = hp.ma(i_map)
        yield filename, fits.ImageHDU(i_map, fits.Header(i_header))


def build_hpmap(filenames, low_mem=True):
    """From a filename list, build a tuple usable with gen_hmap()

    Parameters
    ----------
    filenames: list
        A list of Nmap filenames of healpix maps
    low_mem : bool
        On low memory system, do not read the maps themselves (default: only header)

    Returns
    -------
    tuple list
        A list of tuple which can be used by gen_hpmap    """

    hp_maps = []
    for filename in filenames:
        hp_header = fits.getheader(filename, 1)
        if low_mem is True:
            hp_map = filename
        else:
            hp_map = hp.read_map(filename, verbose=False, nest=None)
            hp_map = hp.ma(hp_map)
        hp_maps.append((filename, hp_map, hp_header))
    return hp_maps


def hpmap_key(hp_map):
    """Generate an key from the hp_map tuple to sort the hp_maps by map
    properties

    Parameters
    ----------
    hp_map: tuple
        A tuple from (build|gen)_hpmap : (filename, healpix map, healpix header)

    Returns
    -------
    str
        A string with the map properties
    """
    i_header = hp_map[2]

    return "%s_%s_%s" % (i_header['NSIDE'], i_header['ORDERING'], hp_celestial(i_header).name)
