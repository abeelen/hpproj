#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""Series of helper function to deal with healpix maps"""

from __future__ import print_function, division

import logging
logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s', level=logging.DEBUG)


import numpy as np
import healpy as hp
from astropy.wcs import WCS

from astropy.io import fits
from astropy.wcs import utils as wcs_utils
from astropy.coordinates import ICRS, Galactic, SkyCoord, UnitSphericalRepresentation, Angle
from astropy import units as u


__all__ = ['build_wcs', 'build_wcs_lonlat',
           'build_wcs_cube', 'build_wcs_cube_lonlat',
           'build_wcs_2pts', 'build_ctype',
           'hp_is_nest', 'hp_celestial',
           'hp_to_wcs', 'hp_to_wcs_ipx',
           'hp_project',
           'gen_hpmap', 'build_hpmap', 'hpmap_key',
           'equiv_celestial']

DEFAULT_SHAPE_OUT = (512, 512)

VALID_PROJ = ['AZP', 'SZP', 'TAN', 'STG', 'SIN',
              'ARC', 'ZPN', 'ZEA', 'AIR', 'CYP',
              'CEA', 'CAR', 'MER', 'COP', 'COE',
              'COD', 'COO', 'SFL', 'PAR', 'MOL',
              'AIT', 'BON', 'PCO', 'TSC', 'CSC',
              'QSC', 'HPX', 'XPH']

VALID_GALACTIC = ['galactic', 'g']
VALID_EQUATORIAL = ['celestial2000', 'equatorial', 'eq', 'c', 'q', 'fk4', 'fk5', 'icrs']


def equiv_celestial(frame):
    """Return an equivalent ~astropy.coordfinates.builtin_frames

    Notes
    -----
    We do not care of the differences between ICRS/FK4/FK5
    """

    frame = frame.lower()
    if frame in VALID_GALACTIC:
        frame = Galactic()
    elif frame in VALID_EQUATORIAL:
        frame = ICRS()
    elif frame in ['ecliptic', 'e']:
        raise ValueError("Ecliptic coordinate frame not yet supported by astropy")
    return frame


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
        the header of the healpix map

    Returns
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


def build_ctype(coordsys, proj_type):
    """Build a valid spatial ctype for a wcs header

    Parameters
    ----------
    coordsys : str ('GALATIC', 'EQUATORIAL')
        the coordinate system of the plate
    proj_type: str ('TAN', 'SIN', 'GSL', ...)
        any projection system supported by WCS

    Returns
    -------
    list:
        a list with the 2 corresponding spatial ctype
    """

    coordsys = coordsys.lower()
    proj_type = proj_type.upper()

    if proj_type not in VALID_PROJ:
        raise ValueError('Unvupported projection')

    if coordsys in VALID_GALACTIC:
        return [coord + proj_type for coord in ['GLON-', 'GLAT-']]
    elif coordsys in VALID_EQUATORIAL:
        return [coord + proj_type for coord in ['RA---', 'DEC--']]
    else:
        raise ValueError('Unsupported coordsys')


def build_wcs(coord, pixsize=0.01, shape_out=DEFAULT_SHAPE_OUT, npix=None, proj_sys='EQUATORIAL', proj_type='TAN'):
    """Construct a :class:`~astropy.wcs.WCS` object for a 2D image

    Parameters
    ----------
    coord : :class:`astropy.coordinate.SkyCoord`
        the sky coordinate of the center of the projection
    pixsize : float
        size of the pixel (in degree)
    shape_out : tuple
        shape of the output map  (n_y,n_x)
    npix : int
        number of pixels in the final square map, the reference pixel will be at the center, superseed shape_out
    proj_sys : str ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the plate (from HEALPIX maps....)
    proj_type : str ('TAN', 'SIN', 'GSL', ...)
        the projection system to use

    Returns
    -------
    WCS: :class:`~astropy.wcs.WCS`
        An corresponding wcs object
    """

    proj_sys = proj_sys.lower()

    if proj_sys in VALID_EQUATORIAL:
        coord = coord.transform_to(ICRS)
        lon = coord.ra.deg
        lat = coord.dec.deg
    elif proj_sys in VALID_GALACTIC:
        coord = coord.transform_to(Galactic)
        lon = coord.l.deg
        lat = coord.b.deg
    else:
        raise ValueError('Unsuported coordinate system for the projection')

    proj_type = proj_type.upper()
    if proj_type not in VALID_PROJ:
        raise ValueError('Unvupported projection')

    if npix:
        shape_out = (npix, npix)

    wcs = WCS(naxis=2)

    # CRPIX IS in Fortran convention
    wcs.wcs.crpix = (np.array(shape_out, dtype=np.float)+1) / 2
    wcs.wcs.cdelt = np.array([-pixsize, pixsize])
    wcs.wcs.crval = [lon, lat]

    wcs.wcs.ctype = build_ctype(proj_sys, proj_type)

    return wcs


def build_wcs_cube(coord, index, pixsize=0.01, shape_out=DEFAULT_SHAPE_OUT, npix=None, proj_sys='EQUATORIAL', proj_type='TAN'):
    """Construct a :class:`~astropy.wcs.WCS` object for a 3D cube, where the 3rd dimension is an index

    Parameters
    ----------
    coord : :class:`astropy.coordinate.SkyCoord`
        the sky coordinate of the center of the projection
    index : int
        reference index
    pixsize : float
        size of the pixel (in degree)
    shape_out : tuple
        shape of the output map (n_y, n_x)
    npix : int
        number of pixels in the final map, the reference pixel will be at the center, override shape_out
    proj_sys : str ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the plate (from HEALPIX maps....)
    proj_type : str ('TAN', 'SIN', 'GSL', ...)
        the projection system to use

    Returns
    -------
    WCS: :class:`~astropy.wcs.WCS`
        An corresponding wcs object
    """

    proj_sys = proj_sys.lower()

    if proj_sys in VALID_EQUATORIAL:
        coord = coord.transform_to(ICRS)
        lon = coord.ra.deg
        lat = coord.dec.deg
    elif proj_sys in VALID_GALACTIC:
        coord = coord.transform_to(Galactic)
        lon = coord.l.deg
        lat = coord.b.deg
    else:
        raise ValueError('Unsuported coordinate system for the projection')

    proj_type = proj_type.upper()
    if proj_type not in VALID_PROJ:
        raise ValueError('Unvupported projection')

    if npix:
        shape_out = (npix, npix)

    wcs = WCS(naxis=3)
    wcs.wcs.crpix = np.append((np.array(shape_out, dtype=np.float)+1) / 2, 1)
    wcs.wcs.cdelt = np.append(np.array([-pixsize, pixsize]), 1)
    wcs.wcs.crval = np.array([lon, lat, index])

    wcs.wcs.ctype = np.append(build_ctype(proj_sys, proj_type), 'INDEX').tolist()

    return wcs


def _lonlat(build_wcs_func):
    """Will decorate the build_wcs function to be able to use it with lon/lat, proj_sys instead of an `astropy.coordinate.SkyCoord` object

    Parameters
    ----------
    build_wcs_func : fct
    a build_wcs_func function (with coords as the first argument

    Returns
    -------
    The same function decorated

    Notes
    -----
    To use this decorator
    build_wcs_lonlat = _lonlat(build_wcs)
    or use @_lonlat on the function declaration

    """

    def decorator(lon, lat, src_frame='EQUATORIAL', **kargs):
        """Transform a function call from (lon, lat, src_frame,*) to (coord, *)"""
        frame = equiv_celestial(src_frame)
        coord = SkyCoord(lon, lat, frame=frame, unit="deg")
        return build_wcs_func(coord, **kargs)

    decorator._coord = build_wcs_func
    decorator.__doc__ = str(build_wcs_func.__doc__).split('\n')[0] +\
        """
    Parameters
    ----------
    lon,lat : floats
        the sky coordinates of the center of projection
    src_frame :  str, ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the longitude and latitude
    """ + '\n'.join(str(build_wcs.__doc__).split('\n')[6:]) + \
        """
    Notes
    -----
    You can access a function using only catalogs with the ._coord() method
    """
    return decorator

build_wcs_lonlat = _lonlat(build_wcs)
build_wcs_cube_lonlat = _lonlat(build_wcs_cube)


def build_wcs_2pts(coords, pixsize=None, shape_out=DEFAULT_SHAPE_OUT, npix=None, proj_sys='EQUATORIAL', proj_type='TAN', relative_pos=(2. / 5, 3. / 5)):
    """Construct a :class:`~astropy.wcs.WCS` object for a 2D image

    Parameters
    ----------
    coord : :class:`astropy.coordinate.SkyCoord`
        the 2 sky coordinates of the projection, they will be horizontal in the resulting wcs
    pixsize : float
        size of the pixel (in degree) (default: None, use relative_pos and shape_out)
    shape_out : tuple
        shape of the output map  (n_y,n_x)
    npix : int
        number of pixels in the final square map, superseed shape_out
    coordsys : str ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the plate (from HEALPIX maps....) will be rotated anyway
    proj_type : str ('TAN', 'SIN', 'GSL', ...)
        the projection system to use, the first coordinate will be the projection center
    relative_pos : tuple
        the relative position of the 2 sources along the x direction [0-1] (will be computed if pixsize is given)

    Returns
    -------
    WCS: :class:`~astropy.wcs.WCS`
        An corresponding wcs object

    Notes
    -----

    By default relative_pos is used to place the sources, and the
    pixsize is derived, but if you define pixsize, then the
    relative_pos will be computed and the sources placed at the center
    of the image

    """

    if npix:
        shape_out = (npix, npix)

    wcs = WCS(naxis=2)

    if not pixsize:
        # Compute pixsize from relative_pos and distance between
        # sources
        pix_distance = np.array(relative_pos) * shape_out[1]
        pix_distance = np.max(pix_distance) - np.min(pix_distance)
        ang_distance = coords[0].separation(coords[1])
        pixsize = ang_distance.deg / pix_distance
    else:
        # Compute relative_pos from pixsize and distance between
        # sources
        ang_distance = coords[0].separation(coords[1])
        pix_distance = ang_distance.deg / pixsize
        relative_pos = pix_distance / shape_out[1]
        # Center it
        relative_pos = 0.5 + np.array([-1., 1]) * relative_pos / 2

    proj_sys = proj_sys.lower()
    if proj_sys in VALID_EQUATORIAL:
        coords = [coord.transform_to(ICRS) for coord in coords]
        lon = coords[0].ra.deg
        lat = coords[0].dec.deg
    elif proj_sys in VALID_GALACTIC:
        coords = [coord.transform_to(Galactic) for coord in coords]
        lon = coords[0].l.deg
        lat = coords[0].b.deg
    else:
        raise ValueError('Unsuported coordinate system for the projection')

    proj_type = proj_type.upper()
    if proj_type not in VALID_PROJ:
        raise ValueError('Unvupported projection')

    # Put the first source on the relative_pos[0]
    wcs.wcs.crpix = np.array([relative_pos[0], 1. / 2], dtype=np.float) * (
        np.array(shape_out, dtype=np.float)[::-1])
    wcs.wcs.crval = [lon, lat]

    wcs.wcs.cdelt = np.array([-pixsize, pixsize])
    # Computes the on-sky position angle (East of North) between this SkyCoord and another.
    rot_angle = (coords[0].position_angle(coords[1]) + Angle(90, unit='deg')).wrap_at('180d')

    logging.debug('... rotating frame with %s deg' % rot_angle.degree)
    wcs.wcs.pc = [[np.cos(rot_angle.radian), np.sin(-rot_angle.radian)],
                  [np.sin(rot_angle.radian), np.cos(rot_angle.radian)]]

    wcs.wcs.ctype = build_ctype(proj_sys, proj_type)

    return wcs


def hp_to_wcs(hp_map, hp_header, wcs, shape_out=DEFAULT_SHAPE_OUT, npix=None, order=0):
    """Project an Healpix map on a wcs header, using nearest neighbors.

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
    frame_in = hp_celestial(hp_header)
    frame_out = wcs_utils.wcs_to_celestial_frame(wcs)

    if not frame_in.is_equivalent_frame(frame_out):
        logging.debug('... converting coordinate system')
        coords = frame_out.realize_frame(
            UnitSphericalRepresentation(alon * u.deg, alat * u.deg)).transform_to(frame_in)
        alon = coords.data.lon.deg
        alat = coords.data.lat.deg

    # Healpix conventions...
    phi = np.radians(alon)
    theta = np.radians(90 - alat)

    if order == 0:
        ipix = hp.ang2pix(hp.npix2nside(len(hp_map)), \
                          theta[mask], phi[mask], \
                          nest=hp_is_nest(hp_header))
        proj_map[mask] = hp_map[ipix]
    elif order == 1:
        proj_map[mask] = hp.get_interp_val(
            hp_map, theta[mask], phi[mask], nest=hp_is_nest(hp_header))
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

    frame_in = hp_celestial(hp_header)
    frame_out = wcs_utils.wcs_to_celestial_frame(wcs)

    nside = hp_header['NSIDE']

    if not frame_in.is_equivalent_frame(frame_out):
        logging.debug('... converting coordinate system')
        coords = frame_out.realize_frame(
            UnitSphericalRepresentation(alon * u.deg, alat * u.deg)).transform_to(frame_in)
        alon = coords.data.lon.deg
        alat = coords.data.lat.deg

    # Healpix conventions...
    phi = np.radians(alon)
    theta = np.radians(90 - alat)

    ipix = hp.ang2pix(nside, theta[mask], phi[mask], nest=hp_is_nest(hp_header))

    return mask, ipix


def hp_project(hp_map, hp_header, coord, pixsize=0.01, npix=512, proj_sys='GALACTIC', proj_type='TAN', order=0, hdu=False):
    """Project an healpix map at a single given position

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
    coord : :class:`astropy.coordinate.SkyCoord`
        the sky coordinate of the center of the projection
    pixsize : float
        size of the pixel (in degree)
    npix : int
        number of pixels in the final map, the reference pixel will be at the center
    proj_sys : str, ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the projection (from HEALPIX maps....)
    proj_type : str ('TAN', 'SIN', 'GSL', ...)
        the projection system to use
    order : int (0|1)
        order of the interpolation 0: nearest-neighbor, 1: bi-linear interpolation
    hdu : bool
        return a :class:`astropy.io.fits.PrimaryHDU` instead of just a ndarray

    Returns
    -------
    array_like
        2D images at lon, lat position
    or :class:`astropy.io.fits.PrimaryHDU` (optionnal)
        containing the array and the corresponding header
    """

    wcs = build_wcs(coord, pixsize, npix=npix, proj_sys=proj_sys, proj_type=proj_type)
    proj_map = hp_to_wcs(hp_map, hp_header, wcs, npix=npix, order=order)

    if hdu:
        return fits.PrimaryHDU(proj_map, wcs.to_header(relax=0x20000))
    else:
        return proj_map


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
            i_map = hp.read_map(i_map, verbose=False)
            i_map = hp.ma(i_map)
        yield (filename, i_map, i_header)


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
            hp_map = hp.read_map(filename, verbose=False)
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
    filename, i_map, i_header = hp_map

    return "%s_%s_%s" % (i_header['NSIDE'], i_header['ORDERING'], hp_celestial(i_header).name)
