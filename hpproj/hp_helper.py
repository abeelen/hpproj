#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""
Series of helper function to deal with healpix maps
"""
from __future__ import print_function, division


import numpy as np
import healpy as hp
from astropy.wcs import WCS

from astropy.io import fits
from astropy.wcs import utils as wcs_utils
from astropy.coordinates import ICRS, Galactic, SkyCoord, UnitSphericalRepresentation, Angle
from astropy import units as u

import logging
logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s', level=logging.DEBUG)

__all__ = ['build_WCS', 'build_WCS_lonlat',
           'build_WCS_cube', 'build_WCS_cube_lonlat',
           'build_WCS_2pts', 'build_ctype',
           'hp_is_nest', 'hp_celestial',
           'hp_to_wcs', 'hp_to_wcs_ipx',
           'hp_project',
           'gen_hpmap', 'build_hpmap',
           'equiv_celestial']

DEFAULT_shape_out = (512, 512)

VALID_PROJ = ['AZP', 'SZP', 'TAN', 'STG', 'SIN',
              'ARC', 'ZPN', 'ZEA', 'AIR', 'CYP',
              'CEA', 'CAR', 'MER', 'COP', 'COE',
              'COD', 'COO', 'SFL', 'PAR', 'MOL',
              'AIT', 'BON', 'PCO', 'TSC', 'CSC',
              'QSC','HPX','XPH']

VALID_GALACTIC = ['galactic', 'g']
VALID_EQUATORIAL = ['celestial2000','equatorial', 'eq', 'c', 'q', 'fk5']

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
    elif frame in [ 'ecliptic', 'e']:
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

    if not proj_type in VALID_PROJ:
        raise ValueError('Unvupported projection')

    if coordsys in VALID_GALACTIC:
        return [ coord+proj_type for coord in ['GLON-', 'GLAT-']]
    elif coordsys in VALID_EQUATORIAL:
        return [ coord+proj_type for coord in ['RA---', 'DEC--']]
    else:
        raise ValueError('Unsupported coordsys')

def build_WCS(coord, pixsize=0.01, shape_out=DEFAULT_shape_out, npix=None, proj_sys='EQUATORIAL', proj_type='TAN'):
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
    if not proj_type in VALID_PROJ:
        raise ValueError('Unvupported projection')

    if npix:
        shape_out = (npix, npix)

    w = WCS(naxis=2)

    # CRPIX IS in Fortran convention
    w.wcs.crpix = np.array(shape_out, dtype=np.float)/2
    w.wcs.cdelt = np.array([-pixsize, pixsize])
    w.wcs.crval = [lon, lat]

    w.wcs.ctype = build_ctype(proj_sys, proj_type)

    return w

def build_WCS_cube(coord, index, pixsize=0.01, shape_out=DEFAULT_shape_out, npix=None, proj_sys='EQUATORIAL', proj_type='TAN'):
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
    if not proj_type in VALID_PROJ:
        raise ValueError('Unvupported projection')


    if npix:
        shape_out = (npix, npix)

    w = WCS(naxis=3)
    w.wcs.crpix = np.append(np.array(shape_out, dtype=np.float)/2, 1)
    w.wcs.cdelt = np.append(np.array([-pixsize, pixsize]), 1)
    w.wcs.crval = np.array([lon, lat, index])

    w.wcs.ctype = np.append( build_ctype(proj_sys, proj_type), 'INDEX').tolist()

    return w


def _lonlat(build_WCS_func):
    """Will decorate the build_WCS function to be able to use it with lon/lat, proj_sys instead of an `astropy.coordinate.SkyCoord` object

    Parameters
    ----------
    build_WCS_func : fct
    a build_WCS_func function (with coords as the first argument

    Return
    ------
    The same function decorated

    Notes
    -----
    To use this decorator
    build_WCS_lonlat = _lonlat(build_WCS)
    or use @_lonlat on the function declaration

    """

    def decorator(lon, lat, src_frame='EQUATORIAL', **kargs):
        frame = equiv_celestial(src_frame)
        coord = SkyCoord(lon,lat, frame=frame, unit="deg")
        return build_WCS_func(coord, **kargs)

    decorator._coord = build_WCS_func
    decorator.__doc__ = str(build_WCS_func.__doc__).split('\n')[0] +\
    """
    Parameters
    ----------
    lon,lat : floats
        the sky coordinates of the center of projection
    src_frame :  str, ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the longitude and latitude
    """ + '\n'.join(str(build_WCS.__doc__).split('\n')[6:]) + \
    """
    Notes
    -----
    You can access a function using only catalogs with the ._coord() method
    """
    return decorator

build_WCS_lonlat = _lonlat(build_WCS)
build_WCS_cube_lonlat = _lonlat(build_WCS_cube)

def build_WCS_2pts(coords, pixsize=None, shape_out=DEFAULT_shape_out, npix=None, proj_sys='EQUATORIAL', proj_type='TAN', relative_pos=(2./5, 3./5)):
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

    w = WCS(naxis=2)

    if not pixsize:
        # Compute pixsize from relative_pos and distance between
        # sources
        pix_distance = np.array(relative_pos)*shape_out[1]
        pix_distance = pix_distance.max() - pix_distance.min()
        ang_distance = coords[0].separation(coords[1])
        pixsize = ang_distance.deg / pix_distance
    else:
        # Compute relative_pos from pixsize and distance between
        # sources
        ang_distance = coords[0].separation(coords[1])
        pix_distance = ang_distance.deg / pixsize
        relative_pos = pix_distance / shape_out[1]
        # Center it
        relative_pos = 0.5+np.array([-1.,1])*relative_pos/2

    proj_sys = proj_sys.lower()
    if proj_sys in VALID_EQUATORIAL:
        coords = [ coord.transform_to(ICRS) for coord in coords]
        lon = coords[0].ra.deg
        lat = coords[0].dec.deg
    elif proj_sys in VALID_GALACTIC:
        coords = [ coord.transform_to(Galactic) for coord in coords]
        lon = coords[0].l.deg
        lat = coords[0].b.deg
    else:
        raise ValueError('Unsuported coordinate system for the projection')

    proj_type = proj_type.upper()
    if not proj_type in VALID_PROJ:
        raise ValueError('Unvupported projection')

    # Put the first source on the relative_pos[0]
    w.wcs.crpix = np.array([relative_pos[0],1./2], dtype=np.float)*(np.array(shape_out, dtype=np.float)[::-1])
    w.wcs.crval = [lon, lat]

    w.wcs.cdelt = np.array([-pixsize, pixsize])
    # Computes the on-sky position angle (East of North) between this SkyCoord and another.
    rot_angle = (coords[0].position_angle(coords[1])+Angle(90, unit='deg')).wrap_at('180d')

    logging.debug('... rotating frame with %s deg'%rot_angle.degree)
    w.wcs.pc = [ [np.cos(rot_angle.radian) , np.sin(-rot_angle.radian)],
                 [np.sin(rot_angle.radian) , np.cos(rot_angle.radian)] ]

    w.wcs.ctype = build_ctype(proj_sys, proj_type)

    return w

def hp_to_wcs(hp_map, hp_header, w, shape_out=DEFAULT_shape_out, npix=None, order=0):
    """Project an Healpix map on a wcs header, using nearest neighbors.

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
    w : :class:`astropy.wcs.WCS`
        wcs object to project with
    shape_out : tuple
        shape of the output map (n_y, n_x)
    npix : int
        number of pixels in the final square map, superseed shape_out
    order : int (0|1)
        order of the interpolation 0: nearest-neighbor, 1: bi-linear interpolation

    Return
    ------
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
    # This as we are using the C convention, the center of the first pixel is -0.5, -0.5
    yy, xx = np.indices(shape_out)-0.5


    alon, alat = w.wcs_pix2world(xx,yy,0)
    mask = ~np.logical_or(np.isnan(alon),np.isnan(alat)) # if pixel lies outside of projected area
    proj_map = np.ma.array(np.zeros(shape_out), mask=~mask, fill_value=np.nan)

    # Determine if we need a rotation for different coordinate systems
    frame_in =  hp_celestial(hp_header)
    frame_out = wcs_utils.wcs_to_celestial_frame(w)

    if not frame_in.is_equivalent_frame(frame_out):
        logging.debug('... converting coordinate system')
        coords = frame_out.realize_frame(UnitSphericalRepresentation(alon*u.deg, alat*u.deg)).transform_to(frame_in)
        alon = coords.data.lon.deg
        alat = coords.data.lat.deg

    # Healpix conventions...
    phi = np.radians(alon)
    theta = np.radians(90-alat)


    if order == 0:
        ipix = hp.ang2pix(hp.npix2nside(len(hp_map)), theta[mask], phi[mask], nest=hp_is_nest(hp_header))
        proj_map[mask] = hp_map[ipix]
    elif order == 1:
        proj_map[mask] = hp.get_interp_val(hp_map, theta[mask], phi[mask], nest=hp_is_nest(hp_header))
    else:
        raise ValueError("Unsupported order for the interpolation")

    return proj_map.filled()

def hp_to_wcs_ipx(hp_header, w, shape_out=DEFAULT_shape_out, npix=None):
    """Return the indexes of pixels of a given wcs and shape_out,
    within a nside healpix map.

    Parameters
    ----------
    hp_header : :class:`astropy.fits.header.Header`
        header of the healpix map, should contain nside and coordsys and ordering
    w : :class:`astropy.wcs.WCS`
        wcs object to project with
    shape_out : tuple
        shape of the output map (n_y, n_x)
    npix : int
        number of pixels in the final square map, superseed shape_out

    Return
    ------
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
    # This as we are using the C convention, the center of the first pixel is -0.5, -0.5
    yy, xx = np.indices(shape_out)-0.5

    alon, alat = w.wcs_pix2world(xx,yy,0)
    mask = ~np.logical_or(np.isnan(alon),np.isnan(alat)) # if pixel lies outside of projected area

    frame_in =  hp_celestial(hp_header)
    frame_out = wcs_utils.wcs_to_celestial_frame(w)

    nside = hp_header['NSIDE']

    if not frame_in.is_equivalent_frame(frame_out):
        logging.debug('... converting coordinate system')
        coords = frame_out.realize_frame(UnitSphericalRepresentation(alon*u.deg, alat*u.deg)).transform_to(frame_in)
        alon = coords.data.lon.deg
        alat = coords.data.lat.deg

    # Healpix conventions...
    phi = np.radians(alon)
    theta = np.radians(90-alat)

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

    Return
    ------
    array_like
        2D images at lon, lat position
    or :class:`astropy.io.fits.PrimaryHDU` (optionnal)
        containing the array and the corresponding header
    """

    w = build_WCS(coord, pixsize, npix=npix, proj_sys=proj_sys, proj_type=proj_type)
    proj_map = hp_to_wcs(hp_map, hp_header, w, npix=npix, order=order)

    if hdu:
        return fits.PrimaryHDU(proj_map, w.to_header(relax=0x20000))
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

    Return
    ------
    tuple
        Return a tuple (filename, healpix map, healpix header) corresponding to the inputed list
    """
    for filename, iMap, iHeader in maps:
        if isinstance(iMap, str):
            iMap = hp.read_map(iMap, verbose=False)
            iMap = hp.ma(iMap)
        yield (filename, iMap, iHeader)

def build_hpmap(filenames, low_mem=True):
    """From a filename list, build a tuple usable with gen_hmap()

    Parameters
    ----------
    filenames: list
        A list of Nmap filenames of healpix maps
    low_mem : bool
        On low memory system, do not read the maps themselves (default: only header)

    Return
    ------
    tuple list
        A list of tuple which can be used by gen_hpmap    """

    hp_maps = []
    for filename in filenames:
        hp_header = fits.getheader(filename, 1)
        if low_mem == True:
            hp_map = filename
        else:
            hp_map = hp.read_map(filename, verbose=False)
            hp_map = hp.ma(hp_map)
        hp_maps.append( (filename, hp_map, hp_header ) )
    return hp_maps

def group_hpmap(maps):
    """Group into a dictionnary the hp_maps according to map properties

    Parameters
    ----------
    hpmap: tuple list
        A list of tuple which can be used by gen_hpmap

    Return
    ------
    dict
        A dictionnary where each key contains hpmap sharing the same properties

    """
    grouped_hpmaps = {}

    for (filename, iMap, iHeader) in maps:
        mapKey = "%s_%s_%s"%(iHeader['NSIDE'], iHeader['ORDERING'],  hp_celestial(iHeader).name)
        iHeader['mapKey'] = mapKey

        if mapKey in grouped_hpmaps.keys():
            grouped_hpmaps[mapKey].append((filename, iMap, iHeader))
        else:
            grouped_hpmaps[mapKey] = [(filename, iMap, iHeader)]

    return grouped_hpmaps
