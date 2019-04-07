#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""Series of helper function to deal with building wcs objects"""

import logging
import numpy as np

from astropy.wcs import WCS
from astropy.coordinates import ICRS, Galactic, Angle, SkyCoord

from .decorator import update_docstring

VALID_PROJ = ['AZP', 'SZP', 'TAN', 'STG', 'SIN',
              'ARC', 'ZPN', 'ZEA', 'AIR', 'CYP',
              'CEA', 'CAR', 'MER', 'COP', 'COE',
              'COD', 'COO', 'SFL', 'PAR', 'MOL',
              'AIT', 'BON', 'PCO', 'TSC', 'CSC',
              'QSC', 'HPX', 'XPH']

DEFAULT_SHAPE_OUT = (512, 512)
VALID_GALACTIC = ['galactic', 'g']
VALID_EQUATORIAL = ['celestial2000', 'equatorial', 'eq', 'c', 'q', 'fk4', 'fk5', 'icrs']

LOGGER = logging.getLogger('hpproj')

__all__ = ['build_wcs', 'build_wcs_cube', 'build_wcs_2pts',
           'build_ctype', 'equiv_celestial', 'rot_frame',
           'build_wcs_profile']


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
        raise ValueError('Unsupported projection')

    if coordsys in VALID_GALACTIC:
        axes = ['GLON-', 'GLAT-']
    elif coordsys in VALID_EQUATORIAL:
        axes = ['RA---', 'DEC--']
    else:
        raise ValueError('Unsupported coordsys')

    return [coord + proj_type for coord in axes]


def rot_frame(coord, proj_sys):
    """Retrieve the proper longitude and latitude

    Parameters
    ----------
    coord : :class:`astropy.coordinate.SkyCoord`
       the sky coordinate of the center of the projection
    proj_sys : str ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the plate (from HEALPIX maps....)

    Returns
    -------
    :class:`~astropy.coordinate.SkyCoord`
        rotated frame
    """

    proj_sys = proj_sys.lower()

    if proj_sys in VALID_EQUATORIAL:
        coord = coord.transform_to(ICRS)
    elif proj_sys in VALID_GALACTIC:
        coord = coord.transform_to(Galactic)
    else:
        raise ValueError('Unsuported coordinate system for the projection')

    return coord


# Cyclic reference if trying to move it to decorator.py
def _lonlat(build_wcs_func):
    """Will decorate the build_wcs function to be able to use it with lon/lat, proj_sys instead of an :class:`astropy.coordinate.SkyCoord` object

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

    def decorator(*args, **kwargs):
        """Transform a function call from (lon, lat, src_frame,*) to (coord, *)"""
        if len(args) > 1 and not isinstance(args[0], SkyCoord):

            # Otherwise proceed to fiddling the arguments..
            lon, lat = args[0:2]
            src_frame = kwargs.pop('src_frame', 'EQUATORIAL').lower()

            # Checks proper arguments values
            assert isinstance(lon, float) or isinstance(lon, int), 'lon must be a float'
            assert isinstance(lat, float) or isinstance(lat, int), 'latitude must be a float'
            assert src_frame in VALID_GALACTIC or src_frame in VALID_EQUATORIAL, 'src_frame must be a valid frame'

            frame = equiv_celestial(src_frame)
            coord = SkyCoord(lon, lat, frame=frame, unit="deg")
            # Ugly fix to modigy the args
            args = list(args)
            args.insert(2, coord)
            args = tuple(args[2:])
        return build_wcs_func(*args, **kwargs)

    decorator._coord = build_wcs_func
    decorator.__doc__ = update_docstring(build_wcs_func, skip=6, head_docstring="""
    Parameters
    ----------
    coord : :class:`astropy.coordinate.SkyCoord`
        the sky coordinate of the center of the projection

        or

    lon,lat : floats
        the sky coordinates of the center of projection and
    src_frame :  keyword, str, ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the longitude and latitude (default EQUATORIAL)""", foot_docstring="""
    Notes
    -----
    You can access a function using only catalogs with the ._coord() method
    """)
    return decorator


def build_wcs_profile(pixsize=0.01):

    wcs = WCS(naxis=1)

    # CRPIX IS in Fortran convention -> first pixel edge is 0
    wcs.wcs.crpix = [0.5]
    wcs.wcs.cdelt = [pixsize]
    wcs.wcs.crval = [0]

    wcs.wcs.ctype = ["RADIUS"]

    return wcs


@_lonlat
def build_wcs(coord, pixsize=0.01, shape_out=DEFAULT_SHAPE_OUT, proj_sys='EQUATORIAL', proj_type='TAN'):
    """Construct a :class:`~astropy.wcs.WCS` object for a 2D image

    Parameters
    ----------
    coord : :class:`astropy.coordinate.SkyCoord`
        the sky coordinate of the center of the projection
    pixsize : float
        size of the pixel (in degree)
    shape_out : tuple
        shape of the output map  (n_y,n_x)
    proj_sys : str ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the plate (from HEALPIX maps....)
    proj_type : str ('TAN', 'SIN', 'GSL', ...)
        the projection system to use

    Returns
    -------
    WCS: :class:`~astropy.wcs.WCS`
        An corresponding wcs object
    """

    assert isinstance(shape_out, (tuple, list, np.ndarray))
    assert len(shape_out) == 2

    coord = rot_frame(coord, proj_sys)

    proj_type = proj_type.upper()
    if proj_type not in VALID_PROJ:
        raise ValueError('Unvupported projection')

    wcs = WCS(naxis=2)

    # CRPIX IS in Fortran convention
    wcs.wcs.crpix = (np.array(shape_out, dtype=np.float) + 1) / 2
    wcs.wcs.cdelt = np.array([-pixsize, pixsize])
    wcs.wcs.crval = [coord.data.lon.deg, coord.data.lat.deg]

    wcs.wcs.ctype = build_ctype(proj_sys, proj_type)

    return wcs


@_lonlat
def build_wcs_cube(coord, index, pixsize=0.01, shape_out=DEFAULT_SHAPE_OUT, proj_sys='EQUATORIAL', proj_type='TAN'):
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
    proj_sys : str ('GALACTIC', 'EQUATORIAL')
        the coordinate system of the plate (from HEALPIX maps....)
    proj_type : str ('TAN', 'SIN', 'GSL', ...)
        the projection system to use

    Returns
    -------
    WCS: :class:`~astropy.wcs.WCS`
        An corresponding wcs object
    """

    coord = rot_frame(coord, proj_sys)

    proj_type = proj_type.upper()
    if proj_type not in VALID_PROJ:
        raise ValueError('Unvupported projection')

    wcs = WCS(naxis=3)
    wcs.wcs.crpix = np.append((np.array(shape_out, dtype=np.float) + 1) / 2, 1)
    wcs.wcs.cdelt = np.append(np.array([-pixsize, pixsize]), 1)
    wcs.wcs.crval = np.array([coord.data.lon.deg, coord.data.lat.deg, index])

    wcs.wcs.ctype = np.append(build_ctype(proj_sys, proj_type), 'INDEX').tolist()

    return wcs


def relative_pixsize(coords, pixsize, shape_out, relative_pos):
    """Compute relative_pos or pixsize depending on the input

    Parameters
    ----------
    coords : class:`astropy.coordinate.SkyCoord`
        the 2 sky coordinates of the projection, they will be horizontal in the resulting wcs
    pixsize : float
        size of the pixel (in degree) (default: None, use relative_pos and shape_out)
    shape_out : tuple
        shape of the output map  (n_y,n_x)
    relative_pos : tuple
        the relative position of the 2 sources along the x direction [0-1] (will be computed if pixsize is given)

    Returns
    -------
    tuple
        (pixsize, relative_pos)
    """

    assert len(coords) == 2 & len(shape_out) == 2 & len(relative_pos) == 2, "Must have a length of 2"

    if pixsize:
        # Compute relative_pos from pixsize and distance between
        # sources
        ang_distance = coords[0].separation(coords[1])
        pix_distance = ang_distance.deg / pixsize
        relative_pos = pix_distance / shape_out[1]
        # Center it
        relative_pos = 0.5 + np.array([-1., 1]) * relative_pos / 2
    else:
        # Compute pixsize from relative_pos and distance between
        # sources
        pix_distance = np.array(relative_pos) * shape_out[1]
        pix_distance = np.max(pix_distance) - np.min(pix_distance)
        ang_distance = coords[0].separation(coords[1])
        pixsize = ang_distance.deg / pix_distance

    return pixsize, relative_pos


def build_wcs_2pts(coords, pixsize=None, shape_out=DEFAULT_SHAPE_OUT, proj_sys='EQUATORIAL', proj_type='TAN', relative_pos=(2. / 5, 3. / 5)):
    """Construct a :class:`~astropy.wcs.WCS` object for a 2D image

    Parameters
    ----------
    coords : class:`astropy.coordinate.SkyCoord`
        the 2 sky coordinates of the projection, they will be horizontal in the resulting wcs
    pixsize : float
        size of the pixel (in degree) (default: None, use relative_pos and shape_out)
    shape_out : tuple
        shape of the output map  (n_y,n_x)
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

    frame = equiv_celestial(proj_sys)

    proj_type = proj_type.upper()
    if proj_type not in VALID_PROJ:
        raise ValueError('Unsupported projection')

    coords = [coord.transform_to(frame) for coord in coords]
    wcs = WCS(naxis=2)
    pixsize, relative_pos = relative_pixsize(coords, pixsize, shape_out, relative_pos)

    # Put the first source on the relative_pos[0]
    wcs.wcs.crpix = np.array([relative_pos[0], 1. / 2], dtype=np.float) * (
        np.array(shape_out, dtype=np.float)[::-1])
    wcs.wcs.crval = [coords[0].data.lon.deg, coords[0].data.lat.deg]

    wcs.wcs.cdelt = np.array([-pixsize, pixsize])
    # Computes the on-sky position angle (East of North) between this SkyCoord and another.
    rot_angle = (coords[0].position_angle(coords[1]) + Angle(90, unit='deg')).wrap_at('180d')

    logging.debug('... rotating frame with %s deg', rot_angle.degree)
    wcs.wcs.pc = [[np.cos(rot_angle.radian), np.sin(-rot_angle.radian)],
                  [np.sin(rot_angle.radian), np.cos(rot_angle.radian)]]

    wcs.wcs.ctype = build_ctype(proj_sys, proj_type)

    return wcs
