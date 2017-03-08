#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""Series of full sky visualization function, with proper wcs header"""

from __future__ import print_function, division

import numpy as np
import healpy as hp

from astropy.wcs import WCS
from astropy import units as u
from astropy.io.fits import ImageHDU
from astropy.coordinates import SkyCoord

from .hp_helper import build_wcs, hp_to_wcs, equiv_celestial

__all__ = ['mollview', 'orthview', 'carview', 'merview',
           'coeview', 'bonview', 'pcoview', 'tscview']

def mollview(hp_map, hp_header, coord=None, npix=360, proj_sys='GALACTIC'):
    """Mollweide projection of the full sky

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
        coord = SkyCoord(0,0,unit='deg',frame=equiv_celestial(proj_sys))

    shape = (npix, int(npix/2))
    w = build_wcs(coord, 360./npix, shape, proj_sys=proj_sys, proj_type='MOL')
    moll = hp_to_wcs(hp_map, dict(hp_header), w, shape[::-1])

    return ImageHDU(moll, w.to_header())


def carview(hp_map, hp_header, coord=None, npix=360, proj_sys='GALACTIC'):
    """Plate carrée projection of the full sky

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
        coord = SkyCoord(0,0,unit='deg',frame=equiv_celestial(proj_sys))

    shape = (npix, int(npix/2))
    w = build_wcs(coord, 360./npix, shape, proj_sys=proj_sys, proj_type='CAR')
    car = hp_to_wcs(hp_map, dict(hp_header), w, shape[::-1])

    return ImageHDU(car, w.to_header())

def merview(hp_map, hp_header, coord=None, npix=360, proj_sys='GALACTIC'):
    """Mercator projection of the full sky

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
        coord = SkyCoord(0,0,unit='deg',frame=equiv_celestial(proj_sys))

    shape = (npix, int(npix/2))
    w = build_wcs(coord, 360./npix, shape, proj_sys=proj_sys, proj_type='MER')
    mer = hp_to_wcs(hp_map, dict(hp_header), w, shape[::-1])

    return ImageHDU(mer, w.to_header())


def coeview(hp_map, hp_header, coord=None, npix=360, proj_sys='GALACTIC'):
    """Conic Equal Area projection of the full sky

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
        coord = SkyCoord(0,0,unit='deg',frame=equiv_celestial(proj_sys))

    shape = (npix, npix)
    w = build_wcs(coord, 360./(npix-1), shape, proj_sys=proj_sys, proj_type='COE')
    w.wcs.set_pv([(1, 1, -20), (2, 1, -70)])
    coe = hp_to_wcs(hp_map, dict(hp_header), w, shape[::-1])


    return ImageHDU(coe, w.to_header())

def bonview(hp_map, hp_header, coord=None, npix=360, proj_sys='GALACTIC'):
    """Bonne's Equal Area projection of the full sky

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
        coord = SkyCoord(0,0,unit='deg',frame=equiv_celestial(proj_sys))

    shape = (npix, int(npix*4./5))
    w = build_wcs(coord, 360./(npix-1), shape, proj_sys=proj_sys, proj_type='BON')
    w.wcs.set_pv([(1,1,0), (2, 1, 45)])
    bon = hp_to_wcs(hp_map, dict(hp_header), w, shape[::-1])

    return ImageHDU(bon, w.to_header())

def pcoview(hp_map, hp_header, coord=None, npix=360, proj_sys='GALACTIC'):
    """Hassler's polyconic projection of the full sky

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
        coord = SkyCoord(0,0,unit='deg',frame=equiv_celestial(proj_sys))

    # TODO: Check proper limits
    shape = (npix,int(7./9*npix))
    w = build_wcs(coord, 360./(npix-1), shape, proj_sys=proj_sys, proj_type='PCO')
    pco = hp_to_wcs(hp_map, dict(hp_header), w, shape[::-1])

    return ImageHDU(pco, w.to_header())

def tscview(hp_map, hp_header, coord=None, npix=360, proj_sys='GALACTIC'):
    """Tangential spherical cube projection of the full sky

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
        coord = SkyCoord(0,0,unit='deg',frame=equiv_celestial(proj_sys))

    # TODO: Check proper limits
    shape = (int(4./3*npix), npix)
    w = build_wcs(coord, 360./(npix-1), shape, proj_sys=proj_sys, proj_type='TSC')
    coe = hp_to_wcs(hp_map, dict(hp_header), w, shape[::-1])

    return ImageHDU(coe, w.to_header())

def orthview(hp_map, hp_header, coord=None, npix=360, proj_sys='GALACTIC'):
    """Slant orthographic projection of the full sky

    Parameters
    ----------
    hp_map : array_like
        healpix map with
    hp_header : :class:`astropy.fits.header.Header`
        corresponding header to project on
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
        coord = SkyCoord(0,0,unit='deg',frame=equiv_celestial(proj_sys))

    shape = (npix, npix)
    w1 = build_wcs(coord, 360./np.pi/(npix-1), shape, proj_sys=proj_sys, proj_type='SIN')
    orth_1 = hp_to_wcs(hp_map, dict(hp_header), w1, shape[::-1])

    coord_opposite = SkyCoord(coord.data.lon + 180*u.deg, -1* coord.data.lat, frame=coord.frame )

    w2 = build_wcs(coord_opposite, 360./np.pi/(npix-1), shape, proj_sys=proj_sys, proj_type='SIN')
    orth_2 = hp_to_wcs(hp_map, dict(hp_header), w2, shape[::-1])

    return (ImageHDU(orth_1, w1.to_header()),
            ImageHDU(orth_2, w2.to_header()))
