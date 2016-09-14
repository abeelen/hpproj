#! /usr/bin/env python

import logging
logger = logging.getLogger('django')

import os, sys
import argparse

import numpy as np
import healpy as hp

try:
    from wcsaxes import WCS # (deprecated)
except ImportError:
    from astropy.wcs import WCS

from astropy.io import fits
from astropy.coordinates import SkyCoord
from photutils import CircularAperture
from photutils import aperture_photometry

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Python3 vs Python2
try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import ConfigParser

# Python3 vs Python2
try:
    from io import BytesIO
except ImportError:
    from cStringIO import StringIO as BytesIO

from base64 import b64encode
# Python3 vs Python2
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

# try:
#     from WebServices.settings import BASE_DIR
# except ImportError:
#     BASE_DIR="../"

###############################################
###############################################
## Following few functions comes from the hp_helper module....
## Do not change

from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from astropy.coordinates import ICRS, Galactic, SkyCoord, UnitSphericalRepresentation, Angle
import astropy.units as u

def hp_celestial(hp_header):
    """Retrieve the celestial system used in healpix maps. From Healpix documentation this can have 3 forms :

    - 'C' or 'Q' : Celestial2000 = eQuatorial,
    - 'E' : Ecliptic,
    - 'G' : Galactic

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
        coordsys = coordsys.lower()
        if coordsys == 'galactic' or coordsys == 'g':
            frame = Galactic()
        elif coordsys == 'celestial2000' or coorsys == 'equatorial' or coordsys =='eq' or coordsys=='c' or coordsys=='q':
            frame = ICRS()
        elif coordsys == 'ecliptic' or coordsys == 'e':
            raise ValueError("Ecliptic coordinate frame not yet supported by astropy")
        return frame
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
        if ordering.lower() == 'nested':
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

    if coordsys == 'galactic' or coordsys == 'g':
        return [ coord+proj_type for coord in ['GLON-', 'GLAT-']]
    elif coordsys == 'equatorial' or coordsys == 'eq':
        return [ coord+proj_type for coord in ['RA---', 'DEC--']]
    else:
        raise ValueError('Unsupported coordsys')

def build_WCS(coord, pixsize=0.01, shape_out=(512, 512), npix=None, proj_sys='EQUATORIAL', proj_type='TAN'):
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

    if proj_sys == 'EQUATORIAL':
        coord = coord.transform_to(ICRS)
        lon = coord.ra.deg
        lat = coord.dec.deg
    elif proj_sys == 'GALACTIC':
        coord = coord.transform_to(Galactic)
        lon = coord.l.deg
        lat = coord.b.deg
    else:
        raise ValueError('Unsuported coordinate system for the projection')

    if npix:
        shape_out = (npix, npix)

    w = WCS(naxis=2)

    w.wcs.crpix = np.array(shape_out, dtype=np.float)/2
    w.wcs.cdelt = np.array([-pixsize, pixsize])
    w.wcs.crval = [lon, lat]

    w.wcs.ctype = build_ctype(proj_sys, proj_type)

    return w

def hp_to_wcs_ipx(hp_header, w, shape_out=(512,1024), npix=None):
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

    yy, xx = np.indices(shape_out)

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

###############################################
###############################################

def cut_sky( lonlat=[0,0],patch=[256,1],coordframe='galactic', maps=None):

    if not maps:
        raise FileNotFoundError("No healpix map to project")


    pixel_size=np.float(patch[1])
    n_pixels=np.float(patch[0])

    if np.str(coordframe)=="galactic":
        coordf = 'GALACTIC'
    elif np.str(coordframe)=="fk5":
        coordf = 'EQUATORIAL'

    # Center of projection
    coord_in = SkyCoord(lonlat[0],lonlat[1], unit=u.deg, frame=coordframe)

    # Build the target WCS header
    w = build_WCS(coord_in, pixsize=pixel_size/60., npix=n_pixels, proj_sys=np.str(coordf), proj_type='TAN')

    # Group the maps by header, keys is NSIDE_ORDERING_COORDSYS
    grouped_maps = {}
    for key in maps.keys():
        iHeader = fits.getheader(maps[key]['filename'],1)
        mapKey = "%s_%s_%s"%(iHeader['NSIDE'], iHeader['ORDERING'], iHeader['COORDSYS'])
        if mapKey in grouped_maps.keys():
            grouped_maps[mapKey].append(key)
        else:
            grouped_maps[mapKey] = [key]

    result = {}
    # Now that we have grouped the map, process each group
    for group in grouped_maps.keys():

        # Construct a basic healpix header from the group key, this
        # will be commun for all the maps in this group, this avoid
        # reading the header once again

        nside, ordering, coordsys = group.split('_')
        hp_header = {'NSIDE': int(nside),
                     'COORDSYS': coordsys,
                     'ORDERING': ordering}

        # Extract mask & pixels, common for all healpix maps of this
        # group
        logger.warning('projecting pixels')
        mask, ipix = hp_to_wcs_ipx(hp_header, w, npix=n_pixels)

        # Set up the figure, common for all healpix maps of this group
        logger.warning('setting figure')
        patch = np.zeros((n_pixels, n_pixels))
        fig=plt.figure()
        wcs_proj=WCS(w.to_header())
        ax_wcs=fig.add_axes([0.1,0.1,0.9,0.9],projection=wcs_proj)
        proj_im = ax_wcs.imshow(patch, interpolation='none', origin='lower' )
        ax_wcs.coords.grid(color='green', linestyle='solid', alpha=0.5)

        if np.str(coordf)=="EQUATORIAL":
            ax_wcs.coords['ra'].set_ticks(color='white')
            ax_wcs.coords['dec'].set_ticks(color='white')
            ax_wcs.coords['ra'].set_axislabel(r'$\alpha_\mathrm{J2000}$')
            ax_wcs.coords['dec'].set_axislabel(r'$\delta_\mathrm{J2000}$')
        elif np.str(coordf)=="GALACTIC":
            ax_wcs.coords['glon'].set_ticks(color='red')
            ax_wcs.coords['glat'].set_ticks(color='red')
            ax_wcs.coords['glon'].set_axislabel(r'$l$')
            ax_wcs.coords['glat'].set_axislabel(r'$b$')

        # Now the actual healpix map reading and projection
        for mapKey in grouped_maps[group]:
            logger.warning('reading '+mapKey)

            hp_map = hp.read_map(maps[mapKey]['filename'], verbose=False, dtype=np.float32)
            patch = np.ma.array(np.zeros((n_pixels, n_pixels)), mask=~mask, fill_value=np.nan)
            patch[mask] = hp_map[ipix]

            logger.warning('updating '+mapKey)
            proj_im.set_data(patch)
            proj_im.set_clim(vmin=patch.min(), vmax=patch.max())

            if 'doContour' in maps[mapKey].keys() and maps[mapKey]['doContour']:
                logger.warning('contouring '+mapKey)
                levels=[patch.max()/3., patch.max()/2.]
                if ((patch.max()-patch.mean()) > 3*patch.std()):
                    proj_cont = ax_wcs.contour(patch,levels=levels,colors="white",interpolation='bicubic')
                else:
                    proj_cont = None

            logger.warning('saving '+mapKey)

            output_map = BytesIO()
            plt.savefig(output_map,bbox_inches='tight', format='png',dpi=75, frameon=False)
            result[mapKey.replace(" ", "")] = {'name': mapKey,
                                               'png': b64encode(output_map.getvalue()).strip()}
            logger.warning(mapKey+ ' done')


            # TODO: Manage the contour cleaning from one map to the
            # next, maybe START by finding the contouring map, and do
            # it first.

   #  logger.warning('reading Xmap')
   # # Rosat Map
   #  filemapx = os.path.join(BASE_DIR,'xmatch/data/map_rosat_70-200_2048.fits')
   #  xmap = hp.read_map(filemapx, verbose=False, dtype=np.float32) #, memmap=True)

   #  logger.warning('updating Xmap')
   #  xpatch = np.ma.array(np.zeros((n_pixels, n_pixels)), mask=~mask, fill_value=np.nan)
   #  xpatch[mask] = xmap[ipix]

   #  # Update figure
   #  proj_im.set_data(xpatch)
   #  proj_im.set_clim(vmin=xpatch.min(), vmax=xpatch.max())

   #  logger.warning('saving Xmap')
   #  outputxmap = BytesIO()
   #  plt.savefig(outputxmap,bbox_inches='tight', format='png',dpi=75, frameon=False)
   #  logger.warning('Xmap done')

   #  # Only the contour
   #  logger.warning('updating Contour')

   #  proj_im.set_visible(False)
   #  if not proj_cont:
   #      ax_wcs.contour(ypatch,levels=levels, transform=ax_wcs.get_transform(wcs_proj),colors="red", interpolation="bicubic")

   #  logger.warning('saving Contour')
   #  outputcmap = BytesIO()
   #  plt.savefig(outputcmap,bbox_inches='tight', format='png', dpi=75, frameon=False)
   #  logger.warning('contour done')



########################################################### APERTURE


    logger.warning('map done')


    # positions = [(n_pixels/2., n_pixels/2.)]
    # apertures = CircularAperture(positions, r=3.0/pixel_size)
    # yphot = aperture_photometry(ypatch-np.median(ypatch), apertures)
    # xphot = aperture_photometry(xpatch-np.median(xpatch), apertures)

    # logger.warning('phot ok')

    return result

    # return {'mapy':b64encode(outputymap.getvalue()).strip(),
    #         'mapx':b64encode(outputxmap.getvalue()).strip(),
    #         'mapc':b64encode(outputxmap.getvalue()).strip(),
    #         'xphot':xphot,
    #         'yphot':yphot,}

def parse_args():
    """Parse arguments from the command line"""
    parser = argparse.ArgumentParser(description="Reproject the spherical sky onto a plane.")
    parser.add_argument('lon', type=float,
                        help='longitude of the projection [deg]')
    parser.add_argument('lat', type=float,
                        help='latitude of the projection [deg]')

    # Removed the default argument from here otherwise the config file
    # wont be used

    parser.add_argument('--npix', nargs=1, type=int,
                        help='number of pixels (default 256)')
    parser.add_argument('--pixsize', nargs=1, type=float,
                        help='pixel size [arcmin] (default 1)')

    parser.add_argument('--coordframe', required=False,
                        help='coordinate frame of the lon. and \
                        lat. of the projection and the projected map \
                        (default: galactic)',
                        choices=['galactic', 'fk5'])

    parser.add_argument('--conffile', required=False,
                        help='Absolute path to a config file')

    parser.add_argument('--mapfilenames', nargs='+', required=False,
                        help='Absolute path to the healpix maps')


    # Do the actual parsing
    args = parser.parse_args()
    #args = parser.parse_args('0 0'.split())

    # Put the list of filenames into the same structure as the config
    # file, we are loosing the doContour keyword but...
    if args.mapfilenames:
        args.maps = dict([ (os.path.basename(filename), dict([('filename', filename) ]) ) for filename in args.mapfilenames ])
    else:
        args.maps = None

    return args


def parse_config(conffile=None):
    """Parse options from a configuration file."""
    config = ConfigParser()

    # Look for cutsky.cfg at several locations
    conffiles = [ os.path.join(directory, 'cutsky.cfg') for directory
                  in [os.curdir, os.path.join(os.path.expanduser("~"), '.config/cutsky')] ]

    # If specifically ask for a config file, then put it at the
    # beginning ...
    if conffile:
        conffiles.insert(0, conffile)

    found = config.read(conffiles)

    if not found:
        raise FileNotFoundError

    # Basic options, same as the option on the command line
    options = {}
    if config.has_option('cutsky','npix'):
        options['npix'] = config.getint('cutsky', 'npix')
    if config.has_option('cutsky', 'pixsize'):
        options['pixsize'] = config.getfloat('cutsky', 'pixsize')
    if config.has_option('cutsky', 'coordframe'):
        options['coordframe'] = config.get('cutsky', 'coordframe')

    # Map list, only get the one which will be projected
    # Also check if contours are requested
    mapsToCut = {}
    for section in config.sections():
        if section != 'cutsky':
            mapInfo = {}
            if config.has_option(section, 'doCut'):
                if config.getboolean(section, 'doCut'):
                    if config.has_option(section, 'filename'):
                        mapInfo['filename'] = config.get(section, 'filename')
                    if config.has_option(section, 'doContour'):
                        mapInfo['doContour'] = config.getboolean(section, 'doContour')
                    mapsToCut[section] = mapInfo

    options['maps'] = mapsToCut

    return options

def main():
    # Mainly for test purpose

    from base64 import b64decode

    try:
        args = parse_args()
    except SystemExit:
        sys.exit()

    try:
        config = parse_config(args.conffile)

    except FileNotFoundError:
        config = {}

    # This is where the default arguments are set

    npix = args.npix or config.get('npix') or 256
    pixsize = args.pixsize or config.get('pixsize') or 1
    coordframe = args.coordframe or config.get('coordframe') or 'galactic'

    maps = args.maps or config.get('maps') or None

    result = cut_sky(lonlat=[args.lon, args.lat], patch=[npix, pixsize], coordframe=coordframe, maps=maps)

    for mapKey in result.keys():
        output = open(mapKey+'.png', 'wb')
        output.write(b64decode(result[mapKey]['png']))
        output.close()


if __name__ == '__main__':
    main()
