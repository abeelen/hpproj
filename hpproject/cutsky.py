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

from hpproject.hp_helpers import build_WCS, hp_to_wcs_ipx


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
