#! /usr/bin/env python

import logging
logger = logging.getLogger('django')

import warnings
import os, sys
import argparse

import numpy as np
import healpy as hp

try: # prama: no cover
    from wcsaxes import WCS # (deprecated)
except ImportError: # pragma: no cover
    from astropy.wcs import WCS

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from photutils import CircularAperture
from photutils import aperture_photometry

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try: # pragma: py3
    from configparser import ConfigParser
except ImportError: # pragma: py2
    from ConfigParser import ConfigParser

try: # pragma: py3
    from io import BytesIO
except ImportError: # pragma: py2
    from cStringIO import StringIO as BytesIO

from base64 import b64encode

try: # pragma: py3
    FileNotFoundError
except NameError: # pragma: py2
    FileNotFoundError = IOError

from .hp_helper import build_WCS, hp_to_wcs_ipx, VALID_PROJ
from .hp_helper import build_hpmap, group_hpmap, gen_hpmap

DEFAULT_npix = 256
DEFAULT_pixsize = 1
DEFAULT_coordframe = 'galactic'
DEFAULT_ctype = 'TAN'

class CutSkySquare:
    """
    Container for Healpix maps and cutsky methods

    ...

    Attributes
    ----------
    npix : int
        the number of pixels for the square maps
    pixsize : float
        the size of the pixels [arcmin]
    ctype : str
        a valid projection type (default : TAN)
    maps : dictonnary
        a grouped dictionnary of gen_hpmap tuples (filename, map, header)


    """
    def __init__(self, maps, npix=DEFAULT_npix, pixsize=DEFAULT_pixsize, ctype=DEFAULT_ctype, low_mem=True):
        """Initialization of a CutSkySquare class

        Parameters
        ----------
        maps : list of tuple
            list of tuple (filename, {opt}) where filename is the full
            path to the healpix map and {opt} is a dictionnary with
            "optional" header for the file (see Notes).
        npix : int
            the number of pixels for the square maps
        pixsize : float
            the size of the pixels [arcmin]
        ctype : str
            a valid projection type (default : TAN)
        low_mem: bool
            a boolean deciding if all the maps are loaded in memory or
            not (default: True, the maps are not loaded on init)

        Notes
        -----

        The {opt} dictionnary MUST containt, at least, the key
        'legend' which will be used to uniquely identify the cutted
        map, other possible keys 'doContour' with a boolean value to
        contour the map latter one

        """

        # Define the basic parameters for the output maps
        self.npix = npix
        self.pixsize = pixsize
        self.ctype = ctype

        # Build an hp_map list and extend the headers of the maps with
        # the optionnal values
        filenames, opts = [filename for filename, opt in maps], [opt for filename, opt in maps]
        hp_map = build_hpmap(filenames, low_mem=low_mem)
        for (filename, iMap, iHeader), opt in zip(hp_map, opts):
            iHeader.extend(opt)

        # group them by map properties for efficiencies reasons
        self.maps = group_hpmap(hp_map)

    def cutsky_fits(self,lonlat = [0, 0], coordframe = DEFAULT_coordframe):
        """Efficiently cut the healpix maps and return cutted fits file with proper header

        Parameters
        ----------
        lonlat : array of 2 floats
            the longitude and latitude of the center of projection [deg]
        coordframe : str
            the coordinate frame used for the position AND the projection

        Returns
        -------
        list of dictionnaries
            the dictionnary as two keys 'legend' (the opts{'legend'} see
            __init()) and 'fits' an ~astropy.io.fits.ImageHDU
        """

        # Center of projection
        coord_in = SkyCoord(lonlat[0],lonlat[1], unit=u.deg, frame=coordframe)

        # Build the target WCS header
        w = build_WCS(coord_in, pixsize=self.pixsize/60., npix=self.npix, proj_sys=coordframe, proj_type=self.ctype)

        cut_fits = []
        for group in self.maps.keys():
            logger.warning('projecting '+group)

            maps = self.maps[group]

            # Construct a basic healpix header from the group key, this
            # will be commun for all the maps in this group
            hp_header = maps[0][2]

            # Extract mask & pixels, common for all healpix maps of this
            # group
            mask, ipix = hp_to_wcs_ipx(hp_header, w, npix=self.npix)

            # Set up the figure, common for all healpix maps of this group
            logger.warning('cutting maps')

            # Now the actual healpix map reading and projection
            for filename, iMap, iHeader in gen_hpmap(maps):
                legend = iHeader['legend']
                patch = np.ma.array(np.zeros((self.npix, self.npix)), mask=~mask, fill_value=np.nan)
                patch[mask] = iMap[ipix]
                header = w.to_header()
                header.append(('filename',filename) ) #, 'original healpix filename')
                header.append(('legend', legend) ) #, 'cutsky legend')
                if 'doContour' in iHeader.keys():
                    header.append(('doContour', iHeader['doContour']))

                cut_fits.append( {'legend': legend,
                                  'fits': fits.ImageHDU(patch, header)} )

        return cut_fits

    def cutsky_png(self, lonlat = [0, 0], coordframe = DEFAULT_coordframe):
        """Efficiently cut the healpix maps and return cutted fits file with proper header and corresponding png

        Parameters
        ----------
        lonlat : array of 2 floats
            the longitude and latitude of the center of projection [deg]
        coordframe : str
            the coordinate frame used for the position AND the projection

        Returns
        -------
        list of dictionnaries
            the dictionnary as two keys 'legend' (the opts{'legend'} see
            __init()), 'fits' an ~astropy.io.fits.ImageHDU,
            'png', a b61encoded png image of the fits

        """

        # Actual cutting
        cut_fits = self.cutsky_fits(lonlat=lonlat, coordframe=coordframe)

        # Common WCS for all cut maps
        w = WCS(cut_fits[0]['fits'])

        # Plotting
        patch = np.zeros((self.npix, self.npix))
        fig=plt.figure()

        wcs_proj=WCS(w.to_header())
        ax_wcs=fig.add_axes([0.1,0.1,0.9,0.9],projection=wcs_proj)
        proj_im = ax_wcs.imshow(patch, interpolation='none', origin='lower' )
        ax_wcs.coords.grid(color='green', linestyle='solid', alpha=0.5)

        if np.str(coordframe)=="EQUATORIAL":
            ax_wcs.coords['ra'].set_ticks(color='white')
            ax_wcs.coords['dec'].set_ticks(color='white')
            ax_wcs.coords['ra'].set_axislabel(r'$\alpha_\mathrm{J2000}$')
            ax_wcs.coords['dec'].set_axislabel(r'$\delta_\mathrm{J2000}$')
        elif np.str(coordframe)=="GALACTIC":
            ax_wcs.coords['glon'].set_ticks(color='red')
            ax_wcs.coords['glat'].set_ticks(color='red')
            ax_wcs.coords['glon'].set_axislabel(r'$l$')
            ax_wcs.coords['glat'].set_axislabel(r'$b$')

        cut_png = {}
        for cut_fit in cut_fits :

            legend = cut_fit['legend']

            logger.debug('plotting '+ legend)

            patch = cut_fit['fits'].data
            patch_header = cut_fit['fits'].header

            proj_im.set_data(patch)
            proj_im.set_clim(vmin=patch.min(), vmax=patch.max())

            if 'doContour' in patch_header.keys() and patch_header['doContour']:
                logger.warning('contouring '+legend)
                levels=[patch.max()/3., patch.max()/2.]
                if ((patch.max()-patch.mean()) > 3*patch.std()):
                    proj_cont = ax_wcs.contour(patch,levels=levels,colors="white",interpolation='bicubic')
                else:
                    proj_cont = None

            logger.debug('saving '+legend)

            output_map = BytesIO()
            plt.savefig(output_map,bbox_inches='tight', format='png',dpi=75, frameon=False)
            cut_png[legend.replace(" ", "")] = {'legend': legend,
                                                'fits': cut_fit['fits'],
                                                'png': b64encode(output_map.getvalue()).strip() }
            logger.warning(legend+ ' done')

        return cut_png

def cutsky(lonlat=[0, 0], patch=[256, 1], coordframe='galactic', ctype=DEFAULT_ctype, maps=None):
    """Old interface to cutsky -- Here for compability"""

    warnings.warn("deprecated", DeprecationWarning)

    if not maps:
        raise FileNotFoundError("No healpix map to project")

    CutThoseMaps = CutSkySquare(maps, npix=patch[0], pixsize=patch[1], ctype=ctype)
    result = CutThoseMaps.cutsky_png(lonlat=lonlat, coordframe=coordframe)

    return result

# def cutsky( lonlat=[0,0],patch=[256,1],coordframe='galactic', ctype=DEFAULT_ctype, maps=None):


#     pixel_size=np.float(patch[1])
#     n_pixels=np.float(patch[0])

#     if np.str(coordframe)=="galactic":
#         coordf = 'GALACTIC'
#     elif np.str(coordframe)=="fk5":
#         coordf = 'EQUATORIAL'

#     # Center of projection
#     coord_in = SkyCoord(lonlat[0],lonlat[1], unit=u.deg, frame=coordframe)

#     # Build the target WCS header
#     w = build_WCS(coord_in, pixsize=pixel_size/60., npix=n_pixels, proj_sys=np.str(coordf), proj_type=ctype)

#     # Group the maps by header, keys is NSIDE_ORDERING_COORDSYS
#     grouped_maps = {}
#     for key in maps.keys():
#         iHeader = fits.getheader(maps[key]['filename'],1)
#         mapKey = "%s_%s_%s"%(iHeader['NSIDE'], iHeader['ORDERING'], iHeader['COORDSYS'])
#         if mapKey in grouped_maps.keys():
#             grouped_maps[mapKey].append(key)
#         else:
#             grouped_maps[mapKey] = [key]

#     result = {}
#     # Now that we have grouped the map, process each group
#     for group in grouped_maps.keys():

#         # Construct a basic healpix header from the group key, this
#         # will be commun for all the maps in this group, this avoid
#         # reading the header once again

#         nside, ordering, coordsys = group.split('_')
#         hp_header = {'NSIDE': int(nside),
#                      'COORDSYS': coordsys,
#                      'ORDERING': ordering}

#         # Extract mask & pixels, common for all healpix maps of this
#         # group
#         logger.warning('projecting pixels')
#         mask, ipix = hp_to_wcs_ipx(hp_header, w, npix=n_pixels)

#         # Set up the figure, common for all healpix maps of this group
#         logger.warning('setting figure')
#         patch = np.zeros((n_pixels, n_pixels))
#         fig=plt.figure()
#         wcs_proj=WCS(w.to_header())
#         ax_wcs=fig.add_axes([0.1,0.1,0.9,0.9],projection=wcs_proj)
#         proj_im = ax_wcs.imshow(patch, interpolation='none', origin='lower' )
#         ax_wcs.coords.grid(color='green', linestyle='solid', alpha=0.5)

#         if np.str(coordf)=="EQUATORIAL":
#             ax_wcs.coords['ra'].set_ticks(color='white')
#             ax_wcs.coords['dec'].set_ticks(color='white')
#             ax_wcs.coords['ra'].set_axislabel(r'$\alpha_\mathrm{J2000}$')
#             ax_wcs.coords['dec'].set_axislabel(r'$\delta_\mathrm{J2000}$')
#         elif np.str(coordf)=="GALACTIC":
#             ax_wcs.coords['glon'].set_ticks(color='red')
#             ax_wcs.coords['glat'].set_ticks(color='red')
#             ax_wcs.coords['glon'].set_axislabel(r'$l$')
#             ax_wcs.coords['glat'].set_axislabel(r'$b$')

#         # Now the actual healpix map reading and projection
#         for mapKey in grouped_maps[group]:
#             logger.warning('reading '+mapKey)

#             hp_map = hp.read_map(maps[mapKey]['filename'], verbose=False, dtype=np.float32)
#             patch = np.ma.array(np.zeros((n_pixels, n_pixels)), mask=~mask, fill_value=np.nan)
#             patch[mask] = hp_map[ipix]

#             logger.warning('updating '+mapKey)
#             proj_im.set_data(patch)
#             proj_im.set_clim(vmin=patch.min(), vmax=patch.max())

#             if 'doContour' in maps[mapKey].keys() and maps[mapKey]['doContour']:
#                 logger.warning('contouring '+mapKey)
#                 levels=[patch.max()/3., patch.max()/2.]
#                 if ((patch.max()-patch.mean()) > 3*patch.std()):
#                     proj_cont = ax_wcs.contour(patch,levels=levels,colors="white",interpolation='bicubic')
#                 else:
#                     proj_cont = None

#             logger.warning('saving '+mapKey)

#             output_map = BytesIO()
#             plt.savefig(output_map,bbox_inches='tight', format='png',dpi=75, frameon=False)
#             result[mapKey.replace(" ", "")] = {'name': mapKey,
#                                                'png': b64encode(output_map.getvalue()).strip()}
#             logger.warning(mapKey+ ' done')


#             # TODO: Manage the contour cleaning from one map to the
#             # next, maybe START by finding the contouring map, and do
#             # it first.

#    #  logger.warning('reading Xmap')
#    # # Rosat Map
#    #  filemapx = os.path.join(BASE_DIR,'xmatch/data/map_rosat_70-200_2048.fits')
#    #  xmap = hp.read_map(filemapx, verbose=False, dtype=np.float32) #, memmap=True)

#    #  logger.warning('updating Xmap')
#    #  xpatch = np.ma.array(np.zeros((n_pixels, n_pixels)), mask=~mask, fill_value=np.nan)
#    #  xpatch[mask] = xmap[ipix]

#    #  # Update figure
#    #  proj_im.set_data(xpatch)
#    #  proj_im.set_clim(vmin=xpatch.min(), vmax=xpatch.max())

#    #  logger.warning('saving Xmap')
#    #  outputxmap = BytesIO()
#    #  plt.savefig(outputxmap,bbox_inches='tight', format='png',dpi=75, frameon=False)
#    #  logger.warning('Xmap done')

#    #  # Only the contour
#    #  logger.warning('updating Contour')

#    #  proj_im.set_visible(False)
#    #  if not proj_cont:
#    #      ax_wcs.contour(ypatch,levels=levels, transform=ax_wcs.get_transform(wcs_proj),colors="red", interpolation="bicubic")

#    #  logger.warning('saving Contour')
#    #  outputcmap = BytesIO()
#    #  plt.savefig(outputcmap,bbox_inches='tight', format='png', dpi=75, frameon=False)
#    #  logger.warning('contour done')



# ########################################################### APERTURE


#     logger.warning('map done')

#     # positions = [(n_pixels/2., n_pixels/2.)]
#     # apertures = CircularAperture(positions, r=3.0/pixel_size)
#     # yphot = aperture_photometry(ypatch-np.median(ypatch), apertures)
#     # xphot = aperture_photometry(xpatch-np.median(xpatch), apertures)

#     # logger.warning('phot ok')

#     return result

#     # return {'mapy':b64encode(outputymap.getvalue()).strip(),
#     #         'mapx':b64encode(outputxmap.getvalue()).strip(),
#     #         'mapc':b64encode(outputxmap.getvalue()).strip(),
#     #         'xphot':xphot,
#     #         'yphot':yphot,}

def parse_args(args):
    """Parse arguments from the command line"""
    parser = argparse.ArgumentParser(description="Reproject the spherical sky onto a plane.")
    parser.add_argument('lon', type=float,
                        help='longitude of the projection [deg]')
    parser.add_argument('lat', type=float,
                        help='latitude of the projection [deg]')

    # Removed the default argument from here otherwise the config file
    # wont be used

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--npix', type=int,
                        help='number of pixels (default 256)')
    group.add_argument('--radius', type=float,
                       help='radius of the requested region [deg] ')

    parser.add_argument('--pixsize', type=float,
                        help='pixel size [arcmin] (default 1)')

    parser.add_argument('--coordframe', required=False,
                        help='coordinate frame of the lon. and \
                        lat. of the projection and the projected map \
                        (default: galactic)',
                        choices=['galactic', 'fk5'])
    parser.add_argument('--ctype', required=False,
                        help='any projection code supported by wcslib\
                         (default:TAN)',
                        choices=VALID_PROJ)

    parser.add_argument('--mapfilenames', nargs='+', required=False,
                        help='Absolute path to the healpix maps')

    parser.add_argument('--conf', required=False,
                        help='Absolute path to a config file')



    # Do the actual parsing
    parsed_args = parser.parse_args(args)
    #parsed_args = parser.parse_args('0 0'.split())

    # Put the list of filenames into the same structure as the config
    # file, we are loosing the doContour keyword but...
    if parsed_args.mapfilenames:
        parsed_args.maps = [ ( filename,
                               dict( [('legend', os.path.basename(filename)) ]) )
                             for filename in
                             parsed_args.mapfilenames ]
    else:
        parsed_args.maps = None

    return parsed_args


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
    if config.has_option('cutsky', 'ctype'):
        options['ctype'] = config.get('cutsky', 'ctype')

    # Map list, only get the one which will be projected
    # Also check if contours are requested
    mapsToCut = []
    for section in config.sections():
        if section != 'cutsky':
            if config.has_option(section, 'doCut') and config.has_option(section, 'filename'):
                if config.getboolean(section, 'doCut'):
                    filename = config.get(section, 'filename')
                    opt = {'legend': section}
                    if config.has_option(section, 'doContour'):
                        opt['doContour'] = config.getboolean(section, 'doContour')
                    mapsToCut.append((filename, opt))

    options['maps'] = mapsToCut

    return options

def combine_args(args, config):
    """
    Combine the different sources of arguments (command line,
    configfile or default arguments
    """

    # This is where the default arguments are set
    pixsize = args.pixsize or config.get('pixsize') or \
              DEFAULT_pixsize
    coordframe = args.coordframe or config.get('coordframe') or \
                 DEFAULT_coordframe
    ctype = args.ctype or config.get('ctype') or \
                 DEFAULT_ctype

    # npix and radius are mutually exclusive, thus if radius is set we
    # need to compute npix
    if args.radius:
        args.npix = int(args.radius/(float(pixsize)/60))

    npix = args.npix or config.get('npix') or \
           DEFAULT_npix

    maps = args.maps or config.get('maps') or \
           None

    return npix, pixsize, coordframe, ctype, maps


def main(): # pragma: no cover
    # Mainly for test purpose

    args = "0. 0. --mapfilenames hpproj/data/CMB_I_SMICA_128_R2.00.fits          hpproj/data/HFI_SkyMap_353_256_R2.00_RING.fits  hpproj/data/HFI_SkyMap_857_128_R2.00_NEST.fits hpproj/data/HFI_SkyMap_100_128_R2.00_RING.fits  hpproj/data/HFI_SkyMap_545_128_R2.00_RING.fits --npix 256 --pixsize 2".split()

    from base64 import b64decode

    try:
        args = parse_args(sys.argv[1:])
    except SystemExit:
        sys.exit()

    try:
        config = parse_config(args.conf)
    except FileNotFoundError:
        config = {}

    npix, pixsize, coordframe, ctype, maps = combine_args(args, config)

    CutThoseMaps = CutSkySquare(maps, npix=npix, pixsize=pixsize, ctype=ctype)
    result = CutThoseMaps.cutsky_png(lonlat=[args.lon, args.lat], coordframe=coordframe)

    result = new_cutsky(lonlat=[args.lon, args.lat],
                        patch=[npix, pixsize], coordframe=coordframe,
                        ctype=ctype, maps=maps)

    # result = cutsky(lonlat=[args.lon, args.lat],
    #                 patch=[npix, pixsize], coordframe=coordframe,
    #                 ctype=ctype, maps=maps)

    for mapKey in result.keys():
        output = open(mapKey+'.png', 'wb')
        output.write(b64decode(result[mapKey]['png']))
        output.close()


if __name__ == '__main__':
    main()
