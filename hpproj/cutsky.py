#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""cutsky module, mainly use :mod:`hpproj.hp_helper` functions"""

import warnings
import logging

import os
import sys
from base64 import b64encode
from itertools import groupby

try:  # pragma: py3
    from io import BytesIO
except ImportError:  # pragma: py2
    from cStringIO import StringIO as BytesIO

import numpy as np

# try: # prama: no cover
# from wcsaxes import WCS # (deprecated)
# except ImportError: # pragma: no cover
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord

from photutils import CircularAperture
from photutils import aperture_photometry

from .wcs_helper import VALID_EQUATORIAL
from .wcs_helper import equiv_celestial, build_wcs
from .hp_helper import build_hpmap, gen_hpmap
from .hp_helper import hp_to_wcs_ipx, hpmap_key

from .parse import ini_main
from .parse import DEFAULT_NPIX, DEFAULT_PIXSIZE, DEFAULT_COORDFRAME, DEFAULT_CTYPE

import matplotlib.pyplot as plt

try:  # pragma: py3
    FileNotFoundError
except NameError:  # pragma: py2
    FileNotFoundError = IOError

LOGGER = logging.getLogger('django')

DEFAULT_PATCH = [256, 1.]


class CutSky(object):

    """
    Container for Healpix maps and cut_* methods

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
        a grouped dictionnary of gen_hpmap tuples (filename, map, header) (see :func:~init)


    """

    def __init__(self, maps=None, npix=DEFAULT_NPIX, pixsize=DEFAULT_PIXSIZE, ctype=DEFAULT_CTYPE, low_mem=True):
        """Initialization of a CutSky class

        Parameters
        ----------
        maps : list of tuple or dictionary
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
        The list of dictionnary describe the maps to be projected :
        ```
        [(full_filename_to_healpix_map.fits, {'legend': legend,
                                              'doContour': True}), # optionnal
          ... ]
        ```
        The {opt} dictionnary MUST containt, at least, the key
        'legend' which will be used to uniquely identify the cutted
        map, other possible keys 'doContour' with a boolean value to
        contour the map latter one

        Alternatively one can use the old interface as a dictionnary :
        ```
        {legend: {'filename': full_filename_to_healpix_map.fits,
                    'doContour': True }, # optionnal
         ... }
        ```
        """

        if maps is None:
            raise FileNotFoundError("No healpix map to project")

        if isinstance(maps, dict):
            maps = to_new_maps(maps)

        # Define the basic parameters for the output maps
        self.npix = npix
        self.pixsize = pixsize
        self.ctype = ctype

        # Build an hp_map list and extend the headers of the maps with
        # the optionnal values
        filenames, opts = zip(*maps)
        hp_map = build_hpmap(filenames, low_mem=low_mem)
        for (filename, i_map, i_header), opt in zip(hp_map, opts):
            # Insure that we do have a doContour key internally
            if 'doContour' not in opt.keys():
                opt['doContour'] = False
            i_header.extend(opt)

        # group them by map properties for efficiencies reasons
        hp_map.sort(key=hpmap_key)
        self.maps = hp_map

        # Save intermediate results
        self.maps_selection = None
        self.cuts = None
        self.lonlat = None
        self.coordframe = DEFAULT_COORDFRAME

    def cut_fits(self, lonlat=None, coordframe=DEFAULT_COORDFRAME, maps_selection=None):
        """Efficiently cut the healpix maps and return cutted fits file with proper header

        Parameters
        ----------
        lonlat : array of 2 floats
            the longitude and latitude of the center of projection [deg]
        coordframe : str
            the coordinate frame used for the position AND the projection
        maps_selection : list
            optionnal list of the 'legend' or filename of the map to
            select a sub-sample of them.

        Returns
        -------
        list of dictionnaries
            the dictionnary has 2 keys :
            * 'legend' (the opts{'legend'} see __init())
            * 'fits' an :class:`~astropy.io.fits.ImageHDU`
        """

        assert len(lonlat) == 2, "lonlat must have 2 elements"

        self.maps_selection = maps_selection

        # Center of projection
        coord_in = SkyCoord(
            lonlat[0], lonlat[1], unit=u.deg, frame=equiv_celestial(coordframe))

        # Build the target WCS header
        wcs = build_wcs(coord_in, pixsize=self.pixsize / 60.,
                        shape_out=(self.npix, self.npix), proj_sys=coordframe, proj_type=self.ctype)

        cuts = []
        for group, maps in groupby(self.maps, key=hpmap_key):
            LOGGER.info('projecting ' + group)

            # Construct a basic healpix header from the group key, this
            # will be commun for all the maps in this group
            nside, ordering, coordsys = group.split('_')
            hp_header = {'NSIDE': int(nside),
                         'ORDERING': ordering,
                         'COORDSYS': coordsys}

            # Extract mask & pixels, common for all healpix maps of this
            # group
            mask, ipix = hp_to_wcs_ipx(hp_header, wcs, shape_out=(self.npix, self.npix))

            # Set up the figure, common for all healpix maps of this group
            LOGGER.info('cutting maps')

            # Now the actual healpix map reading and projection
            for filename, i_hdu in self._to_process(gen_hpmap(maps)):
                legend = i_hdu.header['legend']
                patch = np.ma.array(
                    np.zeros((self.npix, self.npix)), mask=~mask, fill_value=np.nan)
                patch[mask] = i_hdu.data[ipix]
                header = wcs.to_header()
                header.append(('filename', filename))
                header.append(('legend', legend))
                header.append(('doContour', i_hdu.header['doContour']))

                cuts.append({'legend': legend,
                             'fits': fits.ImageHDU(patch, header)})
        self.lonlat = lonlat
        self.coordframe = coordframe
        self.cuts = cuts

        return cuts

    def _to_process(self, gen_hpmap_iter):
        """Iterator to filter the gen_hpmap iterator depending on map selection"""

        for filename, i_hdu in gen_hpmap_iter:
            legend = i_hdu.header['legend']
            if not self.maps_selection or (legend in self.maps_selection or filename in self.maps_selection):
                yield filename, i_hdu

    def _get_cuts(self, lonlat=None, coordframe=None, maps_selection=None):
        """Get map cuts if they are already made, or launch cut_fits

        Parameters
        ----------
        lonlat : array of 2 floats
            the longitude and latitude of the center of projection [deg]
        coordframe : str
            the coordinate frame used for the position AND the projection
        maps_selection : list
            optionnal list of the 'legend' or filename of the map to
            select a sub-sample of them.

        Returns
        -------
        list of dictionnaries
            the dictionnary has 2 keys :
            * 'legend' (the opts{'legend'} see __init())
            * 'fits' an :class:`~astropy.io.fits.ImageHDU`
        """

        if self.lonlat == lonlat and \
           self.coordframe == coordframe and  \
           self.maps_selection == maps_selection and \
           self.cuts:
            # Retrieve previously cut maps
            cuts = self.cuts
        else:
            # Or cut the maps
            cuts = self.cut_fits(
                lonlat=lonlat, coordframe=coordframe, maps_selection=maps_selection)

        return cuts

    def cut_png(self, lonlat=None, coordframe=DEFAULT_COORDFRAME, maps_selection=None):
        """Efficiently cut the healpix maps and return cutted fits file with proper header and corresponding png

        Parameters
        ----------
        lonlat : array of 2 floats
            the longitude and latitude of the center of projection [deg]
        coordframe : str
            the coordinate frame used for the position AND the projection
        maps_selection : list
            optionnal list of the 'legend' or filename of the map to
            select a sub-sample of them.

        Returns
        -------
        list of dictionnaries
            the dictionnary has 3 keys :
            * 'legend' (the opts{'legend'} see __init()),
            * 'fits' an :class:`~astropy.io.fits.ImageHDU`,
            * 'png', a b61encoded png image of the fits

        """

        cuts = self._get_cuts(lonlat, coordframe, maps_selection)

        patch = np.zeros((self.npix, self.npix))

        # Plotting
        old_backend = plt.get_backend()
        plt.switch_backend('Agg')
        fig = plt.figure()

        ax_wcs = fig.add_axes(
            [0.1, 0.1, 0.9, 0.9], projection=WCS(cuts[0]['fits']))
        proj_im = ax_wcs.imshow(patch, interpolation='none', origin='lower')
        ax_wcs.coords.grid(color='green', linestyle='solid', alpha=0.5)

        # DEFAULT_COORDFRAME is galactic
        axes = [('glon', 'red', r'$l$'),
                ('glat', 'red', r'$b$')]

        if np.str(coordframe) in VALID_EQUATORIAL:
            axes = [('ra', 'white', r'$\alpha_\mathrm{J2000}$'),
                    ('dec', 'white', r'$\delta_\mathrm{J2000}$')]

        for axis, color, label in axes:
            ax_wcs.coords[axis].set_ticks(color=color)
            ax_wcs.coords[axis].set_axislabel(label)

        for cut in cuts:

            legend = cut['legend']
            LOGGER.debug('plotting ' + legend)

            patch = cut['fits'].data
            patch_header = cut['fits'].header

            proj_im.set_data(patch)
            proj_im.set_clim(vmin=patch.min(), vmax=patch.max())

            # Insure we do have increasing values even when patch.std == 0
            levels = np.arange(2, 10) * (np.max([patch.std(), 1e-12]))
            if patch_header['doContour']:
                LOGGER.debug('contouring ' + legend)
                ax_wcs.contour(
                    patch, levels=levels, colors="white", interpolation='bicubic')

            # TODO: Retrive and use the contour in other plots

            LOGGER.debug('saving ' + legend)

            # Add the map to the cut dictionnary
            output_map = BytesIO()
            plt.savefig(output_map, bbox_inches='tight',
                        format='png', dpi=75, frameon=False)
            cut['png'] = b64encode(output_map.getvalue()).strip()
            LOGGER.debug(legend + ' done')

        plt.switch_backend(old_backend)
        return cuts

    def cut_phot(self, lonlat=None, coordframe=DEFAULT_COORDFRAME, maps_selection=None):
        """Efficiently cut the healpix maps and return cutted fits file with proper header and corresponding photometry

        Parameters
        ----------
        lonlat : array of 2 floats
            the longitude and latitude of the center of projection [deg]
        coordframe : str
            the coordinate frame used for the position AND the projection
        maps_selection : list
            optionnal list of the 'legend' or filename of the map to
            select a sub-sample of them.

        Returns
        -------
        list of dictionnaries
            the dictionnary has 3 keys :
            * 'legend' (the opts{'legend'} see __init()),
            * 'fits' an :class:`~astropy.io.fits.ImageHDU`,
            * 'phot', the corresponding photometry

        """

        cuts = self._get_cuts(lonlat, coordframe, maps_selection)

        positions = [(self.npix * 1. / 2., self.npix * 1. / 2)]
        apertures = CircularAperture(positions, r=3. / self.pixsize)

        for cut in cuts:
            legend = cut['legend']
            LOGGER.debug('phot on ' + legend)

            patch = cut['fits'].data
            cut['phot'] = aperture_photometry(
                patch - np.median(patch), apertures)

            LOGGER.debug(legend + ' done')

        return cuts

    def cut(self, cut_type, **kwargs):
        """helper function to cut into the maps

        Parameters
        ----------
        cut_type : str (fits|png|phot|votable)
            define what to cut_type
        lonlat : array of 2 floats
            the longitude and latitude of the center of projection [deg]
        coordframe : str
            the coordinate frame used for the position AND the projection
        maps_selection : list
            optionnal list of the 'legend' or filename of the map to
            select a sub-sample of them.

        Returns
        -------
        list of dictionnaries
            the dictionnary output depends on cut_type
        """

        if cut_type == 'fits':
            results = self.cut_fits(**kwargs)
        elif cut_type == 'png':
            results = self.cut_png(**kwargs)
        elif cut_type in ['phot', 'votable']:
            results = self.cut_phot(**kwargs)

        return results


def to_new_maps(maps):
    """Transform old dictionnary type healpix map list used by cutsky to
    list of tuple used by Cutsky

    Parameters
    ----------
    maps : dict
        a dictionnary with key being the legend of the image :
        ```
        {legend: {'filename': full_filename_to_healpix_map.fits,
                    'doContour': True },
         ... }
        ```

    Returns
    -------
    a list of tuple following the new convention:
    ```
    [(full_filename_to_healpix_map.fits, {'legend': legend,
                                          'doContour': True}),
     ... ]
    ```
    """

    warnings.warn("deprecated", DeprecationWarning)

    new_maps = []
    for key in iter(maps.keys()):
        filename = maps[key]['filename']
        opt = {'legend': key}
        if 'doContour' in maps[key]:
            opt['doContour'] = maps[key]['doContour']
        new_maps.append((filename, opt))

    return new_maps


def cutsky(lonlat=None, maps=None, patch=None, coordframe=DEFAULT_COORDFRAME, ctype=DEFAULT_CTYPE):
    """Old interface to cutsky -- Here mostly for compability

    Parameters
    ----------
    lonlat : array of 2 floats
        the longitude and latitude of the center of projection [deg]
    maps: a dict or a list
        either a dictionnary (old interface) or a list of tuple (new
        interface) :
        ```
        {legend: {'filename': full_filename_to_healpix_map.fits,
                  'doContour': True }, # optionnal
         ... }
         ```
         or
         ```
         [(full_filename_to_healpix_map.fits, {'legend': legend,
                                              'doContour': True}), # optionnal
         ... ]
         ```
    patch : array of [int, float]
        [int] the number of pixels and
        [float] the size of the pixel [arcmin]
    coordframe : str
        the coordinate frame used for the position AND the projection
    ctype: str
        a valid projection type (default: TAN)

    Returns
    -------
    list of dictionnaries
        the dictionnary has 4 keys :
        * 'legend' (see maps above),
        * 'fits' an :class:`~astropy.io.fits.ImageHDU`,
        * 'png', a b61encoded png image of the fits
        * 'phot', the corresponding photometry

    """

    assert lonlat is not None, "You must provide a lonlat argument"
    assert len(lonlat) == 2, "lonlat must have 2 elements"
    assert maps is not None, "You must provide an healpix map to project"

    if patch is None:
        patch = DEFAULT_PATCH

    if isinstance(maps, dict):
        maps = to_new_maps(maps)

    cut_those_maps = CutSky(
        maps=maps, npix=patch[0], pixsize=patch[1], ctype=ctype)
    result = cut_those_maps.cut_png(lonlat=lonlat, coordframe=coordframe)
    result = cut_those_maps.cut_phot(lonlat=lonlat, coordframe=coordframe)

    return result


def save_result(output, result):
    """Save the results of the main function"""

    from base64 import b64decode

    filename = os.path.join(output['outdir'], result['legend'])
    if output['fits']:
        # Works only on more recent version of astropy...
        # try:
        #     hdulist = fits.HDUList([fits.PrimaryHDU(), result['fits']])
        #     hdulist.writeto(filename + '.fits', clobber=True)
        # except NotImplementedError:
        # So for now...
        result['fits'].data = result['fits'].data.filled()
        hdulist = fits.HDUList([fits.PrimaryHDU(), result['fits']])
        hdulist.writeto(filename + '.fits', overwrite=True)

    if output['png']:
        png = open(filename + '.png', 'wb')
        png.write(b64decode(result['png']))
        png.close()

    if output['votable']:
        # Need to cast into astropy Table before writing to votable
        phot = Table(result['phot'])
        with open(filename + '.xml', 'w') as file_out:
            phot.write(file_out, format='votable')


def main(argv=None):
    """The main routine."""

    args = ini_main(argv)

    cut_those_maps = CutSky(maps=args['maps'], npix=args['npix'], pixsize=args['pixsize'], ctype=args['ctype'])
    for key in ['fits', 'png', 'votable']:
        if args[key]:
            results = cut_those_maps.cut(
                key, lonlat=[args['lon'], args['lat']], coordframe=args['coordframe'])

    if not os.path.isdir(args['outdir']):
        os.makedirs(args['outdir'])

    for result in results:
        save_result(args, result)


if __name__ == '__main__':
    main(sys.argv[1:])
