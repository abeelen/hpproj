#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

import os
import logging
import json

import pytest

from hpproj import CutSky, cutsky, main, to_new_maps
from hpproj import DEFAULT_NPIX, DEFAULT_PIXSIZE, DEFAULT_CTYPE
from hpproj import hp_is_nest, hp_celestial

import numpy as np
import numpy.testing as npt
import healpy as hp

try:  # pragma: py3
    FileNotFoundError
except NameError:  # pragma: py2
    FileNotFoundError = IOError

logger = logging.getLogger('django')


def test_CutSky_init_exception():
    with pytest.raises(FileNotFoundError):
        my_cutsky = CutSky()

    with pytest.raises(IOError):
        my_custky = CutSky(maps=[('toto.fits', {})])


@pytest.fixture(scope='session')
def generate_hpmap(tmpdir_factory):

    """Generate an uniform healpix map"""

    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside))
    hp_header = {'NSIDE': nside,
                 'ORDERING': 'RING',
                 'COORDSYS': 'C'}
    hp_key = "%s_%s_%s" % (hp_header['NSIDE'], hp_header['ORDERING'], hp_celestial(hp_header).name)

    tmpfile = tmpdir_factory.mktemp("data").join("tmpfile.fits")

    hp.write_map(str(tmpfile), hp_map, nest=hp_is_nest(hp_header), extra_header=hp_header.items())
    return ([(str(tmpfile), {'legend': 'tmpfile'})], hp_map, hp_key)


@pytest.fixture(scope='session')
def generate_mis_hpmap(tmpdir_factory):

    """Generate an uniform healpix map"""

    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside))
    hp_header = {'NSIDE': nside,
                 'ORDERING': 'RING',
                 'COORDSYS': 'C'}
    hp_key = "%s_%s_%s" % (hp_header['NSIDE'], hp_header['ORDERING'], hp_celestial(hp_header).name)

    tmpfile = tmpdir_factory.mktemp("data").join("tmpfile.fits")

    # Removing COORDSYS from header
    hp_header.pop('COORDSYS')
    hp.write_map(str(tmpfile), hp_map, nest=hp_is_nest(hp_header), extra_header=hp_header.items())
    return ([(str(tmpfile), {'legend': 'tmpfile'})], hp_map, hp_key)


# TODO : what happen when file do not exist or are not healpix maps
def test_CutSky_init(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]

    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    assert my_cutsky.npix == DEFAULT_NPIX
    assert my_cutsky.pixsize == DEFAULT_PIXSIZE
    assert my_cutsky.ctype == DEFAULT_CTYPE

    assert my_cutsky.maps[0][0] == filename
    assert my_cutsky.maps[0][1] == filename
    assert my_cutsky.maps[0][2]['legend'] == opt['legend']

    hp_map[0][1]['doContour'] = True
    my_cutsky = CutSky(maps=hp_map, low_mem=False)
    npt.assert_array_equal(my_cutsky.maps[0][1], hp_map_data)
    assert my_cutsky.maps[0][2]['doContour'] is True


def test_CutSky_cut_fits(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]

    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    result = my_cutsky.cut_fits([0, 0])
    assert len(result) == 1
    assert result[0]['legend'] == opt['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, np.ones((my_cutsky.npix, my_cutsky.npix)))


def test_CutSky_cut_fits_assert(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]

    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    with pytest.raises(AssertionError):
        result = my_cutsky.cut_fits([0, 0, 0])


def test_CutSky_cut_fits_selection(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]

    filename2 = filename.replace('.fits', '2.fits')
    opt2 = {'legend': 'tmpfile2', 'doContour': 'False'}
    import shutil
    shutil.copy(filename, filename2)

    hp_maps = [hp_map[0], (filename2, opt2)]
    my_cutsky = CutSky(maps=hp_maps, low_mem=True)
    result = my_cutsky.cut_fits([0, 0])
    assert len(result) == 2

    result = my_cutsky.cut_fits([0, 0], maps_selection=['tmpfile2'])
    assert len(result) == 1
    assert result[0]['legend'] == 'tmpfile2'

    result = my_cutsky.cut_fits([0, 0], maps_selection=[filename2])
    assert len(result) == 1
    assert result[0]['legend'] == 'tmpfile2'

    result = my_cutsky.cut_png([0, 0], maps_selection=[filename2])
    assert len(result) == 1
    assert result[0]['legend'] == 'tmpfile2'


def test_CutSky_cut_png(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]
    hp_map[0][1]['doContour'] = True
    # Will actually not produce a contour in this situation

    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    result = my_cutsky.cut_png([0, 0])
    assert len(result) == 1
    assert result[0]['legend'] == opt['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, np.ones((my_cutsky.npix, my_cutsky.npix)))
    assert result[0]['fits'].header['doContour'] is True
    # CHECK png .... ?

    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    result2 = my_cutsky.cut_fits([0, 0])
    result2 = my_cutsky.cut_png([0, 0])
    assert result[0]['legend'] == result2[0]['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, result2[0]['fits'].data.data)
    assert result[0]['png'] == result2[0]['png']


def test_CutSky_cut_phot(generate_hpmap):
    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]
    hp_map[0][1]['doContour'] = True

    my_cutsky = CutSky(maps=hp_map, low_mem=True)

    # insure assertion when no aperture is given
    result = my_cutsky.cut_phot([0, 0], apertures=None)
    assert result[0]['phot'] is None

    # aperture is a float
    result = my_cutsky.cut_phot([0, 0], apertures=1)
    assert len(result) == 1
    assert result[0]['legend'] == opt['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, np.ones((my_cutsky.npix, my_cutsky.npix)))
    assert result[0]['fits'].header['doContour'] is True

    # Photutils is changing API with v0.6
    aperture_result = result[0]['phot'][0]
    if 'aperture_sum' in aperture_result:
        npt.assert_almost_equal(aperture_result['aperture_sum'], np.pi)
    elif 'aperture_sum_0' in aperture_result:
        npt.assert_almost_equal(aperture_result['aperture_sum_0'], np.pi)

    # aperture is a list
    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    result2 = my_cutsky.cut_fits([0, 0])
    result2 = my_cutsky.cut_phot([0, 0], apertures=[1, 2])
    assert result2[0]['legend'] == opt['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, result2[0]['fits'].data.data)
    npt.assert_almost_equal(result2[0]['phot'][0][3], np.pi)
    npt.assert_almost_equal(result2[0]['phot'][0][4], np.pi * 2**2)

    # aperture is given in hp_map
    hp_map[0][1]['apertures'] = 1
    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    result3 = my_cutsky.cut_phot([0, 0], apertures=None)
    assert result3[0]['legend'] == opt['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, result3[0]['fits'].data.data)
    assert result3[0]['phot'][0][3] == result[0]['phot'][0][3]

    # aperture is given in hp_map as a list
    hp_map[0][1]['apertures'] = [1, 2]
    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    result3 = my_cutsky.cut_phot([0, 0], apertures=None)
    assert result3[0]['legend'] == opt['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, result3[0]['fits'].data.data)
    npt.assert_almost_equal(result3[0]['phot'][0][3], np.pi)
    npt.assert_almost_equal(result3[0]['phot'][0][4], np.pi * 2**2)


class TestCutSky:

    def test_to_new_maps(self):
        old_maps = {'legend': {'filename': 'full_filename_to_healpix_map.fits',
                               'doContour': True}}

        new_maps = to_new_maps(old_maps)
        assert len(new_maps) == 1
        assert isinstance(new_maps[0], tuple)
        assert new_maps[0][0] == old_maps['legend']['filename']
        assert new_maps[0][1] == {'legend': 'legend', 'doContour': True}

    def test_cutsky_exception(self):
        with pytest.raises(AssertionError):
            sub_map = cutsky()
        with pytest.raises(AssertionError):
            sub_map = cutsky(lonlat=[0, 0, 0])
        with pytest.raises(AssertionError):
            sub_map = cutsky(lonlat=[0, 0])

    def test_cutsky(self, generate_hpmap):

        hp_map, hp_map_data, hp_key = generate_hpmap
        filename, opt = hp_map[0]

        # Old interface
        old_hpmap = {opt['legend']: {'filename': filename, 'doContour': True}}

        result = cutsky([0, 0], old_hpmap)
        assert len(result) == 1
        assert result[0]['legend'] == opt['legend']
        npt.assert_array_equal(result[0]['fits'].data.data, np.ones((DEFAULT_NPIX, DEFAULT_NPIX)))
        assert result[0]['fits'].header['doContour'] is True
        assert result[0]['phot'] is None

        #New interface
        new_hpmap = [(filename, {'legend': opt['legend']})]
        result = cutsky([0, 0], new_hpmap)
        assert len(result) == 1
        assert result[0]['legend'] == opt['legend']
        npt.assert_array_equal(result[0]['fits'].data.data, np.ones((DEFAULT_NPIX, DEFAULT_NPIX)))
        assert result[0]['fits'].header['doContour'] is False
        assert result[0]['phot'] is None

    def test_cutsky_misheader(self, generate_mis_hpmap):

        hp_map, hp_map_data, hp_key = generate_mis_hpmap
        filename, opt = hp_map[0]

        new_hpmap = [(filename, {'legend': opt['legend']})]

        with pytest.raises(ValueError):
            result = cutsky([0, 0], new_hpmap)

        new_hpmap = [(filename, {'legend': opt['legend'], 'COORDSYS': 'C'})]
        result = cutsky([0, 0], new_hpmap)

        assert len(result) == 1
        assert result[0]['legend'] == opt['legend']
        npt.assert_array_equal(result[0]['fits'].data.data, np.ones((DEFAULT_NPIX, DEFAULT_NPIX)))
        assert result[0]['fits'].header['doContour'] is False
        assert result[0]['phot'] is None

    def test_main(self, generate_hpmap):

        hp_map, hp_map_data, hp_key = generate_hpmap
        filename, opt = hp_map[0]

        outdir = os.path.join(os.path.dirname(filename), 'output')
        png_file = os.path.join(outdir, opt['legend'] + '.png')
        fits_file = os.path.join(outdir, opt['legend'] + '.fits')
        xml_file = os.path.join(outdir, opt['legend'] + '.xml')

        # default -> --png
        args = "0.0 0.0" + \
               " --mapfilenames " + filename + \
               " --outdir " + outdir

        exit_code = main(args.split())
        assert not os.path.exists(fits_file)
        assert not os.path.exists(xml_file)
        os.remove(png_file)

        # --fist only
        args = "0.0 0.0" + \
               " --mapfilenames " + filename + \
               " --fits " + \
               " --outdir " + outdir

        exit_code = main(args.split())
        assert not os.path.exists(png_file)
        assert os.path.exists(fits_file)
        assert not os.path.exists(xml_file)
        os.remove(fits_file)

        # --fits & --votable
        args = "0.0 0.0" + \
               " --mapfilenames " + filename + \
               " --fits --votable 1" + \
               " --outdir " + outdir

        exit_code = main(args.split())
        assert not os.path.exists(png_file)
        assert os.path.exists(fits_file)
        assert os.path.exists(xml_file)
        os.remove(fits_file)

        # Test all
        args = "0.0 0.0" + \
               " --mapfilenames " + filename + \
               " --fits --png --votable 1 2" + \
               " --outdir " + outdir

        exit_code = main(args.split())
        assert os.path.exists(png_file)
        assert os.path.exists(fits_file)
        assert os.path.exists(xml_file)

        # Test clobber works
        args = "0.0 0.0" + \
               " --mapfilenames " + filename + \
               " --fits --png --votable 1" + \
               " --outdir " + outdir

        exit_code = main(args.split())
        assert os.path.exists(png_file)
        assert os.path.exists(fits_file)
        assert os.path.exists(xml_file)
