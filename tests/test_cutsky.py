#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

import os
import logging
logger = logging.getLogger('django')

import pytest

from hpproj import parse_args, parse_config, combine_args
from hpproj import CutSky, cutsky, main, to_new_maps
from hpproj import DEFAULT_NPIX, DEFAULT_COORDFRAME, DEFAULT_PIXSIZE, DEFAULT_CTYPE

import numpy as np
import numpy.testing as npt
import healpy as hp
from hpproj import hp_is_nest, hp_celestial

try: # pragma: py3
    from configparser import ConfigParser
except ImportError: # pragma: py2
    from ConfigParser import ConfigParser

try: # pragma: py3
    FileNotFoundError
except NameError: # pragma: py2
    FileNotFoundError = IOError

class TestParseArgs:
    def test_parse_args_empty(self):
        with pytest.raises(SystemExit):
            args = parse_args(' '.split())

    def test_parse_args_defaults(self):
        args = parse_args('0.0 0.0 '.split())

        assert args.lon == 0.0
        assert args.lat == 0.0
        assert args.coordframe is None
        assert args.ctype is None
        assert args.npix is None
        assert args.mapfilenames is None
        assert args.maps is None
        assert args.pixsize is None
        assert args.radius is None

        assert args.verbose is False
        assert args.quiet is False
        assert args.verbosity is None

        assert args.fits is False
        assert args.png is False
        assert args.votable is False

    def test_parse_args_mapfilenames(self):
        args = parse_args('0.0 0.0 --mapfilenames blah1 blah2'.split())

        assert args.mapfilenames == ['blah1', 'blah2']
        assert args.maps == [('blah1', {'legend': 'blah1'}),
                             ('blah2', {'legend': 'blah2'})]

    @pytest.mark.parametrize("verbosity, level",
                             [('', None),
                              ('--verbose', logging.DEBUG),
                              ('--quiet', logging.ERROR)
                             ])
    def test_parse_args_verbosity(self, verbosity, level):
        args = parse_args(('0.0 0.0 '+verbosity).split())
        assert args.verbosity == level

class TestParserConfig:
    def test_parser_config_found(self, tmpdir):
        # Insure empty file
        conffile = tmpdir.mkdir("conf").join("cutsky.cfg")
        with pytest.raises(FileNotFoundError):
            config = parse_config(str(conffile))

    @pytest.fixture(scope='session')
    def generate_default_conffile(self, tmpdir_factory):
        conffile = tmpdir_factory.mktemp("conf").join("cutsky.cfg")
        config = ConfigParser()
        config.add_section('cutsky')
        config.set('cutsky', 'npix', str(DEFAULT_NPIX))
        config.set('cutsky', 'pixsize', str(DEFAULT_PIXSIZE))
        config.set('cutsky', 'coordframe', DEFAULT_COORDFRAME)
        config.set('cutsky', 'ctype', DEFAULT_CTYPE)
        config.write(conffile.open(mode='w', ensure=True))

        return conffile, config

    def test_parser_config(self, generate_default_conffile):

        conffile, config = generate_default_conffile
        test_config = parse_config(str(conffile))

        assert test_config.get('npix') == DEFAULT_NPIX
        assert test_config.get('pixsize') == DEFAULT_PIXSIZE
        assert test_config.get('coordframe') == DEFAULT_COORDFRAME
        assert test_config.get('ctype') == DEFAULT_CTYPE
        assert test_config.get('verbosity') is None
        assert test_config.get('fits') is None
        assert test_config.get('png') is None
        assert test_config.get('votable') is None

    def test_parser_config_maps(self, generate_default_conffile):

        conffile, config = generate_default_conffile

        # Every other sections describe a map
        # 'map 1' should be present
        config.add_section('map 1')
        config.set('map 1', 'filename', 'filename1.fits')
        config.set('map 1', 'doCut', str(True))
        config.set('map 1', 'doContour', str(True))

        # 'map 2' should not be present
        config.add_section('map 2')
        config.set('map 2', 'filename', 'filename2.fits')
        config.set('map 2', 'doCut', str(False))

        # 'map 3' should not be present
        config.add_section('map 3')
        config.set('map 3', 'filename', 'filename3.fits')

        config.write(conffile.open(mode='w', ensure=True))
        test_config = parse_config(str(conffile))

        assert test_config.get('maps') == [('filename1.fits',
                                            {'legend': 'map 1',
                                             'doContour':True}),
                                           ('filename3.fits',
                                            {'legend': 'map 3'})
                                          ]

    def test_parser_config_maps_interpolation(self, generate_default_conffile):
        conffile, config = generate_default_conffile

        config.set('cutsky', 'mapdir', 'toto')
        config.add_section('map 4')
        config.set('map 4', 'filename', '${cutsky:mapdir}/filename4.fits')

        config.write(conffile.open(mode='w', ensure=True))
        test_config = parse_config(str(conffile))

        assert test_config.get('maps')[2] == ('toto/filename4.fits', {'legend': 'map 4'})

    @pytest.mark.parametrize("verbosity, level",
                             [('verbose', logging.DEBUG),
                              ('debug', logging.DEBUG),
                              ('quiet', logging.ERROR),
                              ('error', logging.ERROR),
                              ('info', logging.INFO),
                              (str(logging.INFO), logging.INFO),
                              ('aze', None)
                             ])
    def test_parser_verbosity(self, generate_default_conffile, verbosity, level):
        conffile, config = generate_default_conffile
        config.set('cutsky', 'verbosity', verbosity)
        config.write(conffile.open(mode='w', ensure=True))
        test_config = parse_config(str(conffile))

        assert test_config.get('verbosity') == level

    @pytest.mark.parametrize("key, value",
                             [('fits', True),
                              ('png', True),
                              ('votable', True),
                              ('outdir', 'toto')
                             ])
    def test_parser_output(self, generate_default_conffile, key, value):
        conffile, config = generate_default_conffile
        config.set('cutsky', key, str(value))
        config.write(conffile.open(mode='w', ensure=True))
        test_config = parse_config(str(conffile))

        assert test_config.get(key) == value


class TestCombineArgs:
    def test_combine_args_defaults(self):
        args = parse_args('0.0 0.0 '.split())
        (npix, pixsize, coordframe, ctype, maps, output) = combine_args(args, {})

        assert npix == DEFAULT_NPIX
        assert coordframe == DEFAULT_COORDFRAME
        assert pixsize == DEFAULT_PIXSIZE

    def test_combine_args_radius(self):
        args = parse_args('0.0 0.0 --pixsize 1 --radius 60'.split())
        (npix, pixsize, coordframe, ctype, maps, output) = combine_args(args, {})
        assert pixsize == 1
        assert npix == 3600

    @pytest.mark.parametrize("cmd, conf, level",
                             [('--verbose', {},
                               logging.DEBUG),
                              ('--quiet', {},
                               logging.ERROR),
                              ('', {'verbosity': logging.DEBUG},
                               logging.DEBUG),
                              ('--quiet', {'verbosity': logging.DEBUG},
                               logging.ERROR)
                             ])
    def test_combine_args_verbosity(self, cmd, conf, level):
        args = parse_args(('0.0 0.0 '+cmd).split())
        (npix, pixsize, coordframe, ctype, maps, output) = combine_args(args, conf)
        assert logger.level == level

    @pytest.mark.parametrize("cmd, conf, result",
                             [('--png', {},
                               {'fits': False, 'png': True,
                                'votable': False, 'outdir': '.'}),
                              ('--png --fits --votable', {},
                               {'fits': True, 'png': True,
                                'votable': True, 'outdir': '.'}),
                              ('--png --outdir toto', {'fits': True},
                               {'fits': True, 'png': True,
                                'votable': False, 'outdir': 'toto'}),
                             ])
    def test_combine_args_output(self, cmd, conf, result):
        args = parse_args(('0.0 0.0 '+cmd).split())
        (npix, pixsize, coordframe, ctype, maps, output) = combine_args(args, conf)
        assert output == result


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
    hp_key = "%s_%s_%s"%(hp_header['NSIDE'], hp_header['ORDERING'], hp_celestial(hp_header).name)

    tmpfile = tmpdir_factory.mktemp("data").join("tmpfile.fits")

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


def test_CutSky_cut_fits_selection(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]

    filename2 = filename.replace('.fits', '2.fits')
    opt2 = {'legend': 'tmpfile2'}
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
    result = my_cutsky.cut_phot([0, 0])
    assert len(result) == 1
    assert result[0]['legend'] == opt['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, np.ones((my_cutsky.npix, my_cutsky.npix)))
    assert result[0]['fits'].header['doContour'] is True
    assert result[0]['phot'][0]['aperture_sum'] == 0.0

    my_cutsky = CutSky(maps=hp_map, low_mem=True)
    result2 = my_cutsky.cut_fits([0, 0])
    result2 = my_cutsky.cut_phot([0, 0])
    assert result[0]['legend'] == result2[0]['legend']
    npt.assert_array_equal(result[0]['fits'].data.data, result2[0]['fits'].data.data)
    assert result[0]['phot'][0][0] == result2[0]['phot'][0][0]

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
        with pytest.raises(ValueError):
            sub_map = cutsky()
        with pytest.raises(FileNotFoundError):
            sub_map = cutsky(lonlat=[0, 0])

    def test_cutsky(self, generate_hpmap):

        hp_map, hp_map_data, hp_key = generate_hpmap
        filename, opt = hp_map[0]
        old_hpmap = {opt['legend']: {'filename': filename, 'doContour': True}}

        result = cutsky([0, 0], old_hpmap)
        assert len(result) == 1
        assert result[0]['legend'] == opt['legend']
        npt.assert_array_equal(result[0]['fits'].data.data, np.ones((DEFAULT_NPIX, DEFAULT_NPIX)))
        assert result[0]['fits'].header['doContour'] is True
        assert result[0]['phot'][0]['aperture_sum'] == 0.0

    def test_main(self, generate_hpmap):

        hp_map, hp_map_data, hp_key = generate_hpmap
        filename, opt = hp_map[0]

        outdir = os.path.join(os.path.dirname(filename), 'output')
        png_file = os.path.join(outdir, opt['legend']+'.png')
        fits_file = os.path.join(outdir, opt['legend']+'.fits')
        xml_file = os.path.join(outdir, opt['legend']+'.xml')

        # default -> --png
        args = "0.0 0.0"+ \
               " --mapfilenames "+ filename + \
               " --outdir "+ outdir

        exit_code = main(args.split())
        assert os.path.exists(png_file)
        assert not os.path.exists(fits_file)
        assert not os.path.exists(xml_file)
        os.remove(png_file)

        # --fist only
        args = "0.0 0.0"+ \
               " --mapfilenames "+ filename + \
               " --fits " + \
               " --outdir "+ outdir

        exit_code = main(args.split())
        assert not os.path.exists(png_file)
        assert os.path.exists(fits_file)
        assert not os.path.exists(xml_file)
        os.remove(fits_file)

        # --fits & --votable
        args = "0.0 0.0"+ \
               " --mapfilenames "+ filename + \
               " --fits --votable" + \
               " --outdir "+ outdir

        exit_code = main(args.split())
        assert not os.path.exists(png_file)
        assert os.path.exists(fits_file)
        assert os.path.exists(xml_file)
        os.remove(fits_file)

        # Test all
        args = "0.0 0.0"+ \
               " --mapfilenames "+ filename + \
               " --fits --png --votable" + \
               " --outdir "+ outdir

        exit_code = main(args.split())
        assert os.path.exists(png_file)
        assert os.path.exists(fits_file)
        assert os.path.exists(xml_file)

        # Test clobber works
        args = "0.0 0.0"+ \
               " --mapfilenames "+ filename + \
               " --fits --png --votable" + \
               " --outdir "+ outdir

        exit_code = main(args.split())
        assert os.path.exists(png_file)
        assert os.path.exists(fits_file)
        assert os.path.exists(xml_file)
