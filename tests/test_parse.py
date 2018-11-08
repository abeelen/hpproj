#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

import logging

import pytest

from hpproj import DEFAULT_NPIX, DEFAULT_COORDFRAME, DEFAULT_PIXSIZE, DEFAULT_CTYPE
from hpproj import parse

# .parse import parse.parse_args, parse.parse_config, parse.combine_args_config_config


try:  # pragma: py3
    from configparser import ConfigParser
except ImportError:  # pragma: py2
    from ConfigParser import ConfigParser

try:  # pragma: py3
    FileNotFoundError
except NameError:  # pragma: py2
    FileNotFoundError = IOError

logger = logging.getLogger('django')


class TestParseArgs:
    def test_parse_args_empty(self):
        with pytest.raises(SystemExit):
            args = parse.parse_args(' '.split())

    def test_parse_args_defaults(self):
        args = parse.parse_args('0.0 0.0 '.split())

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
        assert args.votable is None

    def test_parse_args_mapfilenames(self):
        args = parse.parse_args('0.0 0.0 --mapfilenames blah1 blah2'.split())

        assert args.mapfilenames == ['blah1', 'blah2']
        assert args.maps == [('blah1', {'legend': 'blah1', 'doContour': False}),
                             ('blah2', {'legend': 'blah2', 'doContour': False})]

    @pytest.mark.parametrize("verbosity, level",
                             [('', None),
                              ('--verbose', logging.DEBUG),
                              ('--quiet', logging.ERROR)])
    def test_parse_args_verbosity(self, verbosity, level):
        args = parse.parse_args(('0.0 0.0 ' + verbosity).split())
        assert args.verbosity == level


class TestParserConfig:
    def test_parser_config_found(self, tmpdir):
        # Insure empty file
        conffile = tmpdir.mkdir("conf").join("cutsky.cfg")
        with pytest.raises(FileNotFoundError):
            config = parse.parse_config(str(conffile))

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
        test_config = parse.parse_config(str(conffile))

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
        test_config = parse.parse_config(str(conffile))

        assert test_config.get('maps') == [('filename1.fits',
                                            {'legend': 'map 1',
                                             'doContour': True}),
                                           ('filename3.fits',
                                            {'legend': 'map 3',
                                             'doContour': False})]

    def test_parser_config_maps_interpolation(self, generate_default_conffile):
        conffile, config = generate_default_conffile

        config.set('cutsky', 'mapdir', 'toto')
        config.add_section('map 4')
        config.set('map 4', 'filename', '${cutsky:mapdir}/filename4.fits')

        config.write(conffile.open(mode='w', ensure=True))
        test_config = parse.parse_config(str(conffile))

        assert test_config.get('maps')[2] == ('toto/filename4.fits', {'legend': 'map 4', 'doContour': False})

    @pytest.mark.parametrize("verbosity, level",
                             [('verbose', logging.DEBUG),
                              ('debug', logging.DEBUG),
                              ('quiet', logging.ERROR),
                              ('error', logging.ERROR),
                              ('info', logging.INFO),
                              (str(logging.INFO), logging.INFO),
                              ('aze', None)])
    def test_parser_verbosity(self, generate_default_conffile, verbosity, level):
        conffile, config = generate_default_conffile
        config.set('cutsky', 'verbosity', verbosity)
        config.write(conffile.open(mode='w', ensure=True))
        test_config = parse.parse_config(str(conffile))

        assert test_config.get('verbosity') == level

    @pytest.mark.parametrize("key, value",
                             [('fits', True),
                              ('png', True),
                              ('votable', 1),
                              ('outdir', 'toto')])
    def test_parser_output(self, generate_default_conffile, key, value):
        conffile, config = generate_default_conffile
        config.set('cutsky', key, str(value))
        config.write(conffile.open(mode='w', ensure=True))
        test_config = parse.parse_config(str(conffile))

        assert test_config.get(key) == value


class TestCombineArgs:
    def test_combine_args_config_defaults(self):
        args = parse.parse_args('0.0 0.0 '.split())
        result_args = parse.combine_args_config(args, {})

        assert result_args['lon'] == 0.0
        assert result_args['lat'] == 0.0
        assert result_args['npix'] == DEFAULT_NPIX
        assert result_args['coordframe'] == DEFAULT_COORDFRAME
        assert result_args['pixsize'] == DEFAULT_PIXSIZE

    def test_combine_args_config_radius(self):
        args = parse.parse_args('0.0 0.0 --pixsize 1 --radius 60'.split())
        result_args = parse.combine_args_config(args, {})
        assert result_args['pixsize'] == 1
        assert result_args['npix'] == 3600

    @pytest.mark.parametrize("cmd, conf, level",
                             [('--verbose', {},
                               logging.DEBUG),
                              ('--quiet', {},
                               logging.ERROR),
                              ('', {'verbosity': logging.DEBUG},
                               logging.DEBUG),
                              ('--quiet', {'verbosity': logging.DEBUG},
                               logging.ERROR)])
    def test_combine_args_config_verbosity(self, cmd, conf, level):
        args = parse.parse_args(('0.0 0.0 ' + cmd).split())
        result_args = parse.combine_args_config(args, conf)
        assert logger.level == level

    @pytest.mark.parametrize("cmd, conf, result",
                             [('--png', {}, {'fits': False, 'png': True, 'votable': None, 'outdir': '.'}),
                              ('--png --fits --votable 1', {}, {'fits': True, 'png': True, 'votable': [1.0], 'outdir': '.'}),
                              ('--png --fits --votable 1 2', {}, {'fits': True, 'png': True, 'votable': [1.0, 2.0], 'outdir': '.'}),
                              ('--png --outdir toto', {'fits': True}, {'fits': True, 'png': True, 'votable': None, 'outdir': 'toto'}),
                              ('', {}, {'fits': False, 'png': True, 'votable': None, 'outdir': '.'}), ]
                             )
    def test_combine_args_config_output(self, cmd, conf, result):
        args = parse.parse_args(('0.0 0.0 ' + cmd).split())
        result_args = parse.combine_args_config(args, conf)
        assert result_args['fits'] == result['fits']
        assert result_args['png'] == result['png']
        assert result_args['votable'] == result['votable']
        assert result_args['outdir'] == result['outdir']
