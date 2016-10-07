import logging
logger = logging.getLogger('django')

import pytest

from .. import parse_args, parse_config, combine_args
from .. import cutsky, CutSkySquare
from .. import DEFAULT_npix, DEFAULT_coordframe, DEFAULT_pixsize, DEFAULT_ctype

import numpy as np
import numpy.testing as npt
import healpy as hp
from .. import hp_is_nest, hp_celestial

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

        assert(args.lon == 0.0)
        assert(args.lat == 0.0)
        assert(args.coordframe == None)
        assert(args.ctype == None)
        assert(args.npix == None)
        assert(args.mapfilenames == None)
        assert(args.maps == None)
        assert(args.pixsize == None)
        assert(args.radius == None)
        assert(args.verbosity == logging.INFO)

    def test_parse_args_mapfilenames(self):
        args = parse_args('0.0 0.0 --mapfilenames blah1 blah2'.split())

        assert(args.mapfilenames == ['blah1', 'blah2'])
        assert(args.maps == [('blah1', {'legend': 'blah1'} ),
                             ('blah2', {'legend': 'blah2'} ) ]  )

    @pytest.mark.parametrize("verbosity, level",
                             [ ('', logging.INFO),
                               ('--verbose', logging.DEBUG),
                               ('--quiet', logging.ERROR)
                             ] )
    def test_parse_args_verbosity(self, verbosity, level):
        args = parse_args(('0.0 0.0 '+verbosity).split())
        assert(args.verbosity == level)

class TestParserConfig:
    def test_parser_config_found(self,tmpdir):
        # Insure empty file
        conffile = tmpdir.mkdir("conf").join("cutsky.cfg")
        with pytest.raises(FileNotFoundError):
            config = parse_config(str(conffile))

    @pytest.fixture(scope='session')
    def generate_default_conffile(self,tmpdir_factory):
        conffile = tmpdir_factory.mktemp("conf").join("cutsky.cfg")
        config = ConfigParser()
        config.add_section('cutsky')
        config.set('cutsky','npix', DEFAULT_npix)
        config.set('cutsky','pixsize', DEFAULT_pixsize)
        config.set('cutsky','coordframe', DEFAULT_coordframe)
        config.set('cutsky','ctype', DEFAULT_ctype)
        config.write(conffile.open(mode='w', ensure=True))

        return conffile, config

    def test_parser_config(self,generate_default_conffile):

        conffile, config = generate_default_conffile
        test_config = parse_config(str(conffile))

        assert(test_config.get('npix') == DEFAULT_npix)
        assert(test_config.get('pixsize') == DEFAULT_pixsize)
        assert(test_config.get('coordframe') == DEFAULT_coordframe)
        assert(test_config.get('ctype') == DEFAULT_ctype)
        assert(test_config.get('verbosity') == None)

    def test_parser_config_maps(self, generate_default_conffile):

        conffile, config = generate_default_conffile

        # Every other sections describe a map
        # 'map 1' should be present
        config.add_section('map 1')
        config.set('map 1', 'filename', 'filename1.fits')
        config.set('map 1', 'doCut', True)
        config.set('map 1', 'doContour', True)

        # 'map 2' should not be present
        config.add_section('map 2')
        config.set('map 2', 'filename', 'filename2.fits')
        config.set('map 2', 'doCut', False)

        # 'map 3' should not be present
        config.add_section('map 3')
        config.set('map 3', 'filename', 'filename2.fits')
        config.write(conffile.open(mode='w', ensure=True))

        test_config = parse_config(str(conffile))

        assert(test_config.get('maps') == [('filename1.fits',
                                            { 'legend': 'map 1',
                                              'doContour':True} ) ] )

    @pytest.mark.parametrize("verbosity, level",
                             [ ('verbose', logging.DEBUG),
                               ('debug', logging.DEBUG),
                               ('quiet', logging.ERROR),
                               ('error', logging.ERROR),
                               ('info', logging.INFO) ] )
    def test_parser_verbosity(self, generate_default_conffile, verbosity, level):
        conffile, config = generate_default_conffile
        config.set('cutsky', 'verbosity', verbosity)
        config.write(conffile.open(mode='w', ensure=True))
        test_config = parse_config(str(conffile))

        assert(test_config.get('verbosity') == level)

class TestCombineArgs:
    def test_combine_args_defaults(self):
        args = parse_args('0.0 0.0 '.split())
        (npix, pixsize, coordframe, ctype, maps) = combine_args(args, {})

        assert(npix == DEFAULT_npix)
        assert(coordframe == DEFAULT_coordframe)
        assert(pixsize == DEFAULT_pixsize)

    def test_combine_args_radius(self):
        args = parse_args('0.0 0.0 --pixsize 1 --radius 60'.split())
        (npix, pixsize, coordframe, ctype, maps) = combine_args(args, {})
        assert(pixsize == 1)
        assert(npix == 3600)

    @pytest.mark.parametrize("args, conf, level",
                             [('0.0 0.0 --verbose', {}, logging.DEBUG),
                              ('0.0 0.0 --quiet', {}, logging.ERROR),
                              ('0.0.0.0', {'verbosity': logging.DEBUG}, logging.DEBUG),
                              ('0.0.0.0 --quiet', {'verbosity': logging.DEBUG}, logging.ERROR)
                             ])
    def test_combine_args_verbosity(self, args, conf, level):
        args = parse_args('0.0 0.0 --verbose'.split())
        (npix, pixsize, coordframe, ctype, maps) = combine_args(args, {})
        assert(logger.level == logging.DEBUG)

def test_cutsky_exception():
    with pytest.raises(FileNotFoundError):
        sub_map = cutsky()


def test_CutSkySquare_init_exception():
    with pytest.raises(TypeError):
        cutsky = CutSkySquare()

    with pytest.raises(IOError):
        custky = CutSkySquare([('toto.fits', {})])

@pytest.fixture(scope='session')
def generate_hpmap(tmpdir_factory):

    """Generate an uniform healpix map"""

    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside))
    hp_header={'NSIDE': nside,
               'ORDERING': 'RING',
               'COORDSYS': 'C'}
    hp_key = "%s_%s_%s"%(hp_header['NSIDE'], hp_header['ORDERING'],  hp_celestial(hp_header).name)

    tmpfile = tmpdir_factory.mktemp("data").join("tmpfile.fits")

    hp.write_map(str(tmpfile), hp_map, nest = hp_is_nest(hp_header), extra_header = hp_header.items())
    return ([(str(tmpfile), {'legend': 'tmpfile'}) ], hp_map, hp_key)

def test_CutSkySquare_init(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]

    cutsky = CutSkySquare(hp_map, low_mem=True)
    assert(cutsky.npix == DEFAULT_npix)
    assert(cutsky.pixsize == DEFAULT_pixsize)
    assert(cutsky.ctype == DEFAULT_ctype)
    assert(cutsky.maps.keys() == [hp_key])
    assert(cutsky.maps[hp_key][0][0] == filename)
    assert(cutsky.maps[hp_key][0][1] == filename)
    assert(cutsky.maps[hp_key][0][2]['legend'] == opt['legend'])

    hp_map[0][1]['doContour'] = True
    cutsky = CutSkySquare(hp_map, low_mem=False)
    npt.assert_array_equal(cutsky.maps[hp_key][0][1], hp_map_data)
    assert(cutsky.maps[hp_key][0][2]['doContour'] == True)

def test_CutSkySquare_cutsky_fits(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]
    hp_map[0][1]['doContour'] = True

    cutsky = CutSkySquare(hp_map, low_mem=True)
    cut_fits = cutsky.cutsky_fits([0,0])
    assert(len(cut_fits) == 1)
    assert(cut_fits[0]['legend'] == opt['legend'])
    npt.assert_array_equal(cut_fits[0]['fits'].data.data, np.ones((cutsky.npix,cutsky.npix)))
    assert(cut_fits[0]['fits'].header['doContour'] == True)

def test_CutSkySquare_cutsky_png(tmpdir):
    # Test for png : create empy plots ???...
    pass
def test_CutSkySquare_cutsky_phot(tmpdir):
    # Test for photometry
    pass

def test_cutsky(generate_hpmap):

    hp_map, hp_map_data, hp_key = generate_hpmap
    filename, opt = hp_map[0]
    old_hpmap = {opt['legend']: {'filename': filename}}

    result = cutsky([0,0], maps=old_hpmap)
