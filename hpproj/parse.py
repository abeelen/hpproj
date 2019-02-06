#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

"""parse module, to be used by :func:`cutsky.main`, mainly here to reduce Cyclomatic complexcity"""

from __future__ import print_function, division

import os
import sys
import argparse
import logging
import json

from .wcs_helper import VALID_PROJ

try:  # pragma: py3
    from configparser import ConfigParser, ExtendedInterpolation
    PATTERN = None
except ImportError:  # pragma: py2
    import re
    from ConfigParser import ConfigParser
    PATTERN = re.compile(r"\$\{(.*?)\}")

try:  # pragma: py3
    FileNotFoundError
except NameError:  # pragma: py2
    FileNotFoundError = IOError

LOGGER = logging.getLogger('hpproj')

DEFAULT_NPIX = 256
DEFAULT_PIXSIZE = 1
DEFAULT_COORDFRAME = 'galactic'
DEFAULT_CTYPE = 'TAN'

SPECIAL_OPT = {'docontour': {'function': ConfigParser.getboolean, 'default': False}}


__all__ = ['parse_args', 'parse_config', 'combine_args_config', 'ini_main']


def parse_args(args):
    """Parse arguments from the command line"""

    parser = argparse.ArgumentParser(
        description="Reproject the spherical sky onto a plane.")
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

    input_map = parser.add_argument_group(
        'input maps', description="one of the two options must be present")

    input_map.add_argument('--mapfilenames', nargs='+', required=False,
                           help='absolute path to the healpix maps')
    input_map.add_argument('--conf', required=False,
                           help='absolute path to a config file')

    out = parser.add_argument_group('output')
    out.add_argument('--fits', action='store_true', help='output fits file')
    out.add_argument('--png', action='store_true',
                     help='output png file (Default: True if nothing else)')
    out.add_argument('--votable', nargs='+', type=float, help='list of aperture [arcmin] to make circular aperture photometry', metavar='aperture')
    out.add_argument('--outdir', required=False,
                     help='output directory (default:".")')

    general = parser.add_argument_group('general')
    verb = general.add_mutually_exclusive_group()
    verb.add_argument(
        '-v', '--verbose', action='store_true', help='verbose mode')
    verb.add_argument('-q', '--quiet', action='store_true', help='quiet mode')

    # Do the actual parsing
    parsed_args = parser.parse_args(args)

    # Put the list of filenames into the same structure as the config
    # file, we are loosing the docontour keyword but...
    if parsed_args.mapfilenames:
        parsed_args.maps = [(filename,
                             dict([('legend', os.path.splitext(os.path.basename(filename))[0]), ('docontour', False)]))
                            for filename in
                            parsed_args.mapfilenames]
    else:
        parsed_args.maps = None

    if parsed_args.verbose:
        parsed_args.verbosity = logging.DEBUG
    elif parsed_args.quiet:
        parsed_args.verbosity = logging.ERROR
    else:
        parsed_args.verbosity = None

    return parsed_args


def find_config(conffile=None):
    """Read the configuration file."""

    if PATTERN:  # pragma: py2
        config = ConfigParser()

    else:  # pragma: py3
        config = ConfigParser(interpolation=ExtendedInterpolation())

    # Look for cutsky.cfg at several locations
    conffiles = [os.path.join(directory, 'cutsky.cfg') for directory
                 in [os.curdir, os.path.join(os.path.expanduser("~"), '.config/cutsky')]]

    # If specifically ask for a config file, then put it at the
    # beginning ...
    if conffile:
        conffiles.insert(0, conffile)

    found = config.read(conffiles)

    if not found:
        raise FileNotFoundError

    return config


def parse_config_basic(config):
    """Parse basic options from the configuration file."""

    # Basic options, same as the option on the command line
    options = {}
    if config.has_option('cutsky', 'npix'):
        options['npix'] = config.getint('cutsky', 'npix')
    if config.has_option('cutsky', 'pixsize'):
        options['pixsize'] = config.getfloat('cutsky', 'pixsize')

    for key in ['coordframe', 'ctype', 'outdir']:
        if config.has_option('cutsky', key):
            options[key] = config.get('cutsky', key)

    return options


def parse_config_basic_output(config, options):
    """Parse basic output options from the configuration file."""

    for key in ['fits', 'png']:
        if config.has_option('cutsky', key) and config.get('cutsky', key):
            options[key] = True
    if config.has_option('cutsky', 'votable'):
        options['votable'] = json.loads(config.get('cutsky', 'votable'))

    return options


def parse_config_verbosity(config):
    """Parse the verbosity level from the config file."""

    allowed_levels = {'verbose': logging.DEBUG,
                      'debug': logging.DEBUG,
                      'quiet': logging.ERROR,
                      'error': logging.ERROR,
                      'info': logging.INFO}

    verbosity = None

    if config.has_option('cutsky', 'verbosity'):
        level = str(config.get('cutsky', 'verbosity')).lower()

        if level in allowed_levels.keys():
            verbosity = allowed_levels[level]
        elif level.isdigit():
            verbosity = int(level)

    return verbosity


def parse_filename(config, section):
    """Parse filename from a config file py2/py3 compatible"""

    filename = config.get(section, 'filename')
    if PATTERN:  # pragma: v2
        def rep_key(mesg):
            """re.sub-pattern to find an extendedInterpolation in py2"""
            section, key = re.split(':', mesg.group(1))
            return config.get(section, key)
        filename = PATTERN.sub(rep_key, filename)

    return filename


def parse_config_map_opt(config, section):
    """Parse map option in a configuration file."""

    opt = {'legend': section}

    for key in config.options(section):
        if key in ['filename', 'docut']:
            # parsed elsewhere
            pass
        elif key not in SPECIAL_OPT:
            opt[key.lower()] = config.get(section, key)

    for key in SPECIAL_OPT:
        if config.has_option(section, key):
            get_function = SPECIAL_OPT[key]['function']
            opt[key] = get_function(config,
                                    section, key)
        else:
            opt[key] = SPECIAL_OPT[key]['default']

    return opt


def parse_config_select_maps(config, sections):
    """Retrieve the section of the maps to project from a configuration file."""

    good_sections = []
    for key in sections:
        if not config.has_option(key, 'docut') or config.getboolean(key, 'docut'):
            good_sections.append(key)

    return good_sections


def parse_config_maps(config):
    """Parse the maps options from the configuration file."""

    # Map list, only get the one which will be projected
    # Also check if contours are requested
    maps_to_cut = []

    map_sections = [key for key in config.sections() if key != 'cutsky' and config.has_option(key, 'filename')]

    map_sections = parse_config_select_maps(config, map_sections)

    for section in map_sections:

        filename = parse_filename(config, section)

        opt = parse_config_map_opt(config, section)

        maps_to_cut.append((filename, opt))

    return maps_to_cut


def parse_config(conffile=None):
    """Parse options from a configuration file."""

    config = find_config(conffile)

    options = parse_config_basic(config)

    options = parse_config_basic_output(config, options)

    options['verbosity'] = parse_config_verbosity(config)
    options['maps'] = parse_config_maps(config)

    return options


def combine_args_config(args, config):
    """
    Combine the different sources of arguments (command line,
    configfile or default arguments

    Notes
    -----
    The `radius` arguments have priority and `npix` will be computed
    based on `pixelsize`
    """

    arguments = [('pixsize', DEFAULT_PIXSIZE),
                 ('coordframe', DEFAULT_COORDFRAME),
                 ('ctype', DEFAULT_CTYPE),
                 ('verbosity', logging.INFO),
                 ('maps', None),
                 ('fits', False),
                 ('png', False),
                 ('votable', None),
                 ('outdir', '.'), ]

    combined_args = vars(args)
    for key, default_value in arguments:
        combined_args[key] = combined_args[key] or config.get(key) or default_value

    LOGGER.setLevel(combined_args['verbosity'])

    # setting the radius will define the number of pixels
    if args.radius:
        args.npix = int(args.radius / (float(combined_args['pixsize']) / 60))

    combined_args['npix'] = args.npix or config.get('npix') or DEFAULT_NPIX

    if not (combined_args['fits'] or combined_args['png'] or (combined_args['votable'] is not None)):
        combined_args['png'] = True

    return combined_args


def ini_main(argv):
    """Initialize the main function"""

    if argv is None:  # pragma: no cover
        argv = sys.argv[1:]

    try:
        args = parse_args(argv)
    except SystemExit:  # pragma: no cover
        sys.exit()

    try:
        config = parse_config(args.conf)
    except FileNotFoundError:
        config = {}

    return combine_args_config(args, config)
