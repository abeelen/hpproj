#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

import pytest

from hpproj import equiv_celestial, build_ctype
from hpproj import build_wcs, build_wcs_cube, build_wcs_2pts
from hpproj import build_wcs_profile
from hpproj import get_lonlat

import numpy as np
from numpy import testing as npt

from astropy.coordinates import ICRS, Galactic, SkyCoord, Angle


class TesTEquivCelestical:

    def test_equiv_celestial_exception(self):
        frames = ['ecliptic', 'Ecliptic', 'e', 'E']
        for frame in frames:
            with pytest.raises(ValueError):
                result = equiv_celestial(frame)

    def test_equiv_celestial(self):

        frames = [('galactic', Galactic()),
                  ('g', Galactic()),
                  ('celestial2000', ICRS()),
                  ('equatorial', ICRS()),
                  ('eq', ICRS()),
                  ('c', ICRS()),
                  ('q', ICRS()),
                  ('fk5', ICRS())]

        for frame, result in frames:
            assert result.is_equivalent_frame(equiv_celestial(frame))


class TestBuildCtype:

    @pytest.mark.parametrize("coordsys, proj_type",
                             [('unknown', 'TAN'),
                              ('G', 'BLA')])
    def test_build_ctype_exception(self, coordsys, proj_type):
        with pytest.raises(ValueError):
            ctype = build_ctype(coordsys, proj_type)

    @pytest.mark.parametrize("test, result",
                             [(('Galactic', 'TAN'), ['GLON-TAN', 'GLAT-TAN']),
                              (('G', 'tan'), ['GLON-TAN', 'GLAT-TAN']),
                              (('EQUATORIAL', 'SiN'), ['RA---SIN', 'DEC--SIN']),
                              ])
    def test_build_ctype(self, test, result):
        coordsys, proj_type = test
        ctype = build_ctype(coordsys, proj_type)
        assert ctype == result


class TestGetLonLat:

    @pytest.mark.parametrize("SkyCoord, proj_type, lonlat",
                             [(SkyCoord(0, 0, unit="deg", frame="icrs"), 'equatorial', (0.0, 0.0)),
                              (SkyCoord(0, 0, unit="deg", frame="icrs"), 'galactic', (96.33728336969006, -60.188551946914465))
                              ])
    def test_getlonlat(self, SkyCoord, proj_type, lonlat):
        result = get_lonlat(SkyCoord, proj_type)
        npt.assert_allclose(lonlat, result)


class TestBuildWCS:

    def test_build_wcs_exception(self):

        params = [('EQ', 'BLA'), ('E', 'TAN')]

        for proj_sys, proj_type in params:
            with pytest.raises(ValueError):
                build_wcs(SkyCoord(0, 0, unit='deg'),
                          proj_sys=proj_sys, proj_type=proj_type)

    def test_build_wcs(self):

        # TODO: Parametrize

        coord, pixsize, shape_out = SkyCoord(0, 0, unit='deg'), 1, [512, 1024]
        wcs = build_wcs(coord, pixsize, shape_out)

        assert wcs.wcs.naxis == 2
        npt.assert_array_equal(wcs.wcs.crval, [0, 0])
        npt.assert_array_equal(wcs.wcs.cdelt, [-1, 1])
        npt.assert_array_equal(wcs.wcs.crpix, [256.5, 512.5])
        npt.assert_array_equal(wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN'])

        wcs = build_wcs(coord, pixsize, shape_out, proj_sys='G')

        assert wcs.wcs.naxis == 2
        npt.assert_array_equal(
            wcs.wcs.crval, [coord.galactic.l.deg, coord.galactic.b.deg])
        npt.assert_array_equal(wcs.wcs.ctype, ['GLON-TAN', 'GLAT-TAN'])

        coord = coord.galactic
        wcs = build_wcs(coord, pixsize, shape_out, proj_sys='EQ')

        assert wcs.wcs.naxis == 2
        # Unfortunatly astropy coordinate transformation are that precise
        npt.assert_allclose(wcs.wcs.crval, [0, 0], atol=1e-14)
        npt.assert_array_equal(wcs.wcs.cdelt, [-1, 1])
        npt.assert_array_equal(wcs.wcs.crpix, [256.5, 512.5])
        npt.assert_array_equal(wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN'])

    def test_build_wcs_cube_exception(self):

        params = [('EQ', 'BLA'), ('E', 'TAN')]

        for proj_sys, proj_type in params:
            with pytest.raises(ValueError):
                build_wcs_cube(
                    SkyCoord(0, 0, unit='deg'), 0, proj_sys=proj_sys, proj_type=proj_type)

    def test_build_wcs_cube(self):

        coord, index, pixsize, shape_out = SkyCoord(
            0, 0, unit='deg'), 1, 1, [512, 1024]

        wcs = build_wcs_cube(coord, index, pixsize, shape_out)

        assert wcs.wcs.naxis == 3
        npt.assert_array_equal(wcs.wcs.crval, [0, 0, 1])
        npt.assert_array_equal(wcs.wcs.cdelt, [-1, 1, 1])
        npt.assert_array_equal(wcs.wcs.crpix, [256.5, 512.5, 1])
        npt.assert_array_equal(
            wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN', 'INDEX'])

        wcs = build_wcs_cube(
            coord, index, pixsize, shape_out, proj_sys='G')

        assert wcs.wcs.naxis == 3
        npt.assert_array_equal(
            wcs.wcs.crval, [coord.galactic.l.deg, coord.galactic.b.deg, 1])
        npt.assert_array_equal(wcs.wcs.cdelt, [-1, 1, 1])
        npt.assert_array_equal(wcs.wcs.crpix, [256.5, 512.5, 1])
        npt.assert_array_equal(
            wcs.wcs.ctype, ['GLON-TAN', 'GLAT-TAN', 'INDEX'])

        coord = coord.galactic
        wcs = build_wcs_cube(coord, index, pixsize, shape_out, proj_sys='EQ')

        assert wcs.wcs.naxis == 3
        # Unfortunatly astropy coordinate transformation are that precise
        npt.assert_allclose(wcs.wcs.crval, [0, 0, 1], atol=1e-14)
        npt.assert_array_equal(wcs.wcs.cdelt, [-1, 1, 1])
        npt.assert_array_equal(wcs.wcs.crpix, [256.5, 512.5, 1])
        npt.assert_array_equal(
            wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN', 'INDEX'])

    # TODO: Test decorator
    def test_decorator_lonlat(self):
        lon, lat = 0, 0
        coord, pixsize, shape_out = SkyCoord(
            lon, lat, unit='deg'), 1, [512, 1024]
        wcs = build_wcs(coord, pixsize, shape_out)

        w_lonlat = build_wcs(lon, lat)
        # Hum... something is wrong.... dont know why...
        pass


def test_build_wcs_2pts_exception():

    params = [('EQ', 'BLA'), ('E', 'TAN')]

    coords = [SkyCoord(0, 0, unit='deg'),
              SkyCoord(1, 0, unit='deg')]

    for proj_sys, proj_type in params:
        with pytest.raises(ValueError):
            build_wcs_2pts(coords, 1, proj_sys=proj_sys, proj_type=proj_type)


def test_build_wcs_2pts():

    coords = [SkyCoord(0, 0, unit='deg'),
              SkyCoord(1, 0, unit='deg')]

    coords_angle = (coords[0].position_angle(
        coords[1]) + Angle(90, unit='deg')).wrap_at('180d').rad

    rot_matrix = [[np.cos(coords_angle), np.sin(-coords_angle)],
                  [np.sin(coords_angle), np.cos(coords_angle)]]

    pixsize, shape_out = 0.1, [128, 100]
    wcs = build_wcs_2pts(coords, pixsize=pixsize, shape_out=shape_out)

    relative_pos = [0.5 - (
        coords[1].separation(coords[0]).deg / pixsize) / shape_out[1] / 2,
        0.5 + (coords[1].separation(coords[0]).deg / pixsize) / shape_out[1] / 2]

    assert wcs.wcs.naxis == 2
    npt.assert_array_equal(wcs.wcs.crval, [0, 0])
    npt.assert_array_equal(wcs.wcs.cdelt, [-1 * pixsize, pixsize])
    npt.assert_array_equal(
        wcs.wcs.crpix, [relative_pos[0] * shape_out[1], shape_out[0] / 2])
    npt.assert_array_equal(wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN'])
    npt.assert_array_equal(wcs.wcs.pc, rot_matrix)

    shape_out, relative_pos = [128, 100], (2. / 5, 3. / 5)
    wcs = build_wcs_2pts(
        coords, shape_out=shape_out, relative_pos=relative_pos)

    pixsize = coords[0].separation(coords[1]).deg / (
        np.sum(np.array(relative_pos) * np.array([-1, 1])) * shape_out[1])

    assert wcs.wcs.naxis == 2
    npt.assert_array_equal(wcs.wcs.crval, [0, 0])
    npt.assert_allclose(wcs.wcs.cdelt, [-1 * pixsize, pixsize])
    npt.assert_array_equal(
        wcs.wcs.crpix, [shape_out[1] * relative_pos[0], shape_out[0] / 2])
    npt.assert_array_equal(wcs.wcs.ctype, ['RA---TAN', 'DEC--TAN'])

    wcs = build_wcs_2pts(coords, shape_out=shape_out, relative_pos=relative_pos, proj_sys='GALACTIC')
    npt.assert_array_equal(wcs.wcs.ctype, ['GLON-TAN', 'GLAT-TAN'])


def test_build_wcs_profile():

    pixsize = 0.1
    wcs = build_wcs_profile(pixsize)

    npt.assert_equal(wcs.wcs.crval, [0])
    npt.assert_equal(wcs.wcs.crpix, [0.5])
    npt.assert_equal(wcs.wcs.cdelt, [pixsize])
    assert wcs.wcs.ctype[0] == "RADIUS"
