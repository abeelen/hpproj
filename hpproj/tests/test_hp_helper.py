#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

from .. import equiv_celestial, hp_celestial, hp_is_nest, build_ctype
from .. import build_WCS, build_WCS_cube, build_WCS_2pts
from .. import build_WCS_lonlat
from .. import hp_to_wcs, hp_to_wcs_ipx
from .. import hp_project, gen_hpmap, hpmap_key

import pytest

import numpy as np
from numpy import testing as npt
import healpy as hp
from astropy.coordinates import ICRS, Galactic, SkyCoord, Angle
from astropy.io import fits

class TestHPCelestical:
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

        for frame, result  in frames:
            assert(result.is_equivalent_frame(equiv_celestial(frame)))

    @pytest.mark.parametrize("hp_header", [{},
                     {'COORDSYS': 'ecliptic'},
                     {'COORDSYS': 'ECLIPTIC'},
                     {'COORDSYS': 'e'},
                     {'COORDSYS': 'E'}, ])
    def test_hp_celestial_exception(self, hp_header):
        with pytest.raises(ValueError):
            frame = hp_celestial(hp_header)

    @pytest.mark.parametrize("hp_header,result",
                             [ ({'COORDSYS': 'G'}, Galactic()),
                               ({'COORDSYS': 'Galactic'}, Galactic()),
                               ({'COORDSYS': 'Equatorial'}, ICRS()),
                               ({'COORDSYS': 'EQ'}, ICRS()),
                               ({'COORDSYS': 'celestial2000'}, ICRS()),
                             ])
    def test_hp_celestial(self, hp_header, result):
        frame = hp_celestial(hp_header)
        assert(frame.is_equivalent_frame(result))

class TestHPNest:
    def test_hp_is_nest_exception(self):
        hp_headers = [{}, {'ORDERING': 'Unknown'}]

        for hp_header in hp_headers:
            with pytest.raises(ValueError):
                is_nest = hp_is_nest(hp_header)

    @pytest.mark.parametrize("hp_header, result",
                             [ ({'ORDERING': 'nested'}, True),
                               ({'ORDERING': 'NESTED'}, True),
                               ({'ORDERING': 'ring'}, False),
                             ])
    def test_hp_is_nest(self, hp_header, result):
        is_nest = hp_is_nest(hp_header)
        assert(is_nest == result)

class TestBuildCtype:
    @pytest.mark.parametrize("coordsys, proj_type",
                             [('unknown', 'TAN'),
                              ('G', 'BLA')])
    def test_build_ctype_exception(self, coordsys, proj_type):
        with pytest.raises(ValueError):
            ctype = build_ctype(coordsys, proj_type)

    @pytest.mark.parametrize("test, result",
                             [ ( ('Galactic', 'TAN'), ['GLON-TAN', 'GLAT-TAN'] ),
                               ( ('G', 'tan'), ['GLON-TAN', 'GLAT-TAN'] ),
                               ( ('EQUATORIAL', 'SiN'), ['RA---SIN', 'DEC--SIN'] ),
                             ])
    def test_build_ctype(self, test, result):
        coordsys, proj_type = test
        ctype = build_ctype(coordsys, proj_type)
        assert(ctype == result)

class TestBuildWCS:
    def test_build_WCS_exception(self):

        params = [ ('EQ', 'BLA'), ('E','TAN') ]

        for proj_sys, proj_type in params:
            with pytest.raises(ValueError):
                build_WCS(SkyCoord(0,0, unit='deg'),proj_sys=proj_sys, proj_type=proj_type)

    def test_build_WCS(self):

        #TODO: Parametrize

        coord, pixsize, shape_out = SkyCoord(0,0,unit='deg'), 1, [512, 1024]
        w = build_WCS(coord, pixsize, shape_out)

        assert(w.wcs.naxis == 2)
        npt.assert_array_equal(w.wcs.crval, [0,0])
        npt.assert_array_equal(w.wcs.cdelt, [-1,1])
        npt.assert_array_equal(w.wcs.crpix, [256,512])
        npt.assert_array_equal(w.wcs.ctype, ['RA---TAN', 'DEC--TAN'])


        w = build_WCS(coord, pixsize, shape_out, npix=512, proj_sys='G')

        assert(w.wcs.naxis == 2)
        npt.assert_array_equal(w.wcs.crval, [coord.galactic.l.deg,coord.galactic.b.deg])
        npt.assert_array_equal(w.wcs.cdelt, [-1,1])
        npt.assert_array_equal(w.wcs.crpix, [256,256])
        npt.assert_array_equal(w.wcs.ctype, ['GLON-TAN', 'GLAT-TAN'])

        coord = coord.galactic
        w = build_WCS(coord, pixsize, shape_out, proj_sys='EQ')

        assert(w.wcs.naxis == 2)
        # Unfortunatly astropy coordinate transformation are that precise
        npt.assert_allclose(w.wcs.crval, [0,0], atol=5e-15)
        npt.assert_array_equal(w.wcs.cdelt, [-1,1])
        npt.assert_array_equal(w.wcs.crpix, [256,512])
        npt.assert_array_equal(w.wcs.ctype, ['RA---TAN', 'DEC--TAN'])

    def test_build_WCS_cube_exception(self):

        params = [ ('EQ', 'BLA'), ('E','TAN') ]

        for proj_sys, proj_type in params:
            with pytest.raises(ValueError):
                build_WCS_cube(SkyCoord(0,0, unit='deg'),0, proj_sys=proj_sys, proj_type=proj_type)

    def test_build_WCS_cube(self):

        coord, index, pixsize, shape_out = SkyCoord(0,0,unit='deg'), 1, 1, [512, 1024]

        w = build_WCS_cube(coord, index, pixsize, shape_out)

        assert(w.wcs.naxis == 3)
        npt.assert_array_equal(w.wcs.crval, [0,0, 1])
        npt.assert_array_equal(w.wcs.cdelt, [-1,1,1])
        npt.assert_array_equal(w.wcs.crpix, [256,512, 1])
        npt.assert_array_equal(w.wcs.ctype, ['RA---TAN', 'DEC--TAN', 'INDEX'])

        w = build_WCS_cube(coord, index, pixsize, shape_out, npix=512, proj_sys='G')

        assert(w.wcs.naxis == 3)
        npt.assert_array_equal(w.wcs.crval, [coord.galactic.l.deg,coord.galactic.b.deg, 1])
        npt.assert_array_equal(w.wcs.cdelt, [-1,1,1])
        npt.assert_array_equal(w.wcs.crpix, [256,256, 1])
        npt.assert_array_equal(w.wcs.ctype, ['GLON-TAN', 'GLAT-TAN', 'INDEX'])

        coord = coord.galactic
        w = build_WCS_cube(coord, index, pixsize, shape_out, proj_sys='EQ')

        assert(w.wcs.naxis == 3)
        # Unfortunatly astropy coordinate transformation are that precise
        npt.assert_allclose(w.wcs.crval, [0,0, 1], atol=5e-15)
        npt.assert_array_equal(w.wcs.cdelt, [-1,1,1])
        npt.assert_array_equal(w.wcs.crpix, [256,512, 1])
        npt.assert_array_equal(w.wcs.ctype, ['RA---TAN', 'DEC--TAN', 'INDEX'])

    def test_decorator_lonlat(self):
        lon, lat = 0, 0
        coord, pixsize, shape_out = SkyCoord(lon,lat,unit='deg'), 1, [512, 1024]
        w = build_WCS(coord, pixsize, shape_out)

        w_lonlat = build_WCS_lonlat(lon, lat)
        # Hum... something is wrong.... dont know why...
        pass


def test_build_WCS_2pts_exception():

    params = [ ('EQ', 'BLA'), ('E','TAN') ]

    coords = [SkyCoord(0,0, unit='deg'),
              SkyCoord(1,0, unit='deg')]

    for proj_sys, proj_type in params:
        with pytest.raises(ValueError):
            build_WCS_2pts(coords,1, proj_sys=proj_sys, proj_type=proj_type)

def test_build_WCS_2pts():

    coords = [SkyCoord(0,0, unit='deg'),
              SkyCoord(1,0, unit='deg')]

    coords_angle = (coords[0].position_angle(coords[1])+Angle(90, unit='deg')).wrap_at('180d').rad

    rot_matrix = [ [np.cos(coords_angle) , np.sin(-coords_angle)],
                 [np.sin(coords_angle) , np.cos(coords_angle)] ]


    pixsize, shape_out  = 0.1, [128,100]
    w = build_WCS_2pts(coords, pixsize=pixsize, shape_out=shape_out)

    relative_pos = [ 0.5-(coords[1].separation(coords[0]).deg/pixsize)/shape_out[1]/2,
                     0.5+(coords[1].separation(coords[0]).deg/pixsize)/shape_out[1]/2  ]

    assert(w.wcs.naxis == 2)
    npt.assert_array_equal(w.wcs.crval, [0,0])
    npt.assert_array_equal(w.wcs.cdelt, [-1*pixsize,pixsize])
    npt.assert_array_equal(w.wcs.crpix, [relative_pos[0]*shape_out[1], shape_out[0] / 2] )
    npt.assert_array_equal(w.wcs.ctype, ['RA---TAN', 'DEC--TAN'])
    npt.assert_array_equal(w.wcs.pc, rot_matrix)

    shape_out, relative_pos = [128,100], (2./5, 3./5)
    w = build_WCS_2pts(coords, shape_out=shape_out, relative_pos=relative_pos)

    pixsize = coords[0].separation(coords[1]).deg / (np.sum(np.array(relative_pos)*np.array([-1,1])) * shape_out[1])

    assert(w.wcs.naxis == 2)
    npt.assert_array_equal(w.wcs.crval, [0,0])
    npt.assert_allclose(w.wcs.cdelt, [-1*pixsize,pixsize])
    npt.assert_array_equal(w.wcs.crpix, [shape_out[1]*relative_pos[0], shape_out[0]/2])
    npt.assert_array_equal(w.wcs.ctype, ['RA---TAN', 'DEC--TAN'])

    w = build_WCS_2pts(coords, npix = shape_out[0], relative_pos=relative_pos, proj_sys='GALACTIC')
    npt.assert_array_equal(w.wcs.crpix, [shape_out[0]*relative_pos[0], shape_out[0]/2])
    npt.assert_array_equal(w.wcs.ctype, ['GLON-TAN', 'GLAT-TAN'])


def test_hp_to_wcs_exception():

    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside))
    hp_header={'NSIDE': nside,
               'ORDERING': 'RING',
               'COORDSYS': 'C'}

    coord, pixsize, shape_out = SkyCoord(0,0,unit='deg'), 1, [512, 512]
    w = build_WCS(coord, pixsize, shape_out)

    # Test order > 1
    with pytest.raises(ValueError):
        sub_map = hp_to_wcs(hp_map, hp_header, w, shape_out=shape_out, order=2)

def test_hp_to_wcs():
    # hp_to_wcs(hp_map, hp_header, w, shape_out=DEFAULT_shape_out, npix=None, order=0):

    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside))
    hp_header={'NSIDE': nside,
               'ORDERING': 'RING',
               'COORDSYS': 'C'}

    coord, pixsize, shape_out = SkyCoord(0,0,unit='deg'), np.degrees(hp.nside2resol(nside)), [512, 512]
    w = build_WCS(coord, pixsize, shape_out)

    # Test order = 0
    sub_map = hp_to_wcs(hp_map, hp_header, w, shape_out=shape_out, order=0)
    assert(sub_map.shape == tuple(shape_out))
    npt.assert_array_equal(sub_map, 1)

    # Test npix option
    sub_map = hp_to_wcs(hp_map, hp_header, w, npix=512, order=0)
    assert(sub_map.shape == tuple(shape_out))
    npt.assert_array_equal(sub_map, 1)

    # Test order = 1
    sub_map = hp_to_wcs(hp_map, hp_header, w, shape_out=shape_out, order=1)
    npt.assert_allclose(sub_map, 1, rtol=1e-15) # hp.get_interp_val precision

    # Test specific pixel Better use an odd number for this, because
    # build_WCS put the reference at the center of the image, which in
    # case of even number leaves it between 4 pixels and hp.ang2pix
    shape_out = [3,3]

    w = build_WCS(coord, pixsize, shape_out)

    lon, lat = coord.ra.deg,coord.dec.deg
    phi, theta = np.radians(lon), np.radians(90-lat)
    ipix = hp.ang2pix(nside, theta, phi, nest=hp_is_nest(hp_header))
    hp_map[ipix] = 0
    sub_map = hp_to_wcs(hp_map, hp_header, w, shape_out=shape_out, order=0)
    i_x, i_y = w.all_world2pix(lon,lat,0)
    assert(sub_map[int(np.ceil(i_y)),int(np.ceil(i_x))] == 0)

    # Test different frame
    w = build_WCS(coord, pixsize, shape_out, proj_sys="G")
    sub_map = hp_to_wcs(hp_map, hp_header, w, shape_out=shape_out)
    lon, lat = coord.galactic.l.deg,coord.galactic.b.deg
    i_x, i_y = w.all_world2pix(lon,lat,0)
    assert(sub_map[int(np.ceil(i_y)),int(np.ceil(i_x))] == 0)


def test_hp_to_wcs_ipx():

    nside = 2**6
    hp_header={'NSIDE': nside,
               'ORDERING': 'RING',
               'COORDSYS': 'C'}

    coord, pixsize, shape_out = SkyCoord(0,0,unit='deg'), 0.1, [1, 1]
    w = build_WCS(coord, pixsize, shape_out)

    # Basic test
    sub_mask, sub_ipx = hp_to_wcs_ipx(hp_header, w, shape_out=shape_out)
    lon, lat = coord.ra.deg,coord.dec.deg
    phi, theta = np.radians(lon), np.radians(90-lat)
    ipix = hp.ang2pix(nside, theta, phi, nest=hp_is_nest(hp_header))

    assert(sub_mask.all() == True)
    npt.assert_array_equal(sub_ipx, ipix)

    # Test different frame
    w = build_WCS(coord, pixsize, npix=1, proj_sys="G")
    sub_mask, sub_ipx = hp_to_wcs_ipx(hp_header, w, npix=1, shape_out=shape_out)
    assert(sub_mask.all() == True)
    npt.assert_array_equal(sub_ipx, ipix)


def test_hp_project():
    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside))
    hp_header={'NSIDE': nside,
               'ORDERING': 'RING',
               'COORDSYS': 'C'}

    coord, pixsize, npix = SkyCoord(0,0,unit='deg'), np.degrees(hp.nside2resol(nside)), 512
    w = build_WCS(coord, pixsize, npix=npix)

    # Test basic
    sub_map = hp_project(hp_map, hp_header, coord, pixsize, npix)
    assert(type(sub_map) is np.ndarray)
    assert(sub_map.shape == (npix, npix))
    npt.assert_array_equal(sub_map, 1)

    # Test HDUs
    sub_map = hp_project(hp_map, hp_header, coord, pixsize, npix, hdu=True)
    assert(type(sub_map) is fits.hdu.image.PrimaryHDU)


def test_gen_hpmap():

    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside))
    hp_header={'NSIDE': nside,
               'ORDERING': 'RING',
               'COORDSYS': 'C'}

    maps = [ ('map'+str(i), hp_map*i, hp_header) for i in range(3)]

    for i, (name, hp_map, hp_header) in enumerate(gen_hpmap(maps)):
        assert(name == 'map'+str(i))
        npt.assert_array_equal(hp_map, i)

def test_hpmap_key():

    hp_map = ('dummy', 'dummy', {'NSIDE': 32,
                                'ORDERING': 'RING',
                                'COORDSYS': 'C'} )
    key = hpmap_key(hp_map)

    assert(isinstance(key, str))
    assert(key == u'32_RING_icrs')

# def test_group_hpmap():

#     nside = 2**6

#     hp_headers = [ {'NSIDE': nside,
#                     'ORDERING': 'RING',
#                     'COORDSYS': 'C'},
#                    {'NSIDE': nside,
#                     'ORDERING': 'NEST',
#                     'COORDSYS': 'C'},
#                    {'NSIDE': nside/2,
#                     'ORDERING': 'RING',
#                     'COORDSYS': 'C'},
#                    {'NSIDE': nside,
#                     'ORDERING': 'RING',
#                     'COORDSYS': 'G'} ]

#     hp_keys = ["%s_%s_%s"%(hp_header['NSIDE'], hp_header['ORDERING'],  hp_celestial(hp_header).name) for hp_header in hp_headers]

#     maps = [ ('dummy_'+str(i),'dummy_'+str(i),hp_header) for i,hp_header in enumerate(hp_headers) ]
#     maps.append(('dummy_4', 'dummy_4', hp_headers[0]))

#     grouped_maps = group_hpmap(maps)

#     # First and last should be grouped
#     assert(grouped_maps[hp_keys[0]] == [maps[0], maps[-1]])

#     # Test the singletons (all but first and last)
#     for i,key in enumerate(hp_keys[1:]):
#         assert(len(grouped_maps[key]) == 1 )
#         assert(grouped_maps[key][0] == maps[i+1] )
