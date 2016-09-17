from .. import hp_celestial, hp_is_nest, build_ctype
from .. import build_WCS, build_WCS_cube, build_WCS_2pts

import numpy as np
from numpy import testing as npt
import healpy as hp
from astropy.coordinates import ICRS, Galactic, SkyCoord, Angle

def test_hp_celestial_exception():

    hp_headers= [{},
                 {'COORDSYS': 'ecliptic'},
                 {'COORDSYS': 'ECLIPTIC'},
                 {'COORDSYS': 'e'},
                 {'COORDSYS': 'E'}, ]

    for hp_header in hp_headers:
        try:
            frame = hp_celestial(hp_header)
        except ValueError:
            assert True
        else:
            assert False, "Expected ValueError"

def test_hp_celestial():
    hp_headers = [ ({'COORDSYS': 'G'}, Galactic()),
                   ({'COORDSYS': 'Galactic'}, Galactic()),
                   ({'COORDSYS': 'Equatorial'}, ICRS()),
                   ({'COORDSYS': 'EQ'}, ICRS()),
                   ({'COORDSYS': 'celestial2000'}, ICRS()),
                 ]

    for hp_header, result in hp_headers:
        frame = hp_celestial(hp_header)
        assert(frame.is_equivalent_frame(result))


def test_hp_is_nest_exception():
    hp_headers = [{}, {'ORDERING': 'Unknown'}]

    for hp_header in hp_headers:
        try:
            is_nest = hp_is_nest(hp_header)
        except ValueError:
            assert True
        else:
            assert False, "Expected ValueError"

def test_hp_is_nest():
    hp_headers = [({'ORDERING': 'nested'}, True),
                  ({'ORDERING': 'NESTED'}, True),
                  ({'ORDERING': 'ring'}, False),
    ]

    for hp_header, result in hp_headers:
        is_nest = hp_is_nest(hp_header)
        assert(is_nest == result)


def test_build_ctype_exception():
    tests = [('unknown', 'TAN'),
             ('G', 'BLA')]

    for coordsys, proj_type in tests:
        try:
            ctype = build_ctype(coordsys, proj_type)
        except ValueError:
            assert True
        else:
            assert False, "Expected ValueError"

def test_build_ctype():
    tests = [ ( ('Galactic', 'TAN'), ['GLON-TAN', 'GLAT-TAN'] ),
              ( ('G', 'tan'), ['GLON-TAN', 'GLAT-TAN'] ),
              ( ('EQUATORIAL', 'SiN'), ['RA---SIN', 'DEC--SIN'] ),
    ]

    for (coordsys, proj_type), result in tests:
        ctype = build_ctype(coordsys, proj_type)
        assert(ctype == result)

def test_build_WCS_exception():

    params = [ ('EQ', 'BLA'), ('E','TAN') ]

    for proj_sys, proj_type in params:
        try:
            build_WCS(SkyCoord(0,0, unit='deg'),proj_sys=proj_sys, proj_type=proj_type)
        except ValueError:
            assert True
        else:
            assert False, "Expected ValueError"

def test_build_WCS():

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



def test_build_WCS_cube_exception():

    params = [ ('EQ', 'BLA'), ('E','TAN') ]

    for proj_sys, proj_type in params:
        try:
            build_WCS_cube(SkyCoord(0,0, unit='deg'),0, proj_sys=proj_sys, proj_type=proj_type)
        except ValueError:
            assert True
        else:
            assert False, "Expected ValueError"

def test_build_WCS_cube():

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


def test_build_WCS_2pts_exception():

    params = [ ('EQ', 'BLA'), ('E','TAN') ]

    coords = [SkyCoord(0,0, unit='deg'),
              SkyCoord(1,0, unit='deg')]

    for proj_sys, proj_type in params:
        try:
            build_WCS_2pts(coords,1, proj_sys=proj_sys, proj_type=proj_type)
        except ValueError:
            assert True
        else:
            assert False, "Expected ValueError"


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


def hp_to_wcs():
    # hp_to_wcs(hp_map, hp_header, w, shape_out=DEFAULT_shape_out, npix=None, order=0):

    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside))
    hp_header={'NSIDE': nside,
               'ORDERING': 'RING',
               'COORDSYS': 'C'}


    coord, pixsize, shape_out = SkyCoord(0,0,unit='deg'), 1, [512, 512]
    w = build_WCS(coord, pixsize, shape_out)

    sub_map = hp_to_wcs(hp_map, hp_header, w, shape_out=shape_out, order=0)
    assert(sub_map.shape == tuple(shape_out))
    npt.assert_array_equal(sub_map, 1)

    sub_map = hp_to_wcs(hp_map, hp_header, w, shape_out=shape_out, order=1)
    npt.assert_allclose(sub_map, 1, rtol=1e-15) # hp.get_interp_val precision

    lon, lat = 0, 0
    phi, theta = np.radians(lon), np.radians(90-lat)
    ipix = hp.ang2pix(nside, theta, phi, nest=hp_is_nest(hp_header))
    hp_map[ipix] = 0
    sub_map = hp_to_wcs(hp_map, hp_header, w, shape_out=shape_out, order=0)
    i_x, i_y = w.all_world2pix(alon,alat,0)
    assert(sub_map[int(np.floor(i_y)),int(np.floor(i_x))] == 0)
