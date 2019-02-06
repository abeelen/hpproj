#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2016 IAS / CNRS / Univ. Paris-Sud
# LGPL License - see attached LICENSE file
# Author: Alexandre Beelen <alexandre.beelen@ias.u-psud.fr>

import pytest

from hpproj import hp_celestial, hp_is_nest
from hpproj import hp_to_wcs, hp_to_wcs_ipx
from hpproj import hp_project, gen_hpmap, hpmap_key
from hpproj import hp_to_profile, hp_profile
from hpproj import hp_to_aperture, hp_photometry
from hpproj import wcs_to_profile
from hpproj import hp_stack

from hpproj import build_wcs, build_wcs_profile

import numpy as np
from numpy import testing as npt
import healpy as hp
from astropy.coordinates import ICRS, Galactic, SkyCoord, Angle
from astropy.io import fits
# from astropy.modeling import models, fitting # This make an optionnal dependancy on scipy...
from astropy import units as u
# from scipy.special import erf


class TestHPCelestical:

    @pytest.mark.parametrize("hp_header", [{},
                                           {'coordsys': 'ecliptic'},
                                           {'coordsys': 'ECLIPTIC'},
                                           {'coordsys': 'e'},
                                           {'coordsys': 'E'}, ])
    def test_hp_celestial_exception(self, hp_header):
        with pytest.raises(ValueError):
            frame = hp_celestial(fits.Header(hp_header))

    @pytest.mark.parametrize("hp_header,result",
                             [({'coordsys': 'G'}, Galactic()),
                              ({'coordsys': 'Galactic'}, Galactic()),
                              ({'coordsys': 'Equatorial'}, ICRS()),
                              ({'coordsys': 'EQ'}, ICRS()),
                              ({'coordsys': 'celestial2000'}, ICRS()),
                              ])
    def test_hp_celestial(self, hp_header, result):
        frame = hp_celestial(fits.Header(hp_header))
        assert frame.is_equivalent_frame(result)


class TestHPNest:

    def test_hp_is_nest_exception(self):
        hp_headers = [{}, {'ordering': 'Unknown'}]

        for hp_header in hp_headers:
            with pytest.raises(ValueError):
                is_nest = hp_is_nest(fits.Header(hp_header))

    @pytest.mark.parametrize("hp_header, result",
                             [({'ordering': 'nested'}, True),
                              ({'ordering': 'NESTED'}, True),
                              ({'ordering': 'ring'}, False),
                              ])
    def test_hp_is_nest(self, hp_header, result):
        is_nest = hp_is_nest(fits.Header(hp_header))
        assert is_nest == result


@pytest.fixture(scope='session')
def uniform_hp_hdu():
    nside = 2**6
    hp_map = np.ones(hp.nside2npix(nside), dtype=np.float)
    hp_header = {'nside': nside,
                 'ordering': 'RING',
                 'coordsys': 'C'}

    return fits.ImageHDU(hp_map, fits.Header(hp_header))


@pytest.fixture(scope='session')
def gaussian_hp_hdu():
    nside = 2**6
    # Trick to get the center in the middle of the healpix pixel
    i_pix = hp.ang2pix(nside, 0, 0, nest=False, lonlat=True)
    lon, lat = hp.pix2ang(nside, i_pix, nest=False, lonlat=True)
    coord = SkyCoord(lon, lat, unit="deg")
    sigma = 5 * np.degrees(hp.nside2resol(nside))

    hp_map = np.zeros(hp.nside2npix(nside), dtype=np.float)
    hp_header = fits.Header({'nside': nside,
                             'ordering': 'RING',
                             'coordsys': 'C'})

    i_pix = hp.query_disc(nside, hp.dir2vec(coord.ra.deg, coord.dec.deg, lonlat=True), 20 * sigma, nest=False)
    lon_arr, lat_arr = hp.pix2ang(nside, i_pix, lonlat=True, nest=False)
    dist = SkyCoord(lon_arr, lat_arr, unit="deg").separation(coord).to(u.deg).value
    hp_map[i_pix] += np.exp(-dist**2 / (2 * sigma**2))

    return sigma, coord, fits.ImageHDU(hp_map, fits.Header(hp_header))


@pytest.fixture(scope='session')
def gaussians_hp_hdu():
    np.random.seed(0)

    nside = 2**8
    n_gaussian = 10
    sigma = 3 * np.degrees(hp.nside2resol(nside))

    hp_map = np.zeros(hp.nside2npix(nside), dtype=np.float)
    hp_header = {'nside': nside,
                 'ordering': 'RING',
                 'coordsys': 'G',
                 'UNIT': 'MJy/deg**2'}

    i_pix = np.random.uniform(0, hp.nside2npix(nside), n_gaussian).astype(int)
    lon, lat = hp.pix2ang(nside, i_pix, nest=False, lonlat=True)
    coords = SkyCoord(lon, lat, unit="deg", frame="galactic")

#    coords = SkyCoord(np.random.uniform(0, 360, n_gaussian), np.degrees(np.arcsin(np.random.uniform(-1, 1, n_gaussian))), unit="deg")
    for coord in coords:
        i_pix = hp.query_disc(nside, hp.dir2vec(coord.l.deg, coord.b.deg, lonlat=True), 20 * sigma, nest=False)
        lon_arr, lat_arr = hp.pix2ang(nside, i_pix, lonlat=True, nest=False)
        dist = SkyCoord(lon_arr, lat_arr, unit="deg", frame="galactic").separation(coord).to(u.deg).value
        hp_map[i_pix] += np.exp(-dist**2 / (2 * sigma**2))

    return sigma, coords, fits.ImageHDU(hp_map, fits.Header(hp_header))


def test_hp_stack_single(gaussians_hp_hdu):

    sigma, coords, hp_hdu = gaussians_hp_hdu
    shape_out = (129, 129)
    pixsize = np.degrees(hp.nside2resol(hp_hdu.header['nside'])) / 3
    hdu = hp_stack(hp_hdu, coords[0], pixsize=pixsize, shape_out=shape_out)

    assert isinstance(hdu, fits.ImageHDU)
    assert hdu.data.shape == shape_out
    assert np.abs(hdu.header['CDELT1']) == pixsize

    npt.assert_allclose(hdu.data.max(), 1, rtol=1e-4)


def test_hp_stack_keep(gaussians_hp_hdu):

    sigma, coords, hp_hdu = gaussians_hp_hdu
    shape_out = (129, 129)
    pixsize = np.degrees(hp.nside2resol(hp_hdu.header['nside'])) / 3
    hdu = hp_stack(hp_hdu, coords, pixsize=pixsize, shape_out=shape_out, keep=True)

    assert isinstance(hdu, fits.ImageHDU)
    assert hdu.data.shape == coords.shape + shape_out
    assert np.abs(hdu.header['CDELT1']) == pixsize

    npt.assert_allclose(hdu.data.max(axis=(1, 2)), 1, rtol=1e-4)


def test_hp_stack_pixsize(gaussians_hp_hdu):

    sigma, coords, hp_hdu = gaussians_hp_hdu
    shape_out = (129, 129)
    pixsize = np.degrees(hp.nside2resol(hp_hdu.header['nside'])) / 3
    pixsize = [pixsize] * coords.shape[0]
    hdu = hp_stack(hp_hdu, coords, pixsize=pixsize, shape_out=shape_out, keep=True)

    assert isinstance(hdu, fits.ImageHDU)
    assert hdu.data.shape == coords.shape + shape_out
    assert np.abs(hdu.header['CDELT1']) == pixsize[0]

    npt.assert_allclose(hdu.data.max(axis=(1, 2)), 1, rtol=1e-4)

    # TODO: Probably shall use a gaussians_pixsize fixture to properly check for that.


def test_hp_stack(gaussians_hp_hdu):

    sigma, coords, hp_hdu = gaussians_hp_hdu
    shape_out = (129, 129)
    pixsize = np.degrees(hp.nside2resol(hp_hdu.header['nside'])) / 3
    hdu = hp_stack(hp_hdu, coords, pixsize=pixsize, shape_out=shape_out)

    assert isinstance(hdu, fits.ImageHDU)
    assert hdu.data.shape == shape_out
    assert np.abs(hdu.header['CDELT1']) == pixsize

    npt.assert_allclose(hdu.data.max(), 1, rtol=1e-4)

    pixsize = np.abs(hdu.header['CDELT1'])
    sigma_pix = sigma / pixsize

    def CircularGaussian2D(x, y, amplitude, x_mean, y_mean, sigma):
        denom = 2 * sigma**2
        return amplitude * np.exp(- ((x - x_mean)**2 + (y - y_mean)**2) / denom)

    y, x = np.indices(hdu.shape, dtype=np.float)
    model = CircularGaussian2D(x, y, amplitude=1, x_mean=(hdu.shape[1] - 1) / 2, y_mean=(hdu.shape[0] - 1) / 2, sigma=sigma_pix)

    # Due to strong pixelization....
    npt.assert_allclose(hdu.data, model, atol=1e-1, rtol=1e-2)

    # This would imply a requirement on scipy...
    # g = models.Gaussian2D(x_mean=hdu.shape[1] / 2, y_mean=hdu.shape[0] / 2, x_stddev=sigma / pixsize)
    # g.theta.fixed = True
    # g.x_stddev.fixed = True
    # g.y_stddev.tied = lambda g: g.x_stddev

    # fit_g = fitting.LevMarLSQFitter()
    # p = fit_g(g, x, y, hdu.data)

    # npt.assert_allclose(p.amplitude, 1, rtol=1e-2)
    # npt.assert_allclose(p.x_mean, (hdu.shape[1] - 1) / 2, rtol=1e-4)
    # npt.assert_allclose(p.y_mean, (hdu.shape[0] - 1) / 2, rtol=1e-4)


def test_wcs_to_profile():

    npix = 65
    sigma = 4
    y_arr, x_arr = np.indices((npix, npix), dtype=np.float)
    data = np.exp(- (x_arr - (npix - 1) / 2.)**2 / (2 * sigma**2) - (y_arr - (npix - 1) / 2.)**2 / (2 * sigma**2))
    wcs = build_wcs(0, 0, src_frame='EQUATORIAL', pixsize=1, shape_out=(npix, npix))
    hdu = fits.ImageHDU(data, wcs.to_header())

    wcs = build_wcs_profile(2)
    shape_out = npix / 3
    profile = wcs_to_profile(hdu, wcs, shape_out)

    r_in = wcs.wcs_pix2world(np.arange(shape_out) - 0.5, 0)[0]
    r_out = wcs.wcs_pix2world(np.arange(shape_out) + 0.5, 0)[0]

    # TODO :
    # Pixel integrated gaussian
    # gauss_pix = sigma * np.sqrt(2 * np.pi) / (2 * (r_out - r_in)) * (erf(r_out / (np.sqrt(2) * sigma)) - erf(r_in / (np.sqrt(2) * sigma)))
    # npt.assert_allclose(profile, gauss_pix, rtol=3e-2)
    pass


def test_hp_to_profile_uniform(uniform_hp_hdu):

    hp_hdu = uniform_hp_hdu
    coord, pixsize, shape_out = SkyCoord(0, 0, unit='deg'), 1, 10

    wcs = build_wcs_profile(pixsize)
    profile = hp_to_profile(hp_hdu, wcs, coord, shape_out=shape_out)

    assert isinstance(profile, np.ndarray)
    assert profile.shape[0] == shape_out
    npt.assert_equal(profile, 1)

    profile, std_profile = hp_to_profile(hp_hdu, wcs, coord, shape_out=shape_out, std=True)
    npt.assert_equal(std_profile, 0)


def test_hp_profile_uniform(uniform_hp_hdu):

    hp_hdu = uniform_hp_hdu
    coord, pixsize, shape_out = SkyCoord(0, 0, unit='deg'), 1, 10

    profile = hp_profile(hp_hdu, coord, pixsize=pixsize, npix=shape_out)

    assert isinstance(profile, fits.ImageHDU)
    assert profile.header['NAXIS'] == 1
    assert profile.header['NAXIS1'] == shape_out
    assert profile.header['CDELT1'] == pixsize

    assert profile.data.shape == (shape_out, )
    npt.assert_equal(profile.data, 1)


def test_hp_profile_gaussian(gaussian_hp_hdu):

    sigma, coord, hp_hdu = gaussian_hp_hdu
    pixsize, shape_out = np.degrees(hp.nside2resol(hp_hdu.header['nside'])), 20

    wcs = build_wcs_profile(pixsize)

    profile = hp_to_profile(hp_hdu, wcs, coord, shape_out=shape_out)

    r_in = wcs.wcs_pix2world(np.arange(shape_out) - 0.5, 0)[0]
    r_out = wcs.wcs_pix2world(np.arange(shape_out) + 0.5, 0)[0]

    # Pixel integrated gaussian
    # gauss_pix = sigma * np.sqrt(2 * np.pi) / (2 * (r_out - r_in)) * (erf(r_out / (np.sqrt(2) * sigma)) - erf(r_in / (np.sqrt(2) * sigma)))
    r_mid = (r_in + r_out) / 2
    gauss_pix = np.exp(-(r_mid**2 / (2 * sigma**2)))
    npt.assert_allclose(profile, gauss_pix, rtol=3e-2)


def test_hp_to_wcs_exception(uniform_hp_hdu):

    hp_hdu = uniform_hp_hdu

    coord, pixsize, shape_out = SkyCoord(0, 0, unit='deg'), 1, [512, 512]
    wcs = build_wcs(coord, pixsize, shape_out)

    # Test order > 1
    with pytest.raises(ValueError):
        sub_map = hp_to_wcs(hp_hdu, wcs, shape_out=shape_out, order=2)


def test_hp_to_wcs(uniform_hp_hdu):
    # hp_to_wcs(hp_map, hp_header, wcs, shape_out=DEFAULT_shape_out,
    # npix=None, order=0):
    hp_hdu = uniform_hp_hdu

    nside = hp_hdu.header['nside']
    coord, pixsize, shape_out = SkyCoord(0, 0, unit='deg'), np.degrees(hp.nside2resol(nside)), [512, 512]
    wcs = build_wcs(coord, pixsize, shape_out)

    # Test order = 0
    sub_map = hp_to_wcs(hp_hdu, wcs, shape_out=shape_out, order=0)
    assert sub_map.shape == tuple(shape_out)
    npt.assert_array_equal(sub_map, 1)

    # Test order = 1
    sub_map = hp_to_wcs(hp_hdu, wcs, shape_out=shape_out, order=1)
    npt.assert_allclose(sub_map, 1, rtol=1e-15)  # hp.get_interp_val precision

    # Test specific pixel Better use an odd number for this, because
    # build_wcs put the reference at the center of the image, which in
    # case of even number leaves it between 4 pixels and hp.ang2pix
    shape_out = [3, 3]

    wcs = build_wcs(coord, pixsize, shape_out)

    lon, lat = coord.ra.deg, coord.dec.deg
    phi, theta = np.radians(lon), np.radians(90 - lat)
    ipix = hp.ang2pix(nside, theta, phi, nest=hp_is_nest(hp_hdu.header))
    hp_hdu.data[ipix] = 0
    sub_map = hp_to_wcs(hp_hdu, wcs, shape_out=shape_out, order=0)
    i_x, i_y = wcs.all_world2pix(lon, lat, 0)
    assert sub_map[int(np.floor(i_y + 0.5)), int(np.floor(i_x + 0.5))] == 0

    # Test different frame
    wcs = build_wcs(coord, pixsize, shape_out, proj_sys="G")
    sub_map = hp_to_wcs(hp_hdu, wcs, shape_out=shape_out)
    lon, lat = coord.galactic.l.deg, coord.galactic.b.deg
    i_x, i_y = wcs.all_world2pix(lon, lat, 0)
    assert sub_map[int(np.floor(i_y + 0.5)), int(np.floor(i_x + 0.5))] == 0


def test_hp_to_wcs_ipx(uniform_hp_hdu):

    hp_hdu = uniform_hp_hdu
    hp_header = hp_hdu.header
    nside = hp_header['nside']

    coord, pixsize, shape_out = SkyCoord(0, 0, unit='deg'), 0.1, [1, 1]
    wcs = build_wcs(coord, pixsize, shape_out)

    # Basic test
    sub_mask, sub_ipx = hp_to_wcs_ipx(hp_header, wcs, shape_out=shape_out)
    lon, lat = coord.ra.deg, coord.dec.deg
    phi, theta = np.radians(lon), np.radians(90 - lat)
    ipix = hp.ang2pix(nside, theta, phi, nest=hp_is_nest(hp_header))

    npt.assert_array_equal(sub_mask, True)
    npt.assert_array_equal(sub_ipx, ipix)

    # Test different frame
    wcs = build_wcs(coord, pixsize, shape_out=(1, 1), proj_sys="G")
    sub_mask, sub_ipx = hp_to_wcs_ipx(
        hp_header, wcs, shape_out=shape_out)
    npt.assert_array_equal(sub_mask, True)
    npt.assert_array_equal(sub_ipx, ipix)


def test_hp_project(uniform_hp_hdu):
    hp_hdu = uniform_hp_hdu

    coord, pixsize, npix = SkyCoord(0, 0, unit='deg'), np.degrees(hp.nside2resol(hp_hdu.header['nside'])), 512

    # Test HDU
    sub_map = hp_project(hp_hdu, coord, pixsize, (npix, npix))
    assert isinstance(sub_map, fits.hdu.image.ImageHDU)
    assert sub_map.data.shape == (npix, npix)


def test_hpmap_decorator(uniform_hp_hdu):
    hp_hdu = uniform_hp_hdu
    hp_header = hp_hdu.header
    nside = hp_header['nside']

    coord, pixsize, npix = SkyCoord(
        0, 0, unit='deg'), np.degrees(hp.nside2resol(nside)), 512

    # Test HDU
    sub_map = hp_project(hp_hdu.data, hp_header, coord, pixsize, (npix, npix))
    assert type(sub_map) is fits.hdu.image.ImageHDU
    assert sub_map.data.shape == (npix, npix)


def test_gen_hpmap(uniform_hp_hdu):

    hp_hdu = uniform_hp_hdu
    hp_map = np.ones(hp.nside2npix(hp_hdu.header['nside']))
    maps = [('map' + str(i), hp_map * i, hp_hdu.header) for i in range(3)]

    for i, (name, hp_hdu) in enumerate(gen_hpmap(maps)):
        assert name == 'map' + str(i)
        npt.assert_array_equal(hp_hdu.data, i)


def test_hpmap_key():

    hp_map = ('dummy', 'dummy', fits.Header({'nside': 32,
                                             'ordering': 'RING',
                                             'coordsys': 'C'}))
    key = hpmap_key(hp_map)

    assert isinstance(key, str)
    assert key == u'32_RING_icrs'

# def test_group_hpmap():

#     nside = 2**6

#     hp_headers = [{'nside': nside,
#                     'ordering': 'RING',
#                     'coordsys': 'C'},
#                    {'nside': nside,
#                     'ordering': 'NEST',
#                     'coordsys': 'C'},
#                    {'nside': nside/2,
#                     'ordering': 'RING',
#                     'coordsys': 'C'},
#                    {'nside': nside,
#                     'ordering': 'RING',
#                     'coordsys': 'G'} ]

# hp_keys = ["%s_%s_%s"%(hp_header['nside'], hp_header['ordering'],
# hp_celestial(hp_header).name) for hp_header in hp_headers]

#     maps = [('dummy_'+str(i),'dummy_'+str(i),hp_header) for i,hp_header in enumerate(hp_headers) ]
#     maps.append(('dummy_4', 'dummy_4', hp_headers[0]))

#     grouped_maps = group_hpmap(maps)

# First and last should be grouped
#     assert grouped_maps[hp_keys[0]] == [maps[0], maps[-1]]

# Test the singletons (all but first and last)
#     for i,key in enumerate(hp_keys[1:]):
#         assert len(grouped_maps[key]) == 1
#         assert grouped_maps[key][0] == maps[i+1]


def test_hp_to_aperture(gaussian_hp_hdu):

    sigma, coord, hp_hdu = gaussian_hp_hdu
    apertures = [Angle(5 * sigma, unit="deg")]

    npix, aper = hp_to_aperture(hp_hdu, coord, apertures)
    assert npix.shape == (1, )
    assert aper.shape == npix.shape

    # Check for total area
    pix_surf = (np.degrees(hp.nside2resol(hp_hdu.header['nside'])) * u.degree)**2
    area = aper * pix_surf
    npt.assert_allclose(area.value, 2 * np.pi * sigma**2, rtol=3e-3)

    # Multiple coords
    coords = SkyCoord([0, 10], [0, 10], unit="deg", frame="galactic")
    npix, aper = hp_to_aperture(hp_hdu, coords, apertures)
    assert npix.shape == coords.shape + (len(apertures), )
    assert aper.shape == npix.shape

    npt.assert_allclose(aper, 0, atol=1e-51)

    # Several apertures
    apertures = Angle(sigma, unit="deg") * np.arange(1, 20)
    npix, aper = hp_to_aperture(hp_hdu, coord, apertures)
    assert npix.shape == (len(apertures), )
    assert aper.shape == npix.shape


def test_hp_photometry(gaussians_hp_hdu):

    sigma, coords, hp_hdu = gaussians_hp_hdu

    # outer radius choosen as to avoid a close source
    apertures = Angle(sigma, "deg") * [6, 15, 18]
    result = hp_photometry(hp_hdu, coords, apertures)

    pix_surf = (np.degrees(hp.nside2resol(hp_hdu.header['nside'])) * u.degree)**2
    area = result['brigthness'] * result['n_pix'] * pix_surf

    # High tolerance as some of the sources are "blended"
    npt.assert_allclose(area.value, 2 * np.pi * sigma**2, rtol=1e-3)
