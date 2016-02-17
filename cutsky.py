#! /anaconda/bin/python

import logging
logger = logging.getLogger('django')

import healpy as hp
from astropy import wcs
import numpy as np
from astropy.coordinates import SkyCoord
from photutils import CircularAperture
from photutils import aperture_photometry
from astropy import coordinates as coord
import astropy.units as u
import cStringIO

import os

try:
    from WebServices.settings import BASE_DIR
except ImportError:
    BASE_DIR="../"

def build_WCS(lon, lat, pixsize=0.01, npix=512, coordsys='ECLIPTIC', proj_type='TAN'):
    """Construct a WCS object for a 2D map
    Parameters
    ----------
    lon, lat : float
        longitude and latitude of the projection center (in degree)
    pixsize : float
        size of the pixel (in degree)
    npix : int
        number of pixels in the final map, the reference pixel will be at the center
    coordsys : str ('GALACTIC', 'ECLIPTIC')
        the coordinate system of the plate (from HEALPIX maps....)
    proj_type : str ('TAN', 'SIN', 'GSL', ...)
        the projection system to use
    Return
    ------
    WCS
        An astropy.wcs.WCS object
    From
    ----
        A.Beelen
   """
    w = wcs.WCS(naxis=2)

    w.wcs.crpix = np.ones(2.)*npix/2
    w.wcs.cdelt = np.array([-pixsize, pixsize])
    w.wcs.crval = np.array([lon, lat])

    if coordsys == 'GALACTIC':
        w.wcs.ctype = [ coord+proj_type for coord in ['GLON-', 'GLAT-']]
    elif coordsys == 'ECLIPTIC':
        w.wcs.ctype = [ coord+proj_type for coord in ['RA---', 'DEC--']]

    return w




def hp_ipx(w, npix,coordsys, nside):
    """Return the nside map healpix index of a wcs header, using nearest neighbors.

    Parameters
    ----------
    w : astropy.wcs.WCS
        wcs object to project to
    npix : int
        size of the map to return
    coordsys : str ('GALATIC', 'EQUATORIAL')
        the coordinate system of the plate
    nside : int
        the desired healpix nside indexes

    Return
    ------
    array_like
        the mask to be used for the map
    array_like
        the projected healpix pixel index

    Note
    ----
    The map could then easily be constructed using

    mask, ipix = np.ma.array(np.zeros((npix, npix)), mask=~mask, fill_value=np.nan)
    proj_map[mask] = healpix_map[ipix]

    """

    xx, yy     = np.meshgrid(np.arange(npix), np.arange(npix))
    alon, alat = w.wcs_pix2world(xx,yy,0)

    if coordsys=='GALACTIC':
        coord='galactic'
    elif coordsys=='ECLIPTIC':
        coord='fk5'

    position    = SkyCoord(alon,alat, frame=coord, unit="deg")
    alon2,alat2 = position.galactic.l.value, position.galactic.b.value

    mask = ~np.logical_or(np.isnan(alon2),np.isnan(alat2)) # if pixel lies outside of projected area
    #proj_map = np.ma.array(np.zeros((npix, npix)), mask=~mask, fill_value=np.nan)
    ipix = hp.ang2pix(nside, np.radians(90-alat2[mask]), np.radians(alon2[mask]))
    #proj_map[mask] = hp_map[ipix]

    return ( mask, ipix)

def hp_project(hp_map, w, npix,coordsys):
    """Project an Healpix map on a wcs header, using nearest neighbors.

    Parameters
    ----------
    hp_map : array_like
        healpix map to project from
    w : astropy.wcs.WCS
        wcs object to project to
    npix : int
        size of the map to return
    Return
    ------
    array_like
        the projected map in a 2D array of npix x npix
    From
    ----
        A.Beelen, M.douspis
    """
    xx, yy     = np.meshgrid(np.arange(npix), np.arange(npix))
    alon, alat = w.wcs_pix2world(xx,yy,0)

    if coordsys=='GALACTIC':
        coord='galactic'
    elif coordsys=='ECLIPTIC':
        coord='fk5'

    position    = SkyCoord(alon,alat, frame=coord, unit="deg")
    alon2,alat2 = position.galactic.l.value, position.galactic.b.value

    mask = ~np.logical_or(np.isnan(alon2),np.isnan(alat2)) # if pixel lies outside of projected area
    proj_map = np.ma.array(np.zeros((npix, npix)), mask=~mask, fill_value=np.nan)
    ipix = hp.ang2pix(hp.npix2nside(len(hp_map)), np.radians(90-alat2[mask]), np.radians(alon2[mask]))
    proj_map[mask] = hp_map[ipix]

    return proj_map.filled()



    ###############################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wcsaxes import WCS



def cut_sky( lonlat=[0,0],patch=[256,1],coordframe='galactic'):


    cglo   = [lonlat[0]]
    cgla   = [lonlat[1]]
    index  = np.arange(np.size(cgla))
    coord_in = coord.SkyCoord(cglo,cgla, unit=u.deg, frame=coordframe)

    glon = coord_in.galactic.l.value[0]
    glat = coord_in.galactic.b.value[0]

    pixel_size=np.float(patch[1])
    n_pixels=np.float(patch[0])


    if np.str(coordframe)=="galactic":
        coordf = 'GALACTIC'

    if np.str(coordframe)=="fk5":
        coordf = 'ECLIPTIC'

    doxmap = True

    w       = build_WCS(cglo[0],cgla[0], pixsize=pixel_size/60., npix=n_pixels, coordsys=np.str(coordf), proj_type='TAN')

    # Set up the figure

    blank = np.zeros((n_pixels, n_pixels))
    logger.warning('setting figure')
    fig=plt.figure()
    wcs_proj=WCS(w.to_header())
    ax_wcs=fig.add_axes([0.1,0.1,0.9,0.9],projection=wcs_proj)
    proj_im = ax_wcs.imshow(blank, interpolation='none', origin='lower' )
    ax_wcs.coords.grid(color='green', linestyle='solid', alpha=0.5)

    if np.str(coordf)=="ECLIPTIC":
        ax_wcs.coords['ra'].set_ticks(color='white')
        ax_wcs.coords['dec'].set_ticks(color='white')
        ax_wcs.coords['ra'].set_axislabel(r'$\alpha_\mathrm{J2000}$')
        ax_wcs.coords['dec'].set_axislabel(r'$\delta_\mathrm{J2000}$')
    elif np.str(coordf)=="GALACTIC":
        ax_wcs.coords['glon'].set_ticks(color='red')
        ax_wcs.coords['glat'].set_ticks(color='red')
        ax_wcs.coords['glon'].set_axislabel(r'$l$')
        ax_wcs.coords['glat'].set_axislabel(r'$b$')

    logger.warning('reading Ymap')
    # MILCA map
    filemap = os.path.join(BASE_DIR,'xmatch/data/MILCA_TSZ_2048_spectral_spacial_local_10arcmin.fits')
    ymap    = hp.read_map(filemap, verbose=False, dtype=np.float32) #, memmap=True)

    logger.warning('projecting pixels')
    # Extract mask & pixels, good for all healpix maps with same nside
    mask, ipix = hp_ipx(w, n_pixels, np.str(coordf), hp.npix2nside(len(ymap)))

    ypatch = np.ma.array(np.zeros((n_pixels, n_pixels)), mask=~mask, fill_value=np.nan)
    ypatch[mask] = ymap[ipix]

    logger.warning('updating Ymap')

    # Update figure
    proj_im.set_data(ypatch)
    proj_im.set_clim(vmin=ypatch.min(), vmax=ypatch.max())


    logger.warning('contouring Ymap')
    levels=[ypatch.max()/3., ypatch.max()/2.]
    if ((ypatch.max()-ypatch.mean()) > 3*ypatch.std()):
        proj_cont = ax_wcs.contour(ypatch,levels=levels,colors="white",interpolation='bicubic')
    else:
        proj_cont = None

    logger.warning('saving Ymap')
    outputymap = cStringIO.StringIO()
    plt.savefig(outputymap,bbox_inches='tight', format='png',dpi=75, frameon=False)
    logger.warning('Ymap done')

    logger.warning('reading Xmap')
   # Rosat Map
    filemapx = os.path.join(BASE_DIR,'xmatch/data/map_rosat_70-200_2048.fits')
    xmap    = hp.read_map(filemapx, verbose=False, dtype=np.float32) #, memmap=True)

    logger.warning('updating Xmap')
    xpatch = np.ma.array(np.zeros((n_pixels, n_pixels)), mask=~mask, fill_value=np.nan)
    xpatch[mask] = xmap[ipix]

    # Update figure
    proj_im.set_data(xpatch)
    proj_im.set_clim(vmin=xpatch.min(), vmax=xpatch.max())

    logger.warning('saving Xmap')
    outputxmap = cStringIO.StringIO()
    plt.savefig(outputxmap,bbox_inches='tight', format='png',dpi=75, frameon=False)
    logger.warning('Xmap done')

    # Only the contour
    logger.warning('updating Contour')

    proj_im.set_visible(False)
    if not proj_cont:
        ax_wcs.contour(ypatch,levels=levels, transform=ax_wcs.get_transform(wcs_proj),colors="red", interpolation="bicubic")

    logger.warning('saving Contour')
    outputcmap = cStringIO.StringIO()
    plt.savefig(outputcmap,bbox_inches='tight', format='png', dpi=75, frameon=False)
    logger.warning('contour done')



########################################################### APERTURE


    logger.warning('map ok')


    positions = [(n_pixels/2., n_pixels/2.)]
    apertures = CircularAperture(positions, r=3.0/pixel_size)
    yphot = aperture_photometry(ypatch-np.median(ypatch), apertures)
    xphot = aperture_photometry(xpatch-np.median(xpatch), apertures)

    logger.warning('phot ok')


    return {'mapy':outputymap.getvalue().encode("base64").strip(),
            'mapx':outputxmap.getvalue().encode("base64").strip(),
            'mapc':outputxmap.getvalue().encode("base64").strip(),
            'xphot':xphot,
            'yphot':yphot,}

if __name__ == '__main__':
    a = cut_sky()
