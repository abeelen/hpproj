#! /anaconda/bin/python


# import matplotlib.image as mpimg
import healpy as hp
from astropy import wcs
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from photutils import CircularAperture
from photutils import aperture_photometry
from astropy import coordinates as coord
import astropy.units as u
import cStringIO
 
# import cross_match

import os
from WebServices.settings import BASE_DIR

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
    
    yy, xx     = np.meshgrid(np.arange(npix), np.arange(npix))
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
    yy, xx     = np.meshgrid(np.arange(npix), np.arange(npix))
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


    w       = build_WCS(glon,glat, pixsize=pixel_size/60., npix=n_pixels, coordsys=np.str(coordf), proj_type='TAN')

    filemap = os.path.join(BASE_DIR,'xmatch/data/MILCA_TSZ_2048_spectral_spacial_local_10arcmin.fits')
    ymap    = hp.read_map(filemap, verbose=False, dtype=np.float32) #, memmap=True)

    # ypatch  = hp_project(ymap, w, n_pixels,np.str(coordf))
    mask, ipix = hp_ipx(w, n_pixels, np.str(coordf), hp.npix2nside(len(ymap)))
    ypatch = np.ma.array(np.zeros((n_pixels, n_pixels)), mask=~mask, fill_value=np.nan)
    ypatch[mask] = ymap[ipix]
    
    fig=plt.figure()
    wcs_proj=WCS(w.to_header())
    ax1_wcs=fig.add_axes([0.1,0.1,0.9,0.9],projection=wcs_proj)
    ax1_wcs.imshow(ypatch, interpolation='none', origin='lower' )
    levels=[ypatch.max()/3., ypatch.max()/2.]
    print((ypatch.max()-ypatch.mean()) , 3*np.std(ypatch))
    if ((ypatch.max()-ypatch.mean()) > 3*np.std(ypatch)):
        ax1_wcs.contour(ypatch,levels=levels,colors="white",interpolation='bicubic')
        
    ax1_wcs.coords.grid(color='green', linestyle='solid', alpha=0.5)
    if np.str(coordf)=="ECLIPTIC":
        ax1_wcs.coords['ra'].set_ticks(color='white')
        ax1_wcs.coords['dec'].set_ticks(color='white')
        ax1_wcs.coords['ra'].set_axislabel(r'$\alpha_\mathrm{J2000}$')
        ax1_wcs.coords['dec'].set_axislabel(r'$\delta_\mathrm{J2000}$')
        
    if np.str(coordf)=="GALACTIC":
        ax1_wcs.coords['glon'].set_ticks(color='red')
        ax1_wcs.coords['glat'].set_ticks(color='red')
        ax1_wcs.coords['glon'].set_axislabel(r'$l$')
        ax1_wcs.coords['glat'].set_axislabel(r'$b$')


    print('map y ok')
    #plt.savefig('outymap.png',bbox_inches='tight')
    outputymap = cStringIO.StringIO()
    plt.savefig(outputymap,bbox_inches='tight', format='png',dpi=75)

    # del(ymap)

    filemapx = os.path.join(BASE_DIR,'xmatch/data/map_rosat_70-200_2048.fits')
    xmap    = hp.read_map(filemapx, verbose=False, dtype=np.float32) #, memmap=True)

#    xpatch  = hp_project(xmap, w, n_pixels,np.str(coordf))
    xpatch = np.ma.array(np.zeros((n_pixels, n_pixels)), mask=~mask, fill_value=np.nan)
    xpatch[mask] = xmap[ipix]

    
    fig=plt.figure()
    wcs_proj=WCS(w.to_header())
    ax2_wcs=fig.add_axes([0.1,0.1,0.9,0.9],projection=wcs_proj)
    ax2_wcs.imshow(xpatch, interpolation='none', origin='lower')
    if ((ypatch.max()-ypatch.mean()) > 3*np.std(ypatch)):
        ax2_wcs.contour(ypatch,levels=levels, transform=ax2_wcs.get_transform(wcs_proj),colors="white",interpolation='bicubic')
    ax2_wcs.coords.grid(color='green', linestyle='solid', alpha=0.5)
    if np.str(coordf)=="ECLIPTIC":
        ax2_wcs.coords['ra'].set_ticks(color='white')
        ax2_wcs.coords['dec'].set_ticks(color='white')
        ax2_wcs.coords['ra'].set_axislabel(r'$\alpha_\mathrm{J2000}$')
        ax2_wcs.coords['dec'].set_axislabel(r'$\delta_\mathrm{J2000}$')
    if np.str(coordf)=="GALACTIC":
        ax2_wcs.coords['glon'].set_ticks(color='red')
        ax2_wcs.coords['glat'].set_ticks(color='red')
        ax2_wcs.coords['glon'].set_axislabel(r'$l$')
        ax2_wcs.coords['glat'].set_axislabel(r'$b$')
    print('map x ok')           
        #plt.savefig('outxmap.png',bbox_inches='tight')
    outputxmap = cStringIO.StringIO()
    plt.savefig(outputxmap,bbox_inches='tight', format='png',dpi=75)
    # del(xmap)

    ## fig=plt.figure()
    ## ax3_wcs=fig.add_axes([0.1,0.1,0.9,0.9],projection=wcs_proj)
    ## ax3_wcs.contour(ypatch,levels=levels, transform=ax3_wcs.get_transform(wcs_proj),colors="red")
    ## ax3_wcs.coords.grid(color='green', linestyle='solid', alpha=0.5)
    ## if np.str(coordf)=="ECLIPTIC":
    ##     ax3_wcs.coords['ra'].set_ticks(color='white')
    ##     ax3_wcs.coords['dec'].set_ticks(color='white')
    ##     ax3_wcs.coords['ra'].set_axislabel(r'$\alpha_\mathrm{J2000}$')
    ##     ax3_wcs.coords['dec'].set_axislabel(r'$\delta_\mathrm{J2000}$')
    ## if np.str(coordf)=="GALACTIC":
    ##     ax3_wcs.coords['glon'].set_ticks(color='red')
    ##     ax3_wcs.coords['glat'].set_ticks(color='red')
    ##     ax3_wcs.coords['glon'].set_axislabel(r'$l$')
    ##     ax3_wcs.coords['glat'].set_axislabel(r'$b$')
  
 #   outputcmap = cStringIO.StringIO()
 #   plt.savefig(outputcmap,bbox_inches='tight', format='png')




########################################################### APERTURE


    print('map ok')


    positions = [(n_pixels/2., n_pixels/2.)]
    apertures = CircularAperture(positions, r=3.0/pixel_size)
    yphot = aperture_photometry(ypatch-np.median(ypatch), apertures)
    xphot = aperture_photometry(xpatch-np.median(xpatch), apertures)

    print('phot ok', xphot)

    
    return {'mapy':outputymap.getvalue().encode("base64").strip(),
            'mapx':outputxmap.getvalue().encode("base64").strip(),
            'mapc':outputxmap.getvalue().encode("base64").strip(),
            'xphot':xphot,
            'yphot':yphot,}
