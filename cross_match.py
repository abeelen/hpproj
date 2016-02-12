import config_pact
import numpy as np
#import matplotlib.pyplot as plt
from astropy import coordinates as coord
import astropy.units as u
from astropy.io import fits
from astropy.io import ascii
#import pandas as pd
import csv
from astropy import wcs
from astropy import coordinates as coord
import configparser
from astropy.table import vstack as astvstack
from astropy.table import hstack as asthstack
from astropy.table import Table, join
import os
#from astroquery.ned import Ned



def cross_match( file_input,lonlat=[6.7,30.45],coordframe='galactic'):

    """ can be run doing
    import cross_match as cm
    tab = cm.main( "cross_match_TOTO.ini")
    """

    if not file_input:
        print('------------------------------')
        print('Need an input file as argument')
        print('------------------------------')
        return

    #reads the ini file

    conf = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    conf.read(file_input)


    # CATALOG INPUT to be checked with the ones in .ini
    # TODO: should be given as argument

    if conf['inputs']['cat']:
	#cat=fits.open('PACT_cat_equ_reg1.fits')
        cc    = fits.getdata(conf.get('inputs','cat'), conf.getint('inputs','ext'))
        cgla   = cc['glat'].squeeze()
        cglo   = cc['glon'].squeeze()
        index  = np.arange(np.size(cgla))
        coord_in = coord.SkyCoord(cglo,cgla, unit=u.deg, frame='galactic')
    else:
        cglo   = [lonlat[0]]
        cgla   = [lonlat[1]]
        index  = np.arange(np.size(cgla))
        coord_in = coord.SkyCoord(cglo,cgla, unit=u.deg, frame=coordframe)

    tab_input_cat = Table()
    tab_input_cat.add_column(Table.Column(name='INDEX', data=index))
    tab_input_cat.add_column(Table.Column(name='GLON',  data=cglo))
    tab_input_cat.add_column(Table.Column(name='GLAT',  data=cgla))

    new = tab_input_cat

    match_result = {}

    for cat in conf.get('global','cat_list').split():
        if conf.has_section(cat):
            lon = conf.get(cat,'lon')
            lat = conf.get(cat,'lat')
            frame = conf.get(cat,'frame')
            fields = conf.get(cat,'fields').split()

            dist_threshold  = conf.getfloat(cat, 'dist_threshold_arcmin')

            cc = fits.getdata(conf.get(cat,'file'), conf.getint(cat,'ext'))
            coord_cc = coord.SkyCoord(cc[lon], cc[lat], unit=u.deg, frame=frame)

            temp_id, temp_sep, trash=coord_in.match_to_catalog_sky(coord_cc)

            # selecting associations with distance <  sz_dist_theshold_arcmin
            wheregood = temp_sep.arcmin.squeeze() < dist_threshold
            #defining new columns related to the cross match

            if wheregood.any():
                if np.sum(wheregood)>1:
                    x_id_in_cat = index[wheregood.squeeze()]
                    x_id        = temp_id[wheregood]
                    x_coord     = coord_cc[x_id]
                    x_sep       = temp_sep[wheregood]
                elif np.sum(wheregood)==1:
                    x_id_in_cat = [0]
                    x_id        = [temp_id*1]
                    x_coord     = coord_cc[x_id]
                    x_sep       = temp_sep


                # fields = [str(x) for x in fields] # removes u
                ccm_tab  = Table(cc[x_id])         # make cat as Table
                ccm_tab.keep_columns(fields)        # extract the fields

                for field in fields:
                    ccm_tab.rename_column(field, cat.upper()+'_'+field)

                # adding columns related to the crossmatch
                ccm_tab.add_column(Table.Column(name='INDEX', data=x_id_in_cat))
                ccm_tab.add_column(Table.Column(name=cat.upper()+'_ID', data=x_id))
                ccm_tab.add_column(Table.Column(name=cat.upper()+'_SEP', data=x_sep.arcmin))


            else:
                ccm_tab  = Table(masked=True)
                ccm_tab.add_column(Table.Column(name='INDEX', data=[0]))
                ccm_tab.add_column(Table.Column(name= cat.upper()+'_ID', data=[-1]))
                ccm_tab.add_column(Table.Column(name= cat.upper()+'_SEP', data=[np.nan]))

            match_result[cat.upper()] = ccm_tab
            if conf.get('outputs', 'verbose'):
                print('\nResults of cross correlation with '+cat.upper())
                print(ccm_tab)



    # put everything into the output table...
    for matched_cat in match_result.keys():
        new = join(new, match_result[matched_cat], keys='INDEX', join_type='outer')
        new[matched_cat.upper()+'_ID'].fill_value = -1

###### NED on the not X matched
######

####OUTPUTS TODO global table


    outputs_filename = conf.get('outputs','filename')
    filename, extension = os.path.splitext(outputs_filename)

    #print('######################################')
    #print('File saved in', outputs_filename)
    #print('######################################')

    if (extension=='.txt'):
        ascii.write(new,outputs_filename,
            format='commented_header', fast_writer=False)
    if (extension=='.html'):
        ascii.write(new,outputs_filename,
            format='html', fast_writer=False)

    ## if (extension=='.fits'):
    ##     new.write(conf.outputs_filename,overwrite=True)

    ## if (extension=='.csv'):
    ##     ascii.write(new, conf.outputs_filename,
    ##                 format='csv', fast_writer=False)


    # Should return match_cat instead...
    return(new)


#if __name__ == "__main__":

#    import sys

#    main( file_input = list(sys.argv[1:]))
