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
import config_pact
from astropy.table import vstack as astvstack
from astropy.table import hstack as asthstack
from astropy.table import Table, join
import os 
#from astroquery.ned import Ned



def cross_match( file_input,glonglat=[0.,0.],coordframe='galactic'):

    """ can be run doing
    import cross_match as cm
    tab = cm.main( "cross_match_TOTO.ini")
    """

    if file_input == []:
        print('------------------------------')
        print('Need an input file as argument')
        print('------------------------------')
        return

    

####INIs
    
    #reads the ini file

    conf = config_pact.config_pact(input_file=file_input)


    # CATALOG INPUT to be checked with the ones in .ini
    # TODO: should be given as argument



    if conf.input_cat:
	#cat=fits.open('PACT_cat_equ_reg1.fits')
        cat    = fits.open(conf.input_cat)
        cc     = cat[1].data
        cgla   = cc['glat'].squeeze()
        cglo   = cc['glon'].squeeze()
        index  = np.arange(np.size(cgla))
        pacts1 = coord.SkyCoord(cglo,cgla, unit=u.deg, frame='galactic')
    else:
        cglo   = [glonglat[0]]
        cgla   = [glonglat[1]]
        index  = np.arange(np.size(cgla))
        pacts1 = coord.SkyCoord(cglo,cgla, unit=u.deg, frame=coordframe)
        #print('')
        #print('Checking object: ', pacts1)
        #print('')
    
    tab_input_cat = Table()
    tab_input_cat.add_column(Table.Column(name='INDEX', data=index))
    tab_input_cat.add_column(Table.Column(name='GLON',  data=cglo))
    tab_input_cat.add_column(Table.Column(name='GLAT',  data=cgla))
    
    new = tab_input_cat

    
#####SZDB

    if (conf.szdb):
        
        catsz      = fits.open(conf.szdb)
        ccsz       = catsz[1].data
        coord_szdb = coord.SkyCoord(ccsz['glon'],ccsz['glat'], unit=u.deg, frame='galactic')

        # cross matching
        temp_id_szdb, temp_sep_szdb, trash=pacts1.match_to_catalog_sky(coord_szdb)

        temp_id_szdb  = temp_id_szdb.squeeze()
        

        # selecting associations with distance <  sz_dist_theshold_arcmin
        wheregood = temp_sep_szdb.arcmin.squeeze() < conf.sz_dist_theshold_arcmin
        #defining new columns related to the cross match

        if np.sum(wheregood)>0:
            if np.sum(wheregood)>1:
                x_id_szdb_in_cat = np.arange(np.size(cgla))[wheregood]
                x_id_szdb        = temp_id_szdb[wheregood]
                x_coord_szdb     = coord_szdb[x_id_szdb]
                x_sep_szdb       = (temp_sep_szdb.squeeze())[wheregood]
            elif np.sum(wheregood)==1:
                x_id_szdb_in_cat = [0]
                x_id_szdb        = [temp_id_szdb*1]
                x_coord_szdb     = coord_szdb[x_id_szdb]
                x_sep_szdb       = temp_sep_szdb
            
            

        # building infos for selected fields (sz_fields)
        #xszdb = ccsz.take(x_id_szdb)
        
        #xszdb =Table()
        #for fi in conf.sz_fields:
        #    xszdb.add_column(Table.Column(name=fi, data=ccsz[fi][x_id_szdb]))
#ou
        
            fields = [str(x) for x in conf.sz_fields] # removes u
            xszdb  = Table(ccsz[x_id_szdb])           # make cat as Table
            xszdb.keep_columns(fields)                # extract the fields

        # adding columns related to the crossmatch
            xszdb.add_column(Table.Column(name='INDEX', data=x_id_szdb_in_cat))
            xszdb.add_column(Table.Column(name='SZDB_ID', data=x_id_szdb))
            xszdb.add_column(Table.Column(name='SZDB_SEP', data=x_sep_szdb.arcmin))
                 
            for fi in conf.sz_fields:
                xszdb.rename_column(fi, 'SZDB_'+fi)

        
            new = join(new, xszdb,keys='INDEX', join_type='outer')
            new['SZDB_ID'].fill_value=-1
       
            if (conf.verbose):
                print('######################################')
                print('Results of cross correlation with SZDB')
                print(new)
        else:
            xszdb  = Table(masked=True)

            xszdb.add_column(Table.Column(name='INDEX', data=[0]))
            xszdb.add_column(Table.Column(name='SZDB_ID', data=[-1]))
            xszdb.add_column(Table.Column(name='SZDB_SEP', data=[-1.0]))
            new = join(new, xszdb,keys='INDEX', join_type='outer')
            new['SZDB_ID'].fill_value=-1
            new['SZDB_ID'].mask=True
        #xszdb.keep_columns(['RA','DEC']) # becareful remove everything else
        # save Table or add Table to the others(X, Op)?        


####PXCC

    if (conf.pxcc):

        catx      = fits.open(conf.pxcc)
        ccx       = catx[1].data
        coord_pxcc = coord.SkyCoord(ccx['glon'],ccx['glat'],
                                    unit=u.deg, frame='galactic')
 
        temp_id_pxcc, temp_sep_pxcc, trash = pacts1.match_to_catalog_sky(
                                             coord_pxcc)

        temp_id_pxcc  = temp_id_pxcc.squeeze()

        wheregood = temp_sep_pxcc.arcmin.squeeze() < conf.x_dist_theshold_arcmin

        if np.sum(wheregood)>0:
            if np.sum(wheregood)>1:
                x_id_pxcc_in_cat = np.arange(np.size(cgla))[wheregood]
                x_id_pxcc        = temp_id_pxcc[wheregood]
                x_coord_pxcc     = coord_pxcc[x_id_pxcc]
                x_sep_pxcc       = (temp_sep_pxcc.squeeze())[wheregood]
            elif np.sum(wheregood)==1:
                x_id_pxcc_in_cat = [0]
                x_id_pxcc        = [temp_id_pxcc*1]
                x_coord_pxcc     = coord_pxcc[x_id_pxcc]
                x_sep_pxcc       = temp_sep_pxcc
 
            fields = [str(x) for x in conf.x_fields] # removes u
            xpxcc  = Table(ccx[x_id_pxcc])           # make cat as Table
            xpxcc.keep_columns(fields)                # extract the fields

            xpxcc.add_column(Table.Column(name='INDEX', data=x_id_pxcc_in_cat))
            xpxcc.add_column(Table.Column(name='PXCC_ID', data=x_id_pxcc))
            xpxcc.add_column(Table.Column(name='PXCC_SEP', data=x_sep_pxcc.arcmin))

        
            for fi in conf.x_fields:
                xpxcc.rename_column(fi, 'PXCC_'+fi)
                
            new = join(new, xpxcc,keys='INDEX', join_type='outer')
            new['PXCC_ID'].fill_value=-1
       
            if (conf.verbose):
                print('######################################')
                print('Results of cross correlation with PXCC')
                print(new)
        else:
            xpxcc  = Table(masked=True)

            xpxcc.add_column(Table.Column(name='INDEX', data=[0]))
            xpxcc.add_column(Table.Column(name='PXCC_ID', data=[-1]))
            xpxcc.add_column(Table.Column(name='PXCC_SEP', data=[-1.0]))
            new = join(new, xpxcc,keys='INDEX', join_type='outer')
            new['PXCC_ID'].fill_value=-1
            new['PXCC_ID'].mask=True
       

#####REDMAPPER
            
    if (conf.rdm):

        catrdm      = fits.open(conf.rdm)
        ccrdm       = catrdm[1].data
        coord_rdm = coord.SkyCoord(ccrdm['ra'],ccrdm['dec'],
                                    unit=u.deg, frame='fk5')
 
        temp_id_rdm, temp_sep_rdm, trash = pacts1.match_to_catalog_sky(
                                             coord_rdm)

        temp_id_rdm  = temp_id_rdm.squeeze()

        wheregood = temp_sep_rdm.arcmin.squeeze() < conf.rdm_dist_theshold_arcmin

        if np.sum(wheregood)>0:
            if np.sum(wheregood)>1:
                x_id_rdm_in_cat = np.arange(np.size(cgla))[wheregood]
                x_id_rdm        = temp_id_rdm[wheregood]
                x_coord_rdm     = coord_rdm[x_id_rdm]
                x_sep_rdm       = (temp_sep_rdm.squeeze())[wheregood]
            elif np.sum(wheregood)==1:
                x_id_rdm_in_cat = [0]
                x_id_rdm        = [temp_id_rdm*1]
                x_coord_rdm     = coord_rdm[x_id_rdm]
                x_sep_rdm       = temp_sep_rdm

            fields = [str(rdm) for rdm in conf.rdm_fields] # removes u
            xrdm  = Table(ccrdm[x_id_rdm])           # make cat as Table
            xrdm.keep_columns(fields)                # erdmtract the fields

            xrdm.add_column(Table.Column(name='INDEX', data=x_id_rdm_in_cat))
            xrdm.add_column(Table.Column(name='RDM_ID', data=x_id_rdm))
            xrdm.add_column(Table.Column(name='RDM_SEP', data=x_sep_rdm.arcmin))

            for fi in conf.rdm_fields:
                xrdm.rename_column(fi, 'RDM_'+fi)

            new = join(new, xrdm,keys='INDEX', join_type='outer')
            new['RDM_ID'].fill_value=-1

        
            if (conf.verbose):
                print('######################################')
                print('Results of cross correlation with RDM')
                print(new)
        else:
            xrdm  = Table(masked=True)

            xrdm.add_column(Table.Column(name='INDEX', data=[0]))
            xrdm.add_column(Table.Column(name='RDM_ID', data=[-1]))
            xrdm.add_column(Table.Column(name='RDM_SEP', data=[-1.0]))
            new = join(new, xrdm,keys='INDEX', join_type='outer')
            new['RDM_ID'].fill_value=-1
            new['RDM_ID'].mask=True


#####WEN
            
    if (conf.wen):

        catwen      = fits.open(conf.wen)
        ccwen       = catwen[1].data
        coord_wen = coord.SkyCoord(ccwen['ra'],ccwen['dec'],
                                    unit=u.deg, frame='fk5')
 
        temp_id_wen, temp_sep_wen, trash = pacts1.match_to_catalog_sky(
                                             coord_wen)

        temp_id_wen  = temp_id_wen.squeeze()

        wheregood = temp_sep_wen.arcmin.squeeze() < conf.wen_dist_theshold_arcmin

        if np.sum(wheregood)>0:
            if np.sum(wheregood)>1:
                x_id_wen_in_cat = np.arange(np.size(cgla))[wheregood]
                x_id_wen        = temp_id_wen[wheregood]
                x_coord_wen     = coord_wen[x_id_wen]
                x_sep_wen       = (temp_sep_wen.squeeze())[wheregood]
            elif np.sum(wheregood)==1:
                x_id_wen_in_cat = [0]
                x_id_wen        = [temp_id_wen*1]
                x_coord_wen     = coord_wen[x_id_wen]
                x_sep_wen       = temp_sep_wen
               
            fields = [str(wen) for wen in conf.wen_fields] # removes u
            xwen  = Table(ccwen[x_id_wen])           # make cat as Table
            xwen.keep_columns(fields)                # ewentract the fields

            xwen.add_column(Table.Column(name='INDEX', data=x_id_wen_in_cat))
            xwen.add_column(Table.Column(name='WEN_ID', data=x_id_wen))
            xwen.add_column(Table.Column(name='WEN_SEP', data=x_sep_wen.arcmin))
            
            for fi in conf.wen_fields:
                xwen.rename_column(fi, 'WEN_'+fi)

            new = join(new, xwen,keys='INDEX', join_type='outer')
            new['WEN_ID'].fill_value=-1


            if (conf.verbose):
                print('######################################')
                print('Results of cross correlation with WEN')
                print(new)

        else:
            xwen  = Table(masked=True)

            xwen.add_column(Table.Column(name='INDEX', data=[0]))
            xwen.add_column(Table.Column(name='WEN_ID', data=[None] ))
            xwen.add_column(Table.Column(name='WEN_SEP', data=[-1.0]))

            new = join(new, xwen,keys='INDEX', join_type='outer')
            new['WEN_ID'].fill_value=-1
            new['WEN_ID'].mask=True



    
    #print((new['SZDB_ID'].mask)[0], (new['SZDB_ID'].mask)[100])
    #print((new['SZDB_ID'])[0], (new['SZDB_ID'])[100])
    
    
        
###### NED on the not X matched
######

####OUTPUTS TODO global table


    
    filename, extension = os.path.splitext(conf.outputs_filename)

    #print('######################################')
    #print('File saved in', conf.outputs_filename)
    #print('######################################')
    
    if (extension=='.txt'):
        ascii.write(new,conf.outputs_filename,
            format='commented_header', fast_writer=False)
    if (extension=='.html'):
        ascii.write(new,conf.outputs_filename,
            format='html', fast_writer=False)
        
    ## if (extension=='.fits'):
    ##     new.write(conf.outputs_filename,overwrite=True)

    ## if (extension=='.csv'):
    ##     ascii.write(new, conf.outputs_filename,
    ##                 format='csv', fast_writer=False)


    




    return(new)
    

#if __name__ == "__main__":

#    import sys

#    main( file_input = list(sys.argv[1:]))
