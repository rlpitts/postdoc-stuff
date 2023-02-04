#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 15:14:43 2019

@author: pitts

Medley of handy functions for things I frequently needed to do with fits files.
In particular, I've merged the contents of mini_formatter.py with this code so
that all the tools I needed for dealing with FIR/submm observatories are in one
place. The original mini_formatter.py is still present as a legacy version, but
you shouldn't need it anymore. I only kept it because I haven't tried running
this file and there were a few minor additions to the version of flux_norm here
that weren't in the original. It should work, though.

About half of the functions are just for converting Herschel files to single-
extension files that most other fits handling software can handle. The Herschel
data are MEF files with ~10 extensions, of which several are images, others are
tables, and a few are just extra headers - efficient packages, but not readily
compatible with other software, and their in-house processing package HIPE was
just too slow, unwieldy, and required learning JAVA. Moreover, a lot of the
most essential header data were scattered across multiple extensions. This code
helps collate what's useful.

Functions included:

>>> obj_img_match()
    Takes a dict or list of SkyCoord objects, compares them to a list of fits
    files, and returns a dict where keys are object names/numbers and values 
    are lists of all fits files with valid data at that object's location.

>>> set_beam()
    Helper function for mini_formatter to copy beam parameters and related data
    to the primary image header from various other headers in Herschel data.
  * IMPORTANT NOTE: this function relies on 'resolns.tbl', a file containing a
    table of beam dimensions and position angles for PACS images dependent on 
    the scan speed and wavelength (the file originally also contained data for
    SPIRE, but the small radial asymmetries ended up not being worth the hassle
    for my purposes).

>>> flux_norm()
    Helper function that converts flux data to one of a handful of standard
    units (MJy/sr, Jy/beam, Jy/px, or Jy/asec^2) and edits the fits header to
    match (note: header and data must be extracted and input separately)

>>> mini_formatter()
    Top-level function that handles the standardization of fits headers, data
    units, and invalid pixel handling, and writes Herschel uncertainty maps to
    separate files.

>>> convolver()
    Quick and dirty function for convolving images to your choice of resolution
    in arcsec (default = 37, to match SPIRE 500 micron images); note that it
    does NOT do any resampling.

>>> cropper()
    Quick and dirty function for cropping images around target coordinates to
    squares of a specified width in arcsec 

"""

import numpy as np
import glob, re, math, sys
from astropy import wcs, units
from astropy.io import fits as pf
from astropy.coordinates import SkyCoord as skyc
from collections import defaultdict


def obj_img_match(coords, flist, name_convention='skycoord', list_unused=False,
                  threshold=0., flag=0., ignore_invalid=False, print_invalid=False):
    '''
    Takes a dict or list of SkyCoord objects, compares them to a list of fits
    files, and returns a dict where keys are object names/numbers and values 
    are lists of all fits files with valid data at that object's location.
    
    Inputs
    -------
    coords : dict or list
        Dict of object names and their associated coordinates converted to 
        Skycoord objects, or just a list of the Skycoord objects
    flist : list
        fits files in which to search for objects given in coords
    name_convention : (optional) string
        if coords is a list, specify how to construct catalog
        names to use as keys in the return dict. Characters must be legal
        for dict keys (surprisingly, +/- and . are allowed).
        Default 'skycoord' means program will convert input coordinates for
        each object into 16-character strings. Otherwise, you can specify an 
        arbritrary character string to append numbers to in the order of coords.
    list_unused : bool
        If True, returned dict will have an extra key-value pair listing all
        files that contained no objects with valid data under the key 'unused'.
    threshold : float
        Any data less than or equal to this value are considered invalid.
        Default is 0. To include 0 and negative numbers, set to -np.Inf.
    flag : float or int
        Invalid data fill value to check for. Data exactly equal to this value 
        are considered invalid.
        Default is 0, but -1 is also common and can easily be subbed in.
    ignore_invalid : bool
        If True, check only if footprints of images in flist contain objects
        in coords (not recommended unless you know that your images have no
        padding & invalid data are confined to a few isolated pixels).
        By default, obj_img_match only adds an image as a match for an object
        if data at the object's location are finite, >threshold, and !=flag.
    print_invalid : bool
        If True, print a notice when an image's footprint contains an object
        but data at that location are invalid according to filters above
                
    Returns
    -------
    output : dict
        A dict of object designations paired with lists of all files containing
        valid data at those objects' locations. 
    '''
    
    redic = defaultdict(list)
    if type(coords)==list:
        if name_convention=='skycoord':
            names = []
            for c in coords:
                if ('ICRS' or 'FK') in str(c.frame):
                    c1 = re.sub('\s|[a-z]','',c.to_string('hmsdms')).split('.')
                    c2 = 'J'+c1[0]+c1[1][-7:]
                elif ('Gal' or 'gal') in str(c.frame):
                    c1 = c.to_string()
                    c2 = c1.replace(' ','' if '-' in c1 else '+')
                names.append(c2)
        else:
            names = [name_convention+str(i+1) for i in range(len(coords))]
        cdic=dict(zip(names,coords))
    elif type(coords) is dict:
        cdic=coords
        
    for f in flist:
        HDU = pf.open(f)
        hdr = HDU[0].header
        w = wcs.WCS(hdr)
        ticker = len(cdic)
        
        try: 
            for k,i in cdic.items():
                x,y=np.squeeze(i.to_pixel(w))
                if i.contained_by(w):
                    dat = HDU[0].data
                    px,py = int(round(y)),int(round(x))
                    if (ignore_invalid is False and np.isfinite(dat[px,py]) and
                        dat[px,py]>threshold and dat[px,py]!=flag) or ignore_invalid is True:
                        redic[k].append(f)
                    else:
                        ticker-=1
                else:
                    ticker-=1
            if ticker==0 and list_unused is True:
                redic['unused'].append(f)
        except AttributeError:
            return 'Error: check format of coords'
        
    return redic

def set_beam(hd,fname): #hd = header, fname=Herschel file name
''' Helper function for mini_formatter to copy beam parameters and related data
    to the primary image header from various other headers in Herschel data.
  * IMPORTANT NOTE: this function relies on 'resolns.tbl', a file containing a
    table of beam dimensions and position angles for PACS images dependent on 
    the scan speed and wavelength (the file originally also contained data for
    SPIRE, but the small radial asymmetries ended up not being worth the hassle
    for my purposes).
'''
    if 'MAP' in fname:
        pacsbm = np.genfromtxt('resolns.tbl',dtype=None,names=True,encoding='ascii',
                               missing_values='-',filling_values=0)
        #pacspa = np.genfromtxt('PACS_PAsByFile.tbl',dtype=None,names=True,missing_values='nan')
    	#go look at http://docs.astropy.org/en/stable/api/astropy.coordinates.SkyCoord.html#astropy.coordinates.SkyCoord.position_angle
        om = 'Para' if 'Para' in hd['INSTMODE'] else hd['SCANSP']
        if 'MAPR' in fname:
            band='R'
        elif 'MAPB' in fname:
            band='B'
        elif 'MAPG' in fname:
            band='G'
        try:
            for i in range(len(pacsbm)):
                if pacsbm['band'][i]==band and str(pacsbm['mode'][i]) in om:
                    x = i
	    #names including '.' are legal but may cause issues later; genfromtxt uses NameValidator()
        except IndexError:
            print(fname, ': ', om, ' mode not recognized. Aborting.')
            sys.exit()
        hd.set('BMAJ',pacsbm['bmaj'][x]/3600,'FWHM in deg')
        hd.set('BMIN',pacsbm['bmin'][x]/3600,'FWHM in deg',after='BMAJ')
        hd.set('BPA',round(pacsbm['pa'][x],1),'beam PA, deg (Equatorial)',after='BMIN')
	#numbers in {} [before ':'] are indices in args of format
	#(required if there are alphanumeric codes after ':' for at least 1 arg in .format)
        #hd.set('PAUNIT','deg    ',after='BPA')
    elif 'spire' in fname: #NOTE: you don't want to use anything with 'psrc' or 'diag' in the name
        #idk why but this loop was originally inserting something in the SPIRE beam that was not BMAJ or BMIN
        if 'plw' in fname:
            spbm = 35.4/3600.
        elif 'pmw' in fname:
            spbm = 24.2/3600.
        elif 'psw' in fname:
            spbm = 17.9/3600.
        hd.set('BMAJ',spbm,'FWHM in deg')
        hd.set('BMIN',spbm,'FWHM in deg', after='BMAJ')
        hd.set('BPA',0.0,'beam PA, deg (SPIRE beam has <10% ellipticity)', after='BMIN')
        #hd.set('PAUNIT','deg    ',after='BPA')
#    else:
#        for k,v in fwhms.items():
#            if k in fname:
#                if 'BMAJ' not in hd.keys():
#                    hd.set('BMAJ',v/3600,'FWHM in deg')
#                    hd.set('BMIN',v/3600,'FWHM in deg', after='BMAJ')
#                    hd.set('BPA',0.0,'beam PA (always 0 if symmetric)', after='BMIN')
#                    hd.set('PAUNIT','deg    ',after='BPA')
#                elif hd['BMAJ']>0.6:
#                    #replaces values not already in deg
#                    hd.set('BMAJ',v/3600,'FWHM in deg')
#                    hd.set('BMIN',v/3600,'FWHM in deg', after='BMAJ')
#                if 'BPA' not in hd.keys(): #catch anything missed by above if-elif statement
#                    hd.set('BPA',0.0,'beam PA (always 0 if symmetric)', after='BMIN')
#                    hd.set('PAUNIT','deg    ',after='BPA')
    return 'bmaj: {0:0.8f}, bmin: {1:0.8f}, bpa:  {2:0.1f}'.format(hd['BMAJ'],hd['BMIN'],hd['BPA'])

#-----------------------------------------------------------

def flux_norm(hdr,dat,omega,beam_info=None): #MUST BE RUN AFTER SETBEAM() if using Herschel data
''' Helper function that converts flux data to one of a handful of standard
    units (MJy/sr, Jy/beam, Jy/px, or Jy/asec^2) and edits the fits header to
    match (note: header and data must be extracted and input separately)

    omega = str, angular area unit; use 'beam' to output in Jy/beam, 
    sr to output in MJy/sr, & asec or arcsec to output in Jy/arcsec^2
'''
    #see http://irsa.ipac.caltech.edu/data/SPITZER/docs/spitzermission/missionoverview/spitzertelescopehandbook/19/
    # and maybe http://newton.cx/~peter/2011/12/reference-the-ultimate-resolved-source-cheatsheet/
    hd=hdr.copy()
    bunit = hd['BUNIT']
    if (('Jy/beam' in omega) or ('Jy/bm' in omega)):
        if beam_info is None and 'BMAJ' not in hd.keys():
            raise ValueError('Need beam parameters to convert to Jy/beam')
        elif len(beam_info)==1 and 'BMAJ' not in hd.keys():
                hd.set('BMAJ',np.squeeze(beam_info)/3600,'FWHM in deg')
                hd.set('BMIN',np.squeeze(beam_info)/3600,'FWHM in deg',after='BMAJ')
                hd.set('BPA',0.0,'beam PA, deg',after='BMIN')
        elif len(beam_info)==2 and 'BMIN' not in hd.keys():
                hd.set('BMAJ',beam_info[0]/3600,'FWHM in deg')
                hd.set('BMIN',beam_info[1]/3600,'FWHM in deg',after='BMAJ')
                hd.set('BPA',0.0,'beam PA, deg',after='BMIN')   
        elif len(beam_info)==3 and 'BMAJ' not in hd.keys():
                hd.set('BMAJ',beam_info[0]/3600,'FWHM in deg')
                hd.set('BMIN',beam_info[1]/3600,'FWHM in deg',after='BMAJ')
                hd.set('BPA',beam_info[2] if 'BPA' not in hd.keys() else hd['BPA'],
                       'beam PA, deg',after='BMIN')        
            
    if omega in bunit:
        print('flux already normalized')
        return hd,dat

    else: #convert everything to Jy/deg^2 first; Jy/deg^2 is a nice neutral unit
        if 'QTTY' in str(hd):
            hd.remove('QTTY____')
            #not sure but if I get rid of it & the problem w/ Miriad goes away, that could've been confusion source

        if 'CDELT1' in hd.keys():
            x,y=float(hd['CDELT1']),float(hd['CDELT2']) #pixel dims (x,y) in deg
        else: #x,y should be in deg by the end
            if not np.isclose(float(hd['CD2_1']),0.0,rtol=1e-06):
                cd11=math.radians(float(hd['CD1_1']))
                #cd12=math.radians(float(hd['CD1_2']))
                cd21=math.radians(float(hd['CD2_1']))
                #cd22=math.radians(float(hd['CD2_2']))
                theta=math.atan(cd21/cd11)
                x=( float(hd['CD2_1']) - float(hd['CD1_1']) )/( math.cos(theta) - math.sin(theta) )
                y=( float(hd['CD1_2']) + float(hd['CD2_2']) )/( math.cos(theta) - math.sin(theta) )
            else:
                x=float(hd['CD1_1'])
                y=float(hd['CD2_2'])

        pixa = abs(x*y) #deg^2
#        if x > hd['BMAJ']:
#            raise ValueError('Error: pixel size greater than beam fwhm')
#            sys.exit()

        if (any(s in bunit for s in ('Jy/pixel', 'Jy/pix', 'Jy/px')) and
            not any(s in omega for s in ('Jy/pixel', 'Jy/pix', 'Jy/px'))): #this is HPACS
            djy = dat / pixa

        elif 'MJy' in bunit and 'MJy' not in omega: #this is HSPIRE, Glimpse, & MIPS
            djy = dat * 10.**6 * ( (np.pi/180.)**2 )

        elif (('Jy/beam' in bunit) or ('Jy/bm' in bunit)) and not (('Jy/beam' in omega) or ('Jy/bm' in omega)): #this is ATLASGAL
            #after setbeam(), bmaj, bmin should be in degrees
            #Fix according to http://herschel.esac.esa.int/hcss-doc-13.0/load/spire_drg/html/ch06s09.html ?
            try:
                ba =  np.pi * float(hd['BMAJ']) * float(hd['BMIN']) / ( 4*math.log(2) )
            except KeyError:
                try:
                    ba =  np.pi * float(hd['B_EFF']) **2 / ( 4*math.log(2) )
                except KeyError:
                    sys.exit('beam parameters not found')
            djy = dat / ba

        elif 'DN' in bunit and ('WISE' or 'wise') in str(hd): #this is WISE
            dn_to_jy = {1:1.9350E-06,2:2.7048E-06,3:1.8326e-06,4:5.2269E-05} #DN/pixel-->Jy/pixel
            beam_area = {1:  6.08   * 5.60 * np.pi / (4*np.log(2)),
	    	     2:  6.84   * 6.12 * np.pi / (4*np.log(2)),
	    	     3:  7.36   * 6.08 * np.pi / (4*np.log(2)),
	    	     4:  11.99  * 11.65* np.pi / (4*np.log(2))} #sq arcsec
            #bmaj_min_pa = {1:(6.08,5.60,3),2:(6.84,6.12,15),3:(7.36,6.08,6),4:(11.99,11.65,0)}
	        #bmaj/bmin in arcsec (change at addition of OMEGABM), bpa in deg
    	    #PSFs in .csh code assume azimuthally averaged FWHMs, but that should be OK - the 2D PSFs are square anyway
            band = hd['BAND']
            hd.set('OMEGABM',beam_area[band]/(3600.**2),'beam area (deg^2)')
            hd.set('DNTOJY',dn_to_jy[band],'from http://wise2.ipac.caltech.edu/docs/release/prelim/expsup/sec2_3f.html')
            djy = dat * dn_to_jy[band] * float(hd['OMEGABM']) / pixa

        elif 'W/m^2-sr' in bunit and 'W/m^2-sr' not in omega: #this is MSX
            band = hd['BAND']
            Wm2_to_jy = {'A':7.133E+12,'C':2.863E+13,'D':3.216E+13,'E':2.476E+13}
            passband = {'A':'6.8 - 10.8','C':'11.1 - 13.2','D':'13.5 - 15.9','E':'18.2 - 25.1'}
            wl={'A':8.28E-06,'C':1.213E-05,'D':1.465E-05,'E':2.134E-05}
            hd.set('WAVELENG',wl[band],'central wavelength', after='BAND')
            hd.set('BANDFWHM',passband[band],'upper and lower FWHM limits', after='WAVELENG')
            hd.set('WM2TOJY',Wm2_to_jy[band],'W/m^2sr to Jy/sr conversion factor (see comments)',after='BUNIT')
            hd.set('COMMENT','radiance to flux conversion relies on isophotal assumption; adjust based on S_nu')
            djy = dat * Wm2_to_jy[band] * ( (np.pi/180.)**2 )

        else:
            raise ValueError('Unit not recognized')
            sys.exit()

    #By here BMAJ/MIN are in deg
        if 'sr' in omega:
            newdat = ( djy * 10.**-6 ) * (180./np.pi)**2
            hd.set('BUNIT','MJy/sr')
        elif ('arcsec' in omega) or ('asec' in omega):
            newdat = djy / 3600.**2
            hd.set('BUNIT','Jy/asec^2')
        elif 'beam' in omega:
            if 'OMEGABM' in hd.tostring():
                newdat = djy * float(hd['OMEGABM'])
            else:
                bma = np.pi * float(hd['BMAJ']) * float(hd['BMIN']) / ( 4*math.log(2) )
                newdat = djy * bma
            hd.set('BUNIT','Jy/beam')
        elif ('px' in omega) or ('pix' in omega):
            newdat = djy * pixa
            hd.set('BUNIT','Jy/px')
        hd.set('HISTORY','flux density units converted from {}'.format(bunit),after='BUNIT')
        return hd,newdat

def mini_formatter(flist, omega='beam'):
'''
    Top-level function that handles the standardization of fits headers, data
    units, and invalid pixel handling, and writes Herschel uncertainty maps to
    separate files.

    flist : list of fits files to process (recommended to use glob)
    omega : str, angular area unit used to determine requested flux unit to
        convert to with flux_norm(); "beam" means the default unit is Jy/beam
'''
    cmap = [('level','lvl'),('30pxmp_',''),('25HPPJSMAP','jsm'),('25HPPUNIMAP','uni'),
            ('13422','id'),('plw145','_xR'),('pmw145','_xG'),('psw145','_xB'), ('.fits.gz','')]

    for fi in flist:
        print(fi)
        hdu_in=pf.open(fi)
        hdr,dat = hdu_in['image'].header, hdu_in['image'].data
        phdr = hdu_in[0].header #only used to copy necessary keywords to main header
        hdr.set('TELESCOP',phdr['TELESCOP'],'Name of telescope',after='META_0')
        hdr.set('INSTRUME',phdr['INSTRUME'],'Instrument attached to this product',after='TELESCOP')
        hdr.set('DESC',phdr['DESC'],'Name of this product',after='INSTRUME')
        hdr.set('CUSMODE',phdr['CUSMODE'],'Common Uplink System observation mode',after='DESC')
        hdr.set('INSTMODE',phdr['INSTMODE'],'Instrument Mode',after='DESC')
        if 'hpacs' in fi:
            hdr.set('OFFSET',phdr['META_8'],after='INSTRUME')
        if 'extd' in fi and 'Para' not in phdr['INSTMODE']:
            hdr.set('SCANSPD',phdr['SCANSPD'],phdr.comments['SCANSPD'],after='INSTMODE')
            hdr.set('SCANVEL',phdr['SCANVEL'],phdr.comments['SCANVEL'],after='INSTMODE')
        fi2=fi #copy
        for old,new in cmap:
            fi2 = fi2.replace(old,new) if old in fi2 else fi2
        seq = fi2.rsplit('\\')
        seq1=seq[-1].split('_')
        nm='_'.join([seq1[0],seq1[1],seq[0],seq1[-1][-5:]])
        try:
            edat = hdu_in['error'].data
        except KeyError:
            try:
                edat = hdu_in['stDev'].data
            except KeyError:
                edat = None
        hdu_in.close() #copies of hdr & data now in memory, don't need file open anymore
    	
    	#now for some weird gymnastics because some in-place modifications don't work.
        dat0=np.copy(dat) #built-in copy function doesn't cut it here
        hdr0=hdr.copy()
        set_beam(hdr0,fi) #hdr0 modified in-place
        hdr1,dat1 = flux_norm(hdr0,dat0,omega)
        #print('see if copying inside flux_norm worked: ',hdr0['bunit'])
        #it worked!
        uhdr1,udat1 = flux_norm(hdr0,edat,omega)
        uhdr1.set('QTTY','UNCERTAINTY',after='TELESCOP')
        check = float(hdr1['CDELT1']) if 'CDELT' in hdr1.tostring() else abs(float(hdr1['CD1_1']))
        if check > hdr1['BMAJ']:
            raise ValueError('Error: pixel size greater than beam fwhm')
            sys.exit()
        if (np.array(dat1)==np.array(dat0)).all() and 'flux density units converted from' in hdr.tostring():
    	    raise IOError('unit conversion failed to execute')
    	    sys.exit()
        else:
            #NaNs are useful since we're working in python
            dat1[np.where(dat1==0)] = np.NaN
            udat1[np.where(udat1==0)] = np.NaN            
            new_hdu = pf.PrimaryHDU(data = dat1,header = hdr1)
            pf.HDUList(new_hdu).writeto(fi[:11]+nm+'.fits',overwrite=True)
            if edat is not None:
                new_uhdu = pf.PrimaryHDU(data = udat1,header = uhdr1)
                pf.HDUList(new_uhdu).writeto(fi[:11]+nm+'_err.fits',overwrite=True)               
            print('New header & data written to {}(_err).fits'.format(nm))
    return None

from astropy.convolution import Gaussian2DKernel, convolve

def convolver(flist, fwhm=37.):
'''
    Quick and dirty function for convolving images to your choice of resolution
    in arcsec (default = 37, to match SPIRE 500 micron images); note that it
    does NOT do any resampling.

    flist : list of fits files to process (glob pattern recommended)
    fwhm : float, FWHM in arcsec of the desired final beam size.
'''
    for fi in flist:
        hdu_in=pf.open(fi)
        hdr,dat = hdu_in[0].header, hdu_in[0].data
        hdu_in.close()
        sigma=(fwhm/2.35482) / abs(float(hdr['cdelt2'])*3600.)
        kern=Gaussian2DKernel(sigma)
        cim = convolve(dat,kern)
        hdr.set('BMAJ',fwhm/3600.,'deg')
        hdr.set('BMIN',fwhm/3600.,'deg')
        hdr['BPA']=0.0
        newf = fi.split('.fits')[0]+'_conv.fits'
        newhdu=pf.PrimaryHDU(header=hdr,data=cim)
        newhdu.writeto(newf,overwrite=True)
        print('Wrote convolved image to '+newf)
    return None
     
from astropy.nddata import Cutout2D
def cropper(flist,skycoords,alphw,deltw,rname=None,frame='ICRS'):
'''
    Quick and dirty function for cropping images around target coordinates to
    squares of a specified width in arcsec

    flist : list of fits files to process (glob pattern recommended)
    skycoords : SkyCoord object, coordinates at which to center the cutout
    alphaw, deltaw : full widths of the cutouts in arcsec, (names imply that
        axes are aligned with ICRS coordinate system, but that needn't be true)
    rname : string to insert into output file name, which is constructed using
        the input file name, to distinguish 2+ cutouts from the same image
    frame : str, coordinate system to convert the resulting image to (does not
        rotate the image!)
'''
    for f in flist:
        hdu=pf.open(f)
        hdr,dat=hdu[0].header,hdu[0].data  
        cards = list(hdr.keys())
        if 'DEC' in hdr['CTYPE2']:
            frame = 'ICRS' if 'ICRS' in str(hdr) else 'FK5'
            if not ('RADESYS' or 'RADECSYS') in cards:
                hdu[0].header.set('RADESYS',frame,
                   after='EQUINOX' if 'EQUINOX' in cards else 'EPOCH')
        elif 'GLON' in str(hdr):
            frame = 'Galactic'
        
        w=wcs.WCS(hdr)
        cutout=Cutout2D(dat,skycoords,(alphw*units.arcsec,deltw*units.arcsec),wcs=w)
        h=cutout.wcs.to_header()
        hdr['naxis2'],hdr['naxis1']=cutout.data.shape
        hdr['crpix1']=h['crpix1']
        hdr['crpix2']=h['crpix2']
        hdr['crval1']=h['crval1']
        hdr['crval2']=h['crval2']
        hdr.set('LONPOLE',h['lonpole'],after='EQUINOX' if 'EQUINOX' in cards else 'EPOCH')
        hdr.set('LATPOLE',h['latpole'],after='LONPOLE')
        newhdu = pf.PrimaryHDU(data=cutout.data,header=hdr)
        newhdu.writeto(f.split('.fits')[0]+'_cutout{}.fits'.format(rname),overwrite=True)

        try:
            uhdu=pf.open(f.split('interp.')[0]+'err.fits' if 'interp' in f else f.split('.')[0]+'_err.fits')
            uhdu[0].header.set('RADESYS',frame, after='EQUINOX')
            ucutout=Cutout2D(uhdu[0].data,skycoords,(alphw*units.arcsec,deltw*units.arcsec),wcs=w)        
            newuhdu=pf.PrimaryHDU(data=ucutout.data,header=hdr)    
            newuhdu.writeto(f.split('.fits')[0]+'_cutout{}_err.fits'.format(rname),overwrite=True)
        except FileNotFoundError:
            pass
        
        print('wrote '+f.split('.fits')[0]+'_cutout{}.fits'.format(rname))            
            
    return '...done.'

