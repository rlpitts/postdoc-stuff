'''
Gaussian decomposition code from apphotcyg.py cleaned up for 1st year students.
This one has an additional 7-component Gaussian function for N48.

I was not familiar with most functools modules when I made this. I'm sure there's a
better way to automate spawning of new Gaussian fits with partial() or decorators

Requires a number of research-generated files to work: 
- pilscygx-sma-ds9.reg (region file that describes circles around sources as seen by SMA) or
	pilscygx-ellps-ds9.reg (describes ellipses around sources as seen by Herschel)
- formerly proprietary science images of PILS-Cygnus objects with the SMA telescope, or
	appropriately-sized cutouts of Herschel images
This code has been tested both on SMA data and cutouts of Herschel data, with changes to the
initial specs of the Gaussians for the latter.
'''
import numpy as np
import glob#, math, sys
from astropy import wcs, units
from astropy.io import fits as pf
from astropy.coordinates import SkyCoord as skyc
#import photutils as pu
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit as cfit
from scipy.optimize import OptimizeWarning
import scipy.interpolate as terp
import scipy.integrate as integ
#import natconst as nc
cc  = 2.9979e10      # Light speed             [cm/s]
hh  = 6.6262e-27     # Planck's constant       [erg.s]
AU  = 1.496e13       # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
MS  = 1.99e33        # Solar mass              [g]
kk  = 1.3807e-16     # Bolzmann's constant     [erg/K]

flist = glob.glob('Pils-Cygnus/*/*final*.fits')

ra,dec, fwhms, names=np.genfromtxt('pilscygx-sma-ds9.reg', usecols=[1,2,3,4], dtype=None, unpack=True, comments=None,
				converters={3: lambda s: str(s).replace('d',''), 4: lambda s: str(s).replace('#','')},
				encoding='ascii')

cygcdmain = dict(zip(names,[skyc(ra[i],dec[i],frame='icrs') for i in range(len(ra))]))
cygdbg={'N12':0.0039,'N30':0.021,'N38':0.015,'N48':0.010,'N51':0.011,
        'N53':0.007,'N54':0.004,'N63':0.0083,'S8':0.007,'S26':0.0064}

def find_kappa(wavlen,mantle='thin',densexp=6):
    if mantle not in ['bare','thin','thick'] or not (5<=densexp<=8):
        return 'Error: could not find dust opacity table; check inputs.'
    wav,dop=np.loadtxt('dustoptabs/mrn{}{}.txt'.format(mantle,str(densexp)),
                       unpack=True)
    return np.interp(wavlen,wav,dop)

def planck(nu,T):
    return (2. * nu**3 * hh / (cc**2)) / (np.exp(hh * nu / (T * kk)) - 1.)

def calc_M870um(T, intflux, dist=1700., kappa=1.8896):
    '''T in [K], intflux in [Jy], dist in pc,
    kappa in cgs units (default assumes density of 1e-6 & thin ice mantles)'''
    S = intflux*1e-23
    BT = planck(cc/870e-4,T)
    D = dist * pc
    return S * 100. * D**2 / (kappa * BT * 1.99e33)

def gauss2D(x, y, xm, ym, major, minor, amplitude, pa):
    """
    Copied from AstroLyze
    Calculates a 2D Gaussian at position x y.
    
    Parameters
    ----------
    x : float or numpy.ndarray
        the x-axis value/values where the Gaussian is to be caluclated.
    y : float or numpy.ndarray
        the y-axis value/values where the Gaussian is to be calculated.

    major, minor : float
        The standard deviation of the Gaussian in x and y direction.
    pa : float
        The position angle of the Gaussian in degrees w.r.t. y-axis of array
    xm, ym:
        The Gaussian mean / offset in x and y direction from 0.
    amplitude :
        The height of the Gaussian.

    Returns
    -------
    gauss : float or np.ndarray
        The y value for the specified Gaussian distribution evaluated at x.

    Notes
    -----
    The function used to describe the Gaussian is :

        f = (amplitude * exp (-1 (a*(x-xm)^2 + 2*b*(x-xm)*(y-ym)
            + c*(y-ym)^2)))

    where:

        a = cos(pa)**2/(2*major**2) + sin(pa)**2/(2*minor**2) \\
        b = (-1*sin(2*pa)/(4*major**2))+(sin(2*pa)/(4*minor**2)) \\
        c = sin(pa)**2/(2*major**2) + cos(pa)**2/(2*minor**2) \\
    """
    pa = pa * np.pi / 180.
    a = np.cos(pa) ** 2 / (2. * major ** 2) + np.sin(pa) ** 2 / (2. * minor ** 2)
    b = ((-1. * np.sin(2 * pa) / (4. * major ** 2)) + (np.sin(2. * pa) / (4 * minor **
         2)))
    c = np.sin(pa) ** 2 / (2. * major ** 2) + np.cos(pa) ** 2 / (2. * minor ** 2)
    gauss = a * (x - xm) ** 2
    gauss += 2. * b * (x - xm) * (y - ym)
    gauss += c * (y - ym) ** 2
    gauss = np.exp(-1 * gauss)
    gauss = amplitude * gauss
    return gauss

def singgauss2d(V, rapx, depx, Maj, Min, A, R): #, B
    y,x=V
    return gauss2D(x,y,rapx,depx,Maj,Min,A,R).ravel()   
    
def doublegauss(V,rapx1,depx1,maj1,min1,A1,rot1,
                  rapx2,depx2,maj2,min2,A2,rot2):
    y,x=V
    Avec = [A1,A2]
    ravec = [rapx1,rapx2]
    devec = [depx1,depx2]
    majvec = [maj1,maj2]
    minvec = [min1,min2]
    rotvec = [rot1,rot2]
    return np.sum([gauss2D(x,y,ravec[i],devec[i],majvec[i],minvec[i],
                           Avec[i],rotvec[i]) for i in range(len(Avec))],axis=0).ravel()

def tripgauss(V,rapx1,depx1,maj1,min1,A1,rot1,
                rapx2,depx2,maj2,min2,A2,rot2,
                rapx3,depx3,maj3,min3,A3,rot3):
    y,x=V
    Avec = [A1,A2,A3]
    ravec = [rapx1,rapx2,rapx3]
    devec = [depx1,depx2,depx3]
    majvec = [maj1,maj2,maj3]
    minvec = [min1,min2,min3]
    rotvec = [rot1,rot2,rot3]
    return np.sum([gauss2D(x,y,ravec[i],devec[i],majvec[i],minvec[i],
                           Avec[i],rotvec[i]) for i in range(len(Avec))],axis=0).ravel()

def quadgauss(V,rapx1,depx1,maj1,min1,A1,rot1,
                rapx2,depx2,maj2,min2,A2,rot2,
                rapx3,depx3,maj3,min3,A3,rot3,
                rapx4,depx4,maj4,min4,A4,rot4):
    y,x=V
    Avec = [A1,A2,A3,A4]
    ravec = [rapx1,rapx2,rapx3,rapx4]
    devec = [depx1,depx2,depx3,depx4]
    majvec = [maj1,maj2,maj3,maj4]
    minvec = [min1,min2,min3,min4]
    rotvec = [rot1,rot2,rot3,rot4]
    return np.sum([gauss2D(x,y,ravec[i],devec[i],majvec[i],minvec[i],
                           Avec[i],rotvec[i]) for i in range(len(Avec))],axis=0).ravel()

#def quingauss(V,rapx1,depx1,maj1,min1,A1,rot1,
#                rapx2,depx2,maj2,min2,A2,rot2,
#                rapx3,depx3,maj3,min3,A3,rot3,
#                rapx4,depx4,maj4,min4,A4,rot4,
#                rapx5,depx5,maj5,min5,A5,rot5):
#    y,x=V
#    Avec = [A1,A2,A3,A4,A5]
#    ravec = [rapx1,rapx2,rapx3,rapx4,rapx5]
#    devec = [depx1,depx2,depx3,depx4,depx5]
#    majvec = [maj1,maj2,maj3,maj4,maj5]
#    minvec = [min1,min2,min3,min4,min5]
#    rotvec = [rot1,rot2,rot3,rot4,rot5]
#    return np.sum([gauss2D(x,y,ravec[i],devec[i],majvec[i],minvec[i],
#                           Avec[i],rotvec[i]) for i in range(len(Avec))],axis=0).ravel()
    
def septgauss(V,rapx1,depx1,maj1,min1,A1,rot1,
                rapx2,depx2,maj2,min2,A2,rot2,
                rapx3,depx3,maj3,min3,A3,rot3,
                rapx4,depx4,maj4,min4,A4,rot4,
                rapx5,depx5,maj5,min5,A5,rot5,
                rapx6,depx6,maj6,min6,A6,rot6,
                rapx7,depx7,maj7,min7,A7,rot7,):
    y,x=V
    Avec = [A1,A2,A3,A4,A5,A6,A7]
    ravec = [rapx1,rapx2,rapx3,rapx4,rapx5,rapx6,rapx7]
    devec = [depx1,depx2,depx3,depx4,depx5,depx6,depx7]
    majvec = [maj1,maj2,maj3,maj4,maj5,maj6,maj7]
    minvec = [min1,min2,min3,min4,min5,min6,min7]
    rotvec = [rot1,rot2,rot3,rot4,rot5,rot6,rot7]
    return np.sum([gauss2D(x,y,ravec[i],devec[i],majvec[i],minvec[i],
                           Avec[i],rotvec[i]) for i in range(len(Avec))],axis=0).ravel()

def fit_cyggauss(img,obj=None,pshow=False):
    if obj is None:
        obj=img.split('/')[1][4:]
    resdic = {} # result dictionary
    HDU = pf.open(img)
    dat = np.squeeze(HDU[0].data)
    hdr = HDU[0].header
    if 'NAXIS3' in hdr.keys():
        del hdr['PC0*']
        hdr['NAXIS'] = 2
        for i in [3,4]:
            del hdr['NAXIS'+str(i)]
            del hdr['CTYPE'+str(i)]
            del hdr['CRVAL'+str(i)]
            del hdr['CDELT'+str(i)]
            del hdr['CRPIX'+str(i)]
            del hdr['CUNIT'+str(i)]
    w = wcs.WCS(hdr)
    olist = [k for k in cygcdmain.keys() if k.startswith(obj)]
    print(olist)
    ngauss = len(olist)
    
    cpd = dict((k,np.squeeze(i.to_pixel(w))) for (k,i) in cygcdmain.items() if i.contained_by(w))
    print(cpd)
    if 'CD2_1' in hdr.keys():
        crota2=np.arctan(float(hdr['cd2_1'])/float(hdr['cd1_1']))*180./np.pi
        cd1=float(hdr['cd1_1'])/np.cos(crota2)
        cd2=float(hdr['cd2_2'])/np.cos(crota2)
        cd = np.mean((abs(cd1),abs(cd2))) #cdelts should be basically equal
    else:
        try:
            crota2=float(hdr['crota2'])
        except KeyError:
            try:
                crota2=float(hdr['PA'])
            except KeyError:
                crota2=0.0
        cd1=float(hdr['cdelt1'])
        cd2=float(hdr['cdelt2'])
        cd = np.mean((abs(cd1),abs(cd2)))
    bmaj = hdr['BMAJ'] / cd
    bmin = hdr['BMIN'] / cd
    #bpa = hdr['bpa']+crota2#pa is w.r.t. y axis of array, not of WCS

    #convert data to Jy/px
    pxa = cd**2
    bma = np.pi*hdr['BMAJ']*hdr['BMIN']/(4*np.log(2))
    newdat = dat * pxa/bma
    bglvl=cygdbg[obj] * pxa/bma
    mask = np.where(newdat>5.*bglvl) #don't need unit coversion for indices
    
    raps = np.array([cpd[key][0] for key in olist])
    deps = np.array([cpd[key][1] for key in olist])
    fwhms_maj=bmaj*np.ones(len(olist))
    fwhms_min=bmin*np.ones(len(olist))

    nax1,nax2 = np.shape(dat)
    a,d = np.mgrid[0:nax1, 0:nax2]
    
    if ngauss==1:
        bounds =[[raps[0]-5.,deps[0]-5.,0.,0.,np.nanmin(newdat),0.],
                 [raps[0]+5,deps[0]+5,2.*fwhms_maj[0],2.*fwhms_min[0],np.nanmax(newdat),180.]]
        mgp,mgcov = cfit(singgauss2d,
                         (a[mask].ravel(),d[mask].ravel()),
                         newdat[mask].ravel(), bounds=bounds, #p0=p0, sigma=newu.ravel()
                         sigma=bglvl/newdat[mask].ravel())
        resdic['model'] = singgauss2d((a, d), *mgp)

    elif ngauss==2:
        bounds =[[raps[0]-2.,deps[0]-2.,0.,0.,np.nanmin(newdat),0.,
                  raps[1]-2,deps[1]-2,0.,0.,np.nanmin(newdat),0.],
                 [raps[0]+2,deps[0]+2,fwhms_maj[0],fwhms_min[0],np.nanmax(newdat),180.,
                  raps[1]+2,deps[1]+2,fwhms_maj[1],fwhms_min[1],np.nanmax(newdat),180.]]
        mgp,mgcov = cfit(doublegauss,(a[mask].ravel(),d[mask].ravel()),
                     newdat[mask].ravel(), bounds=bounds,
                     sigma=bglvl/newdat[mask].ravel())
        resdic['model'] = doublegauss((a, d), *mgp)#.reshape(a.shape[0], d.shape[1])

    elif ngauss==3:
        bounds =[[raps[0]-2.,deps[0]-2.,0.5,0.5,np.nanmin(newdat),0.,
                  raps[1]-2,deps[1]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[2]-2,deps[2]-2,0.5,0.5,np.nanmin(newdat),0.0],
                 [raps[0]+2,deps[0]+2,fwhms_maj[0],fwhms_min[0],np.nanmax(newdat),180.,
                  raps[1]+2,deps[1]+2,fwhms_maj[1],fwhms_min[1],np.nanmax(newdat),180.,
                  raps[2]+2,deps[2]+2,fwhms_maj[2],fwhms_min[2],np.nanmax(newdat),180.]]
        mgp,mgcov = cfit(tripgauss,(a[mask].ravel(),d[mask].ravel()),
                     newdat[mask].ravel(), bounds=bounds,
                     sigma=bglvl/newdat[mask].ravel())
        resdic['model'] = tripgauss((a, d), *mgp)#.reshape(a.shape[0], d.shape[1])
        
    elif ngauss==4:
        bounds =[[raps[0]-2,deps[0]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[1]-2,deps[1]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[2]-2,deps[2]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[3]-2,deps[3]-2,0.5,0.5,np.nanmin(newdat),0.],
                 [raps[0]+2,deps[0]+2,fwhms_maj[0],fwhms_min[0],np.nanmax(newdat),180.,
                  raps[1]+2,deps[1]+2,fwhms_maj[1],fwhms_min[1],np.nanmax(newdat),180.,
                  raps[2]+2,deps[2]+2,fwhms_maj[2],fwhms_min[2],np.nanmax(newdat),180.,
                  raps[3]+2,deps[3]+2,fwhms_maj[3],fwhms_min[3],np.nanmax(newdat),180.]]
        mgp,mgcov = cfit(quadgauss,(a[mask].ravel(),d[mask].ravel()),
                     newdat[mask].ravel(),  bounds=bounds,
                     sigma=bglvl/newdat[mask].ravel())
        resdic['model'] = quadgauss((a, d), *mgp)#.reshape(a.shape[0], d.shape[1])

    elif ngauss>4:
        bounds =[[raps[0]-2,deps[0]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[1]-2,deps[1]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[2]-2,deps[2]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[3]-2,deps[3]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[4]-2,deps[4]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[5]-2,deps[5]-2,0.5,0.5,np.nanmin(newdat),0.,
                  raps[6]-2,deps[6]-2,0.5,0.5,np.nanmin(newdat),0.,
                 ],
                 [raps[0]+2,deps[0]+2,fwhms_maj[0],fwhms_min[0],np.nanmax(newdat),180.,
                  raps[1]+2,deps[1]+2,fwhms_maj[1],fwhms_min[1],np.nanmax(newdat),180.,
                  raps[2]+2,deps[2]+2,fwhms_maj[2],fwhms_min[2],np.nanmax(newdat),180.,
                  raps[3]+2,deps[3]+2,fwhms_maj[3],fwhms_min[3],np.nanmax(newdat),180.,
                  raps[4]+2,deps[4]+2,fwhms_maj[4],fwhms_min[4],np.nanmax(newdat),180.,
                  raps[5]+2,deps[5]+2,fwhms_maj[5],fwhms_min[5],np.nanmax(newdat),180.,
                  raps[6]+2,deps[6]+2,fwhms_maj[6],fwhms_min[6],np.nanmax(newdat),180.]]
        mgp,mgcov = cfit(septgauss,(a[mask].ravel(),d[mask].ravel()),
                     newdat[mask].ravel(),  bounds=bounds,
                     sigma=bglvl/newdat[mask].ravel())
        resdic['model'] = septgauss((a, d), *mgp)#.reshape(a.shape[0], d.shape[1])

    resdic['resid'] = newdat.ravel()-resdic['model']
    resdic['parms'] = mgp
    resdic['perrs'] = np.sqrt(np.diag(mgcov))
    resdic['rmsd'] = np.sqrt( np.nansum(abs(newdat.ravel()**2-resdic['model']**2)) )           
    if ngauss==1:#
        resdic['integ'] = integ.nquad(gauss2D,[[0,nax1],[0,nax2]],
                                      args=mgp)[0]
        
        print('Sum of single model gaussian over data: ',resdic['integ']/np.sum(newdat))
        resdic['interr'] = resdic['integ']*np.sqrt( (resdic['perrs'][2]/resdic['parms'][2])**2. +
                                                      (resdic['perrs'][3]/resdic['parms'][3])**2. +
                                                      (resdic['perrs'][4]/resdic['parms'][4])**2. )
    else:
        dummy = [integ.nquad(gauss2D,[[0,nax1],[0,nax2]],
                             args=(mgp[0+6*i:6+6*i]))[0] for i in range(len(olist)) ]
        resdic['integ'] = dict(zip(olist, dummy))
        dummy2 = [ resdic['integ'][olist[i]]*np.sqrt( (resdic['perrs'][2+6*i]/resdic['parms'][2+6*i])**2. +
                                                      (resdic['perrs'][3+6*i]/resdic['parms'][3+6*i])**2. +
                                                      (resdic['perrs'][4+6*i]/resdic['parms'][4+6*i])**2. )
                  for i in range(len(olist)) ]
        resdic['interr'] = dict(zip(olist, dummy2))

    if pshow is True:
        if len(olist)>2:
            title='{}-{}'.format(olist[0],olist[-1])
        elif len(olist)==2:
            title='{} & {}'.format(olist[0],olist[-1])
        else:
            title=olist[0]
        fig, [ax1, ax2, ax3] = plt.subplots(ncols=3, subplot_kw={'projection': w})
        dplot = ax1.imshow(newdat, origin='lower',cmap=mpl.cm.magma)
        gplot = ax2.imshow(resdic['model'].reshape(a.shape[0], d.shape[1]), origin='lower',cmap=mpl.cm.magma)
        gresplot = ax3.imshow(resdic['resid'].reshape(a.shape[0], d.shape[1]), origin='lower',cmap=mpl.cm.magma)
        plt.colorbar(dplot, ax=ax1)
        plt.colorbar(gplot, ax=ax2)
        plt.colorbar(gresplot, ax=ax3)
        ax1.set_title('Data (Jy/px), '+title)
        ax2.set_title('Multi-gauss fit, '+title)
        ax3.set_title('Residuals')
    print(img,': \n RMSD: ',resdic['rmsd'],
          '\n Integrals: ',resdic['integ'],
          '\n Uncertainties: ',resdic['interr'])
    return resdic
