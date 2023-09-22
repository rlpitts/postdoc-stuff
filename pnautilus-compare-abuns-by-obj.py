#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 15:50:59 2022

@author: pitts
"""

import os, re#, pdb
import numpy as np
import sys # to be able to retrieve arguments of the script
import pylab as pl
import matplotlib.ticker as ticks
from itertools import cycle
from scipy.integrate import trapz
lines = ["-","--",":","-."]

ccycler = cycle(['maroon','tab:red','tab:orange','gold','teal','cornflowerblue','b','indigo','k'])#, 'tab:gray'])

default_species_list='H2CS,OCS,SO,SO2,CS,CCS,C3S,CH3SH,NS,NCSH,HS,H2S,S,CH3OH,HCOOH,CH3CN,CH3OCH3,HCOOCH3,H2CO,HC3N,HNCO'

def abvT_by_obj(folders,species_list,T_min = None,T_max = None):
    if len(folders)>4:
        return 'This program can display chemical abundances for no more than 4 objects/simulations at once.'
    linecycler = cycle(lines[:len(folders)])
    species_name = species_list.replace(' ', '').split(',')
    fig = pl.figure()
    pl.clf()
    fig.subplots_adjust(left=0.12, bottom=0.1, right=0.96, top=0.95, wspace=0.26, hspace=0.26)
    plot_ab = fig.add_subplot(1, 1, 1)
    
    for j,f in enumerate(folders):
        oname = f.split('_')[-1]
        ls=next(linecycler) #idk why you have to do this once first.
        #ref_time = [] # time in years
        abundances = {} # Array containing Species abundances (relative to H) [number ratio].
        
        # For each species, we read the corresponding file and store its values
        for name in species_name:
            filename = "/home/pitts/pnautilus-1.1/{}/ab/{}.ab".format(f,name)
            #filePath = os.path.join(f,'ab',filename)          
            tmp_abundance = np.loadtxt(filename, skiprows=1, usecols=1, dtype=float)
            abundances[name] = tmp_abundance
        time, logAv, logn, logT = np.loadtxt('/home/pitts/pnautilus-1.1/{}/structure_evolution.dat'.format(f),
                                             comments='!',usecols=[0,1,2,3],unpack=True)
        #ref_time = time[1:]
        #Av = 10**logAv[1:]
        #ncol = 10**logn[1:]
        temp = 10**logT[1:]
        pl.gca().set_prop_cycle(None)
        cs=next(ccycler)
        for i,species in enumerate(species_name):
            pn = plot_ab.loglog(temp, abundances[species], label=oname+': '+species, linestyle=ls, color=cs)
            plot_ab.set_xlim(15.0 if T_min is None else T_min, 200.0 if T_max is None else T_max)
            #plot_ab.gca().invert_xaxis()
            plot_ab.set_xlabel("$T$ [K]",fontsize=14)
            # fx = interp1d(temp,ncol, fill_value='extrapolate')
            #plot_ab.legend(bbox_to_anchor=(1.04,1-j/len(folders)), loc="upper left",title=oname ) 
            plot_ab.legend(loc='upper left',bbox_to_anchor=(1.04,1))#,ncol=2
            if i<len(species_name)-1:
                cs=next(ccycler)
        # ax2=plot_ab.twiny()
        # ax2.set_xscale('log')
        # ax2.set_xlim(plot_ab.get_xlim())
        # ax2.minorticks_off()
        # # print(plot_ab.get_xticks())
        # ax2.set_xticks(plot_ab.get_xticks())
        # #ax2.set_xticks(newmins,minor=True)
        # ax2.set_xticklabels(['{:.1e}'.format(tic) for tic in fx(plot_ab.get_xticks())])
        # ax2.set_xlabel('$n$ [cm$^{-3}$]')
    plot_ab.tick_params(length=6, axis='both', labelsize=12)
    plot_ab.tick_params(length=3, which='minor', axis='y')
    plot_ab.set_ylabel("Abundance [X/H$_2$]",fontsize=14)
    plot_ab.yaxis.set_minor_locator(ticks.LogLocator(base = 10.0, subs='all'))
    plot_ab.grid(True)
    pl.show()
    return None

AU = 1.496e13
resarr_correct={'N12': {'menv': 130., 'munc':30., 'lum': 360., 'lunc':[40.,50.],
                        'plrho': -1.2, 'punc':[0.3,0.1], 'rout': 25000., 'dist':1400.},
                'N30': {'menv': 190., 'munc':50., 'lum': 14000.,'lunc':3000.,
                        'plrho': -1.0, 'punc':0.1, 'rout': 30000., 'dist':1400.},
                'N48': {'menv': 50., 'munc':10.,'lum': 950.,'lunc':200.,
                        'plrho': -0.6, 'punc':0, 'rout': 16000., 'dist':1400.},
                 'N51': {'menv': 90., 'munc':10.,'lum': 750.,'lunc':70.,
                         'plrho': -1.1,'punc':0.1,  'rout': 25000., 'dist':1400.},
                 'N53': {'menv': 240., 'munc':[40.,20.],'lum': 170.,'lunc':[20.,30.],
                         'plrho': -1.0,'punc':0.2, 'rout': 15000., 'dist':1400.},
                 'N63': {'menv': 80.,'munc':[30.,10.],'lum': 310.,'lunc':[40.,80.],
                         'plrho': -1.2,'punc':[0.3,0.2], 'rout': 25000., 'dist':1400.},
                 'S8': {'menv': 90.,'munc':10.,'lum': 3300.,'lunc':500.,
                        'plrho': -1.6,'punc':-1, 'rout': 40000., 'dist':1400.},
                 'S26': {'menv': 530.,'munc':[110.,150.],'lum': 60000.,'lunc':2000.,
                         'plrho': -1.3,'punc':0.1, 'rout': 80000., 'dist':3300.}}

import glob
def integrate_abunds(folder,species,rin_au=20.0,rout_au=None,sum_ices=False):
    nh, r  = np.loadtxt('{}/structure_evolution_{}.dat'.format(folder.split('/')[0],'Av5' if 'Av5' in folder else 'Av10'), 
                        usecols=[2,4], dtype=None, encoding='ascii', 
                        converters={2: lambda ln: 10**float(ln),
                                    4: lambda r: AU*10**float(r.replace('!',''))},
                       skiprows=3,unpack=True) #first row of data shouldn't be used^
    # Av = N(H2)/2.21e21 cm^-2
    # rreg = np.arange(min(r),max(r)+200*AU, 200*AU)
    # nhreg = np.interp(rreg,r[::-1],nh[::-1]) #interp can only handle x increasing
    rout=rout_au*AU if rout_au is not None else max(r)
    r=r[:len(glob.glob(folder+'/abundances*.out'))]
    nh=nh[1:len(glob.glob(folder+'/abundances*.out'))]
    # only count t=0 point for r, not n
    inds = np.where(np.logical_and(r[1:]>=rin_au*AU,r[1:]<rout))
    nhi=nh[inds]
    ra=np.insert(r[inds],0,rout)
    if type(species) is list:
        Ncol = {}
        for s in sorted(species):
            filename = "{}/ab/{}.ab".format(folder,s)
            abundance = np.loadtxt(filename, skiprows=1, usecols=1, dtype=float)
            # print(len(r[::-1]),len(abundance))
            # abreg = np.interp(rreg,r[1::-1],abundance[::-1])
            # Ncol[s] = 2*np.trapz(nhreg*abreg,x=rreg)
            ab=abundance[inds]
            # note that r includes the point at t=0, which is cut from all other arrays
            if sum_ices is True and s.replace('K','J') in list(Ncol.keys()):
                Ncol[s.replace('K','J')]+=2*np.pi*sum(ab[i]*nhi[i]*abs(ra[i+1]-ra[i]) for i in range(1,len(ab)))
            else:
                Ncol[s]=2*np.pi*sum(ab[i]*nhi[i]*abs(ra[i+1]-ra[i]) for i in range(1,len(ab)))
            ## quick and dirty, but should work
    else:
        abundance = np.loadtxt("{}/ab/{}.ab".format(folder,species),
                               skiprows=1, usecols=1, dtype=float)
        ab=abundance[inds]
        Ncol[s]=2*np.pi*sum(ab[i]*nhi[i]*abs(ra[i+1]-ra[i]) for i in range(len(ab)))
        # abreg = np.interp(rreg,r[1::-1],abundance[::-1])
        # Ncol = 2*np.trapz(nhreg*abreg,x=rreg)        
    # return 2*np.trapz(nhreg,x=rreg), Ncol
            
    ntot = 2*np.pi*sum(nhi[i]*abs(ra[i+1]-ra[i]) for i in range(len(ab)))
    return ntot, Ncol

def abunds2tex(obj,species_list=default_species_list,outfi='abunds_tex.txt',rout=2000.,
               comp_spec=None, get_ncol=False):
    if get_ncol is True and comp_spec is not None:
        return 'Error: Mutually exclusive kwargs. Unset either comp_spec or get_ncol.'
    if comp_spec is not None and comp_spec not in species_list.split(','):
        species_list+=','+comp_spec    
    fl=sorted(glob.glob('/home/pitts/pnautilus-1.1/S3phaseYSO_{}/ss1e6y*'.format(obj)))
    hdr = 'Species & Observed '
    for p in fl:
        s = '05' if 'Av5' in p else '10'
        s+= '3' if '1e3G0' in p else '0'
        s+= '1' if 'OCSedit' in p else '0'
        s+= '-16' if '4e-16' in p else '-17'
        hdr+='& {} '.format(s)
    hdr+='\\\\ \n'
    abunds={k:[] for k in species_list.split(',')}
    f=open('/home/pitts/pnautilus-1.1/S3phaseYSO_{}/{}{}'.format(obj,obj,outfi), 'w')
    f.write(hdr)
    for d in fl:#[len(fl)//2:]+fl[:len(fl)//2]:
        wdir = d.split('/home/pitts/pnautilus-1.1/')[1]
        nh,ncol = integrate_abunds(wdir, species_list.split(','), rout_au=rout)
        for k,nx in list(ncol.items()):
            if comp_spec is not None:
                abunds[k].append('{:.1e}'.format(nx/ncol[comp_spec]).replace('e-','\\times10^{-'))
            else:
                #newrow=['{:.1e}'.format(nx if get_ncol is True else nx/nh).replace(*sr) for
                #        sr in [['e-','\\times10^{-'],['-0','-'],['e+','\\times10^{+'],['+0','+']]]
                #abunds[k].append(newrow[0])
                abunds[k].append('{:.1e}}})'.format(nx/nh).replace('e-','(10^{-').replace('-0','-'))
    for k,v in abunds.items():
        f.write(k.replace('2','$_2$').replace('3','$_3$').replace('-0','-')+' & \t & $'+'}$ & $'.join(v)+'}$\\\\'+'\n') 
    f.close()
    return 'Wrote {}'.format('/home/pitts/pnautilus-1.1/S3phaseYSO_{}/{}{}'.format(obj,obj,outfi))
        
def abunds2ascii(obj,species_list=default_species_list,comp_spec=None,rout=2000.,
                 outfi='abunds_ascii.txt',get_ncol=False):
    if get_ncol is True and comp_spec is not None:
        return 'Error: Mutually exclusive kwargs. Unset either comp_spec or get_ncol.'
    if comp_spec is not None and comp_spec not in species_list.split(','):
        species_list+=comp_spec
    fl=sorted(glob.glob('/home/pitts/pnautilus-1.1/S3phaseYSO_{}/ss1e6y*'.format(obj)))
    hdr = 'Species'
    for p in fl:
        s = '05' if 'Av5' in p else '10'
        s+= '3' if '1e3G0' in p else '0'
        s+= '1' if 'OCSedit' in p else '0'
        s+= '-16' if '4e-16' in p else '-17'
        hdr+='\t{} '.format(s)
    hdr+='\n'
    abunds={k:[] for k in species_list.split(',')}
    f=open('/home/pitts/pnautilus-1.1/S3phaseYSO_{}/{}{}'.format(obj,obj,outfi), 'w')
    f.write(hdr)
    for d in fl:#[len(fl)//2:]+fl[:len(fl)//2]:
        wdir = d.split('/home/pitts/pnautilus-1.1/')[1]
        nh,ncol = integrate_abunds(wdir, species_list.split(','),rout_au=rout)
        for k,nx in list(ncol.items()):
            if comp_spec is not None:
                abunds[k].append('{:.1e}'.format(nx/ncol[comp_spec]))
            else:
                abunds[k].append('{:.1e}'.format(nx if get_ncol is True else nx/nh))
    for k,v in abunds.items():
        f.write(k+'\t'+'\t'.join(v)+'\n') 
    f.close()
    return 'Wrote {}'.format('/home/pitts/pnautilus-1.1/S3phaseYSO_{}/{}{}'.format(obj,obj,outfi))

from collections import OrderedDict as od
def getSbudget_at(folder,T=None,R=None):
    abunds = open("/home/pitts/pnautilus-1.1/{}/abundances.tmp".format(folder), 'r').readlines()
    abdict = {}
    for row in abunds:
        if '!' not in row and '1.00000E-99' not in row:
            k,v = row.split(' =  ')
            v = float(v.strip())
            if re.search('S(?!i)',k) is not None:
                if k.replace('K','J') in list(abdict.keys()):
                    abdict[k.replace('K','J')]=abdict[k.replace('K','J')]+v
                else:
                    abdict[k]=v
    if T is not None or R is not None:
        temp, ra  = np.loadtxt('/home/pitts/pnautilus-1.1/{}/structure_evolution.dat'.format(folder), usecols=[3,4], 
                               dtype=None, encoding='ascii', converters={3: lambda ln: 10**float(ln),
                                                   4: lambda r: 10**float(r.replace('!',''))},
                               skiprows=3,unpack=True) #first row of data shouldn't be used^
        temp,ra=temp[1:],ra[1:]
        if T is not None:
            ijk = np.where(abs(temp-T)==min(abs(temp-T)))
        elif R is not None:
            ijk = np.where(abs(ra-R)==min(abs(ra-R)))
        for k in abdict.keys():
            abundfi = np.loadtxt("/home/pitts/pnautilus-1.1/{}/ab/{}.ab".format(folder,k),
                               skiprows=1, usecols=1, dtype=float)
            abdict[k]=abundfi[ijk]
            if 'J' in k:
                abundK = np.loadtxt("/home/pitts/pnautilus-1.1/{}/ab/{}.ab".format(folder,k.replace('J','K')),
                               skiprows=1, usecols=1, dtype=float)
                abdict[k]=abdict[k]+abundK[ijk]
        icetot=0.
        gastot=0.
        for k, v in abdict.items():
            if 'J' in k:
                icetot+=v[0]
            else:
                gastot+=v[0]
        odic = od({})
        for k, v in sorted(abdict.items(), key=lambda item: item[1])[::-1]:
            odic[k]=[v[0],v[0]/icetot if 'J' in k else v[0]/gastot]
    return odic,gastot,icetot

def piechart_integrated_S_budget(folder,rin=None,rout=None,pie=True):
    dummy = open("{}/abundances.tmp".format(folder), 'r').readlines()
    speclist=[]
    for row in dummy:
        if ('!' not in row and '1.00000E-99' not in row and 
            re.search('S(?!i)',row.split(' =  ')[0]) is not None):
            # if not any(s in row for s in ['H2CCS','H2C3S','HNCS','HCNS','NCSH','HNS','HSN','HCCCHS',
            #                               'NH2CH2SH','NH2CHS','HNCHSH','H2NS','NH2SH','HSCO']):
                speclist.append(row.split(' =  ')[0])
    
    nh2,ncol=integrate_abunds(folder,speclist,rin_au=rin,rout_au=rout,sum_ices=True)
    
    icetot=0.
    gastot=0.
    for k, v in ncol.items():
        if 'J' in k:
            icetot+=v/nh2
        else:
            gastot+=v/nh2
    odic = od({})
    iqty,ilab,ioth=[],[],0
    gqty,glab,goth=[],[],0
    
    for k, v in sorted(ncol.items(), key=lambda item: item[1])[::-1]:
        va=v/nh2
        if 'J' in k:
            odic[k]=[va,va/icetot]
            if va/icetot>=0.01 and len(iqty)<9:
                iqty.append(va/icetot)
                ilab.append('{}: {:.0%}'.format(k.replace('J','')
                                             .replace('2','$_2$')
                                             .replace('3','$_3$')
                                             .replace('4','$_4$')+'(s)',
                            va/icetot))
            else:
                ioth+=va/icetot
        else:
            odic[k]=[va,va/gastot]
            if va/gastot>=0.01 and len(gqty)<9:
                gqty.append(va/gastot)
                glab.append('{}: {:.0%}'.format(k.replace('2','$_2$')
                                             .replace('3','$_3$')
                                             .replace('4','$_4$')+'(g)',
                            va/gastot))       
            else:
                goth+=va/gastot               
    iqty.append(ioth)
    ilab.append('Other: {:.0%}'.format(ioth))
    gqty.append(goth)
    glab.append('Other: {:.0%}'.format(goth))
    fig,axes=pl.subplots(nrows=1,ncols=2,figsize=(10.,4.))
    totallphases=gastot+icetot
    axes[0].set_title('S-Gases ({}% of S)'.format(str(int(round(gastot*100/totallphases,0)))))
    axes[0].pie(gqty,labels=glab,normalize=True)
    axes[1].set_title('S-Ices ({}% of S)'.format(str(int(round(icetot*100/totallphases,0)))))
    axes[1].pie(iqty,labels=ilab,normalize=True)
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    pl.subplots_adjust(wspace=0.5, left=0.12, right=0.88, 
                       top=0.99, bottom=0.01)
    s = '05' if 'Av5' in folder else '10'
    s+= '3' if '1e3G0' in folder else '0'
    s+= '1' if 'OCSedit' in folder else '0'
    s+= '-16' if '4e-16' in folder else '-17'
    #pl.title('Model ID {}-{}'.format(folder[-3:],s))
    pl.show()
    return odic,gastot,icetot