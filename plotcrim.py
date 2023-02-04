#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 13:41:30 2021

@author: pitts
"""
import pylab as pl
import numpy as np
import emcee, corner
import scipy.optimize as sciop
from scipy import stats

from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')

tab = np.genfromtxt('crimiercomp2.txt', encoding='ascii', dtype=None, missing_values='...',
                    filling_values=-9999, names=True, delimiter='\t')
#row order is - (lower error bars), + (upper error bars)
ebartab = {'L_bol':[[90.0, 4e3, 1000.0, 300.0, 200.0, 40.0, 1000.0, 4e4],
                    [80.0, 1e3, 600.0, 200.0, 200.0, 30.0, 200.0, 6e3]],
           'M_env':[[30.0, 50.0, 10.0, 10.0, 20.0, 10.0, 10.0, 80.0],
                    [30.0, 50.0, 10.0, 10.0, 40.0, 30.0, 10.0, 80.0]],
           'R_out':[[6e3, 7e3, 5e3, 5e3, 5e3, 1.3e4, 4e3, 3e4],
                    [1.5e4, 1e4, 6e3, 5e3, 2e3, 5e3, 7e3, 1e4],
                    [False,False,False,False,False,False,True,True], #lower lims, upper lims
                    [False,False,True,False,True,False,False,False]],
           'p_rho':[[0.3, 0.1, 0.4, 0.1, 0.2, 0.3, 0.0, 0.1],
                    [0.1, 0.1, 0.0, 0.1, 0.2, 0.1, 0.2, 0.1],
                    [False,False,False,False,False,False,True,False],
                    [False,False,True,False,False,False,False,False]],
           'n_1000AU':[[6.7e6, 3.1e6, 1.0e6, 2.4e6, 1.8e7, 3.9e6, 2.3e6, 1.6e7],
                       [10.4e6, 6.9e6, 1.0, 4e6, 7e7, 11.7e6, 26.3e6, 3.4e7],
                       [False,False,False,False,False,False,True,False],
                       [False,False,True,False,False,False,False,False]]#,
           #'n_10K':[[],[]]
           }

#all out slopes will be power laws of the form logy = mlogx+logb, i.e. y = b*x^m
def plaw(x,a,b):
    return (x**a)*b

from astropy.modeling.powerlaws import BrokenPowerLaw1D as bplaw

#assume gaussian errors and constant underestimation factor
#(some are better represented by log-normal errors, but it's too complicated to try to do them separately)
def ln_like(theta, x, y, yerr):
    a, b, log_f = theta
    sigma2 = yerr ** 2 + plaw(x,a,b) ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - plaw(x,a,b)) ** 2 / sigma2 + np.log(sigma2))

def ln_prior(theta):
    a, b, log_f = theta
    if (0.0 < a < 5.0) and (-2.0 < np.log10(b) < 6.0) and (-15 < log_f < 2.0):
        return 0.0 #log(0)=1
    else:
        return -np.inf
    
def ln_prob(theta, x, y, yerr):
    lp = ln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(theta, x, y, yerr)

nll = lambda *args: -ln_like(*args)

#try fixing the x-axis break point at L = 60 L_sun since it's not well-constrained
def bln_like(theta, x, y, yerr):
    a1, a2, by, log_f = theta
    sigma2 = yerr ** 2 + bplaw.evaluate(x, by,60.0,-a1,-a2) ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - bplaw.evaluate(x,by,60.0,-a1,-a2)) ** 2 / sigma2 + np.log(sigma2))

def bln_prior(theta):
    a1, a2, by, log_f = theta
    if (-5.0 < a1 < 5.0) and (-5.0 < a2 < 5.0) and (1.0 < np.log10(by) < 9.0) and (log_f < 2.0):
        return 0.0 #log(0)=1
    else:
        return -np.inf
    
def bln_prob(theta, x, y, yerr):
    lp = bln_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + bln_like(theta, x, y, yerr)

bnll = lambda *args: -bln_like(*args)

### MCMC fit: M_env vs L

initM = np.array([0.5, 0.1, -3.0])# + 0.1 * np.random.randn(3)
binitM = np.array([0.8, 0.5, 100.0,  -1.0])# + 0.1 * np.random.randn(3)

indsM = np.where(np.logical_and(tab['M_env_upl']>0, tab['ms']!=',1'))

poptM = sciop.minimize(nll, initM, method = 'Nelder-Mead',
                       args=(tab['L_bol'][indsM],tab['M_env'][indsM],tab['M_env_upl'][indsM]))


posM = poptM.x + 1e-1 * np.random.randn(32, len(initM))
nwalkers, ndim = posM.shape
samplerM = emcee.EnsembleSampler(nwalkers, ndim, ln_prob,
                                 args=(tab['L_bol'][indsM],
                                       tab['M_env'][indsM],
                                       tab['M_env_upl'][indsM]))
samplerM.run_mcmc(posM, 5000, progress=True)
flat_samplesM = samplerM.get_chain(discard=100, thin=15, flat=True)
figm = corner.corner(flat_samplesM, labels=[r'$\xi$',r'$M_o$ [$M_{\odot}$]',r'ln$\epsilon_M$'],
                     truths=poptM.x, quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize':14})

bpoptM = sciop.minimize(bnll, binitM, method = 'Nelder-Mead',
                       args=(tab['L_bol'][indsM],tab['M_env'][indsM],tab['M_env_upl'][indsM]))
bposM = bpoptM.x + 1e-1 * np.random.randn(40, len(binitM))
nwalkers, ndim = bposM.shape
bsamplerM = emcee.EnsembleSampler(nwalkers, ndim, bln_prob,
                                 args=(tab['L_bol'][indsM],
                                       tab['M_env'][indsM],
                                       tab['M_env_upl'][indsM]))
bsamplerM.run_mcmc(bposM, 5000, progress=True)
bflat_samplesM = bsamplerM.get_chain(discard=100, thin=15, flat=True)
bfigm = corner.corner(bflat_samplesM, labels=[r'$\xi_1$',r'$\xi_2$',r'$M_{bk}$ [$M_{\odot}$]',r'ln$\epsilon_M$'],
                      truths=bpoptM.x, quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize':14})

psm = np.array([np.percentile(flat_samplesM[:, i], [16, 50, 84]) for i in range(len(initM))])
bpsm = np.array([np.percentile(bflat_samplesM[:, i], [16, 50, 84]) for i in range(len(binitM))])

# LRtestM = 2* ( bln_like(bpsm[:,1], tab['L_bol'][indsM], tab['M_env'][indsM], tab['M_env_upl'][indsM])/5 -
#                ln_like(psm[:,1], tab['L_bol'][indsM], tab['M_env'][indsM], tab['M_env_upl'][indsM])/3)
# print('M vs. L:\n single power law: ', psm, '\n broken power law: ', bpsm, '\n', LRtestM)
print('\n M vs. L:\n single power law: ', stats.ks_2samp(sorted(plaw(np.logspace(-1,np.log10(3e6)),poptM.x[0],poptM.x[1])),
                                                      tab['M_env'][indsM]))
print('\n M vs. L:\n broken power law: ', stats.ks_2samp(sorted(bplaw.evaluate(np.logspace(-1,np.log10(3e6)), bpoptM.x[2],60.0,-bpoptM.x[0],-bpoptM.x[1])),
                                                      tab['M_env'][indsM]))
rssm = np.sum( (tab['M_env'][indsM] - plaw(tab['L_bol'][indsM],poptM.x[0],poptM.x[1]))**2 )
brssm = np.sum( (tab['M_env'][indsM] - bplaw.evaluate(tab['L_bol'][indsM], bpoptM.x[2],60.0,-bpoptM.x[0],-bpoptM.x[1]))**2 )
print('\n M vs. L F-test: ',brssm/rssm)
### MCMC fit: R_out vs L

initR = np.array([0.15, 5e3, -1.0])# + 0.1 * np.random.randn(3)
binitR = np.array([0.5, 0.2, 2e4, -1.0])# + 0.1 * np.random.randn(3)

indsR = np.where(np.logical_and(tab['R_out_upl']>0, tab['ms']!=',1'))

poptR = sciop.minimize(nll, initR, method = 'Nelder-Mead',
                       args=(tab['L_bol'][indsR],tab['R_out'][indsR],tab['R_out_upl'][indsR]))
posR = poptR.x * abs(np.random.randn(32, len(initR)))
nwalkers, ndim = posR.shape
samplerR = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, 
                                 args=(tab['L_bol'][indsR],
                                       tab['R_out'][indsR],
                                       tab['R_out_upl'][indsR]))
samplerR.run_mcmc(posR, 5000, progress=True)
flat_samplesR = samplerR.get_chain(discard=100, thin=15, flat=True)
figr = corner.corner(flat_samplesR, labels=[r'$\alpha$',r'$R_o$ [AU]',r'ln$\epsilon_R$'],
                     truths=poptR.x, quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize':14})

bpoptR = sciop.minimize(bnll, binitR, method = 'Nelder-Mead',
                       args=(tab['L_bol'][indsR],tab['R_out'][indsR],tab['R_out_upl'][indsR]))
bposR = bpoptR.x * abs(np.random.randn(40, len(binitR)))
nwalkers, ndim = bposR.shape
bsamplerR = emcee.EnsembleSampler(nwalkers, ndim, bln_prob, 
                                 args=(tab['L_bol'][indsR],
                                       tab['R_out'][indsR],
                                       tab['R_out_upl'][indsR]))
bsamplerR.run_mcmc(bposR, 5000, progress=True)
bflat_samplesR = bsamplerR.get_chain(discard=100, thin=15, flat=True)
bfigr = corner.corner(bflat_samplesR, labels=[r'$\alpha_1$',r'$\alpha_2$',r'$R_{bk}$ [AU]',r'ln$\epsilon_R$'],
                     truths=bpoptR.x, quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize':14})

psr = np.array([np.percentile(flat_samplesR[:, i], [16, 50, 84]) for i in range(len(initR))])
bpsr = np.array([np.percentile(bflat_samplesR[:, i], [16, 50, 84]) for i in range(len(binitR))])

# LRtestR = 2* ( bln_like(bpsr[:,1], tab['L_bol'][indsR], tab['R_out'][indsR], tab['R_out_upl'][indsR]) -
#                ln_like(psr[:,1], tab['L_bol'][indsR], tab['R_out'][indsR], tab['R_out_upl'][indsR]))
# print('R vs. L: ', psr, '\n', bpsr, '\n', LRtestR)
print('\n R vs. L:\n single power law: ', stats.ks_2samp(sorted(plaw(np.logspace(-1,np.log10(3e6)),poptR.x[0],poptR.x[1])),
                                                      tab['R_out'][indsR]))
print('\n R vs. L:\n broken power law: ', stats.ks_2samp(sorted(bplaw.evaluate(np.logspace(-1,np.log10(3e6)), bpoptR.x[2],60.0,-bpoptR.x[0],-bpoptR.x[1])),
                                                      tab['R_out'][indsR]))
rssr = np.sum( (tab['R_out'][indsR] - plaw(tab['L_bol'][indsR],poptR.x[0],poptR.x[1]))**2 )
brssr = np.sum( (tab['R_out'][indsR] - bplaw.evaluate(tab['L_bol'][indsR], bpoptR.x[2],60.0,-bpoptR.x[0],-bpoptR.x[1]))**2 )
print('\n R vs. L F-test: ',brssr/rssr)

### MCMC fit: p and n(1000AU) vs L

def line(x,a,b):
    return a*x+b

def ln_like1(theta, x, y, yerr):
    a, b, log_f = theta
    sigma2 = yerr ** 2 + line(x,a,b) ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - line(x,a,b)) ** 2 / sigma2 + np.log(sigma2))

def ln_like2(theta, x, y, yerr):
    a, b = theta
    #sigma2 = yerr ** 2 + line(x,a,b) ** 2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - line(x,a,b)) ** 2 / yerr ** 2 + np.log(yerr ** 2))

def ln_prior1(theta):
    a, b, log_f = theta
    if (-1.0 < a < 1.0) and (0. < b < 10.0) and (-20 < log_f < 1.0):
        return 0.0 #log(0)=1
    else:
        return -np.inf

def ln_prior2(theta):
    a, b = theta
    if (-1.0 < a < 1.0) and (0. < b < 10.0):# and (-20 < log_f < 1.0):
        return 0.0 #log(0)=1
    else:
        return -np.inf
    
def ln_prob1(theta, x, y, yerr):
    lp = ln_prior1(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like1(theta, x, y, yerr)    
    
def ln_prob2(theta, x, y, yerr):
    lp = ln_prior2(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like2(theta, x, y, yerr)

nll1 = lambda *args: -ln_like1(*args)
nll2 = lambda *args: -ln_like2(*args)

### MCMC fit: p vs L

linitp = np.array([1.4, -0.4])
initp = np.array([0.15, 1.0, -1.0])

indsp = np.where(np.logical_and(tab['p_rho_lol']>0, tab['ms']!=',1'))

poptp = sciop.minimize(nll1, initp, method = 'Nelder-Mead',
                       args=(np.log10(tab['L_bol'][indsp]),
                             tab['p_rho'][indsp],
                             tab['p_rho_lol'][indsp]))#0.35*np.ones(len(tab['p_rho_lol'][indsp]))))
posp = poptp.x + 1e-1 * np.random.randn(32, len(initp))
#posn[:,1]=10**np.random.uniform(3,7,32)
#posn = np.vstack((np.random.uniform(0.3,0.6,32),10**np.random.uniform(3,7,32),np.random.uniform(-0.6,0.1,32)))
nwalkers, ndim = posp.shape
samplerp = emcee.EnsembleSampler(nwalkers, ndim, ln_prob1,
                                 args=(np.log10(tab['L_bol'][indsp]),
                                       tab['p_rho'][indsp],
                                       tab['p_rho_lol'][indsp]))#0.35*np.ones(len(tab['p_rho_lol'][indsp]))))
samplerp.run_mcmc(posp, 5000, progress=True)
flat_samplesp = samplerp.get_chain(discard=100, thin=15, flat=True)
figp = corner.corner(flat_samplesp, labels=[r'$\lambda$',r'$p_o$ [cm$^{-3}$]', r'ln$\epsilon_n$'],
                     truths=poptp.x, quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize':14})

psp = np.array([np.percentile(flat_samplesp[:, i], [16, 50, 84]) for i in range(len(initp))])
rssp = np.sum( (tab['p_rho'][indsp] - line(np.log10(tab['L_bol'][indsp]),poptp.x[0],poptp.x[1]))**2 )
print('\n p vs. L F-test: ', rssp/np.sum( (tab['p_rho'][indsp] - np.average(tab['p_rho'][indsp], weights=tab['p_rho_lol'][indsp]))**2 ))

stderr = np.sqrt((np.std(tab['p_rho'][indsp])**2)*len(tab['p_rho'][indsp])/(len(tab['p_rho'][indsp])-2))
tt = poptp.x[0]/stderr
pval = stats.t.sf(np.abs(poptp.x[0]/stderr), len(tab['p_rho'][indsp])-2)*2
print('\n p_rho vs L p-value: ',pval)

### MCMC fit: n(1000AU) vs L

initn = np.array([0.3, 5.0])#, -1.0]) #+ 0.1 * np.random.randn(3)
binitn = np.array([0.75, 0.3, 1e7, -1.0])#, -1.0])

indsn = np.where(tab['n_1000AU_lol']>0)
poptn = sciop.minimize(nll2, initn, method = 'Nelder-Mead',
                       args=(np.log10(tab['L_bol'][indsn]),
                             np.log10(tab['n_1000AU'][indsn]),
                             np.log10(tab['n_1000AU_lol'][indsn])))
posn = poptn.x + 2e-1 * np.random.randn(32, len(initn))
#posn[:,1]=10**np.random.uniform(3,7,32)
#posn = np.vstack((np.random.uniform(0.3,0.6,32),10**np.random.uniform(3,7,32),np.random.uniform(-0.6,0.1,32)))
nwalkers, ndim = posn.shape
samplern = emcee.EnsembleSampler(nwalkers, ndim, ln_prob2,
                                 args=(np.log10(tab['L_bol'][indsn]),
                                       np.log10(tab['n_1000AU'][indsn]),
                                       np.log10(tab['n_1000AU_lol'][indsn])))
samplern.run_mcmc(posn, 5000, progress=True)
flat_samplesn = samplern.get_chain(discard=100, thin=15, flat=True)
fign = corner.corner(flat_samplesn, labels=[r'$\eta$',r'log$n_o$ [cm$^{-3}$]'],# r'ln$\epsilon_n$'],
                     truths=poptn.x, quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize':14})

bpoptn = sciop.minimize(bnll, binitn, method = 'Nelder-Mead',
                        args=(tab['L_bol'][indsn],tab['n_1000AU'][indsn],tab['n_1000AU_lol'][indsn]))
bposn = bpoptn.x + 2e-1 * np.random.randn(40, len(binitn))
#posn[:,1]=10**np.random.uniform(3,7,32)
#posn = np.vstack((np.random.uniform(0.3,0.6,32),10**np.random.uniform(3,7,32),np.random.uniform(-0.6,0.1,32)))
nwalkers, ndim = bposn.shape
bsamplern = emcee.EnsembleSampler(nwalkers, ndim, bln_prob,
                                  args=(tab['L_bol'][indsn],tab['n_1000AU'][indsn],tab['n_1000AU_lol'][indsn]))
bsamplern.run_mcmc(bposn, 5000, progress=True)
bflat_samplesn = bsamplern.get_chain(discard=100, thin=15, flat=True)
bfign = corner.corner(bflat_samplesn, labels=[r'$\eta_1$', r'$\eta_2$',r'$n_{bk}$ [cm$^{-3}$]', r'ln$\epsilon_n$'],
                      truths=bpoptn.x, quantiles=[0.16, 0.5, 0.84], label_kwargs={'fontsize':14})

psn = np.array([np.percentile(flat_samplesn[:, i], [16, 50, 84]) for i in range(len(initn))])
bpsn = np.array([np.percentile(bflat_samplesn[:, i], [16, 50, 84]) for i in range(len(binitn))])

# LRtestn = 2* ( bln_like(bpsn[:,1], tab['L_bol'][indsn], tab['n_1000AU'][indsn], tab['n_1000AU_lol'][indsn]) -
#                ln_like2(psn[:,1], np.log10(tab['L_bol'][indsn]), np.log10(tab['n_1000AU'][indsn]), np.log10(tab['n_1000AU_lol'][indsn])))
# print('n vs. L: ', psn, '\n', bpsn, '\n', LRtestn)
print('\n n vs. L:\n single power law: ', stats.ks_2samp(sorted(10**line(np.linspace(-1,np.log10(3e6)),poptn.x[0],poptn.x[1])),
                                                      tab['n_1000AU'][indsn]))
print('\n n vs. L:\n broken power law: ', stats.ks_2samp(sorted(bplaw.evaluate(np.logspace(-1,np.log10(3e6)), bpoptR.x[2],60.0,-bpoptR.x[0],-bpoptR.x[1])),
                                                      tab['n_1000AU'][indsn]))
rssn = np.sum( (tab['n_1000AU'][indsn] - 10**line(np.log10(tab['L_bol'][indsn]),poptn.x[0],poptn.x[1]))**2 )
brssn = np.sum( (tab['n_1000AU'][indsn] - bplaw.evaluate(tab['L_bol'][indsn], bpoptn.x[2],60.0,-bpoptn.x[0],-bpoptn.x[1]))**2 )
print('\n n vs. L F-test: ',brssr/rssr)

### PLOTS #######################################
x0=np.logspace(-1,np.log10(3e6))

fig,axes=pl.subplots(nrows=2,ncols=2)
axes[0,0].errorbar(tab['L_bol'][:8].astype(float),tab['M_env'][:8].astype(float),
                   xerr = ebartab['L_bol'], yerr=ebartab['M_env'], fmt='k.',
                   capsize=2, zorder=20, ecolor='k',  ls='none')
inds=np.where(tab['M_env']>0)
cinds = np.where(np.logical_and(tab['M_env']>0,np.logical_or(tab['clr']=='tab:olive',tab['clr']=='tab:gray')))
axes[0,0].errorbar(tab['L_bol'][cinds].astype(float),tab['M_env'][cinds].astype(float),
                    xerr = [tab['L_bol_lol'][cinds],tab['L_bol_upl'][cinds]],
                    yerr = [tab['M_env_lol'][cinds],tab['M_env_upl'][cinds]], fmt='',
                    capsize=0, zorder=0, ecolor=tab['clr'][cinds],  ls='none', alpha=0.2)
axes[0,0].plot(x0,plaw(x0,poptM.x[0],poptM.x[1]),'k-',
                label='single power law')#r'$M_{\mathrm{env}}=M_oL^{\xi}$')#\n $\xi={:.2}^{{+{:.2}}}_{{-{:.2}}}$,\
               #     $M_{{o}}/M_{{\odot}}={:.2}^{{+{:.2}}}_{{-{:.2}}}$'.format(psm[0,1], psm[0,2]-psm[0,1], psm[0,1]-psm[0,0], psm[1,1], psm[1,2]-psm[1,1], psm[1,1]-psm[1,0]))
axes[0,0].plot(x0,bplaw.evaluate(x0, bpoptM.x[2],60.0,-bpoptM.x[0],-bpoptM.x[1]),'k--',
                label='broken power law')#r'$M_{\mathrm{env}}=M_{bk}\left(\frac{L}{60\,L_{\odot}}\right)^{\xi}$')
axes[0,0].grid(True,zorder=0)
for j,m in enumerate(set(tab['ms'][inds])):
    try:
        inds=np.where(np.logical_and( np.logical_and(tab['M_env']>0, tab['ms']==m), tab['ms']!=',1'))
        axes[0,0].scatter(tab['L_bol'][inds].astype(float),tab['M_env'][inds].astype(float),
                      c=tab['clr'][inds], s=float(m[1:]),  marker=m[0],
                      zorder=20 if 'k' in tab['clr'][inds] else j+2)
    except:
        print(j,m)
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].tick_params(length=6, axis='both', labelsize=12)
#axes[0,0].tick_params(length=3, which='minor', axis='both')
axes[0,0].set_xlabel('$L_{\mathrm{bol}}$ [$L_{\odot}$]',fontsize=14)
axes[0,0].set_ylabel('$M_{\mathrm{env}}$ [$M_{\odot}$]',fontsize=14)
axes[0,0].set_xlim(0.1,3e6)
axes[0,0].legend(loc=0,fontsize=12)

axes[0,1].errorbar(tab['L_bol'][:8].astype(float),tab['R_out'][:8].astype(float),
                   xerr = ebartab['L_bol'], yerr=ebartab['R_out'][:2], 
                   lolims=ebartab['R_out'][2], uplims = ebartab['R_out'][3], fmt='k.',
                   capsize=2, zorder=20, ecolor='k', ls='none')
inds=np.where(tab['R_out']>0)
cinds = np.where(np.logical_and(tab['R_out_lol']>0,tab['clr']=='tab:olive'))
#cinds2 = np.where(np.logical_and(tab['R_out']>0,tab['clr']=='tab:gray'))
axes[0,1].errorbar(tab['L_bol'][cinds].astype(float),tab['R_out'][cinds].astype(float),
                    xerr = [tab['L_bol_lol'][cinds],tab['L_bol_upl'][cinds]], fmt='',
                    yerr = [tab['R_out_lol'][cinds],tab['R_out_upl'][cinds]], #lolims=True, <--this is a problem for some reason
                    capsize=0, zorder=0, ecolor=tab['clr'][cinds],  ls='none', alpha=0.2)
# axes[0,1].errorbar(tab['L_bol'][cinds2].astype(float),tab['R_out'][cinds2].astype(float),
#                     xerr = [tab['L_bol_lol'][cinds2],tab['L_bol_upl'][cinds2]], fmt='',
#                     yerr = [tab['R_out_lol'][cinds2],tab['R_out_upl'][cinds2]], 
#                     capsize=0, zorder=0, ecolor=tab['clr'][cinds2],  ls='none')
axes[0,1].plot(x0,plaw(x0,poptR.x[0],poptR.x[1]),'k-',
                label='single power law')#r'$R_{\mathrm{out}}=R_oL^{\alpha}$')
axes[0,1].plot(x0,bplaw.evaluate(x0, bpoptR.x[2],60.0,-bpoptR.x[0],-bpoptR.x[1]),'k--',
                label='broken power law')#r'$R_{\mathrm{out}}=R_{bk}\left(\frac{L}{60\,L_{\odot}}\right)^{\alpha}$')
axes[0,1].grid(True,zorder=0)
for j,m in enumerate(set(tab['ms'][inds])):
    inds=np.where(np.logical_and( np.logical_and(tab['R_out']>0, tab['ms']==m), tab['ms']!=',1'))
    axes[0,1].scatter(tab['L_bol'][inds].astype(float),tab['R_out'][inds].astype(float),
                  c=tab['clr'][inds], s=float(m[1:]),  marker=m[0],
                  zorder=20 if 'k' in tab['clr'][inds] else j+2)
axes[0,1].set_xlim(0.1,3e6)
axes[0,1].set_ylim(2e3,7e5)
axes[0,1].set_xscale('log')
axes[0,1].set_yscale('log')
axes[0,1].tick_params(length=6, axis='both', labelsize=12)
#axes[0,1].tick_params(length=3, which='minor', axis='both')
axes[0,1].set_xlabel('$L_{\mathrm{bol}}$ [$L_{\odot}$]',fontsize=14)
axes[0,1].set_ylabel('$R_{\mathrm{out}}$ [AU]',fontsize=14)
axes[0,1].legend(loc=0,fontsize=12)

axes[1,0].errorbar(tab['L_bol'][:8].astype(float),tab['p_rho'][:8].astype(float),
                   xerr = ebartab['L_bol'], yerr=ebartab['p_rho'][:2], fmt='k.',
                   lolims=ebartab['p_rho'][2], uplims = ebartab['p_rho'][3],
                   capsize=2, zorder=20, ecolor='k', ls='none')
inds=np.where(np.logical_and(tab['p_rho']>0.,tab['clr']!='tab:olive'))
# cinds = np.where(np.logical_and(tab['p_rho']>0,tab['clr']=='tab:olive'))
# axes[1,0].errorbar(tab['L_bol'][cinds].astype(float),tab['p_rho'][cinds].astype(float),
#                    xerr = [tab['L_bol_lol'][cinds],tab['L_bol_upl'][cinds]],
#                    yerr = [tab['p_rho_lol'][cinds],tab['p_rho_upl'][cinds]], 
#                    capsize=0, zorder=0, ecolor='tab:olive', ls=None)
meanp = np.average(tab['p_rho'][indsp], weights=tab['p_rho_lol'][indsp])
stdp = np.std(tab['p_rho'][indsp])
axes[1,0].axhline(np.average(tab['p_rho'][indsp], weights=tab['p_rho_lol'][indsp]), color='k', ls='-',
                label=r'$\langle p \rangle$ = {:.1f}'.format(np.average(tab['p_rho'][indsp], weights=tab['p_rho_lol'][indsp])))
axes[1,0].hlines([meanp-stdp,meanp+stdp], 0, 1, transform=axes[1,0].get_yaxis_transform(), color='k', ls=':',
                label=r'$\sigma_{{p}}$ = {:.1}'.format(stdp))
axes[1,0].plot(x0,line(np.log10(x0),poptp.x[0],poptp.x[1]),'k--',
               label='single power law')#r'$n_{1000\mathrm{AU}}=n_oL^{\eta}$')
axes[1,0].grid(True,zorder=0)
for j,m in enumerate(set(tab['ms'][inds])):
    inds=np.where(np.logical_and( np.logical_and(tab['p_rho']>0, tab['ms']==m), tab['ms']!=',1'))
    axes[1,0].scatter(tab['L_bol'][inds].astype(float),tab['p_rho'][inds],
                  c=tab['clr'][inds], s=float(m[1:]),  marker=m[0],
                  zorder=20 if 'k' in tab['clr'][inds] else j+2)
axes[1,0].set_xscale('log')
axes[1,0].tick_params(length=6, axis='both', labelsize=12)
axes[1,0].set_xlim(0.1,3e6)
axes[1,0].set_ylim(-0.1,2.6)
#axes[1,0].tick_params(length=3, which='minor', axis='both')
axes[1,0].set_xlabel('$L_{\mathrm{bol}}$ [$L_{\odot}$]',fontsize=14)
axes[1,0].set_ylabel('$|p|$ (power-law index)',fontsize=14)
axes[1,0].legend(loc=0,fontsize=12)

inds=np.where(tab['n_1000AU']>0)
axes[1,1].errorbar(tab['L_bol'][:8].astype(float),tab['n_1000AU'][:8].astype(float),
                   xerr = ebartab['L_bol'], yerr=ebartab['n_1000AU'][:2], fmt='k.',
                   lolims=ebartab['n_1000AU'][2], uplims = ebartab['n_1000AU'][3],
                   capsize=2, zorder=20, ecolor='k',  ls='none')
cinds = np.where(np.logical_and(tab['n_1000AU']>0,np.logical_or(tab['clr']=='tab:olive',tab['clr']=='tab:gray')))
axes[1,1].errorbar(tab['L_bol'][cinds].astype(float),tab['n_1000AU'][cinds].astype(float),
                    xerr = [tab['L_bol_lol'][cinds],tab['L_bol_upl'][cinds]],
                    yerr = [tab['n_1000AU_lol'][cinds],tab['n_1000AU_lol'][cinds]],
                    fmt='', capsize=0, zorder=0, ecolor=tab['clr'][cinds],  ls='none', alpha=0.2)
axes[1,1].plot(x0,plaw(x0,poptn.x[0],10**poptn.x[1]),'k-',
                label='single power law')#r'$n_{1000\mathrm{AU}}=n_oL^{\eta}$')
axes[1,1].plot(x0,bplaw.evaluate(x0, bpoptn.x[2],60.0,-bpoptn.x[0],-bpoptn.x[1]),'k--',
                label='broken power law')#r'$n_{1000\mathrm{AU}}=n_{bk}\left(\frac{L}{60\,L_{\odot}}\right)^{\eta}$')
axes[1,1].grid(True,zorder=0)
for j,m in enumerate(set(tab['ms'][inds])):
    inds=np.where(np.logical_and( np.logical_and(tab['n_1000AU']>0, tab['ms']==m), tab['ms']!=',1'))
    axes[1,1].scatter(tab['L_bol'][inds].astype(float),tab['n_1000AU'][inds].astype(float),
                  c=tab['clr'][inds], s=float(m[1:]),  marker=m[0],
                  zorder=20 if 'k' in tab['clr'][inds] else j+2)
axes[1,1].set_xscale('log')
axes[1,1].set_yscale('log')
axes[1,1].tick_params(length=6, axis='both', labelsize=12)
axes[1,1].set_xlim(0.1,3e6)
#axes[1,1].tick_params(length=3, which='minor', axis='both')
axes[1,1].set_xlabel('$L_{\mathrm{bol}}$ [$L_{\odot}$]',fontsize=14)
axes[1,1].set_ylabel('$n$(r=1000 AU) (cm$^{-3}$)',fontsize=14)
axes[1,1].legend(loc=0, fontsize=12)