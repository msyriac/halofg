from __future__ import print_function
from orphics import maps,io,cosmology
from enlib import enmap
import numpy as np
import os,sys
import healpy as hp

proot = "/gpfs01/astro/workarea/msyriac/data/planck/"

import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])


try:

    lss,smica_nls = np.loadtxt(proot+"smica_nls.txt",unpack=True)
    lsl,lgmca_nls = np.loadtxt(proot+"lgmca_nls.txt",unpack=True)
    
except:
    mask = hp.read_map(proot+"COM_Mask_Lensing_2048_R2.00.fits")
    fsky = mask.sum()*1./mask.size


    # smica

    smica1 = hp.read_map(proot+"COM_CMB_IQU-smica-field-Int_2048_R2.01_ringhalf-1.fits")*1e6
    smica2 = hp.read_map(proot+"COM_CMB_IQU-smica-field-Int_2048_R2.01_ringhalf-2.fits")*1e6

    autos = (hp.anafast(smica1*mask)+hp.anafast(smica2*mask))/2.
    k12 = hp.anafast(smica1*mask,smica2*mask)

    smica_nls = ((autos-k12)/2.)/fsky
    print(smica_nls)
    lss = np.arange(len(smica_nls))

    # lgmca


    lgmcan = hp.read_map(proot+"WPR2_CMB_noise_muK.fits") * mask
    lgmca_nls = hp.anafast(lgmcan)/fsky
    lsl = np.arange(len(lgmca_nls))

    io.save_cols(proot+"smica_nls.txt",(lss,smica_nls))
    io.save_cols(proot+"lgmca_nls.txt",(lsl,lgmca_nls))
    

cc = cosmology.Cosmology(lmax=6000,pickling=True,dimensionless=False)
ells = np.arange(0,3000,1)
cltt = cc.theory.lCl('TT',ells)

spbeam = maps.gauss_beam(lss,5.0)
lpbeam = maps.gauss_beam(lsl,5.0)

pl = io.Plotter(yscale='log',xlabel="$\\ell$",ylabel="$\\ell(\\ell+1)C^{TT}_{\\ell}/2\\pi\ (\\mu K-\\mathrm{rad})^2$",ftsize=17)
pl.add(ells,cltt*ells*(ells+1.)/2./np.pi,color="k",lw=2)
pl.add(lss,smica_nls*lss*(lss+1.)/2./np.pi/spbeam**2.,label="SMICA")
pl.add(lsl,lgmca_nls*lsl*(lsl+1.)/2./np.pi/lpbeam**2.,label="LGMCA")


abeam = maps.gauss_beam(ells,1.5)
for noise in [6.,10.,20.]:

    lknee = 3000
    alpha = -4.

    nls = cosmology.white_noise_with_atm_func(ells,noise,lknee,alpha,dimensionless=False,TCMB=2.7255e6)

    pl.add(ells,nls*ells*(ells+1.)/2./np.pi/abeam**2.,ls="--",lw=2,label=str(noise)+" $\\mu K$-arcmin")

noise = 45.
abeam = maps.gauss_beam(ells,5.0)
nls = cosmology.white_noise_with_atm_func(ells,noise,0,1,dimensionless=False,TCMB=2.7255e6)
# pl.add(ells,nls*ells*(ells+1.)/2./np.pi/abeam**2.,ls="--",lw=2,label="LGCMA estimate")
    

pl.legend(loc='lower right',labsize=12)
pl._ax.set_xlim(0,3000)
pl._ax.set_ylim(1,1e4)

pl.done(io.dout_dir+"smicalgmca.pdf")

