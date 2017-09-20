import numpy as np
import sys, os, glob
from orphics.analysis.pipeline import mpi_distribute, MPIStats
import orphics.tools.stats as stats
import alhazen.io as aio
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
import warnings
import logging
logger = logging.getLogger()
with io.nostdout():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from enlib import enmap, lensing, resample
from alhazen.quadraticEstimator import Estimator
import alhazen.lensTools as lt
from ConfigParser import SafeConfigParser 
from szar.counts import ClusterCosmology
import enlib.fft as fftfast
import argparse


out_dir = os.environ['WWW']+"plots/"
alex_cmb_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/alex/lensedCMBMaps_set00_"+str(x).zfill(5)+"/order5_periodic_lensedCMB_T_7.fits"
alex_ucmb_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/alex/lensedCMBMaps_set00_"+str(x).zfill(5)+"/unlensedCMB_T_7.fits"
alex_kappa_file = lambda x: "/gpfs01/astro/workarea/msyriac/data/alex/phiMaps_"+str(x).zfill(5)+"/kappaMap_7.fits"

N = 100

template = enmap.read_map("/gpfs01/astro/workarea/msyriac/data/alex/downgraded1arc_template.fits")

kellmin = 100
kellmax = 5000

avgutt = 0.
davgutt = 0.
avgltt = 0.
davgltt = 0.
lbin_edges = np.arange(kellmin,kellmax,200)

for k,i in enumerate(range(59,N)):


    
    print "Reading CMB file ", i ,"..."


    alex_cmb = enmap.read_map(alex_ucmb_file(i))
    if k==0:
        modlmap_dat = alex_cmb.modlmap()
        lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
    
    ltt2d = fmaps.get_simple_power_enmap(alex_cmb)
    ccents,ltt = lbinner_dat.bin(ltt2d)
    avgutt += ltt.copy()

    print "Downgrading..."
    cmb = enmap.ndmap(resample.resample_fft(alex_cmb,template.shape),template.wcs)

    print "Saving file..."
    enmap.write_map(alex_ucmb_file(i).replace("unlensedCMB_T","unlensedCMB_T_1arcpix"),cmb)

    # if k==0:
    #     dmodlmap_dat = cmb.modlmap()
    #     dlbinner_dat = stats.bin2D(dmodlmap_dat,lbin_edges)
    # ltt2d = fmaps.get_simple_power_enmap(cmb)
    # ccents,ltt = dlbinner_dat.bin(ltt2d)
    # davgutt += ltt.copy()
    
    # print "Reading CMB file ..."
    # alex_cmb = enmap.read_map(alex_cmb_file(i))
    # ltt2d = fmaps.get_simple_power_enmap(alex_cmb)
    # ccents,ltt = lbinner_dat.bin(ltt2d)
    # avgltt += ltt.copy()

    # print "Downgrading..."
    # cmb = enmap.ndmap(resample.resample_fft(alex_cmb,template.shape),template.wcs)

    # # print "Saving file..."
    # # enmap.write_map(alex_cmb_file(i).replace("lensedCMB_T","lensedCMB_T_1arcpix"),cmb)

    # ltt2d = fmaps.get_simple_power_enmap(cmb)
    # ccents,ltt = dlbinner_dat.bin(ltt2d)
    # davgltt += ltt.copy()


    #print "Reading kappa file..."
    #alex_cmb = enmap.read_map(alex_kappa_file(i))

    #print "Downgrading..."
    #cmb = enmap.ndmap(resample.resample_fft(alex_cmb,template.shape),template.wcs)

    #print "Saving file..."
    #enmap.write_map(alex_kappa_file(i).replace("kappaMap","kappaMap_1arcpix"),cmb)


#pdiff = (davgltt-avgltt)*100./avgltt
# pdiffu = (davgutt-avgutt)*100./avgutt

# pl = io.Plotter()
# #pl.add(ccents,pdiff)
# pl.add(ccents,pdiffu,ls="--")
# pl.hline()
# pl.done(out_dir+"alex_dgradepdiff.png")
