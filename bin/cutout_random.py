"""

At the moment this gives repeated cutouts!

"""

import healpy as hp
import numpy as np
import os,sys
import orphics.tools.io as io
import orphics.tools.curvedSky as cs
import matplotlib.pyplot as plt
import pandas as pd
import time



timestr = str(int(10* time.time()))

out_dir = os.environ['WWW']+"plots/"
map_root = os.environ['WORK2']+'/data/sehgal/'

save_dir = map_root+"randoms_"+timestr+"/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
else:
    raise

kappa_file = map_root + 'healpix_4096_KappaeffLSStoCMBfullsky.fits'
cmb_file = map_root + '148_lensedcmb_healpix.fits'    
kappa_map = hp.read_map(kappa_file)
cmb_map = hp.read_map(cmb_file)


Nrandoms = 300000
rand_ra = np.random.uniform(0.,90,size=Nrandoms)
rand_dec = np.random.uniform(0.,90,size=Nrandoms)

size_arc = 100.
pix = 0.5
Npix = int(size_arc/pix)
stack = 0.
k = 0

    
for ra,dec in zip(rand_ra,rand_dec):

    
    cutout_kappa = hp.visufunc.gnomview_quick(kappa_map, rot=(ra, dec), coord='C', xsize=Npix, ysize=Npix,reso=pix,return_projected_map=True,fig=0)
    cutout_cmb = hp.visufunc.gnomview_quick(cmb_map, rot=(ra, dec), coord='C', xsize=Npix, ysize=Npix,reso=pix,return_projected_map=True,fig=0)

    np.save(save_dir+"kappa_cutout_"+str(k).zfill(5),np.ma.filled(cutout_kappa,fill_value=np.nan))
    np.save(save_dir+"cmb_cutout_"+str(k).zfill(5),np.ma.filled(cutout_cmb,fill_value=np.nan))
    stack += cutout_kappa
    print ra,dec
    k += 1

stack /= k
io.quickPlot2d(stack,out_dir+"stack.png")
    
    
    
