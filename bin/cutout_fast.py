import healpy as hp
import numpy as np
import os,sys
import orphics.tools.io as io
import orphics.tools.curvedSky as cs
import matplotlib.pyplot as plt
import pandas as pd
import time
import halofg.sehgalInterface as si

out_dir = os.environ['WWW']+"plots/"
cutout_section = "cutout_default"
SimConfig = io.config_from_file("input/sehgal.ini")
ras,decs,m200s,zs,vzs = si.select_from_halo_catalog(SimConfig,M200_min=2e14,z_min=0.6,z_max=np.inf,Nmax=1000)#None)#1000)
#print len(ras)
#sys.exit()
Npix,arc,pix = si.pix_from_config(SimConfig,cutout_section)

save_dir = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cutouts/"

#components = ["tsz","ksz","cib","radio"]
#components = ["tsz","kappa","cib","radio","ksz"]
components = ["ksz"]
for component in components:

    if component=="kappa":
        hpmap = si.get_kappa(SimConfig,section="kappa")
    else:
        hpmap = si.get_components_map_from_config(SimConfig,148,component)


    stamp = 0.
    for k,(ra,dec,vz) in enumerate(zip(ras,decs,vzs)):
        print k+1
        cutout = hp.visufunc.cutout_gnomonic(hpmap, rot=(ra, dec), coord='C', xsize=Npix, ysize=Npix,reso=pix)
        filename = save_dir + component+"_"+str(k)
        np.save(filename,np.array(cutout))
        if component=="ksz":
            Ny,Nx = cutout.shape
            cpix = cutout[Ny/2,Nx/2]
            mul = np.sign(cpix)
        else:
            mul = 1.
        stamp += cutout*mul

    stamp /= k

    io.quickPlot2d(stamp,out_dir+"stamp_"+component+".png")
