from __future__ import print_function
from orphics import maps,io,cosmology
from enlib import enmap
import numpy as np
import os,sys
import halofg.sehgalInterface as si

class SimpleExperiment(object):

    def __init__(self,shape,wcs,freqs,noises,fg_noises,beams=None,beam_150=None):
        """
        freqs - list of GHz bandpass means
        
        """
        
        if beams is None:
            assert beam_150 is not None
            beams = [beam_150*150./freq for freq in freqs]
        


cache_dir = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cache/"
components = ['tsz','irpts','radpts']

arc_width = 30.
px = 0.5
Nmax = None

istack = maps.InterpStack(arc_width,px,proj="car")

bmaps = {}

for component in components:
    bmaps[component] = enmap.read_map(cache_dir+"sehgal_d56_148_"+component+".hdf")

SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()
df,ras,decs,m200,z = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=2e14,M200_max=np.inf,z_min=0.3,z_max=np.inf,Nmax=Nmax,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None)

c = 0
for k,(ra,dec) in enumerate(zip(ras,decs)):
    tsz = istack.cutout(bmaps['tsz'],ra,dec)
    if tsz is None: continue
    cib = istack.cutout(bmaps['irpts'],ra,dec)
    rad = istack.cutout(bmaps['radpts'],ra,dec)
    #print(tsz.shape)
    c += 1
    if c==1:
        io.plot_img(tsz,io.dout_dir+"tsz.png")
        io.plot_img(cib,io.dout_dir+"cib.png")
        io.plot_img(rad,io.dout_dir+"rad.png")
print("done")
