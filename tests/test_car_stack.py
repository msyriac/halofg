from __future__ import print_function
from orphics import maps,io,cosmology,stats
from enlib import enmap,bench
import numpy as np
import os,sys
from halofg.sehgalInterface import StampReader,select_from_halo_catalog
import halofg.sehgalInterface as si

SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()
cutout_section = "cutout_cluster"
sections = ["kappa","148_tsz","148_irpts","148_radpts","148_ksz"]
sreader = StampReader(PathConfig,SimConfig,cutout_section,sections,suffix="car_half_arcmin")

mmin = 4e14
mmax = np.inf
zmin = 0.
zmax = np.inf
nmax = 30
df,ras,decs,m200,z = select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=mmin,M200_max=mmax,z_min=zmin,z_max=zmax,Nmax=nmax,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None,dec_min=1.,dec_max=15.)



# test_hp = si.get_kappa(PathConfig,SimConfig,section="kappa",base_nside=None)

s = stats.Stats()

for k,(ra,dec) in enumerate(zip(ras,decs)):
    print(k)
    stamp = sreader.get_stamp("148_tsz",ra,dec)
    Ny,Nx = stamp.shape

    # tstamp = maps.cutout_gnomonic(test_hp,rot=(ra,dec),coord='C',
    #          xsize=Ny,ysize=Nx,reso=0.5,
    #          nest=False,remove_dip=False,
    #          remove_mono=False,gal_cut=0,
    #          flip='astro')
    

    # io.plot_img(stamp,io.dout_dir+"stest_qwert_"+str(k)+".png")
    # sys.exit()
    s.add_to_stack("tsz",stamp)
    # s.add_to_stack("test",tstamp)

s.get_stacks()
print (s.stacks['tsz'].shape)
io.plot_img(s.stacks['tsz'],io.dout_dir+"stacksz.png")
# io.plot_img(s.stacks['test'],io.dout_dir+"stacktest.png")
