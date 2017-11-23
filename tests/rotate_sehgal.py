import numpy as np
import halofg.sehgalInterface as si
import os,sys
import orphics.io as io
import healpy as hp
from enlib import bench

SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()


eulers = np.loadtxt("data/eulers_98percent.txt")
print eulers.shape
psi,theta,phi = eulers[:,0]

print psi, theta, phi

with bench.show("hpmap load"):
    hp_map = si.get_component_map_from_config(PathConfig,SimConfig,"148_ksz",base_nside=None)

nside = 8192

with bench.show("alm"):
    alm=hp.map2alm(hp_map)
with bench.show("rotalm"):
    hp.rotate_alm(alm,psi*np.pi/180.,theta*np.pi/180.,phi*np.pi/180.)
with bench.show("invalm"):
    hmap_new=hp.alm2map(alm,nside=nside,verbose=False)

