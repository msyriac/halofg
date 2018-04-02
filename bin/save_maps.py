from __future__ import print_function
import healpy as hp
import numpy as np
import os,sys
from orphics import io, maps
import halofg.sehgalInterface as si
from enlib import enmap

import argparse
parser = argparse.ArgumentParser(description='Save maps from Sehgal et. al. sims')
parser.add_argument("MapType", type=str,help='kappa/tsz/ksz/radio/cib')
args = parser.parse_args()

# ACT MAP LOADER
config_yaml_path = "../pvt_config/actpol_maps.yaml"
mreader = maps.ACTPolMapReader(config_yaml_path)
season = "s14"
patch = "deep56"
array = "pa1"
freq = "150"
day_night = "night"
act_coadd,identifier = mreader.get_map(split=-1,season=season,patch=patch,array=array,freq=freq,day_night=day_night,
                                       full_map=False,weight=False,get_identifier=True,t_only=True)

act_coadd = act_coadd[0]
shape,wcs = act_coadd.shape, act_coadd.wcs
SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()

if args.MapType=="kappa":
    hp_map = si.get_kappa(PathConfig,SimConfig,section="kappa",base_nside=None)
else:
    hp_map = si.get_component_map_from_config(PathConfig,SimConfig,args.MapType,base_nside=None)

hp_map -= hp_map.mean()
print(np.mean(hp_map),np.var(hp_map))
imap = maps.enmap_from_healpix(shape,wcs,hp_map,hp_coords="equatorial",interpolate=True)

print(imap.shape)


enmap.write_map("/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cache/sehgal_d56_"+args.MapType+".hdf",imap)

io.plot_img(imap,io.dout_dir+args.MapType+"_sehgald56.png",high_res=True)
