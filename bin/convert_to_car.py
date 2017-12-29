from __future__ import print_function
from enlib import enmap,bench
import matplotlib.pyplot as plt
from orphics import catalogs,io,maps
import healpy as hp
import numpy as np
import halofg.sehgalInterface as si


import argparse
parser = argparse.ArgumentParser(description='Convert healpix maps to CAR')
parser.add_argument("MapName", type=str,help='component map filename')
parser.add_argument("-r", "--res",     type=float,  default=0.5,help="Arcminute resolution.")
parser.add_argument("-d", "--decmax",     type=float,  default=55.,help="Maximum declination in degrees.")
args = parser.parse_args()


SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()





    
df,ra,dec,m200,z = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=0.,M200_max=np.inf,z_min=0.,z_max=np.inf,Nmax=None,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None,mirror=False)
all_halos = ra.size

shape,wcs = enmap.geometry(pos=np.array([[-args.decmax,-5.],[args.decmax,95.]])*np.pi/180.,res=args.res*np.pi/180./60.)
print ("Area sq.deg. : ", enmap.area(shape,wcs)*(180./np.pi)**2.)
cats = catalogs.CatMapper(ra,dec,shape=shape,wcs=wcs)
unique = cats.counts.sum()
print ("All halos : ", all_halos)
print ("Unique halos : ", unique)
print ("Percentage retained : ", unique*100./all_halos , " %")


map_root = PathConfig.get("paths","input_data")
hp_map = hp.read_map(map_root+args.MapName+".fits")


carmap = maps.enmap_from_healpix(shape,wcs,hp_map,"j2000",True)
out_name = map_root+args.MapName+"_car_"+str(args.res)+"_"+str(args.decmax)+".fits"
enmap.write_map(out_name,carmap)

# Test stack
istack = maps.InterpStack(20.,0.5)
df,ra,dec,m200,z = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=1.e14,M200_max=np.inf,z_min=0.,z_max=np.inf,Nmax=100,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None,mirror=True)
stack=0.
for ira,idec in zip(ra,dec):
    stack += istack.cutout_from_file(out_name,shape,wcs,ira,idec)
    
io.plot_img(stack,io.dout_dir+"stack_"+args.MapName+".png")
