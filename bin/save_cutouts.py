import pandas as pd
import numpy as np
import halofg.sehgalInterface as si
import os,sys
import orphics.tools.io as io
from orphics.analysis.pipeline import mpi_distribute, MPIStats
from mpi4py import MPI
import healpy as hp


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

import argparse
parser = argparse.ArgumentParser(description='Save cutouts form Sehgal et. al. sims')
parser.add_argument("Out", type=str,help='Output Directory Name (not path, that\'s specified in ini')
parser.add_argument("Bin", type=str,help='Section of ini specifying mass/z bin from catalog')
parser.add_argument("MapType", type=str,help='kappa/tsz/ksz/radio/cib')

args = parser.parse_args()

SimConfig = io.config_from_file("input/sehgal.ini")


mmin = SimConfig.getfloat(args.Bin,'mass_min')
mmax = SimConfig.getfloat(args.Bin,'mass_max')
zmin = SimConfig.getfloat(args.Bin,'z_min')
zmax = SimConfig.getfloat(args.Bin,'z_max')
nmax = SimConfig.get(args.Bin,'N_max')
nmax = None if nmax=="inf" else int(nmax)

df,ra,dec,m200,z = si.select_from_halo_catalog(SimConfig,catalog_section='catalog_default',M200_min=mmin,M200_max=mmax,z_min=zmin,z_max=zmax,Nmax=nmax,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None)

Nuse = len(ra)
ra = ra.tolist()
dec = dec.tolist()


num_each,each_tasks = mpi_distribute(Nuse,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: print "At most ", max(num_each) , " tasks..."
# What am I doing?
my_tasks = each_tasks[rank]

if args.MapType=="kappa":
    hp_map = si.get_kappa(SimConfig,section="kappa",base_nside=None)
else:
    hp_map = si.get_component_map_from_config(SimConfig,args.MapType,base_nside=None)


Npix,arc,pix =  si.pix_from_config(SimConfig,cutout_section="cutout_default")
import time
k = 0
a = time.time()
Ncheck = 100

save_dir = SimConfig.get("sims","map_root")+args.Out+"/"+args.Bin+"/"
if rank==0:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

comm.Barrier()

    
for index in my_tasks:
    
    cutout = hp.visufunc.cutout_gnomonic(hp_map, rot=(ra[index], dec[index]), coord='C', xsize=Npix, ysize=Npix,reso=pix)
    cutout = np.asarray(cutout.astype(np.float32))
    mpibox.add_to_stack(args.MapType,cutout)
    np.save(save_dir+"cutout_"+args.MapType+"_"+str(index),cutout)

    k+=1

    if rank==0:
        if k%Ncheck==0:
            b = time.time()
            avg = (b-a)*1./Ncheck
            remaining = avg*(len(my_tasks)-k)/60.
            print remaining, " minutes left..."
            a = b


mpibox.get_stacks()            
if rank==0:

    io.quickPlot2d(mpibox.stacks[args.MapType],io.dout_dir+args.MapType+"_"+args.Bin+"_stack.png")
