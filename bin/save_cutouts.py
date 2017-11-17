import pandas as pd
import numpy as np
import halofg.sehgalInterface as si
import os,sys
import orphics.tools.io as io
from orphics.tools.mpi import MPI, mpi_distribute, MPIStats
import orphics.tools.curvedSky as curved
import healpy as hp
import logging, time


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

import argparse
parser = argparse.ArgumentParser(description='Save cutouts from Sehgal et. al. sims')
parser.add_argument("Out", type=str,help='Output Directory Name (not path, that\'s specified in ini')
parser.add_argument("Bin", type=str,help='Section of ini specifying mass/z bin from catalog')
parser.add_argument("MapType", type=str,help='kappa/tsz/ksz/radio/cib')
parser.add_argument("-r", "--random", action='store_true',help='Random.')

args = parser.parse_args()

random = args.random

SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()

logging.basicConfig(filename="saved_cuts"+str(time.time()*10)+".log",level=logging.DEBUG,format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',datefmt='%m-%d %H:%M',filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


mmin = SimConfig.getfloat(args.Bin,'mass_min')
mmax = SimConfig.getfloat(args.Bin,'mass_max')
zmin = SimConfig.getfloat(args.Bin,'z_min')
zmax = SimConfig.getfloat(args.Bin,'z_max')
nmax = SimConfig.get(args.Bin,'N_max')
nmax = None if nmax=="inf" else int(nmax)

df,ra,dec,m200,z = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=mmin,M200_max=mmax,z_min=zmin,z_max=zmax,Nmax=nmax,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None)

Nuse = len(ra)
ra = ra.tolist()
dec = dec.tolist()


num_each,each_tasks = mpi_distribute(Nuse,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)
if rank==0: logging.info("At most "+ str(max(num_each)) + " tasks...")
# What am I doing?
my_tasks = each_tasks[rank]

if args.MapType=="kappa":
    hp_map = si.get_kappa(PathConfig,SimConfig,section="kappa",base_nside=None)
else:
    hp_map = si.get_component_map_from_config(PathConfig,SimConfig,args.MapType,base_nside=None)


Npix,arc,pix =  si.pix_from_config(SimConfig,cutout_section="cutout_128")
import time
k = 0
a = time.time()
Ncheck = 10

plot_dir = PathConfig.get("paths","plots")+args.Out+"/"+args.Bin+"/"
save_dir = PathConfig.get("paths","input_data")+args.Out+"/"+args.Bin+"/"
if rank==0:
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

comm.Barrier()

    
for index in my_tasks:

    if random:
        ra_now = np.random.uniform(0.,90.)
        dec_now = np.random.uniform(0.,90.)
    else:
        ra_now = ra[index]
        dec_now = dec[index]

    cutout = curved.cutout_gnomonic(hp_map, rot=(ra_now, dec_now), coord='C', xsize=Npix, ysize=Npix,reso=pix)
    cutout = np.asarray(cutout.astype(np.float32))
    mpibox.add_to_stack(args.MapType,cutout)
    np.save(save_dir+"cutout_"+args.MapType+"_"+str(index),cutout)

    k+=1

    if rank==0:
        if k%Ncheck==0:
            b = time.time()
            avg = (b-a)*1./Ncheck
            remaining = avg*(len(my_tasks)-k)/60.
            logging.info(str(remaining)+ " minutes left...")
            a = b


mpibox.get_stacks()            
if rank==0:

    io.quickPlot2d(mpibox.stacks[args.MapType],plot_dir+args.MapType+"_"+args.Bin+"_stack.png")
    logging.info("Done!")
