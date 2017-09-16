import healpy as hp
import numpy as np
import os,sys
import orphics.tools.io as io
import orphics.tools.curvedSky as cs
import matplotlib.pyplot as plt
import pandas as pd
import time
import halofg.sehgalInterface as si
from orphics.analysis.pipeline import mpi_distribute, MPIStats
from mpi4py import MPI
import time


# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    

out_dir = os.environ['WWW']+"plots/"
cutout_section = "cutout_cluster"
SimConfig = io.config_from_file("input/sehgal.ini")
cutout_section = "cutout_cluster"
Nmax = 500
ras,decs,m200s,zs = si.select_from_halo_catalog(SimConfig,M200_min=2e14,z_min=0.3,Nmax=Nmax)

Npix,arc,pix = si.pix_from_config(SimConfig,cutout_section)

Ntot = Nmax

# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Ntot,numcores-1)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333000,root=1)

if rank==0: print "At most ", max(num_each) , " tasks..."

# What am I doing?
if rank>0: my_tasks = each_tasks[rank-1] #!!!

ras = ras.tolist()
decs = decs.tolist()


if rank==0:
    hpmap = si.get_kappa(SimConfig)

    for core in range(1,numcores):
        for index in each_tasks[core-1]:
            cutout = hp.visufunc.cutout_gnomonic(hpmap, rot=(ras[index], decs[index]), coord='C', xsize=Npix, ysize=Npix,reso=pix)
            send_dat = np.array(cutout).astype(np.float64)
            comm.Send(send_dat, dest=core, tag=index)
            print "Sending ",index," to core ", core, " / ", numcores
    print "Done sending."

if rank>0:    
    for k,index in enumerate(my_tasks):

        expected_shape = (Npix,Npix)
        data_vessel = np.empty(expected_shape, dtype=np.float64)
        comm.Recv(data_vessel, source=0, tag=index)
        cutout = data_vessel
        
        mpibox.add_to_stack("kappas",cutout)
        time.sleep(10)

    mpibox.get_stacks()

    if rank==1:
        kappastack = mpibox.stacks["kappas"]
        io.quickPlot2d(kappastack,out_dir+"kappastack.png")
    
