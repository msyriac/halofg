from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs,lensing,mpi
from enlib import enmap,lensing as enlensing,resample
import numpy as np
import os,sys
import healpy as hp

import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])


out_dir = "/gpfs01/astro/workarea/msyriac/data/depot/halofg/covclkh/"

comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
index = int(sys.argv[1])
Nsims = int(sys.argv[2])

np.random.seed(index)

tsz_scale = 0.002541306131582873*0.91

num_cutouts = 42
i = index
fstr = "kappa"
fname = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cutouts/"+fstr+"_car_unrotated_tapered_"+str(i)+".hdf"
kappa = enmap.read_map(fname)

fstr = "tsz_148"
fname = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cutouts/"+fstr+"_car_unrotated_tapered_"+str(i)+".hdf"
tsz = enmap.read_map(fname)*tsz_scale

#fstr = "halo_delta"
fstr = "halo_delta_vrestricted"
fname = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cutouts/"+fstr+"_car_unrotated_tapered_"+str(i)+".hdf"
delta = enmap.read_map(fname)

fname = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cutouts/taper_car_unrotated_"+str(i)+".hdf"
taper = enmap.read_map(fname)
w2 = np.mean(taper**2.)
w3 = np.mean(taper**3.)


shape,wcs = kappa.shape,kappa.wcs
fc = maps.FourierCalc(shape,wcs)
cc = cosmology.Cosmology(lmax=4000,pickling=True,dimensionless=False)
theory = cc.theory

modlmap = enmap.modlmap(shape,wcs)

print("Fting...")
p2d,kkappa,_ = fc.power2d(kappa)

bin_edges = np.arange(100,3000,160)
binner = stats.bin2D(modlmap,bin_edges)

cents,p1dkk = binner.bin(p2d)



ells = np.arange(2,3000,1)
clkk = theory.gCl('kk',ells)

p2d,kdelta = fc.f1power(delta,kkappa)
cents,p1dkh = binner.bin(p2d)

p2d,ksz = fc.f1power(tsz,kdelta)
cents,p1dszgal = binner.bin(p2d)

# if rank==0: 
#     pl = io.Plotter(yscale='log')
#     pl.add(cents,-p1dszgal/w2)
#     # pl.hline()
#     pl.done(io.dout_dir+"sehgalclszgal.png")


lmax = modlmap.max()
lells = np.arange(0,lmax,1)
cltt = theory.uCl('TT',lells)
ps = cltt.reshape((1,1,lells.size))

noiseX = 45.0
noiseY = 10.0
# noiseX = 0.01
# noiseY = 0.01
beamX = 5.0
beamY = 1.5
# beamX = 0.01
# beamY = 0.01
pnoiseX = (noiseX*np.pi/180./60.)**2.
pnoiseY = (noiseY*np.pi/180./60.)**2.


kbeamX = maps.gauss_beam(modlmap,beamX)
kbeamY = maps.gauss_beam(modlmap,beamY)

psx = (lells*0.+pnoiseX).reshape((1,1,lells.size))
psy = (lells*0.+pnoiseY).reshape((1,1,lells.size))
mgen = maps.MapGen(shape,wcs,ps)
ngenx = maps.MapGen(shape,wcs,psx)
ngeny = maps.MapGen(shape,wcs,psy)


nX = modlmap*0. + pnoiseX #0.
nY = modlmap*0. + pnoiseY #0.
#kbeamX = modlmap*0.+1.
#kbeamY = modlmap*0.+1.
gradcut = None

# QE
tellmin = 100; tellmax = 3000; kellmin = tellmin ; kellmax = 3096
tmask = maps.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
kmask = maps.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
# qestYY =  lensing.Estimator(shape,wcs,
#                   theory,
#                   theorySpectraForNorm=theory,
#                   noiseX2dTEB=[nY,modlmap*0.+nY,modlmap*0.+nY],
#                   noiseY2dTEB=[nY,modlmap*0.+nY,modlmap*0.+nY],
#                   noiseX_is_total = False,
#                   noiseY_is_total = False,
#                   fmaskX2dTEB=[tmask,tmask,tmask],
#                   fmaskY2dTEB=[tmask,tmask,tmask],
#                   fmaskKappa=kmask,
#                   kBeamX = kbeamY,
#                   kBeamY = kbeamY,
#                   doCurl=False,
#                   TOnly=True,
#                   halo=True,
#                   gradCut=gradcut,
#                   verbose=False,
#                   loadPickledNormAndFilters=None,
#                   savePickledNormAndFilters=None,
#                   uEqualsL=True,
#                   bigell=9000,
#                   mpi_comm=None,
#                   lEqualsU=False)


qestXY =  lensing.Estimator(shape,wcs,
                  theory,
                  theorySpectraForNorm=theory,
                  noiseX2dTEB=[nX,modlmap*0.+nX,modlmap*0.+nX],
                  noiseY2dTEB=[nY,modlmap*0.+nY,modlmap*0.+nY],
                  noiseX_is_total = False,
                  noiseY_is_total = False,
                  fmaskX2dTEB=[tmask,tmask,tmask],
                  fmaskY2dTEB=[tmask,tmask,tmask],
                  fmaskKappa=kmask,
                  kBeamX = kbeamX,
                  kBeamY = kbeamY,
                  doCurl=False,
                  TOnly=True,
                  halo=True,
                  gradCut=gradcut,
                  verbose=False,
                  loadPickledNormAndFilters=None,
                  savePickledNormAndFilters=None,
                  uEqualsL=True,
                  bigell=9000,
                  mpi_comm=None,
                  lEqualsU=False)


Njobs = Nsims 
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
st = stats.Stats(comm=comm)


posmap = enmap.posmap(shape,wcs)
kellmax = 5000
kmask = maps.mask_kspace(shape,wcs,lmax=kellmax)
phi,_ = lensing.kappa_to_phi(enmap.enmap(maps.filter_map(enmap.enmap(kappa,wcs),kmask),wcs),modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
pos = posmap + grad_phi
alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)



for task in my_tasks:

    if rank==0:
        print("Lensing...")
    unlensed = mgen.get_map(seed=task)*taper
    lensed = enlensing.displace_map(unlensed, alpha_pix, order=5)

    # if task==0:
    #     io.plot_img(lensed,io.dout_dir+"lensed_highres.png",high_res=True)
    #     io.plot_img(lensed,io.dout_dir+"lensed.png",high_res=False)

    if rank==0:
        print("Reconstructing...")

    nmapX = ngenx.get_map(seed=int(1e5)+task)
    nmapY = ngeny.get_map(seed=int(1e6)+task)
        
    obsYclean = maps.filter_map(lensed.copy(),kbeamY)+nmapY
    obsY = maps.filter_map(lensed.copy()+tsz.copy(),kbeamY)+nmapY
    obsXclean = maps.filter_map(lensed.copy(),kbeamX)+nmapX
    obsX = maps.filter_map(lensed.copy()+tsz.copy(),kbeamX)+nmapX


    uobsY = maps.filter_map(unlensed.copy(),kbeamY)+nmapY
    uobsX = maps.filter_map(unlensed.copy(),kbeamX)+nmapX
    
    
    urecon = qestXY.kappa_from_map("TT",uobsX,T2DDataY=uobsY)
    recon = qestXY.kappa_from_map("TT",obsXclean,T2DDataY=obsYclean)
    reconYY = qestXY.kappa_from_map("TT",obsX,T2DDataY=obsY)
    reconY = qestXY.kappa_from_map("TT",obsXclean,T2DDataY=obsY)
    p2d,krecon = fc.f1power(recon,kkappa)
    cents,p1d = binner.bin(p2d)

    if task==0:
        io.plot_img(lensed+tsz,io.dout_dir+"lensedsz_highres.png",high_res=True)
        io.plot_img(lensed+tsz,io.dout_dir+"lensedsz.png",high_res=False)


    p2d,kreconYY = fc.f1power(reconYY,kkappa)
    cents,p1dYY = binner.bin(p2d)

    p2d,kreconY = fc.f1power(reconY,kkappa)
    cents,p1dY = binner.bin(p2d)


    p2d = fc.f2power(krecon,kdelta)
    cents,clkhest = binner.bin(p2d)

    p2d = fc.f2power(kreconYY,kdelta)
    cents,clkhestYY = binner.bin(p2d)

    p2d = fc.f2power(kreconY,kdelta)
    cents,clkhestY = binner.bin(p2d)

    st.add_to_stack("mf",urecon)
    st.add_to_stats("ik",p1d)
    st.add_to_stats("ikYY",p1dYY)
    st.add_to_stats("ikY",p1dY)
    st.add_to_stats("hk",clkhest)
    st.add_to_stats("hkYY",clkhestYY)
    st.add_to_stats("hkY",clkhestY)

    diffYY = (p1dYY-p1d)/p1d
    diffY = (p1dY-p1d)/p1d

    st.add_to_stats("ikYYdiff",diffYY)
    st.add_to_stats("ikYdiff",diffY)

    
    hdiffYY = (clkhestYY-clkhest)/clkhest
    hdiffY = (clkhestY-clkhest)/clkhest

    st.add_to_stats("hkYYdiff",hdiffYY)
    st.add_to_stats("hkYdiff",hdiffY)

    if rank==0: print("Rank 0 doing task ", task , " / ", len(my_tasks))
    
st.get_stats()
st.get_stacks()
    
if rank==0:

    mf = st.stacks['mf']
    p2d,kmf = fc.f1power(mf,kdelta)
    cents,p1dmf = binner.bin(p2d)

    clkhest = st.stats['hk']['mean'] - p1dmf
    clkhestYY = st.stats['hkYY']['mean'] - p1dmf
    clkhestY = st.stats['hkY']['mean'] - p1dmf

    #io.save_cols(out_dir+"clkhs_bin_160_"+str(index)+".txt",(cents,clkhest/w3,clkhestYY/w3,clkhestY/w3))
    io.save_cols(out_dir+"clkhs_bin_160_vrestricted_"+str(index)+".txt",(cents,clkhest/w3,clkhestYY/w3,clkhestY/w3))

