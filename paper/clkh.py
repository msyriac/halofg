from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs,lensing,mpi
from enlib import enmap,lensing as enlensing,resample
import numpy as np
import os,sys
import healpy as hp

import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])

np.random.seed(100)

comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Nsims = int(sys.argv[1])

pix = 0.5
box = np.array([[-0.1,-0.1],[15.,90.1]])*np.pi/180.
shape,wcs = enmap.geometry(pos=box,res=pix*np.pi/180./60.)

kappa_name = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/kappa_car_0p5arcmin_1300sqdeg.fits"
try:
    kappa_map = enmap.read_fits(kappa_name)
except:
    hp_map = hp.read_map("/gpfs01/astro/workarea/msyriac/data/sims/sehgal/healpix_4096_KappaeffLSStoCMBfullsky.fits")
    kappa_map = maps.enmap_from_healpix(shape,wcs,hp_map,hp_coords="j2000",interpolate=True)
    enmap.write_map(kappa_name,kappa_map)
    
sz_name = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/148_tsz_car_0p5arcmin_1300sqdeg.fits"
try:
    sz_map = enmap.read_fits(sz_name)
except:
    hp_map = hp.read_map("/gpfs01/astro/workarea/msyriac/data/sims/sehgal/148_tsz_healpix.fits")
    sz_map = maps.enmap_from_healpix(shape,wcs,hp_map,hp_coords="j2000",interpolate=True)
    enmap.write_map(sz_name,sz_map)

    
sz_map *= 0.002541306131582873*0.91

taper,w2 = maps.get_taper_deg(shape,wcs,taper_width_degrees = 1.0,pad_width_degrees = 0.5,weight=None)
pix = 2.0
shape,wcs = enmap.geometry(pos=box,res=pix*np.pi/180./60.)
kappa_map = enmap.enmap(resample.resample_fft(kappa_map*taper,shape),wcs)
sz_map = enmap.enmap(resample.resample_fft(sz_map*taper,shape),wcs)
taper,w2 = maps.get_taper_deg(shape,wcs,taper_width_degrees = 1.0,pad_width_degrees = 0.5,weight=None)


if rank==0: io.plot_img(sz_map,io.dout_dir+"szmap.png",high_res=False)

fc = maps.FourierCalc(shape,wcs)
cc = cosmology.Cosmology(lmax=4000,pickling=True,dimensionless=False)
theory = cc.theory

if rank==0: print(kappa_map.area()*(180./np.pi)**2.)

# io.plot_img(sz_map,io.dout_dir+"sz.png",high_res=True)


# io.plot_img(kappa_map*taper,io.dout_dir+"kappa.png",high_res=True)
modlmap = enmap.modlmap(shape,wcs)

print("Fting...")
p2d,kkappa,_ = fc.power2d(kappa_map)

bin_edges = np.arange(10,3000,80)
binner = stats.bin2D(modlmap,bin_edges)

cents,p1dkk = binner.bin(p2d)



ells = np.arange(2,3000,1)
clkk = theory.gCl('kk',ells)

# pl = io.Plotter(yscale='log')
# pl.add(ells,clkk)
# pl.add(cents,p1dkk/w2)
# pl.done(io.dout_dir+"sehgalclkk.png")


import halofg.sehgalInterface as si

PathConfig = io.load_path_config()
SimConfig = io.config_from_file("input/sehgal.ini")
mmin = 1.e13
zmin = 0.
zmax = 1.5
#zmin = 0.45
#zmax = 0.6
df,ras,decs,m200,z = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=mmin,M200_max=np.inf,z_min=zmin,z_max=zmax,Nmax=None,random_sampling=True,mirror=False)

cmapper = catalogs.CatMapper(ras,decs,shape=shape,wcs=wcs)

delta = cmapper.get_delta()

# io.plot_img(delta,io.dout_dir+"delta.png",high_res=True)

p2d,kdelta = fc.f1power(delta*taper,kkappa)
cents,p1dkh = binner.bin(p2d)

p2d,ksz = fc.f1power(sz_map,kdelta)
cents,p1dszgal = binner.bin(p2d)

if rank==0: 
    pl = io.Plotter(yscale='log')
    pl.add(cents,-p1dszgal/w2)
    # pl.hline()
    pl.done(io.dout_dir+"sehgalclszgal.png")


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
tellmin = modlmap[modlmap>2].min(); tellmax = 3000; kellmin = tellmin ; kellmax = 3096
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
phi,_ = lensing.kappa_to_phi(enmap.enmap(maps.filter_map(enmap.enmap(kappa_map,wcs),kmask),wcs),modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
pos = posmap + grad_phi
alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)



for task in my_tasks:

    if rank==0:
        print("Lensing...")
    unlensed = mgen.get_map(seed=task)
    lensed = enlensing.displace_map(unlensed*taper, alpha_pix, order=5)

    if task==0:
        io.plot_img(lensed,io.dout_dir+"lensed_highres.png",high_res=True)
        io.plot_img(lensed,io.dout_dir+"lensed.png",high_res=False)

    if rank==0:
        print("Reconstructing...")

    nmapX = ngenx.get_map(seed=int(1e5)+task)
    nmapY = ngeny.get_map(seed=int(1e6)+task)
        
    obsYclean = maps.filter_map(lensed.copy(),kbeamY)+nmapY
    obsY = maps.filter_map(lensed.copy()+sz_map.copy(),kbeamY)+nmapY
    obsXclean = maps.filter_map(lensed.copy(),kbeamX)+nmapX
    obsX = maps.filter_map(lensed.copy()+sz_map.copy(),kbeamX)+nmapX
    
    
    recon = qestXY.kappa_from_map("TT",obsXclean,T2DDataY=obsYclean)
    reconYY = qestXY.kappa_from_map("TT",obsX,T2DDataY=obsY)
    reconY = qestXY.kappa_from_map("TT",obsXclean,T2DDataY=obsY)
    p2d,krecon = fc.f1power(recon,kkappa)
    cents,p1d = binner.bin(p2d)

    if task==0:
        io.plot_img(lensed+sz_map,io.dout_dir+"lensedsz_highres.png",high_res=True)
        io.plot_img(lensed+sz_map,io.dout_dir+"lensedsz.png",high_res=False)


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
    
if rank==0:
    
    w3 = np.mean(taper**3.)

    p1d = st.stats['ik']['mean']
    p1dYY = st.stats['ikYY']['mean']
    p1dY = st.stats['ikY']['mean']
    
    pl = io.Plotter(yscale='log')
    pl.add(ells,clkk)
    pl.add(cents,p1dkk/w2)
    pl.add(cents,p1d/w3)
    pl.add(cents,p1dYY/w3)
    pl.add(cents,p1dY/w3)
    pl.done(io.dout_dir+"sehgalclkk.png")

    pl = io.Plotter()
    diffYY,ediffYY = st.stats['ikYYdiff']['mean'],st.stats['ikYYdiff']['errmean'] #(p1dYY-p1d)/p1d
    diffY,ediffY = st.stats['ikYdiff']['mean'],st.stats['ikYdiff']['errmean'] #(p1dY-p1d)/p1d
    pl.add_err(cents,diffYY,yerr=ediffYY,label="unclean grad",marker="o",ls="-")
    pl.add_err(cents,diffY,yerr=ediffY,label="clean grad",marker="o",ls="-")
    pl.hline()
    pl._ax.set_ylim(-0.2,0.2)
    pl.legend()
    pl.done(io.dout_dir+"sehgalclkkdiff.png")

    clkhest = st.stats['hk']['mean']
    clkhestYY = st.stats['hkYY']['mean']
    clkhestY = st.stats['hkY']['mean']

    pl = io.Plotter(yscale='log')
    pl.add(cents,p1dkh/w2)
    pl.add(cents,clkhest/w3)
    pl.add(cents,clkhestYY/w3)
    pl.add(cents,clkhestY/w3)
    pl.done(io.dout_dir+"sehgalclkh.png")


    pl = io.Plotter(ylabel="$\\Delta C^{\\kappa g}_L / C^{\\kappa g}_L$",xlabel="$L$")
    diffYY,ediffYY = st.stats['hkYYdiff']['mean'],st.stats['hkYYdiff']['errmean'] #(p1dYY-p1d)/p1d
    diffY,ediffY = st.stats['hkYdiff']['mean'],st.stats['hkYdiff']['errmean'] #(p1dY-p1d)/p1d
    pl.add_err(cents,diffYY,yerr=ediffYY,label="tSZ contaminated gradient",marker="o",ls="-")
    pl.add_err(cents,diffY,yerr=ediffY,label="Clean gradient",marker="o",ls="-")
    pl.hline()
    pl.legend()
    pl._ax.set_ylim(-0.25,0.4)
    pl._ax.set_xlim(2,2500)
    pl.done(io.dout_dir+"sehgalclkhdiff_2000_same.pdf")
    print(diffY*100.)
    print(diffYY*100.)
