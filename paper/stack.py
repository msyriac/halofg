from __future__ import print_function
from orphics import maps,io,cosmology,stats,mpi,lensing
from enlib import enmap
import numpy as np
import os,sys
import healpy as hp
import astropy.io.fits as fits
from szar.counts import ClusterCosmology

proot = "/gpfs01/astro/workarea/msyriac/data/planck/"
catfile = "/gpfs01/astro/workarea/msyriac/data/catalogs/redmapper_dr8_public_v6.3_catalog.fits"
hdu = fits.open(catfile)

random = False

N = None
#N = 100


ras = hdu[1].data['RA'][:N]
decs = hdu[1].data['DEC'][:N]

if random:
    nrandom = 200000
    ramin = ras.min()
    ramax = ras.max()
    decmin = decs.min()
    decmax = decs.max()
    ras = np.random.uniform(ramin,ramax,nrandom)
    decs = np.random.uniform(decmin,decmax,nrandom)


lgmca = hp.read_map(proot+"WPR2_CMB_muK.fits")
smica = hp.read_map(proot+"COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits")*1e6
f143 = hp.read_map(proot+"HFI_SkyMap_143_2048_R2.02_full.fits")*1e6

lgmca -= lgmca.mean()
smica -= smica.mean()
f143 -= f143.mean()

arc = 120.
pix = 1.0
Npix = int(arc/pix)
noiseX = 45.
noiseY = 45.
beamX = 5.0
beamY = 5.0

shape,wcs = maps.rect_geometry(width_deg=arc/60.,px_res_arcmin=pix)
Npix = shape[0]
assert Npix == shape[1]
modlmap = enmap.modlmap(shape,wcs)
modrmapA = enmap.modrmap(shape,wcs)*180.*60./np.pi
cc = ClusterCosmology(lmax=6000,pickling=True,dimensionless=False)
theory = cc.theory
lmax = modlmap.max()
ells = np.arange(0,lmax,1)
ucltt = theory.uCl('TT',ells)
ps = ucltt.reshape((1,1,ucltt.size))
nY = (noiseY*np.pi/180./60.)**2.
nX = (noiseX*np.pi/180./60.)**2.
kbeamY = maps.gauss_beam(beamY,modlmap)
kbeamX = maps.gauss_beam(beamX,modlmap)
rbin_edges = np.arange(0.,20.,pix*2.)
rbinner = stats.bin2D(modrmapA,rbin_edges)
taper,w2 = maps.get_taper(shape,taper_percent = 18.0,pad_percent = 5.0,weight=None)


# QE
tellmin = modlmap[modlmap>2].min(); tellmax = 3000; kellmin = tellmin ; kellmax = 3100
tmask = maps.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
kmask = maps.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
qest =  lensing.Estimator(shape,wcs,
                  theory,
                  theorySpectraForNorm=theory,
                  noiseX2dTEB=[modlmap*0.+nX,modlmap*0.+nX,modlmap*0.+nX],
                  noiseY2dTEB=[modlmap*0.+nY,modlmap*0.+nY,modlmap*0.+nY],
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
                  gradCut=2000,
                  verbose=False,
                  loadPickledNormAndFilters=None,
                  savePickledNormAndFilters=None,
                  uEqualsL=False,
                  bigell=9000,
                  mpi_comm=None,
                  lEqualsU=False)





comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Nsims = len(ras)
Njobs = Nsims 
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
st = stats.Stats(comm=comm)

# if not(random):
#     mf = {}
#     for comb in ['LF','SF','SS','LL','FF']:
#         mf[comb] = np.load("stack_"+comb+"_random_"+str(random)+".npy")



for task in my_tasks:
    ra = ras[task]
    dec = decs[task]

    cut_smica = maps.cutout_gnomonic(smica, rot=(ra, dec), coord=['G','C'], xsize=Npix, ysize=Npix,reso=pix)
    cut_lgmca = maps.cutout_gnomonic(lgmca, rot=(ra, dec), coord=['G','C'], xsize=Npix, ysize=Npix,reso=pix)
    cut_143 = maps.cutout_gnomonic(f143, rot=(ra, dec), coord=['G','C'], xsize=Npix, ysize=Npix,reso=pix)

    st.add_to_stack("smica",cut_smica)
    st.add_to_stack("lgmca",cut_lgmca)
    st.add_to_stack("143",cut_143)


    recon = {}
    recon['LF'] = qest.kappa_from_map("TT",cut_lgmca*taper,T2DDataY=cut_143*taper)
    recon['SF'] = qest.kappa_from_map("TT",cut_smica*taper,T2DDataY=cut_143*taper)
    recon['SS'] = qest.kappa_from_map("TT",cut_smica*taper,T2DDataY=cut_smica*taper)
    recon['LL'] = qest.kappa_from_map("TT",cut_lgmca*taper,T2DDataY=cut_lgmca*taper)
    recon['FF'] = qest.kappa_from_map("TT",cut_143*taper,T2DDataY=cut_143*taper)

    for comb in ['LF','SF','SS','LL','FF']:
        st.add_to_stack(comb,recon[comb])
        cents,r1d = rbinner.bin(recon[comb])
        st.add_to_stats(comb+"1d",r1d)

    
    if rank==0 and (task+1)%20==0: print("Rank 0 doing task ", task+1 , " / ", len(my_tasks))

st.get_stacks()
st.get_stats()

if rank==0:
    io.plot_img(st.stacks['smica'],io.dout_dir+"smica.png",extent=[-arc/2.,arc/2.,-arc/2.,arc/2.])
    io.plot_img(st.stacks['lgmca'],io.dout_dir+"lgmca.png",extent=[-arc/2.,arc/2.,-arc/2.,arc/2.])
    io.plot_img(st.stacks['143'],io.dout_dir+"f143.png",extent=[-arc/2.,arc/2.,-arc/2.,arc/2.])

    np.save("stack_smica_random_"+str(random)+".npy",np.nan_to_num(np.asarray(st.stacks["smica"])))
    np.save("stack_lgmca_random_"+str(random)+".npy",np.nan_to_num(np.asarray(st.stacks["lgmca"])))
    np.save("stack_143_random_"+str(random)+".npy",np.nan_to_num(np.asarray(st.stacks["143"])))

    
    for comb in ['LF','SF','SS','LL','FF']:
        io.plot_img(st.stacks[comb],io.dout_dir+"recon"+comb+".png",extent=[-arc,arc,-arc,arc])
        np.save("stack_"+comb+"_random_"+str(random)+".npy",st.stacks[comb])
        
