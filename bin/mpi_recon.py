import numpy as np
import sys, os, glob
from orphics.analysis.pipeline import mpi_distribute
import orphics.tools.stats as stats
import alhazen.io as aio
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
import warnings
import logging
logger = logging.getLogger()
with io.nostdout():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from enlib import enmap, lensing
from alhazen.quadraticEstimator import Estimator
import alhazen.lensTools as lt
from ConfigParser import SafeConfigParser 
from szar.counts import ClusterCosmology
import enlib.fft as fftfast
from alhazen.halos import NFWkappa
import argparse

from mpi4py import MPI

parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')

parser.add_argument("dirname", type=str,help='Directory name.')
parser.add_argument("-N", "--nsim",     type=int,  default=None)

args = parser.parse_args()
dirname = args.dirname
Nsims = args.nsim


analysis_section = "analysis_arc"
sim_section = "sims"

simulated_cmb = True
simulated_kappa = True
periodic = True

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


out_dir = os.environ['WWW']+"plots/"
map_root = os.environ['WORK2']+'/data/sehgal/'

save_dir = map_root + dirname

kappa_glob = sorted(glob.glob(save_dir+"/kappa*"))
cmb_glob = sorted(glob.glob(save_dir+"/cmb*"))



Ntotk = len(kappa_glob)
Ntotc = len(cmb_glob)

if rank!=0:
    comm.send(Ntotk,dest=0,tag=99)
    comm.send(Ntotc,dest=0,tag=88)
    Nmin = None
else:
    Ntotks = [Ntotk]
    Ntotcs = [Ntotc]
    for i in range(1,numcores):
        Ntotks.append(comm.recv(source=i,tag=99))
        Ntotcs.append(comm.recv(source=i,tag=88))
    Nmin = min(min(Ntotks,Ntotcs))
Nmin = comm.bcast(Nmin,root=0)
if rank==0: print "Nmin : ", Nmin
cmb_glob = cmb_glob[:Nmin]
kappa_glob = kappa_glob[:Nmin]
Ntot = Nmin


if simulated_cmb and simulated_kappa and (Nsims is not None):
    Ntot = Nsims
    cmb_glob = [""]*Ntot
    kappa_glob = [""]*Ntot

num_each,each_tasks = mpi_distribute(Ntot,numcores)

if rank==0: print "At most ", max(num_each) , " tasks..."

my_tasks = each_tasks[rank]
my_kappa_files = [kappa_glob[i] for i in my_tasks]
my_cmb_files = [cmb_glob[i] for i in my_tasks]


# Read config
iniFile = "input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

expf_name = "experiment_simple"
pol = False
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)    
lmax,tellmin,tellmax,pellmin,pellmax,kellmin,kellmax = aio.ellbounds_from_config(Config,"reconstruction")
parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=True)
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=True)
bin_edges = np.arange(0.,20.,Config.getfloat(analysis_section,"pixel_arcmin")*2.)
binner_dat = stats.bin2D(parray_dat.modrmap*60.*180./np.pi,bin_edges)
binner_sim = stats.bin2D(parray_sim.modrmap*60.*180./np.pi,bin_edges)
lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)


# === COSMOLOGY ===
with io.nostdout():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logger.disabled = True
        cc = ClusterCosmology(lmax=lmax,pickling=True)
        logger.disabled = False
parray_dat.add_cosmology(cc)
theory = cc.theory
gradCut = 2000
TOnly = True
template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
nT = parray_dat.nT
nP = parray_dat.nP
kbeam_dat = parray_dat.lbeam
kbeampass = kbeam_dat
fMaskCMB_T = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=tellmin,lmax=tellmax)
fMaskCMB_P = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=pellmin,lmax=pellmax)
fMask = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=kellmin,lmax=kellmax)

with io.nostdout():
    qest = Estimator(template_dat,
                     cc.theory,
                     theorySpectraForNorm=None,
                     noiseX2dTEB=[nT,nP,nP],
                     noiseY2dTEB=[nT,nP,nP],
                     fmaskX2dTEB=[fMaskCMB_T,fMaskCMB_P,fMaskCMB_P],
                     fmaskY2dTEB=[fMaskCMB_T,fMaskCMB_P,fMaskCMB_P],
                     fmaskKappa=fMask,
                     kBeamX = kbeampass,
                     kBeamY = kbeampass,
                     doCurl=False,
                     TOnly=not(pol),
                     halo=True,
                     uEqualsL=False,
                     gradCut=gradCut,verbose=False,
                     loadPickledNormAndFilters=None,
                     savePickledNormAndFilters=None)

taper_percent = 12.0 if not(periodic) else 0.
pad_percent = 4.0 if not(periodic) else 0.
Ny,Nx = shape_dat
taper = fmaps.cosineWindow(Ny,Nx,lenApodY=int(taper_percent*min(Ny,Nx)/100.),lenApodX=int(taper_percent*min(Ny,Nx)/100.),padY=int(pad_percent*min(Ny,Nx)/100.),padX=int(pad_percent*min(Ny,Nx)/100.))
w2 = np.mean(taper**2.)
w4 = np.mean(taper**4.)
if rank==0:
    io.quickPlot2d(taper,out_dir+"taper.png")
    print "w2 : " , w2


pixratio = Config.getfloat(analysis_section,"pixel_arcmin")/Config.getfloat(sim_section,"pixel_arcmin")
if simulated_cmb or simulated_kappa:
    lens_order = Config.getint(sim_section,"lens_order")
    parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=True)
    parray_sim.add_cosmology(cc)




my_kappa1d_data = []
my_kapparecon_data = []
inp_kappa_stack = 0.
recon_kappa_stack = 0.

random = True if "random" in dirname else False

if random or periodic:
    meanfield = 0.
else:
    meanfield = np.load("/gpfs01/astro/workarea/msyriac/data/sehgal/randoms_15041256892/reconstack.npy")

k = -1
for index,kappa_file,cmb_file in zip(my_tasks,my_kappa_files,my_cmb_files):
    assert kappa_file[-9:]==cmb_file[-9:]
    
    k += 1
    if rank==0: print "Rank ", rank , " doing cutout ", index
    if not(simulated_kappa):
        kappa = enmap.upgrade(enmap.ndmap(np.load(kappa_file),wcs_dat),pixratio)
        phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
        alpha_pix = enmap.grad_pixf(fphi)
    else:
        if k==0:
            massOverh = 2.e14
            zL = 0.7
            overdensity = 180.
            critical = False
            atClusterZ = False
            concentration = 3.2
            comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
            comL = cc.results.comoving_radial_distance(zL)*cc.h
            winAtLens = (comS-comL)/comS
            kappa,r500 = NFWkappa(cc,massOverh,concentration,zL,parray_sim.modrmap* 180.*60./np.pi,winAtLens,
                                      overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
            phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
            alpha_pix = enmap.grad_pixf(fphi)
            

    if not(simulated_cmb):
        cmb = np.load(cmb_file) / 1.072480e+09
    else:
        unlensed = parray_sim.get_unlensed_cmb(seed=index)
        if random and simulated_kappa:
            lensed = unlensed
        else:
            lensed = lensing.lens_map_flat_pix(unlensed.copy(), alpha_pix.copy(),order=lens_order)
        cmb = enmap.downgrade(lensed,pixratio)
                


    kappa = fmaps.filter_map(kappa,kappa*0.+1.,parray_sim.modlmap,lowPass=kellmax,highPass=kellmin)
    cents,kappa1d = binner_sim.bin(kappa)
    my_kappa1d_data.append(kappa1d)
    inp_kappa_stack += kappa


    measured = cmb * taper
    fkmaps = fftfast.fft(measured,axes=[-2,-1])
    qest.updateTEB_X(fkmaps,alreadyFTed=True)
    qest.updateTEB_Y()
    with io.nostdout():
        rawkappa = qest.getKappa("TT").real
    kappa_recon = rawkappa/(taper**2.)-meanfield
    kappa_recon[parray_dat.modrmap*180.*60./np.pi>40.] = 0.
    #kappa_recon = fmaps.filter_map(kappa_recon,kappa_recon*0.+1.,parray_dat.modlmap,lowPass=kellmax,highPass=kellmin)
    cents,kapparecon1d = binner_dat.bin(kappa_recon)
    my_kapparecon_data.append(kapparecon1d)
    recon_kappa_stack += kappa_recon

    if rank==0 and index==0:
        io.quickPlot2d(cmb,out_dir+"cmb.png")
        io.quickPlot2d(measured,out_dir+"mcmb.png")
        io.quickPlot2d(kappa,out_dir+"inpkappa.png")
        io.quickPlot2d(kappa_recon,out_dir+"reconkappa.png")



    

my_kappa1d_data = np.array(my_kappa1d_data)
my_kapparecon_data = np.array(my_kapparecon_data)
if rank!=0:
    assert my_kappa1d_data.shape==(num_each[rank],len(cents))
    comm.Send(my_kappa1d_data, dest=0, tag=13)
    comm.Send(my_kapparecon_data, dest=0, tag=14)
    comm.Send(inp_kappa_stack, dest=0, tag=15)
    comm.Send(recon_kappa_stack, dest=0, tag=16)
else:

    all_kappa1d_data = my_kappa1d_data
    all_kapparecon_data = my_kapparecon_data
    all_inpstack = inp_kappa_stack
    all_reconstack = recon_kappa_stack
    for core in range(1,numcores):
        print "Waiting for core ", core , " / ", numcores
        expected_shape = (num_each[core],len(cents))
        data_vessel = np.empty(expected_shape, dtype=np.float64)
        comm.Recv(data_vessel, source=core, tag=13)
        all_kappa1d_data = np.append(all_kappa1d_data,data_vessel,axis=0)

        data_vessel = np.empty(expected_shape, dtype=np.float64)
        comm.Recv(data_vessel, source=core, tag=14)
        all_kapparecon_data = np.append(all_kapparecon_data,data_vessel,axis=0)


        expected_shape = shape_sim
        data_vessel = np.empty(expected_shape, dtype=np.float64)
        comm.Recv(data_vessel, source=core, tag=15)
        all_inpstack += data_vessel

        expected_shape = shape_dat
        data_vessel = np.empty(expected_shape, dtype=np.float64)
        comm.Recv(data_vessel, source=core, tag=16)
        all_reconstack += data_vessel


    kappa_stats = stats.getStats(all_kappa1d_data)
    kapparecon_stats = stats.getStats(all_kapparecon_data)

    pl = io.Plotter()
    sgn = 1 if simulated_cmb else -1
    pl.addErr(cents,kappa_stats['mean'],yerr=kappa_stats['errmean'],ls="-")
    pl.addErr(cents,sgn*kapparecon_stats['mean'],yerr=kapparecon_stats['errmean'],ls="--")
    pl.done(out_dir+"kappa1d.png")

    if not(random):
        filename = "/profiles_simcmb_"+str(simulated_cmb)+"_simkap_"+str(simulated_kappa)+"_periodic_"+str(periodic)+".txt"
        np.savetxt(save_dir+filename,np.vstack((cents,kappa_stats['mean'],kappa_stats['errmean'],sgn*kapparecon_stats['mean'], \
                                                kapparecon_stats['errmean'])).transpose(), \
                   header="# bin centers (arc) , input_kappa, input_kappa_err, recon_kappa, recon_kappa_err")
            

    io.quickPlot2d(stats.cov2corr(kappa_stats['cov']),out_dir+"kappa_corr.png")
    io.quickPlot2d(all_inpstack/Ntot,out_dir+"inpstack.png")
    io.quickPlot2d(all_reconstack/Ntot,out_dir+"reconstack.png")
    inp = enmap.downgrade(enmap.ndmap(all_inpstack/Ntot,wcs_sim),pixratio) 
    rec = all_reconstack/Ntot
    pdiff = np.nan_to_num((inp-rec)*100./inp)
    io.quickPlot2d(pdiff,out_dir+"pdiffstack.png",lim=20.)
    np.save(save_dir+"/reconstack",all_reconstack/Ntot)
