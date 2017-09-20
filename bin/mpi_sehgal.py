import numpy as np
import sys, os, glob
from orphics.analysis.pipeline import mpi_distribute, MPIStats
import orphics.tools.stats as stats
import orphics.tools.cmb as cmb
import alhazen.io as aio
import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
import warnings
import logging
logger = logging.getLogger()
with io.nostdout():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from enlib import enmap, lensing, resample
from alhazen.quadraticEstimator import Estimator
import alhazen.lensTools as lt
from ConfigParser import SafeConfigParser 
from szar.counts import ClusterCosmology
import enlib.fft as fftfast
import argparse
from mpi4py import MPI
from flipper.fft import fft,ifft

sim_root = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cutouts/"
def load_sehgal_kappa(index):
    filename = sim_root+"kappa_"+str(index)+".npy"
    return np.load(filename)

def load_sehgal_component(comp,index):
    filename = sim_root+comp+"_"+str(index)+".npy"
    return np.load(filename)

# Runtime params that should be moved to command line
analysis_section = "analysis_arc"
sim_section = "sims"
expf_name = "experiment_simple"
cosmology_section = "cc_cluster_high"
recon_section = "reconstruction_cluster_lowell"

postfilter = False
gradweight = False


# Parse command line
parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')
parser.add_argument("-N", "--nsim",     type=int,  default=None)
parser.add_argument("-G", "--gradcut",     type=int,  default=2000)
parser.add_argument('-u',"--tszx", action='store_true', default=False)
parser.add_argument('-v',"--tszy", action='store_true', default=False)
parser.add_argument('-x',"--kszx", action='store_true', default=False)
parser.add_argument('-y',"--kszy", action='store_true', default=False)

args = parser.parse_args()
Nsims = args.nsim
grad_cut = args.gradcut
do_tszx = args.tszx
do_tszy = args.tszy
do_kszx = args.kszx
do_kszy = args.kszy
if grad_cut<1: grad_cut = None

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


# i/o directories
out_dir = os.environ['WWW']+"plots/mass_sehgal_"+str(grad_cut)+"_postfilter_"+str(postfilter)+"_tszx_"+str(do_tszx)+"_tszy_"+str(do_tszy)+"_kszx_"+str(do_kszx)+"_kszy_"+str(do_kszy)  +"_"

    
Ntot = Nsims


# Efficiently distribute sims over MPI cores
num_each,each_tasks = mpi_distribute(Ntot,numcores)
# Initialize a container for stats and stacks
mpibox = MPIStats(comm,num_each,tag_start=333)

if rank==0: print "At most ", max(num_each) , " tasks..."

# What am I doing?
my_tasks = each_tasks[rank]


# Read config
iniFile = "input/recon.ini"
Config = SafeConfigParser()
Config.optionxform=str
Config.read(iniFile)

pol = False
shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(Config,sim_section,analysis_section,pol=pol)
analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)
lb = aio.ellbounds_from_config(Config,recon_section,min_ell)
tellmin = lb['tellminY']
tellmax = lb['tellmaxY']
pellmin = lb['pellminY']
pellmax = lb['pellmaxY']
kellmin = lb['kellmin']
kellmax = lb['kellmax']
parray_dat = aio.patch_array_from_config(Config,expf_name,shape_dat,wcs_dat,dimensionless=False)
parray_sim = aio.patch_array_from_config(Config,expf_name,shape_sim,wcs_sim,dimensionless=False)
bin_edges = np.arange(0.,10.,analysis_resolution)
binner_dat = stats.bin2D(parray_dat.modrmap*60.*180./np.pi,bin_edges)
binner_sim = stats.bin2D(parray_sim.modrmap*60.*180./np.pi,bin_edges)
lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)
lbin_edges = np.arange(kellmin,kellmax,200)
lbinner_dat = stats.bin2D(modlmap_dat,lbin_edges)
lbinner_sim = stats.bin2D(modlmap_sim,lbin_edges)

if postfilter:
    postfilter_sim = cmb.gauss_beam(modlmap_sim,1.5*analysis_resolution)
    postfilter_dat = cmb.gauss_beam(modlmap_dat,1.5*analysis_resolution)
else:
    postfilter_sim = 1.+modlmap_sim*0.
    postfilter_dat = 1.+modlmap_dat*0.

# === COSMOLOGY ===
if rank==0: print "Cosmology..."
theory, cc, lmax = aio.theory_from_config(Config,cosmology_section,dimensionless=False)
assert cc is not None
parray_dat.add_theory(theory,lmax,orphics_is_dimensionless=False)
parray_sim.add_theory(theory,lmax,orphics_is_dimensionless=False)
gradCut = grad_cut
template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
nT = parray_dat.nT
nP = parray_dat.nP
if rank==0: io.quickPlot2d(nT,out_dir+"nt.png")
if rank==0: io.quickPlot2d(parray_dat.lbeam,out_dir+"kbeam.png")
fMaskCMB_T = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=tellmin,lmax=tellmax)
fMaskCMB_P = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=pellmin,lmax=pellmax)
fMask = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=kellmin,lmax=kellmax)

qest = Estimator(template_dat,
                 theory,
                 theorySpectraForNorm=None,
                 noiseX2dTEB=[nT,nP,nP],
                 noiseY2dTEB=[nT,nP,nP],
                 fmaskX2dTEB=[fMaskCMB_T,fMaskCMB_P,fMaskCMB_P],
                 fmaskY2dTEB=[fMaskCMB_T,fMaskCMB_P,fMaskCMB_P],
                 fmaskKappa=fMask,
                 kBeamX = parray_dat.lbeam,
                 kBeamY = parray_dat.lbeam,
                 doCurl=False,
                 TOnly=not(pol),
                 halo=True,
                 uEqualsL=False,
                 gradCut=gradCut,verbose=False,
                 bigell=lmax)

    

pixratio = analysis_resolution/Config.getfloat(sim_section,"pixel_arcmin")
px_dat = analysis_resolution
lens_order = Config.getint(sim_section,"lens_order")
if gradweight:
    Ny,Nx = shape_dat
    xMap,yMap,modRMap,xx,yy = fmaps.get_real_attributes(Ny,Nx,px_dat,px_dat)
    cosmap = xMap/modRMap
    sinmap = yMap/modRMap

k = -1
for index in my_tasks:
    
    k += 1

    kappain = load_sehgal_kappa(index)
    kappa = enmap.ndmap(resample.resample_fft(kappain,shape_sim),wcs_sim)
    phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
    grad_phi = enmap.grad(phi)
            

    if rank==0: print "Generating unlensed CMB for ", k, "..."
    unlensed = parray_sim.get_unlensed_cmb(seed=index)
    if rank==0: print "Lensing..."
    lensed = lensing.lens_map(unlensed.copy(), grad_phi, order=lens_order, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)
    #lensed = lensing.lens_map_flat(unlensed.copy(), phi, order=lens_order)
    if rank==0: print "Downsampling..."
    cmb = lensed if abs(pixratio-1.)<1.e-3 else resample.resample_fft(lensed,shape_dat)
    cmb = enmap.ndmap(cmb,wcs_dat)

    
    
    if rank==0: print "Filtering and binning input kappa..."
    dkappa = enmap.ndmap(fmaps.filter_map(kappa,postfilter_sim,parray_sim.modlmap,lowPass=kellmax,highPass=kellmin),wcs_sim)
    dkappa = dkappa if abs(pixratio-1.)<1.e-3 else enmap.ndmap(resample.resample_fft(dkappa,shape_dat),wcs_dat)
    cents,kappa1d = binner_dat.bin(dkappa)
    mpibox.add_to_stats("input_kappa1d",kappa1d)
    mpibox.add_to_stack("input_kappa2d",dkappa)
    

    if rank==0: print "Reconstructing..."
    fgX = 0.
    fgY = 0.
    if do_tszx or do_tszy: tsz = load_sehgal_component("tsz",index)
    if do_kszx or do_kszy: ksz = load_sehgal_component("ksz",index)
    if do_tszx: fgX += tsz
    if do_tszy: fgY += tsz
    if do_kszx: fgX += ksz
    if do_kszy: fgY += ksz
    
    measuredX = cmb + fgX
    measuredY = cmb + fgY
    mpibox.add_to_stack("cmbX",measuredX)
    mpibox.add_to_stack("cmbY",measuredY)

    
    fkmapsX = fftfast.fft(measuredX,axes=[-2,-1])
    fkmapsY = fftfast.fft(measuredY,axes=[-2,-1])
    qest.updateTEB_X(fkmapsX,alreadyFTed=True)
    qest.updateTEB_Y(fkmapsY,alreadyFTed=True)
    rawkappa = qest.getKappa("TT").real

    if gradweight:
        Gx = ifft(qest.kGradx['T']*qest.N.WXY('TT'),axes=[-2,-1],normalize=True).real
        Gy = ifft(qest.kGrady['T']*qest.N.WXY('TT'),axes=[-2,-1],normalize=True).real
        # Gx = ifft(qest.kGradx['T'],axes=[-2,-1],normalize=True).real
        # Gy = ifft(qest.kGrady['T'],axes=[-2,-1],normalize=True).real
        wt1 = (Gx*cosmap+Gy*sinmap)**2.
        wt2 = (Gy*cosmap+Gx*sinmap)**2.
        if rank==0 and k==0:
            io.quickPlot2d(cosmap,out_dir+"cosmap.png")
            io.quickPlot2d(sinmap,out_dir+"sinmap.png")
            io.quickPlot2d(wt2,out_dir+"wt2.png")
            io.quickPlot2d(wt1,out_dir+"wt1.png")
            io.quickPlot2d(Gx,out_dir+"gx.png")
            io.quickPlot2d(Gy,out_dir+"gy.png")
        cents,wt11d = binner_dat.bin(wt1)
        cents,wt21d = binner_dat.bin(wt2)
            

    kappa_recon = enmap.ndmap(np.nan_to_num(rawkappa),wcs_dat)
    kappa_recon = enmap.ndmap(fmaps.filter_map(kappa_recon,postfilter_dat,parray_dat.modlmap,lowPass=kellmax,highPass=kellmin),wcs_dat)
    kappa_recon -= kappa_recon.mean()
    mpibox.add_to_stack("recon_kappa2d",kappa_recon)
    cents,kapparecon1d = binner_dat.bin(kappa_recon)
    mpibox.add_to_stats("recon_kappa1d",kapparecon1d)
    if gradweight:
        k1dwt1 = kapparecon1d*wt11d
        k1dwt2 = kapparecon1d*wt21d
        mpibox.add_to_stack("numer_kw1",k1dwt1)
        mpibox.add_to_stack("numer_kw2",k1dwt2)
        mpibox.add_to_stack("denom_kw1",wt11d)
        mpibox.add_to_stack("denom_kw2",wt21d)
        
    cpower = fmaps.get_simple_power_enmap(enmap1=kappa_recon,enmap2=dkappa)
    ipower = fmaps.get_simple_power_enmap(enmap1=dkappa)
    lcents, cclkk = lbinner_dat.bin(cpower)
    lcents, iclkk = lbinner_dat.bin(ipower)
    mpibox.add_to_stats("crossp",cclkk)
    mpibox.add_to_stats("inputp",iclkk)

    if rank==0 and index==0:
        io.quickPlot2d(cmb,out_dir+"cmb.png")
        io.quickPlot2d(measuredX,out_dir+"mcmbx.png")
        io.quickPlot2d(measuredY,out_dir+"mcmby.png")
        io.quickPlot2d(dkappa,out_dir+"inpkappa.png")
        io.quickPlot2d(kappa_recon,out_dir+"reconkappa.png")


mpibox.get_stacks()
mpibox.get_stats()



if rank==0:


    cstats = mpibox.stats['crossp']
    istats = mpibox.stats['inputp']


    pl = io.Plotter(scaleY='log')
    pl.addErr(lcents,cstats['mean'],yerr=cstats['errmean'],marker="o",label="recon x cross")
    pl.add(lcents,istats['mean'],marker="x",ls="-",label="input")
    lcents,nlkk = lbinner_dat.bin(qest.N.Nlkk['TT'])
    pl.add(lcents,nlkk,ls="--",label="theory n0")
    pl.legendOn(loc="lower left",labsize=9)
    pl.done(out_dir+"cpower.png")


    pl = io.Plotter()
    ldiff = (cstats['mean']-istats['mean'])*100./istats['mean']
    lerr = cstats['errmean']*100./istats['mean']
    pl.addErr(lcents,ldiff,yerr=lerr,marker="o",ls="-")
    pl._ax.axhline(y=0.,ls="--",color="k")
    pl._ax.set_ylim(-20.,10.)
    pl.done(out_dir+"powerdiff.png")

    if gradweight:
        wtd1 = mpibox.stacks["numer_kw1"]/mpibox.stacks["denom_kw1"]
        wtd2 = mpibox.stacks["numer_kw2"]/mpibox.stacks["denom_kw2"]
    

    kappa_stats = mpibox.stats["input_kappa1d"]
    kapparecon_stats = mpibox.stats["recon_kappa1d"]

    savemat = np.vstack((cents,kapparecon_stats['mean'],kapparecon_stats['errmean'])).T
    np.save(out_dir+"kappa1d.npy",savemat)
    
    pl = io.Plotter(scaleX='log',scaleY='log')

    if gradweight:
        pl.add(cents,wtd1,marker="o",label="wt1")
        pl.add(cents,wtd2,marker="o",label="wt2")
    
    pl.addErr(cents,kappa_stats['mean'],yerr=kappa_stats['errmean'],ls="-")
    pl.addErr(cents,kapparecon_stats['mean'],yerr=kapparecon_stats['errmean'],ls="--")
    pl._ax.set_xlim(0.1,10.)
    pl._ax.set_ylim(0.001,0.63)
    pl.done(out_dir+"kappa1d.png")

    cmbx = mpibox.stacks["cmbX"]
    io.quickPlot2d(cmbx,out_dir+"cmbx.png")
    cmby = mpibox.stacks["cmbY"]
    io.quickPlot2d(cmby,out_dir+"cmby.png")


    io.quickPlot2d(stats.cov2corr(kappa_stats['cov']),out_dir+"kappa_corr.png")

    reconstack = mpibox.stacks["recon_kappa2d"]
    io.quickPlot2d(reconstack,out_dir+"reconstack.png")
    inpstack = mpibox.stacks["input_kappa2d"]
    io.quickPlot2d(inpstack,out_dir+"inpstack.png")
    inp = enmap.ndmap(inpstack if abs(pixratio-1.)<1.e-3 else resample.resample_fft(inpstack,shape_dat),wcs_dat)
    pdiff = np.nan_to_num((inp-reconstack)*100./inp)
    io.quickPlot2d(pdiff,out_dir+"pdiffstack.png",lim=20.)


    
    pl = io.Plotter()
    rec = kapparecon_stats['mean']
    inp = kappa_stats['mean']
    recerr = kapparecon_stats['errmean']
    diff = (rec-inp)*100./inp
    differr = 100.*recerr/inp

    if gradweight:
        diff1 = (wtd1-inp)*100./inp
        diff2 = (wtd2-inp)*100./inp
    
    pl.addErr(cents,diff,yerr=differr,marker="o")

    if gradweight:
        pl.add(cents,diff1,marker="o",label="wt1")
        pl.add(cents,diff2,marker="o",label="wt2")
    pl.legendOn(labsize=8,loc="lower right")
    pl.hline()
    pl._ax.set_ylim(-10,10)
    pl._ax.set_xlim(0.,10.)
    pl.done(out_dir+"diffper.png")



    pl = io.Plotter()
    diff = (rec-inp)/recerr
    pl.add(cents,diff,marker="o",ls="-")
    pl.legendOn(labsize=8,loc="lower right")
    pl.hline()
    pl._ax.set_xlim(0.,10.)
    pl.done(out_dir+"diffbias.png")




