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

# Runtime params that should be moved to command line
analysis_section = "analysis_tiny"
sim_section = "sims_tiny"
expf_name = "experiment_noiseless"
cosmology_section = "cc_cluster_high"
#recon_section = "reconstruction_cluster"
recon_section = "reconstruction_cluster_lowell"

postfilter = False
gradweight = False

def get_nfw(massOverh,modrmap,zL=0.7,concentration=3.2):
    from alhazen.halos import NFWkappa

    #massOverh = 2.e14
    overdensity = 180.
    critical = False
    atClusterZ = False
    comS = cc.results.comoving_radial_distance(cc.cmbZ)*cc.h
    comL = cc.results.comoving_radial_distance(zL)*cc.h
    winAtLens = (comS-comL)/comS
    kappa,r500 = NFWkappa(cc,massOverh,concentration,zL,modrmap* 180.*60./np.pi,winAtLens,
                              overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)

    return kappa


# Parse command line
parser = argparse.ArgumentParser(description='Verify lensing reconstruction.')
parser.add_argument("-N", "--nsim",     type=int,  default=None)
parser.add_argument("-M", "--mass",     type=float,  default=2.e14)
parser.add_argument("-G", "--gradcut",     type=int,  default=2000)
parser.add_argument("-u", "--unlensed", action='store_true',help='Make unlensed.')
parser.add_argument("-f", "--fake", action='store_true',help='Add fake kappa.')
args = parser.parse_args()
Nsims = args.nsim
cluster_mass = args.mass
grad_cut = args.gradcut
nolens = args.unlensed
if args.fake: assert nolens
if grad_cut<1: grad_cut = None

# Get MPI comm
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()    


# i/o directories
out_dir = os.environ['WWW']+"plots/mass_"+str(cluster_mass)+"_"+str(grad_cut)+"_postfilter_"+str(postfilter)+"_unlensed_"+str(nolens)+"_fake_"+str(args.fake)+"_"  # for plots

    
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
bin_edges = np.arange(0.,5.,analysis_resolution*2)
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
#assert cc is not None
parray_dat.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)
parray_sim.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)
gradCut = grad_cut
template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
nT = parray_dat.nT
nP = parray_dat.nP
# if rank==0: io.quickPlot2d(nT,out_dir+"nt.png")
# if rank==0: io.quickPlot2d(parray_dat.lbeam,out_dir+"kbeam.png")
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
    if k==0:

        from alhazen.halos import NFWkappa

        massOverh = cluster_mass
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
        thetaRef = 0.94
        kappaRef,r500Ref = NFWkappa(cc,massOverh,concentration,zL,thetaRef,winAtLens,
                                  overdensity=overdensity,critical=critical,atClusterZ=atClusterZ)
        if rank==0: print "Kappa at ", thetaRef, " arc is ", kappaRef

        #kappa = parray_sim.get_grf_kappa(seed=1)
        phi, fphi = lt.kappa_to_phi(kappa,parray_sim.modlmap,return_fphi=True)
        grad_phi = enmap.grad(phi)
            

    if rank==0: print "Generating unlensed CMB for ", k, "..."
    unlensed = parray_sim.get_unlensed_cmb(seed=index)
    if rank==0: print "Lensing..."
    lensed = unlensed if nolens else lensing.lens_map(unlensed.copy(), grad_phi, order=lens_order, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)
    #lensed = lensing.lens_map_flat(unlensed.copy(), phi, order=lens_order)
    if rank==0: print "Downsampling..."
    cmb = lensed if abs(pixratio-1.)<1.e-3 else resample.resample_fft(lensed,shape_dat)
    cmb = enmap.ndmap(cmb,wcs_dat)
    if rank==0: print "Adding noise..."
    flensed = fftfast.fft(cmb,axes=[-2,-1])
    flensed *= parray_dat.lbeam
    lensedt = fftfast.ifft(flensed,axes=[-2,-1],normalize=True).real
    noise = parray_dat.get_noise_sim(seed=index+10000000)
    lensedt += noise
    cmb = lensedt
        

    
    if rank==0: print "Filtering and binning input kappa..."
    dkappa = enmap.ndmap(fmaps.filter_map(kappa,postfilter_sim,parray_sim.modlmap,lowPass=kellmax,highPass=kellmin),wcs_sim)
    dkappa = dkappa if abs(pixratio-1.)<1.e-3 else enmap.ndmap(resample.resample_fft(dkappa,shape_dat),wcs_dat)
    cents,kappa1d = binner_dat.bin(dkappa)
    mpibox.add_to_stats("input_kappa1d",kappa1d)
    mpibox.add_to_stack("input_kappa2d",dkappa)
    

    if rank==0: print "Reconstructing..."
    measured = cmb
    fkmaps = fftfast.fft(measured,axes=[-2,-1])
    qest.updateTEB_X(fkmaps,alreadyFTed=True)
    qest.updateTEB_Y()
    rawkappa = qest.getKappa("TT").real
    if args.fake: rawkappa += dkappa

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
        io.quickPlot2d(measured,out_dir+"mcmb.png")
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


    first_bin = mpibox.vectors["recon_kappa1d"][:,0]
    hist, bin_edges = np.histogram(first_bin,bins=30)
    bin_cents = (bin_edges[1:]+bin_edges[:-1])/2.
    pl = io.Plotter()
    pl.add(bin_cents,hist)
    pl.done(out_dir+"histogram_first_bin.png")
    

    pl = io.Plotter(scaleX='log',scaleY='log')
    if gradweight:
        pl.add(cents,wtd1,marker="o",label="wt1")
        pl.add(cents,wtd2,marker="o",label="wt2")
    
    pl.addErr(cents,kappa_stats['mean'],yerr=kappa_stats['errmean'],ls="-")
    pl.addErr(cents,kapparecon_stats['mean'],yerr=kapparecon_stats['errmean'],ls="--")
    pl._ax.set_xlim(0.1,10.)
    pl._ax.set_ylim(0.001,0.63)
    pl.done(out_dir+"kappa1d.png")



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




    mean = kapparecon_stats['mean']
    cov = kapparecon_stats['covmean']
    siginv = np.linalg.inv(cov)
    
    chisq = np.dot(np.dot(mean,siginv),mean)
    sigma = np.sqrt(chisq)
    print "S/N null : ",sigma

    sigma_width = cluster_mass/sigma
    range_sigma = 10.
    massmid = cluster_mass
    left = max(massmid-range_sigma*sigma_width,0.)
    right = massmid+range_sigma*sigma_width
    print left,massmid,right
    
    mass_range = np.linspace(left,right,100)
    Likes = []
    for k,m in enumerate(mass_range):
        trial = get_nfw(m,parray_dat.modrmap)
        trial = enmap.ndmap(fmaps.filter_map(trial,trial*0.+1.,parray_dat.modlmap,lowPass=kellmax,highPass=kellmin),wcs_dat)
        cents,theory = binner_dat.bin(trial)
        Likes.append(np.exp(-0.5*stats.fchisq(mean,siginv,theory,amp=1.)))
        if k%10==0: print m

    Likes = np.array(Likes)
    Likes /= Likes.sum()

    maxlike = mass_range[np.where(np.isclose(Likes,Likes.max()))]
    print "Max like mass ", maxlike
    bias = (maxlike-args.mass)*100./args.mass
    print "Bias ", bias," %"
    
    pl = io.Plotter()
    pl.add(mass_range,Likes)
    pl._ax.axvline(x=cluster_mass,ls="--")
    pl._ax.axvline(x=maxlike,ls="-")
    pl.done(out_dir+"likes.png")

    
