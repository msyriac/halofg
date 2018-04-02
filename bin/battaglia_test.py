from __future__ import print_function
import numpy as np
from szar import sims, counts
from orphics import maps,io,cosmology,lensing,stats,mpi
from enlib import enmap,lensing as enlensing,bench,resample
import os,sys
import argparse

# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("Nclusters", type=int,help='Number of simulated clusters.')
parser.add_argument("Amp", type=float,help='Amplitude of mass wrt 1e15.')
parser.add_argument("SZType", type=str,help='None/mean/vary for SZ in Y leg.')
parser.add_argument("-a", "--arc",     type=float,  default=100.,help="Stamp width (arcmin).")
parser.add_argument("-p", "--pix",     type=float,  default=0.1953125,help="Pix width (arcmin).")
parser.add_argument("-d", "--dpix",     type=float,  default=0.5,help="Pix width (arcmin).")
parser.add_argument("-b", "--beam",     type=float,  default=1.0,help="Beam (arcmin).")
parser.add_argument("-n", "--noise",     type=float,  default=1.0,help="Noise (uK-arcmin).")
#parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()
sz_type = args.SZType.lower()
assert sz_type in ['none','mean','vary']


# MPI
comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()


# Paths

PathConfig = io.load_path_config()
pout_dir = PathConfig.get("paths","plots")+"qest_hdv_fg_"+str(args.noise)+"_"+sz_type+"_"
io.mkdir(pout_dir,comm)
output_dir = PathConfig.get("paths","output_data")+"qest_hdv_fg/"
io.mkdir(output_dir,comm)


# Theory
theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
cc = counts.ClusterCosmology(skipCls=True,verbose=False)
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

# Geometry
shape, wcs = maps.rect_geometry(width_arcmin=args.arc,px_res_arcmin=args.pix,pol=False)
modlmap = enmap.modlmap(shape,wcs)
modrmap = enmap.modrmap(shape,wcs)
oshape,owcs = enmap.scale_geometry(shape,wcs,args.pix/args.dpix)
omodlmap = enmap.modlmap(oshape,owcs)
omodrmap = enmap.modrmap(oshape,owcs)

# Binning
bin_edges = np.arange(0.,20.0,args.dpix*2)
binner = stats.bin2D(omodrmap*60.*180./np.pi,bin_edges)

# Noise model
noise_uK_rad = args.noise*np.pi/180./60.
normfact = np.sqrt(np.prod(enmap.pixsize(shape,wcs)))
kbeam = maps.gauss_beam(args.beam,modlmap)
okbeam = maps.gauss_beam(args.beam,omodlmap)


# Simulate
lmax = int(modlmap.max()+1)
ells = np.arange(0,lmax,1)
ps = theory.uCl('TT',ells).reshape((1,1,lmax))
ps_noise = np.array([(noise_uK_rad)**2.]*ells.size).reshape((1,1,ells.size))
mg = maps.MapGen(shape,wcs,ps)
ng = maps.MapGen(oshape,owcs,ps_noise)
kamp_true = args.Amp
kappa = lensing.nfw_kappa(kamp_true*1e15,modrmap,cc,overdensity=180.,critical=False,atClusterZ=False)
phi,_ = lensing.kappa_to_phi(kappa,modlmap,return_fphi=True)
grad_phi = enmap.grad(phi)
posmap = enmap.posmap(shape,wcs)
pos = posmap + grad_phi
alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)
lens_order = 5



# Cluster SZ
if sz_type!="none":
    Nbattaglia = 300
    Nsims = Nbattaglia
    Njobs = Nsims
    num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
    if rank==0: print ("At most ", max(num_each) , " tasks...")
    my_tasks = each_tasks[rank]
    bsims = sims.BattagliaSims(cc.c,rootPath="/gpfs01/astro/workarea/msyriac/data/sims/battaglia/")
    snap = 44
    szs = np.zeros((300,)+oshape)
    for task in my_tasks:
        try:
            szmap = enmap.read_hdf(output_dir+"sz_map_"+str(task)+".hdf")
            assert szmap.shape==oshape
            if rank==0: print("Rank 0 found and loaded saved SZs.")
        except:
            massIndex = task
            kappaMap,szmap,projectedM500,z = bsims.getKappaSZ(snap,massIndex,oshape,owcs,apodWidthArcmin=4.)
            enmap.write_hdf(output_dir+"sz_map_"+str(task)+".hdf",szmap)
        szs[task,:,:] = szmap.copy()
    comm.Barrier()
    for task in range(Nsims):
        if task in my_tasks: continue
        szs[task,:,:] = enmap.read_hdf(output_dir+"sz_map_"+str(task)+".hdf")

    sz_mean = np.mean(szs,axis=0)
    if rank==0:
        io.plot_img(sz_mean,pout_dir+"szmean.png")
        io.plot_img(szs[0],pout_dir+"szfirst.png")
        io.plot_img(szs[150],pout_dir+"szmid.png")
        io.plot_img(szs[299],pout_dir+"szlast.png")

if rank==0: print("Starting sims...")
# Stats
Nsims = args.Nclusters
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
mstats = stats.Stats(comm)
np.random.seed(rank)


# QE
tellmin = omodlmap[omodlmap>2].min(); tellmax = 8000; kellmin = tellmin ; kellmax = 8096
tmask = maps.mask_kspace(oshape,owcs,lmin=tellmin,lmax=tellmax)
kmask = maps.mask_kspace(oshape,owcs,lmin=kellmin,lmax=kellmax)
qest = lensing.qest(oshape,owcs,theory,noise2d=okbeam*0.+(noise_uK_rad)**2.,beam2d=okbeam,kmask=tmask,kmask_K=kmask,pol=False,grad_cut=2000,unlensed_equals_lensed=False)


if sz_type!="none":
    def get_rand_sz():
        randint = np.random.randint(0,Nbattaglia)
        chosen_sz = szs[randint,:,:]
        randint = np.random.randint(0,4)
        return np.rot90(chosen_sz,k=randint)
    

for i,task in enumerate(my_tasks):
    if (i+1)%10==0 and rank==0: print(i+1)

    unlensed = mg.get_map()
    noise_map = ng.get_map()
    lensed = maps.filter_map(enlensing.displace_map(unlensed, alpha_pix, order=lens_order),kbeam)
    fdownsampled = enmap.enmap(resample.resample_fft(lensed,oshape),owcs)
    stamp = fdownsampled  + noise_map
    if task==0: io.plot_img(stamp,pout_dir+"cmb_noisy.png")

    if sz_type=='none':
        ystamp = stamp.copy()
    elif sz_type=='mean':
        ystamp = stamp + maps.filter_map(sz_mean,okbeam)
    elif sz_type=='vary':
        vsz = maps.filter_map(get_rand_sz(),okbeam)
        ystamp = stamp + vsz
        if task<5:
            io.plot_img(vsz,pout_dir+"vsz"+str(task)+".png")
        
    recon = qest.kappa_from_map("TT",stamp,T2DDataY=ystamp)
    cents, recon1d = binner.bin(recon)

    mstats.add_to_stats("recon1d",recon1d)
    mstats.add_to_stack("recon",recon)

mstats.get_stats()
mstats.get_stacks()

if rank==0:

    stack = mstats.stacks['recon']

    recon1d = mstats.stats['recon1d']['mean']
    recon1d_err = mstats.stats['recon1d']['errmean']
    recon1d_cov = mstats.stats['recon1d']['covmean']

    io.plot_img(stack,pout_dir+"stack.png")

    cc = counts.ClusterCosmology(skipCls=True,verbose=False)
    kinp = lensing.nfw_kappa(kamp_true*1e15,omodrmap,cc,overdensity=180.,critical=False,atClusterZ=False)
    # print(cc.h)
    # sys.exit()
    kappa_true = maps.filter_map(kinp,kmask)

    print(kamp_true)
    enmap.write_map("debug_kappa_input_stamp_"+sz_type+".hdf",enmap.enmap(kinp,owcs))
    enmap.write_map("debug_modrmap_stamp_"+sz_type+".hdf",omodrmap)
    enmap.write_map("debug_kmask_stamp_"+sz_type+".hdf",enmap.enmap(kmask,owcs))
    enmap.write_map("debug_kappa_true_stamp_"+sz_type+".hdf",enmap.enmap(kappa_true,owcs))
    
    cents, ktrue1d = binner.bin(kappa_true)
    io.save_cols("debug_kappa_true_prof_"+sz_type+".txt",(cents,ktrue1d))

    arcs,ks = np.loadtxt("input/hdv.csv",unpack=True,delimiter=",")


    pl = io.Plotter()
    pl.add(cents,ktrue1d,lw=2,color="k",label="true")
    pl.add(arcs,ks,lw=2,alpha=0.5,label="hdv profile")
    pl.add_err(cents,recon1d,recon1d_err,ls="--",label="recon")
    pl.hline()
    pl.legend(loc='upper right')
    pl._ax.set_ylim(-0.01,0.2)
    pl.done(pout_dir+"recon1d.png")


    # MASS DEPENDENT BIAS CORRECTION FOR CASES WITH FG
    # if sz_type=="none":
    #     io.save_cols(output_dir+"bias_corr.txt",(cents,ktrue1d-recon1d))
    # else:
    #     cents_in,bias_corr = np.loadtxt(output_dir+"bias_corr.txt",unpack=True)
    #     assert np.all(np.isclose(cents,cents_in))
    #     recon1d += bias_corr

                                        
        
    pl1 = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
    

    arcmax = 10.

    length = cents[cents<=arcmax].size
    cinv = np.linalg.pinv(recon1d_cov[:length,:length])
    diff = ktrue1d[cents<=arcmax] 
    chisq = np.dot(np.dot(diff.T,cinv),diff)
    sn = np.sqrt(chisq)
    print("=== ARCMAX ",arcmax," =====")
    print ("S/N null for 1000 : ",sn*np.sqrt(1000./args.Nclusters))
    pred_sigma = kamp_true/sn

    num_amps = 100
    nsigma = 10.
    kamps = np.linspace(kamp_true-nsigma*pred_sigma,kamp_true+nsigma*pred_sigma,num_amps)
    k1ds = []
    for amp in kamps:
        template = lensing.nfw_kappa(amp*1e15,omodrmap,cc,overdensity=180.,critical=False,atClusterZ=False) # !!!
        kappa_sim = maps.filter_map(template,kmask)
        cents, k1d = binner.bin(kappa_sim)
        k1ds.append(k1d)

    # Fit true for S/N
    lnlikes = []
    for k1d in k1ds:
        diff = (k1d-ktrue1d)[cents<=arcmax] 
        lnlike = -0.5*np.dot(np.dot(diff.T,cinv),diff)
        lnlikes.append(lnlike)


    lnlikes = np.array(lnlikes)
    amaxes = kamps[np.isclose(lnlikes,lnlikes.max())]

    p = np.polyfit(kamps,lnlikes,2)

    c,b,a = p
    mean = -b/2./c
    sigma = np.sqrt(-1./2./c)
    print(mean,sigma)
    sn = (kamp_true/sigma)
    print ("S/N fit for 1000 : ",sn*np.sqrt(1000./args.Nclusters))
    pbias = (mean-kamp_true)*100./kamp_true
    #print ("Bias : ",pbias, " %")
    #print ("Bias : ",(mean-kamp_true)/sigma, " sigma")



    # Fit data for bias
    lnlikes = []
    for k1d in k1ds:

        diff = (k1d-recon1d)[cents<=arcmax] 
        lnlike = -0.5*np.dot(np.dot(diff.T,cinv),diff)
        lnlikes.append(lnlike)

    lnlikes = np.array(lnlikes)
    pl1.add(kamps,lnlikes,label="qe chisquare")
    amaxes = kamps[np.isclose(lnlikes,lnlikes.max())]
    p = np.polyfit(kamps,lnlikes,2)

    pl1.add(kamps,p[0]*kamps**2.+p[1]*kamps+p[2],ls="--",label="qe chisquare fit")
    for amax in amaxes:
        pl1.vline(x=amax,ls="-")
            
    pl1.vline(x=kamp_true,ls="--")
    pl1.legend(loc='upper left')
    pl1.done(pout_dir+"lensed_lnlikes_all.png")

    # QE
    c,b,a = p
    mean = -b/2./c
    sigma = np.sqrt(-1./2./c)
    print(mean,sigma)
    sn = (kamp_true/sigma)
    pbias = (mean-kamp_true)*100./kamp_true
    print ("QE Bias : ",pbias, " %")
    print ("QE Bias : ",(mean-kamp_true)/sigma, " sigma")

    like = np.exp(lnlikes)
    like /= like.max()
    nkamps = np.linspace(kamps.min(),kamps.max(),1000)
    pl2 = io.Plotter(xlabel="$A$",ylabel="$\\mathcal{L}$")

    pl2.add(nkamps,np.exp(-(nkamps-mean)**2./2./sigma**2.),label="QE likelihood from chisquare fit")
    pl2.add(kamps,like,label="QE likelihood",alpha=0.2)




    pl2.vline(x=kamp_true,ls="--")
    pl2.legend(loc='upper left')
    pl2.done(pout_dir+"lensed_likes.png")

    
