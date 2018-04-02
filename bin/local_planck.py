from __future__ import print_function
from orphics import maps,io,cosmology,stats,catalogs,lensing,mpi
from enlib import enmap,bench,fft
import numpy as np
import os,sys,traceback
import healpy as hp
import astropy.io.fits as fits     
import argparse
from scipy.interpolate import interp1d

# Parse command line
parser = argparse.ArgumentParser(description='Do a thing.')
parser.add_argument("region", type=str,help='Positional arg.')
parser.add_argument("-N", "--nrandoms",     type=int,  default=None,help="Number of random locations for meanfield.")
parser.add_argument("-m", "--meanfield",     type=str,  default=None,help="Location of meanfield.")
#parser.add_argument("-f", "--flag", action='store_true',help='A flag.')
args = parser.parse_args()

region = args.region
suffix = "meanfield_" if args.nrandoms is not None else ""
if args.nrandoms is not None: assert args.meanfield is None
io.dout_dir += region+"_"+suffix

if region=="deep56" or region=="deep6":
    wnoise = 10.
elif region=="boss_north":
    wnoise = 15.



comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
    

with bench.show("load maps"):
    sreader = maps.SigurdCoaddReader("../pvt-config/actpol_maps.yaml")
    coadd_main = sreader.get_map(-1,freq="150",day_night="daynight",
                            planck=True,region=region,weight=False,get_identifier=False)[0]
    coadd = enmap.read_fits("/gpfs01/astro/workarea/msyriac/data/act/maps/s16/cache/"+region+"_inpainted.fits")
    
    # io.plot_img(coadd_main,io.dout_dir+"coadd.png")
    # io.plot_img(coadd,io.dout_dir+"coadd_inp.png")
    # sys.exit()
    
    shape,wcs = coadd.shape,coadd.wcs
    # if rank==0: enmap.write_map("/gpfs01/astro/workarea/msyriac/data/planck/template_"+region+".fits",enmap.zeros(shape,wcs))
    # sys.exit()    
#     wmap = sreader.get_map(-1,freq="150",day_night="daynight",
#                            planck=True,region=region,weight=True,get_identifier=False)                          
    splits = []                  
    wmaps = []                    
    for split in range(2):        
        splits.append(sreader.get_map(split,freq="150",day_night="daynight",
                                      planck=True,region=region,weight=False,get_identifier=False)[0])
        wmaps.append(sreader.get_map(split,freq="150",day_night="daynight",planck=True,region=region,weight=True,get_identifier=False))


# smica_file = "/gpfs01/astro/workarea/msyriac/data/planck/HFI_SkyMap_143_2048_R2.02_full.fits"
# smica = hp.read_map(smica_file) *1e6 


# smica_file = "/gpfs01/astro/workarea/msyriac/data/planck/COM_CMB_IQU-smica-field-Int_2048_R2.01_full.fits"
# with io.nostdout():
#     smica = hp.read_map(smica_file) *1e6 

# smica_file = "/gpfs01/astro/workarea/msyriac/data/planck/WPR2_CMB_muK.fits"
# smica = hp.read_map(smica_file)


# smica_file = "/gpfs01/astro/workarea/msyriac/data/planck/smica_"+region+".fits"
# smica = enmap.read_map(smica_file)[0]  *1e6  #[:,::-1] *1e6 

smica_file = "/gpfs01/astro/workarea/msyriac/data/planck/lgmca_"+region+".fits"
smica = enmap.read_map(smica_file)[0]  



# ACT MAP LOADER
config_yaml_path = "../pvt-config/actpol_maps.yaml"
mreader = maps.SimoneC7V5Reader(config_yaml_path)
ls,bells = mreader.get_beam("s15","deep56","pa2","150","night")


Ny,Nx = coadd.shape[-2:]
arc = 100.
if args.nrandoms is not None:
    iys = np.random.randint(0,Ny,args.nrandoms)
    ixs = np.random.randint(0,Nx,args.nrandoms)
    random = True
else:
    random = False
    rfile = "/gpfs01/astro/workarea/msyriac/data/sdss/redmapper_dr8_public_v6.3_catalog.fits"  
    hdu = fits.open(rfile)            

    ras = hdu[1].data['RA']          
    decs = hdu[1].data['DEC']        
    lams = hdu[1].data['lambda']
    zspec = hdu[1].data['Z_SPEC']
    zlam = hdu[1].data['Z_LAMBDA']
    zs = zspec.copy()
    zs[zspec<0] = zlam[zspec<0]
    #print(hdu[1].columns)

    print(zs.mean(),zs.min(),zs.max())
    sys.exit()

    cmapper = catalogs.CatMapper(ras,decs,shape=shape[-2:],wcs=wcs,verbose=False)
    if rank==0:
        cts = enmap.smooth_gauss(cmapper.counts,20.*np.pi/180./60.)
        io.plot_img(cts,io.dout_dir+"precat_hres.png",high_res=True)
    num_clusters = cmapper.counts.sum()
    if rank==0: print(num_clusters)
    riys,rixs = cmapper.pixs[0],cmapper.pixs[1]
    iys = riys[np.logical_and(np.logical_and(np.logical_and(riys>0,riys<Ny),rixs>0),rixs<Nx)]
    ixs = rixs[np.logical_and(np.logical_and(np.logical_and(riys>0,riys<Ny),rixs>0),rixs<Nx)]


    

    # iys = riys
    # ixs = rixs

    # coords = np.rad2deg(enmap.pix2sky(shape,wcs,(iys,ixs)))
    # ras = coords[1]
    # decs = coords[0]


    # if rank==0:
    #     cmapper = catalogs.CatMapper(ras,decs,shape=shape[-2:],wcs=wcs,verbose=False)
    #     cts = enmap.smooth_gauss(cmapper.counts,20.*np.pi/180./60.)
    #     io.plot_img(cts,io.dout_dir+"postcat_hres.png",high_res=True)
    # sys.exit()

Nsims = len(iys)
Njobs = Nsims
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]

save_dir = "/gpfs01/astro/workarea/msyriac/data/act/maps/s16/cache/"

if args.meanfield is not None:
    mf = enmap.read_fits(save_dir+region+"_meanfield.fits")
else:
    mf = 0.


j = 0

s = stats.Stats(comm=comm)    
match = maps.MatchedFilter(shape,wcs)

for k,task in enumerate(my_tasks):
    iy = iys[task]
    ix = ixs[task]

    cut_act = maps.cutout(coadd,arc,iy=iy,ix=ix)              
    cut_act_check = maps.cutout(coadd_main,arc,iy=iy,ix=ix)              
    if cut_act is None:
        # s.add_to_stats("rras",np.array((ra,)))
        # s.add_to_stats("rdecs",np.array((dec,)))
        continue

    modrmap = cut_act.modrmap()*180.*60./np.pi

    pt_arc_cut = 20.
    pt_val_cut = 500.
    if np.any(cut_act_check[modrmap<pt_arc_cut]>pt_val_cut): 
        continue
    
    j+=1

    if j==1:
        ny,nx = cut_act.shape
        assert ny==nx
        Npix = ny
        theory_file_root = "../alhazen/data/Aug6_highAcc_CDM"
        theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                     useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)
        
        modlmap = cut_act.modlmap()
        xtellmin = 20 ; xtellmax = 2000 ; ytellmin = 20 ; ytellmax = 6000 ; kellmin = 20 ; kellmax = 6100
        lshape,lwcs = cut_act.shape,cut_act.wcs
        mP = maps.mask_kspace(lshape,lwcs,lmin=xtellmin,lmax=xtellmax)
        mA = maps.mask_kspace(lshape,lwcs,lmin=ytellmin,lmax=ytellmax)
        mK = maps.mask_kspace(lshape,lwcs,lmin=kellmin,lmax=kellmax)
        taper,w2 = maps.get_taper(lshape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
        bP = maps.gauss_beam(modlmap,5.0)
        bA = interp1d(ls,bells,fill_value=0.,bounds_error=False)(modlmap) #maps.gauss_beam(modlmap,1.5)
        nP = modlmap*0.+(42.*np.pi/180./60.)**2.



    csplits = []
    csplits.append(maps.cutout(splits[0],arc,iy=iy,ix=ix))
    csplits.append(maps.cutout(splits[1],arc,iy=iy,ix=ix))

    cwmaps = []
    cwmaps.append(maps.cutout(wmaps[0],arc,iy=iy,ix=ix))
    cwmaps.append(maps.cutout(wmaps[1],arc,iy=iy,ix=ix))

    nm = maps.NoiseModel(splits=csplits,wmap=cwmaps,mask=taper,kmask=None,iau=False)
    nA = nm.noise2d[0,0]

    if j==1:
        bin_edges = np.arange(300,6000,200)
        binner = stats.bin2D(modlmap,bin_edges)
    cents,n1d = binner.bin(np.sqrt(nA)*60.*180./np.pi)

    wnoise = n1d[np.logical_and(cents>4000,cents<6000)].mean()
    if wnoise>35.:
        j -= 1
        continue

    if not(random): s.add_to_stats("wntt",n1d)

    qest = lensing.Estimator(lshape,lwcs,  
             theory,
             theorySpectraForNorm=None,
             noiseX2dTEB=[nP,nP,nP],
             noiseY2dTEB=[nA,nA,nA],
             noiseX_is_total = False,
             noiseY_is_total = False,
             fmaskX2dTEB=[mP,mP,mP],
             fmaskY2dTEB=[mA,mA,mA],
             fmaskKappa=mK,
             kBeamX = bP,  
             kBeamY = bA,  
             doCurl=False,    
             TOnly=True,     
             halo=True,      
             gradCut=2000,    
             verbose=False)
    nlkk2d = qest.N.Nlkk['TT']
    cents,nkk1d = binner.bin(nlkk2d)
    if not(random): s.add_to_stats("nkk",nkk1d)
    
    if (k+1)%100==0 and rank==0: print(k+1," / ", len(my_tasks), j)

    cut_smica = maps.cutout(smica,arc,iy=iy,ix=ix)              
    # cut_smica = maps.cutout_gnomonic(smica,rot=(ra,dec),coord=['G','C'],
    #                                 xsize=Npix,ysize=Npix,reso=0.5,
    #                                 nest=False,remove_dip=False,
    #                                 remove_mono=False,gal_cut=0,
    #                                 flip='geo')#'astro')

    
    xmap = (cut_smica - cut_smica.mean())*taper
    ymap = (cut_act - cut_act.mean())*taper
    
    recon,krecon = qest.kappa_from_map("TT",xmap,T2DDataY=ymap,alreadyFTed=False,returnFt=True)


    # if rank==0:
    #     io.plot_img(xmap,io.dout_dir+"smica.png")
    #     io.plot_img(ymap,io.dout_dir+"act.png")
    # sys.exit()

    recon -= mf

    template = lensing.nfw_kappa(2e14,modrmap*np.pi/180./60.,cc,overdensity=200.,critical=True,atClusterZ=True)
    k0,k0_var = match.apply(imap=recon,template=template,noise_power=nlkk2d)
    s.add_to_stack("match",np.array([k0,1./k0_var]))

    
    s.add_to_stack("reconft",np.nan_to_num(krecon/nlkk2d))    
    s.add_to_stack("reconwt",np.nan_to_num(1./nlkk2d))    
    s.add_to_stack("recon",recon)    
    s.add_to_stack("smica",xmap)
    s.add_to_stack("act",ymap)

    if j==1:
        rbin_edges = np.arange(0.,10.,0.5)
        rbinner = stats.bin2D(modrmap,rbin_edges)
    rcents,r1d = rbinner.bin(recon)
    s.add_to_stats("r1d",r1d)
        
    

s.get_stacks()
s.get_stats()

if rank==0:

    # print(s.vectors['rras'].shape)
    # print(s.vectors['rdecs'].shape)
    # ras = s.vectors['rras'].ravel()
    # decs = s.vectors['rdecs'].ravel()
    
    # cmapper = catalogs.CatMapper(ras,decs,shape=shape[-2:],wcs=wcs,verbose=False)
    # cts = enmap.smooth_gauss(cmapper.counts,20.*np.pi/180./60.)
    # io.plot_img(cts,io.dout_dir+"rejected_hres.png",high_res=True)

                       
    io.plot_img(s.stacks['smica'],io.dout_dir+"smica.png")
    io.plot_img(s.stacks['act'],io.dout_dir+"act.png")
    io.plot_img(s.stacks['recon'],io.dout_dir+"recon.png")


    
    ftwted = s.stacks['reconft']/s.stacks['reconwt']
    reconwted = fft.ifft(ftwted,axes=[-2,-1],normalize=True).real
    reconwted -= mf
    io.plot_img(reconwted,io.dout_dir+"reconwt.png")


    if random:
        enmap.write_fits(save_dir+region+"_meanfield.fits",enmap.enmap(reconwted,lwcs))
        


    rcents,r1dw = rbinner.bin(reconwted)
    rprof = s.stats['r1d']
    prof,prof_err = rprof['mean'],rprof['errmean']
    pl = io.Plotter()
    pl.add(rcents,r1dw)
    pl.add_err(rcents,prof,yerr=prof_err,ls="--")
    pl.hline()
    pl._ax.set_ylim(-0.02,0.2)
    pl.done(io.dout_dir+"rprof.png")
            
    

    if not(random):
        wntts = np.array(s.vectors['wntt'])
        nkks = np.array(s.vectors['nkk'])


        nclusters = wntts.shape[0]
        print ("Number of clusters stacked : ", nclusters)


        np.save(save_dir+region+"_sum_of_recon_fts.npy",s.stacks['reconft']*nclusters)
        np.save(save_dir+region+"_sum_of_recon_wts.npy",s.stacks['reconwt']*nclusters)

        pl = io.Plotter()
        for k in range(nclusters):
            pl.add(cents,wntts[k,:],alpha=0.1)
        pl.done(io.dout_dir+"ntts.png")


        ells = np.arange(2,7000,1)
        clkk = theory.gCl('kk',ells)
        pl = io.Plotter(yscale='log')
        pl.add(ells,clkk,lw=2,color='k')
        for k in range(nclusters):
            pl.add(cents,nkks[k,:],alpha=0.1)
        pl.done(io.dout_dir+"nkks.png")


        recon1d_cov = rprof['covmean']
        corr = stats.cov2corr(recon1d_cov)
        io.plot_img(corr,io.dout_dir+"corr.png")
        
        arcmax = 10.
        length = rcents[rcents<=arcmax].size
        cinv = np.linalg.pinv(recon1d_cov[:length,:length])
        diff = r1dw[rcents<=arcmax] 
        chisq = np.dot(np.dot(diff.T,cinv),diff)
        sn = np.sqrt(chisq)
        print ("S/N null  : ",sn)
        kamp_true = 0.06
        pred_sigma = kamp_true/sn

        from szar import counts
        cc = counts.ClusterCosmology(skipCls=True)

        pl = io.Plotter()
        pl.add(rcents,r1dw)
        pl.add_err(rcents,prof,yerr=prof_err,ls="--")

        
        num_amps = 200
        nsigma = 6.
        kamps = np.linspace(kamp_true-nsigma*pred_sigma,kamp_true+nsigma*pred_sigma,num_amps)
        k1ds = []
        for amp in kamps:
            template = lensing.nfw_kappa(amp*1e15,modrmap*np.pi/180./60.,cc,overdensity=200.,critical=True,atClusterZ=True)
            kappa_sim = maps.filter_map(template,mK)
            rcents, k1d = rbinner.bin(kappa_sim)
            k1ds.append(k1d)

            pl.add(rcents,k1d,ls="-.")

        pl.hline()
        pl.done(io.dout_dir+"rprof_fit_each.png")
            

        # Fit true for S/N
        lnlikes = []
        for k1d in k1ds:
            diff = (k1d-r1dw)[rcents<=arcmax] 
            lnlike = -0.5*np.dot(np.dot(diff.T,cinv),diff)
            lnlikes.append(lnlike)


        lnlikes = np.array(lnlikes)
        amaxes = kamps[np.isclose(lnlikes,lnlikes.max())]
        print(amaxes)
        kmax = amaxes.ravel()[0]
        
        p = np.polyfit(kamps,lnlikes,2)

        c,b,a = p
        mean = -b/2./c
        sigma = np.sqrt(-1./2./c)
        print(mean,sigma)
        sn = (kmax/sigma)
        print ("S/N fit  : ",sn)


        pl = io.Plotter(xlabel="$A$",ylabel="$\\mathrm{ln}\\mathcal{L}$")
        pl.add(kamps,lnlikes)
        pl.add(kamps,p[0]*kamps**2.+p[1]*kamps+p[2],ls="--")
        pl.vline(x=kmax,ls="--")
        pl.done(io.dout_dir+"lensed_lnlikes_all.png")

        
        template = lensing.nfw_kappa(kmax*1e15,modrmap*np.pi/180./60.,cc,overdensity=200.,critical=True,atClusterZ=True)
        kappa_sim = maps.filter_map(template,mK)
        rcents, k1d = rbinner.bin(kappa_sim)

        pl = io.Plotter()
        pl.add(rcents,r1dw)
        pl.add(rcents,k1d,ls="-.")
        pl.add_err(rcents,prof,yerr=prof_err,ls="--")
        pl.hline()
        pl._ax.set_ylim(-0.02,0.2)
        pl.done(io.dout_dir+"rprof_fit.png")


        
        pl = io.Plotter(xlabel="$A$",ylabel="$\\mathcal{L}$")
        like = np.exp(lnlikes)
        like /= like.max()
        nkamps = np.linspace(kamps.min(),kamps.max(),1000)
        pl.add(nkamps,np.exp(-(nkamps-mean)**2./2./sigma**2.))
        pl.add(kamps,like)
        pl.vline(x=kamp_true,ls="--")
        pl.done(io.dout_dir+"lensed_likes.png")
