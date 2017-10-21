import orphics.tools.io as io
import orphics.analysis.flatMaps as fmaps
from alhazen.quadraticEstimator import Estimator
import alhazen.io as aio
import alhazen.lensTools as lt
from enlib import enmap, resample, lensing, fft
from orphics.analysis.pipeline import mpi_distribute, MPIStats
import numpy as np

class HaloFgPipeline(object):

    def __init__(self,inp_dir,out_dir,Nmax,analysis_section,sims_section,recon_section,cutout_section,catalog_bin,
                 experimentX,experimentY,components,recon_config_file="input/cmb-config/recon.ini",
                 sim_config_file="input/sehgal.ini",mpi_comm=None,cosmology_section="cc_cluster",
                 gradcut=2000,bin_edges=None,verbose=False):
        
        self.Config = io.config_from_file(recon_config_file)
        self.SimConfig = io.config_from_file(sim_config_file)

        # Get MPI comm
        self.comm = mpi_comm
        try:
            self.rank = mpi_comm.Get_rank()
            self.numcores = mpi_comm.Get_size()
        except:
            self.rank = 0
            self.numcores = 1
        
        if verbose and self.rank==0: print "Initializing patches..."
        shape_sim, wcs_sim, shape_dat, wcs_dat = aio.enmaps_from_config(self.Config,
                                                                        sims_section,
                                                                        analysis_section,
                                                                        pol=False)

        
        self.psim = aio.patch_array_from_config(self.Config,experimentX,
                                                shape_sim,wcs_sim,dimensionless=False)
        self.pdatX = aio.patch_array_from_config(self.Config,experimentX,
                                                shape_dat,wcs_dat,dimensionless=False)
        self.pdatY = aio.patch_array_from_config(self.Config,experimentY,
                                                shape_dat,wcs_dat,dimensionless=False)


        arc = self.SimConfig.getfloat(cutout_section,"arc")
        pix = self.SimConfig.getfloat(cutout_section,"px")
        self.kshape, self.kwcs = enmap.rect_geometry(arc,pix,proj="car",pol=False)


        
        if Nmax is not None:
            Ntot = Nmax
        else:
            Ntot = self.SimConfig.get(catalog_bin,'N_max')
            if Ntot=="inf" :
                import glob
                search_path = self.SimConfig.get("sims","map_root")+inp_dir+"/"+catalog_bin+"/*kappa*.npy"
                files = glob.glob(search_path)
                Ntot = len(files)
                assert Ntot>0
            else:
                Ntot = int(Ntot)

                
        num_each,each_tasks = mpi_distribute(Ntot,self.numcores)
        self.mpibox = MPIStats(self.comm,num_each,tag_start=333)
        if self.rank==0: print "At most ", max(num_each) , " tasks..."
        self.clusters = each_tasks[self.rank]

        if verbose and self.rank==0: print "Initializing cosmology..."

        theory, cc, lmax = aio.theory_from_config(self.Config,cosmology_section,dimensionless=False)
        self.pdatX.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)
        self.pdatY.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)
        self.psim.add_theory(cc,theory,lmax,orphics_is_dimensionless=False)

        self.lens_order = self.Config.getint(sims_section,"lens_order")
        self.map_root = self.SimConfig.get("sims","map_root")+inp_dir+"/"+catalog_bin+"/cutout_"

        # FG COMPONENTS INIT
        self.components = components.split(',')
        if len(self.components)==1 and self.components[0].lower().strip()=="none":
            self.components=[]


        # RECONSTRUCTION INIT
        if verbose and self.rank==0: print "Initializing quadratic estimator..."

        min_ell = fmaps.minimum_ell(shape_dat,wcs_dat)
        lb = aio.ellbounds_from_config(self.Config,recon_section,min_ell)
        tellminY = lb['tellminY']
        tellmaxY = lb['tellmaxY']
        pellminY = lb['pellminY']
        pellmaxY = lb['pellmaxY']
        tellminX = lb['tellminX']
        tellmaxX = lb['tellmaxX']
        pellminX = lb['pellminX']
        pellmaxX = lb['pellmaxX']
        kellmin = lb['kellmin']
        kellmax = lb['kellmax']
        lxmap_dat,lymap_dat,modlmap_dat,angmap_dat,lx_dat,ly_dat = fmaps.get_ft_attributes_enmap(shape_dat,wcs_dat)
        lxmap_sim,lymap_sim,modlmap_sim,angmap_sim,lx_sim,ly_sim = fmaps.get_ft_attributes_enmap(shape_sim,wcs_sim)

        template_dat = fmaps.simple_flipper_template_from_enmap(shape_dat,wcs_dat)
        fMaskCMB_TX = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=tellminX,lmax=tellmaxX)
        fMaskCMB_TY = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=tellminY,lmax=tellmaxY)
        fMaskCMB_PX = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=pellminX,lmax=pellmaxX)
        fMaskCMB_PY = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=pellminY,lmax=pellmaxY)
        fMask = fmaps.fourierMask(lx_dat,ly_dat,modlmap_dat,lmin=kellmin,lmax=kellmax)

        with io.nostdout():
            self.qestimator = Estimator(template_dat,
                                        theory,
                                        theorySpectraForNorm=None,
                                        noiseX2dTEB=[self.pdatX.nT,self.pdatX.nP,self.pdatX.nP],
                                        noiseY2dTEB=[self.pdatY.nT,self.pdatY.nP,self.pdatY.nP],
                                        fmaskX2dTEB=[fMaskCMB_TX,fMaskCMB_PX,fMaskCMB_PX],
                                        fmaskY2dTEB=[fMaskCMB_TY,fMaskCMB_PY,fMaskCMB_PY],
                                        fmaskKappa=fMask,
                                        kBeamX = self.pdatX.lbeam,
                                        kBeamY = self.pdatY.lbeam,
                                        doCurl=False,
                                        TOnly=True,
                                        halo=True,
                                        uEqualsL=True,
                                        gradCut=gradcut,verbose=False,
                                        bigell=lmax)


        if verbose and self.rank==0: print "Initializing binner..."
        import orphics.tools.stats as stats
        if bin_edges is None:
            analysis_resolution =  np.min(enmap.extent(shape_dat,wcs_dat)/shape_dat[-2:])*60.*180./np.pi
            bin_edges = np.arange(0.,10.,2.*analysis_resolution)

        self.binner = stats.bin2D(self.pdatX.modrmap*60.*180./np.pi,bin_edges)
        self.plot_dir = self.SimConfig.get("output","plot_dir")+inp_dir+"/"+out_dir+"/"+catalog_bin+"/"
        self.result_dir = self.SimConfig.get("output","result_dir")+inp_dir+"/"+out_dir+"/"+catalog_bin+"/"

    def get_unlensed(self,seed):
        return self.psim.get_unlensed_cmb(seed=seed)

    def get_kappa(self,index,stack=False):
        retmap = np.load(self.map_root+"kappa_"+str(index)+".npy").astype(np.float64)
        assert np.all(retmap.shape==self.kshape)
        if stack:
            self.mpibox.add_to_stack("inpstack",retmap)
        return retmap
        
    def upsample(self,imap):
        return enmap.ndmap(resample.resample_fft(imap,self.psim.shape),self.psim.wcs)

    def get_lensed(self,unlensed,kappa):
        phi = lt.kappa_to_phi(kappa,self.psim.modlmap,return_fphi=False)
        grad_phi = enmap.grad(phi)
        lensed = lensing.lens_map(unlensed, grad_phi, order=self.lens_order, mode="spline", border="cyclic", trans=False, deriv=False, h=1e-7)
        return lensed

    def downsample(self,imap):
        return enmap.ndmap(resample.resample_fft(imap,self.pdatX.shape),self.pdatX.wcs)

    def get_fg_single_band(self,cluster_id,stack=False):
        fg = 0.
        for comp in self.components:
            imap = np.load(self.map_root+comp+"_"+str(cluster_id)+".npy").astype(np.float64)
            assert np.all(imap.shape==self.kshape)
            fg += imap
            if stack:
                self.mpibox.add_to_stack(comp,imap)

        if len(self.components)>0 and (fg.shape!=self.pdatX.shape): fg = enmap.ndmap(resample.resample_fft(fg,self.pdatX.shape),self.pdatX.wcs)
        return fg
            

    def beam(self,XY,imap):
        flensed = fft.fft(imap,axes=[-2,-1])
        assert XY in ["X","Y"]
        lbeam = self.pdatX.lbeam if XY=="X" else self.pdatY.lbeam
        flensed *= lbeam
        return fft.ifft(flensed,axes=[-2,-1],normalize=True).real

    def get_noise(self,XY,seed):
        assert XY in ["X","Y"]
        pdat = self.pdatX if XY=="X" else self.pdatY
        return pdat.get_noise_sim(seed=seed)

    def qest(self,X,Y):
        self.qestimator.updateTEB_X(X,alreadyFTed=False)
        self.qestimator.updateTEB_Y(Y,alreadyFTed=False)
        recon = self.qestimator.getKappa("TT").real
        return recon


    def profile(self,imap2d):
        cents,imap1d = self.binner.bin(imap2d)
        self.cents = cents
        return imap1d
    
    def dump(self):
        io.mkdir(self.plot_dir)
        io.mkdir(self.result_dir)

        reconstack = self.mpibox.stacks['reconstack']
        inpstack = self.mpibox.stacks['inpstack']
        
        io.quickPlot2d(reconstack,self.plot_dir+"reconstack.png")
        io.quickPlot2d(inpstack,self.plot_dir+"inpstack.png")
        for comp in self.components:
            io.quickPlot2d(self.mpibox.stacks[comp],self.plot_dir+comp+"_stack.png")


            
        if (inpstack.shape!=self.pdatX.shape): inpstack = enmap.ndmap(resample.resample_fft(inpstack,self.pdatX.shape),self.pdatX.wcs)
        np.save(self.result_dir+"inpstack.npy",inpstack)
        np.save(self.result_dir+"reconstack.npy",reconstack)
        inp_profile = self.profile(inpstack)
            
        io.save_cols(self.result_dir+"profile.txt",(self.cents,
                                                    inp_profile,
                                                    self.mpibox.stats['recon1d']['mean']))

        np.save(self.result_dir+"covmean.npy",self.mpibox.stats['recon1d']['covmean'])
        np.save(self.result_dir+"cov.npy",self.mpibox.stats['recon1d']['cov'])

                     
        print "Done!"
