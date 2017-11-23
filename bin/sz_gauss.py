from __future__ import print_function
from szar.foregrounds import f_nu,g_nu,fgNoises,fgGenerator
from orphics import io,maps,stats,cosmology
import numpy as np, os, sys
from enlib import enmap,fft

const = cosmology.defaultConstants
theory_file_root = "data/Aug6_highAcc_CDM"
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)


fgs = fgNoises(const,ksz_file='../szar/input/ksz_BBPS.txt',ksz_p_file='../szar/input/ksz_p_BBPS.txt',tsz_cib_file='../szar/input/sz_x_cib_template.dat',ksz_battaglia_test_csv=None,tsz_battaglia_template_csv="../szar/data/sz_template_battaglia.csv")




shape,wcs = maps.rect_geometry(width_deg=10.,px_res_arcmin=2.0)
modlmap = enmap.modlmap(shape,wcs)
lmax = np.max(modlmap)
ells = np.arange(0,lmax,1)
ps = theory.lCl('TT',ells).reshape((1,1,ells.size))
mg = maps.MapGen(shape,wcs,ps)

TCMB = 2.27255e6

N = 100


freqs = np.array([90.,148.,220.,353.])
noises = np.array([1.,1.,1.,1.])/100.
# freqs = np.array([90.,148.])
# noises = np.array([1.,1.])/100.
beam = 1.4
beams = beam *148./freqs
kbeams = []
for beam in beams:
    kbeams.append(maps.gauss_beam(beam,modlmap))


mg_noises = []
for noise in noises:
    ps_noise = ells*0.+(noise*np.pi/180./60.)**2.
    ps_noise = ps_noise.reshape((1,1,ells.size))
    mg_noises.append( maps.MapGen(shape,wcs,ps_noise))

fc = maps.FourierCalc(shape,wcs)
bin_edges = np.arange(30,6000,40)
binner = stats.bin2D(modlmap,bin_edges)


components = ['tsz','cibc','cibp']
fgen = fgGenerator(shape,wcs,components,fgs)




nfreqs = len(noises)
cmb2d = theory.lCl('TT',modlmap)
Covmat = np.tile(cmb2d,(nfreqs,nfreqs,1,1))#,modlmap.shape[0],modlmap.shape[1]))

for i,(kbeam1,freq1,noise1) in enumerate(zip(kbeams,freqs,noises)):
    for j,(kbeam2,freq2,noise2) in enumerate(zip(kbeams,freqs,noises)):
        if i==j:
            Covmat[i,j,:,:] += modlmap*0.+(noise1*np.pi/180./60.)**2./kbeam1**2.
        for component in components:
            Covmat[i,j,:,:] += fgen.get_noise(component,freq1,freq2)
        #(TCMB*(y*np.pi/180./60.)*f_nu(const,freq1))*(TCMB*(y*np.pi/180./60.)*f_nu(const,freq2))+


cinv = np.linalg.inv(Covmat.T).T
print(cinv.shape)

kmask = maps.mask_kspace(shape,wcs, lmin = None, lmax = 6000)

                     
for i in range(N):

    cmb = mg.get_map()
    fgmaps = []
    for component in components:
        fgmaps.append( fgen.get_maps(component,freqs) )
    fgmaps = np.stack(fgmaps).sum(axis=0)


    p2d_cleans = []
    p2d_uncleans = []
    kuncleans = []
    
    pl = io.Plotter(yscale="log")
    for k,(kbeam,freq,mg_noise) in enumerate(zip(kbeams,freqs,mg_noises)):
        fg = fgmaps[k,:,:]
        bcmb = maps.filter_map(cmb+fg,kbeam)
        noise_map = mg_noise.get_map()
        unclean = bcmb+noise_map

        p2dunclean,kunclean,_ = fc.power2d(unclean)
        p2dunclean /= kbeam**2.
        kunclean *= np.nan_to_num(kmask/kbeam)
        p2d_uncleans.append(p2dunclean)

        cents, p1dunclean = binner.bin(p2dunclean)

        kuncleans.append(kunclean)
        
        pl.add(cents,p1dunclean*cents**2.,alpha=0.4,label=str(freq))
        


    kuncleans = np.stack(kuncleans)

    kcleaned = np.einsum('klij,lij->kij',cinv,kuncleans).sum(axis=0) / cinv.sum(axis=(0,1))
    p2d_cleaned = fc.f2power(kcleaned,kcleaned)
    cents, p1dcleaned = binner.bin(p2d_cleaned)
    pl.add(cents,p1dcleaned*cents**2.,ls="none",marker="o")
    pl.add(ells,ps[0,0,:]*ells**2.,color="k",lw=2)
    pl._ax.set_xlim(2,6000)
    pl.legend()
    pl.done("cls.png")
    io.plot_img(unclean,"unclean.png",lim=[-300,300])
    io.plot_img(fft.ifft(kcleaned,axes=[-2,-1],normalize=True).real,"cleaned.png",lim=[-300,300])
    sys.exit()


def make_cinv(cmb2d,kbeams,freqs,noises,const_dict=cosmology.defaultConstants):
    """
    cmb2d -- Theory C_ell_TT in 2D fourier space

    """

    nfreqs = len(noises)
    Covmat = np.tile(cmb2d,(nfreqs,nfreqs,1,1))

    for i,(kbeam1,freq1,noise1) in enumerate(zip(kbeams,freqs,noises)):
        for j,(kbeam2,freq2,noise2) in enumerate(zip(kbeams,freqs,noises)):
            if i==j:
                Covmat[i,j,:,:] += modlmap*0.+(noise1*np.pi/180./60.)**2./kbeam1**2.
            Covmat[i,j,:,:] += (TCMB*(y*np.pi/180./60.)*f_nu(const,freq1))*(TCMB*(y*np.pi/180./60.)*f_nu(const,freq2))+modlmap*0.


    cinv = np.linalg.inv(Covmat.T).T
    print(cinv.shape)

    
def ilc_cmb(kmaps,cinv):
    """Clean a set of microwave observations using ILC to get an estimate of the CMB map.
    
    Accepts
    -------

    kmaps -- (nfreq,Ny,Nx) array of beam-deconvolved fourier transforms at each frequency
    cinv -- (nfreq,nfreq,Ny,Nx) array of the inverted covariance matrix

    Returns
    -------

    Fourier transform of CMB map estimate, (Ny,Nx) array 
    """
    return np.einsum('klij,lij->kij',cinv,kmaps).sum(axis=0) / cinv.sum(axis=(0,1))
