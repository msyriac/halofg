from __future__ import print_function
from szar.foregrounds import f_nu
from orphics import cosmology
from orphics import io,maps,stats
import numpy as np, os, sys
from enlib import enmap,fft
from szar.foregrounds import f_nu,g_nu,fgNoises,fgGenerator


const = cosmology.defaultConstants
theory_file_root = "data/Aug6_highAcc_CDM"
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

fgs = fgNoises(const,ksz_file='../szar/input/ksz_BBPS.txt',ksz_p_file='../szar/input/ksz_p_BBPS.txt',tsz_cib_file='../szar/input/sz_x_cib_template.dat',ksz_battaglia_test_csv=None,tsz_battaglia_template_csv="../szar/data/sz_template_battaglia.csv")
components = ['tsz']


shape,wcs = maps.rect_geometry(width_arcmin=128.,px_res_arcmin=0.5)
fgen = fgGenerator(shape,wcs,components,fgs)

modlmap = enmap.modlmap(shape,wcs)
lmax = np.max(modlmap)
ells = np.arange(0,lmax,1)
ps = theory.lCl('TT',ells).reshape((1,1,ells.size))
mg = maps.MapGen(shape,wcs,ps)

modrmap = enmap.modrmap(shape,wcs)
modlmap = enmap.modlmap(shape,wcs)
sigma_rad = 2.0*np.pi/180./60.
y = 1.e-4 * np.exp(-modrmap**2./2./sigma_rad**2.)
TCMB = 2.27255e6

N = 1000


# WHY DO WE NEED FREQS TO SPAN 220 TO GET LOW RESIDUALS?

freqs = np.array([90.,148.,350.])
noises = np.array([1.,1.,1.])*3./2.
# freqs = np.array([90.,148.])#,220.])
# noises = np.array([1.,1.])/100.#,20.])
beam = 1.4
#beams = [beam]*len(freqs)
beams = beam *148./freqs
kbeams = [maps.gauss_beam(modlmap,abeam) for abeam in beams]

mg_noises = []
for noise in noises:
    ps_noise = ells*0.+(noise*np.pi/180./60.)**2.
    ps_noise = ps_noise.reshape((1,1,ells.size))
    mg_noises.append( maps.MapGen(shape,wcs,ps_noise))


nfreqs = len(noises)
cmb2d = theory.lCl('TT',modlmap)
Covmat = np.tile(cmb2d,(nfreqs,nfreqs,1,1))

for i,(kbeam1,freq1,noise1) in enumerate(zip(kbeams,freqs,noises)):
    for j,(kbeam2,freq2,noise2) in enumerate(zip(kbeams,freqs,noises)):
        if i==j:
            Covmat[i,j,:,:] += modlmap*0.+(noise1*np.pi/180./60.)**2./kbeam1**2.
        for component in components:
            Covmat[i,j,:,:] += fgen.get_noise(component,freq1,freq2,modlmap)


cinv = np.linalg.inv(Covmat.T).T

    
fc = maps.FourierCalc(shape,wcs)
kmask = maps.mask_kspace(shape,wcs, lmin = None, lmax = 6000)

stack = 0.

for i in range(N):

    imap = mg.get_map()
    kuncleans = []
    for kbeam,freq,mg_noise in zip(kbeams,freqs,mg_noises):
        sz = TCMB*y*f_nu(const,freq)
        imap_freq = imap+sz
        imap_freq = maps.filter_map(imap_freq,kbeam)
        noise_map = mg_noise.get_map()
        imap_freq += noise_map
        p2dunclean,kunclean,_ = fc.power2d(imap_freq)
        kunclean *= np.nan_to_num(kmask/kbeam)
        kuncleans.append(kunclean)
        io.plot_img(imap_freq,"map"+str(freq)+".png",lim=[-300,300])
        
    kuncleans = np.stack(kuncleans)
    kcleaned = maps.ilc_cmb(kuncleans,cinv)
    cleaned = fft.ifft(kcleaned,axes=[-2,-1],normalize=True).real
    stack += cleaned
    io.plot_img(cleaned,"map_cleaned.png",lim=[-300,300])
    io.plot_img(imap,"map_clean.png",lim=[-300,300])
        
    sys.exit()
    if i%10==0: print (i)
io.plot_img(stack/N,"stack.png")
