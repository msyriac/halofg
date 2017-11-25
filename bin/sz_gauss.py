from __future__ import print_function
from szar.foregrounds import f_nu,g_nu,fgNoises,fgGenerator
from orphics import io,maps,stats,cosmology
import numpy as np, os, sys
from enlib import enmap,fft



const = cosmology.defaultConstants
theory_file_root = "data/Aug6_highAcc_CDM"
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

shape,wcs = maps.rect_geometry(width_deg=10.,px_res_arcmin=2.0)
modlmap = enmap.modlmap(shape,wcs)
lmax = np.max(modlmap)
ells = np.arange(0,lmax,1)
cmb_ps_1d = theory.lCl('TT',ells) 
ps = cmb_ps_1d.reshape((1,1,ells.size))
mg = maps.MapGen(shape,wcs,ps)

TCMB = 2.27255e6

N = 100


# freqs = np.array([90.,148.,220.,353.])
# noises = np.array([1.,1.,1.,1.])/100.
freqs = np.array([90.,148.])
f_nus = f_nu(const,freqs)
noises = np.array([2.,1.])/100.*100.
beam = 1.4
beams = beam *148./freqs
kbeams = []
kbeams1d = []
for beam in beams:
    kbeams.append(maps.gauss_beam(beam,modlmap))
    kbeams1d.append(maps.gauss_beam(beam,ells))


mg_noises = []
for noise in noises:
    ps_noise = ells*0.+(noise*np.pi/180./60.)**2.
    ps_noise = ps_noise.reshape((1,1,ells.size))
    mg_noises.append( maps.MapGen(shape,wcs,ps_noise))

fc = maps.FourierCalc(shape,wcs)
bin_edges = np.arange(30,6000,80)
binner = stats.bin2D(modlmap,bin_edges)


components = ['tsz']#,'cibc','cibp']


fnoises = fgNoises(const,ksz_file='../szar/input/ksz_BBPS.txt',ksz_p_file='../szar/input/ksz_p_BBPS.txt',tsz_cib_file='../szar/input/sz_x_cib_template.dat',ksz_battaglia_test_csv=None,tsz_battaglia_template_csv="../szar/data/sz_template_battaglia.csv",components=components,lmax=6000)

fgen = fgGenerator(shape,wcs,components,const,ksz_file='../szar/input/ksz_BBPS.txt',ksz_p_file='../szar/input/ksz_p_BBPS.txt',tsz_cib_file='../szar/input/sz_x_cib_template.dat',ksz_battaglia_test_csv=None,tsz_battaglia_template_csv="../szar/data/sz_template_battaglia.csv")




cmb2d = theory.lCl('TT',modlmap)
wnoises = (noises*np.pi/180./60.)**2.
cinv = maps.ilc_cinv(modlmap,cmb2d,kbeams,freqs,wnoises,components,fgen)
cinv_1d = maps.ilc_cinv(ells,cmb_ps_1d,kbeams1d,freqs,wnoises,components,fnoises)
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

    kcleaned = maps.silc(kuncleans,cinv)
    ckcleaned = maps.cilc(kuncleans,cinv,f_nus*0.+1.,f_nus)
    p2d_cleaned = fc.f2power(kcleaned,kcleaned)
    cents, p1dcleaned = binner.bin(p2d_cleaned)
    cp2d_cleaned = fc.f2power(ckcleaned,ckcleaned)
    cents, cp1dcleaned = binner.bin(cp2d_cleaned)
    pl.add(cents,p1dcleaned*cents**2.,ls="none",marker="o")
    pl.add(cents,cp1dcleaned*cents**2.,ls="none",marker="o")
    pl.add(ells,ps[0,0,:]*ells**2.,color="k",lw=2)
    pl.add(ells,maps.silc_noise(cinv_1d)*ells**2.,ls="-.",lw=2,label="silc")
    pl.add(ells,maps.cilc_noise(cinv_1d,f_nus*0.+1.,f_nus)*ells**2.,ls="-.",lw=2,label="cilc sz")
    pl._ax.set_xlim(2,6000)
    pl._ax.set_ylim(1e0,1e5)
    pl.legend(loc="lower left")
    pl.done("cls.png")
    io.plot_img(unclean,"unclean.png",lim=[-300,300])
    io.plot_img(fft.ifft(kcleaned,axes=[-2,-1],normalize=True).real,"cleaned.png",lim=[-300,300])
    sys.exit()

