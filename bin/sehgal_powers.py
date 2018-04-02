from __future__ import print_function
import healpy as hp
import numpy as np
import os,sys
from orphics import io,maps,catalogs,cosmology,stats
import halofg.sehgalInterface as si
from enlib import enmap
from scipy.ndimage.interpolation import rotate
from szar.foregrounds import fgNoises,fgGenerator

SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()

const = cosmology.defaultConstants
theory_file_root = "data/Aug6_highAcc_CDM"
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)

pl = io.Plotter(yscale='log')
ells = np.arange(0,10000,1)
cltt = theory.lCl('TT',ells)

comp = {}
labels = ['tsz','radpts','irpts'] #,'ksz'
colors = ['C0','C1','C2']
cals = [1.3,1.,0.8]
#cals = [1.,1.,1.]
for k,(label,color,cal) in enumerate(zip(labels,colors,cals)):
    print (label)
    imap = enmap.read_map("/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cache/sehgal_d56_148_"+label+".hdf")
    if k==0:
        shape,wcs = imap.shape,imap.wcs
        # kmask = maps.mask_kspace(shape,wcs, lxcut = None, lycut = None, lmin = None, lmax = lmax)
        kbeam = maps.gauss_beam(enmap.modlmap(shape,wcs),1.4)
        taper,w2 = maps.get_taper(shape,taper_percent = 12.0,pad_percent = 3.0,weight=None)
        fc = maps.FourierCalc(shape,wcs)
        modlmap = enmap.modlmap(shape,wcs)
        bin_edges = np.arange(100,10000,200)
        binner = stats.bin2D(modlmap,bin_edges)

        
    #comp[label] = maps.filter_map(imap*taper,kbeam)
    # io.plot_img(comp[label],io.dout_dir+"d56_"+label+".png",high_res=True)
    p2d,_,_ = fc.power2d(imap*taper*cal)
    p2d /= w2
    cents, comp[label] = binner.bin(p2d)
    pl.add(cents,comp[label]*cents**2.,marker="o",label=label,color=color,ls="none")


components = ['tsz','cibp','cibc','radps']
fnoises = fgNoises(const,ksz_file='../szar/input/ksz_BBPS.txt',ksz_p_file='../szar/input/ksz_p_BBPS.txt',tsz_cib_file='../szar/input/sz_x_cib_template.dat',ksz_battaglia_test_csv=None,tsz_battaglia_template_csv="../szar/input/sz_template_battaglia.csv",components=components,lmax=10000)

    
nl = fnoises.get_noise("tsz",148.,148.,ells)
pl.add(ells,nl*ells**2.,color="C0",ls="-")

nl = fnoises.get_noise("cibp",148.,148.,ells)+fnoises.get_noise("cibc",148.,148.,ells)
pl.add(ells,nl*ells**2.,color="C2",ls="-")

nl = fnoises.get_noise("radps",148.,148.,ells)
pl.add(ells,nl*ells**2.,color="C1",ls="-")

pl.add(ells,cltt*ells**2.,color='k',lw=2)
pl._ax.set_ylim(1.,1e5)
pl.legend()
pl.done(io.dout_dir+"comps.png")
