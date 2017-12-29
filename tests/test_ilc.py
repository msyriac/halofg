from __future__ import print_function
import healpy as hp
import numpy as np
import os,sys
from orphics import io,maps,catalogs,cosmology
import halofg.sehgalInterface as si
from enlib import enmap
from scipy.ndimage.interpolation import rotate
from szar.foregrounds import fgNoises,fgGenerator

SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()


zmin = 0.5
zmax = np.inf
# mmin = 1e14
# mmax = 6e14
mmin = 4e14
mmax = np.inf
nmax = None
#deep56: -7.5,38.5,4.,-5.5
#patches: y0,x0,y1,x1 (y=DEC deg, x=RA deg)
df,ras,decs,m200,z = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=mmin,M200_max=mmax,z_min=zmin,z_max=zmax,Nmax=nmax,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None,ra_min=-5.5,ra_max=38.5,dec_min=-7.5,dec_max=4.)

# print(len(ras.tolist()))
# sys.exit()



comp = {}
labels = ['tsz','radpts','ksz','irpts']
cals = [1.,1.,1.0,1.0]
for k,(label,cal) in enumerate(zip(labels,cals)):
    print (label)
    imap = enmap.read_map("/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cache/sehgal_d56_148_"+label+".hdf")
    if k==0:
        shape,wcs = imap.shape,imap.wcs
        # kmask = maps.mask_kspace(shape,wcs, lxcut = None, lycut = None, lmin = None, lmax = lmax)
        kbeam = maps.gauss_beam(enmap.modlmap(shape,wcs),1.4)
        taper,w2 = maps.get_taper(shape,taper_percent = 8.0,pad_percent = 2.0,weight=None)
        
    comp[label] = maps.filter_map(imap*taper*cal,kbeam)
    # io.plot_img(comp[label],io.dout_dir+"d56_"+label+".png",high_res=True)


const = cosmology.defaultConstants
theory_file_root = "data/Aug6_highAcc_CDM"
theory = cosmology.loadTheorySpectraFromCAMB(theory_file_root,unlensedEqualsLensed=False,
                                                    useTotal=False,TCMB = 2.7255e6,lpad=9000,get_dimensionless=False)



arc = 80.5
px = 0.5
Npix = int(arc/px)

cshape,cwcs = maps.rect_geometry(width_arcmin=arc,px_res_arcmin=px)
assert cshape==(Npix,Npix)
modlmap = enmap.modlmap(cshape,cwcs)
modrmap = enmap.modrmap(cshape,cwcs)
lmax = modlmap.max()
ells = np.arange(0,lmax,1)
cls = theory.lCl('TT',ells)
cmb2d = theory.lCl('TT',modlmap)
ps = cls.reshape((1,1,ells.size))
mg = maps.MapGen(cshape,cwcs,ps)
components = ['tsz','cibc','cibp','radps']
fgen = fgGenerator(shape,wcs,components,const,ksz_file='../szar/input/ksz_BBPS.txt',ksz_p_file='../szar/input/ksz_p_BBPS.txt',tsz_cib_file='../szar/input/sz_x_cib_template.dat',ksz_battaglia_test_csv=None,tsz_battaglia_template_csv="../szar/data/sz_template_battaglia.csv")
#cinv = maps.ilc_cinv(modlmap,cmb2d,kbeams,freqs,wnoises,components,fgen)
ckbeam = maps.gauss_beam(enmap.modlmap(cshape,cwcs),1.4)


stacks = {}
count = 0
all_stack = 0.


irs = []
for j,label in enumerate(labels):
    stacks[label] = 0.

for k,(ra,dec) in enumerate(zip(ras,decs)):
    if (k+1)%1000==0: print(ra,dec)

    iy,ix = enmap.sky2pix(shape,wcs,(dec*np.pi/180.,ra*np.pi/180.)).astype(np.int)

    rangle = np.random.uniform(0.,360.)
    
    cmb = mg.get_map()
    fgs = 0.
    reject = False
    for j,label in enumerate(labels):
        cutout = comp[label][iy-Npix//2:iy+Npix//2+1,ix-Npix//2:ix+Npix//2+1].copy()
        if cutout.shape!=(Npix,Npix):
            reject = True
            break
        #cutout = rotate(cutout,rangle,reshape=False)
        # if label=="irpts" :
        #     fcutout = cutout[modrmap<10.*np.pi/180./60.]
        #     if fcutout.mean()>20.: break
        #     irs.append(fcutout.mean())
        stacks[label] += cutout.copy()
        fgs += cutout.copy()
        if j==0: count += 1

    if not(reject):
        all_stack += fgs
        observed = maps.filter_map(cmb,ckbeam) + fgs
        if k<30: io.plot_img(observed,io.dout_dir+"observed_"+str(k)+".png")
    
# import matplotlib.pyplot as plt
# plt.hist(irs,bins=20)
# plt.savefig(io.dout_dir+"irhist.png")
print(count)
for label in labels:
    io.plot_img(stacks[label]/count,io.dout_dir+"stack_"+label+".png")
io.plot_img((all_stack)/count,io.dout_dir+"stack_sum.png")
