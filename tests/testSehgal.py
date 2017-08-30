import healpy as hp
import numpy as np
import os,sys
import orphics.tools.io as io
import orphics.tools.curvedSky as cs
import matplotlib.pyplot as plt
import pandas as pd


out_dir = os.environ['WWW']+"plots/"
map_root = os.environ['WORK2']+'/data/sehgal/'
kappa_file = map_root + 'healpix_4096_KappaeffLSStoCMBfullsky.fits'
cmb_file = map_root + '148_lensedcmb_healpix.fits'
halo_file = map_root + 'halo_nbody.hd5'
kappa_map = hp.read_map(kappa_file)
# cmb_map = hp.read_map(cmb_file)
# cs.quickMapView(kappa_map,saveLoc=out_dir+"sehgalKappa.png")
# cs.quickMapView(cmb_map,saveLoc=out_dir+"sehgalCmb.png")


print "Loading halo catalog..."
#halos = np.loadtxt(halo_file)

df = pd.read_hdf(halo_file)



print df['Z'].shape
print "Loaded halo catalog."

# M200_min = -np.inf #2.0e14 #also tried 2.0e13
# M200_max = np.inf #no upper limit for now
# z_min = 0.
# z_max = np.inf


M200_min = 2e14 #2.0e14 #also tried 2.0e13
M200_max = np.inf #no upper limit for now
z_min = 0.1
z_max = 0.4

Nmax = 40


sel = (df['Z']>z_min) & (df['Z']<z_max) & (df['M200']>M200_min) & (df['M200']<M200_max)
df = df[sel].sample(Nmax) if Nmax is not None else df[sel]

print df['Z'].shape



halos_select_z = df['Z']
print "Number selected",  len(halos_select_z)
halos_select_M200 = df['M200']
halos_select_RA = df['RA']
halos_select_DEC = df['DEC']

mean_mass = halos_select_M200.mean()

# plot z and M200 histograms
plt.clf()
plt.hist(halos_select_z, bins=100, log=False, facecolor='blue')
plt.xlabel(r'$z$', fontsize=18)
plt.ylabel(r"$N_{halos}$",fontsize=18)
plt.savefig(out_dir+'halo_nbody_M200gt5e13_zgt0p1_zlt0p8_hist_z.png')
#
plt.clf()
bins = np.logspace(np.log10(halos_select_M200.min()), np.log10(halos_select_M200.max()), 50)
plt.hist(halos_select_M200, bins=bins, log=True, facecolor='blue')
#plt.xlim( left=M200_min, right=1.0e15 )
plt.xlabel(r'$M_{200} \, [{\rm M_{\odot}}]$', fontsize=18)
plt.ylabel(r"$N_{halos}$",fontsize=18)
plt.gca().set_xscale("log")
plt.savefig(out_dir+'halo_nbody_M200gt5e13_zgt0p1_zlt0p8_hist_M200.png')

size_arc = 100.
pix = 0.5
Npix = int(size_arc/pix)+1
stack = 0.
k = 0
for ra,dec in zip(halos_select_RA,halos_select_DEC):
    cutout = hp.gnomview(kappa_map, rot=(ra, dec), coord='C', xsize=Npix, ysize=Npix,reso=pix,return_projected_map=True,fig=0)
    stack += cutout
    print ra,dec
    k += 1

stack /= k
io.quickPlot2d(stack,out_dir+"stack.png")
    
    
    
