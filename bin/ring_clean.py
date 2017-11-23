from __future__ import print_function
from szar.foregrounds import f_nu
from orphics.cosmology import defaultConstants as const, Cosmology
from orphics import io,maps,stats
import numpy as np, os, sys
from enlib import enmap

nus = np.arange(30.,857,0.1)

print (f_nu(const,90.)/f_nu(const,150.))

fnus = f_nu(const,nus)


lmax = 6000
cc = Cosmology(lmax=lmax,pickling=True,dimensionless=False)
ells = np.arange(0,lmax,1)
ps = cc.theory.lCl('TT',ells).reshape((1,1,ells.size))
shape,wcs = maps.rect_geometry(width_arcmin=128.,px_res_arcmin=0.5)
mg = maps.MapGen(shape,wcs,ps)

modrmap = enmap.modrmap(shape,wcs)
modlmap = enmap.modlmap(shape,wcs)
sigma_rad = 2.0*np.pi/180./60.
y = 1.e-4 * np.exp(-modrmap**2./2./sigma_rad**2.)
TCMB = 2.27255e6

N = 100


freqs = np.array([90.,148.])#,220.])
noises = np.array([10.,10.])#,20.])
beam = 1.4
beams = beam *148./freqs

mg_noises = []
for noise in noises:
    ps_noise = ells*0.+(noise*np.pi/180./60.)**2.
    ps_noise = ps_noise.reshape((1,1,ells.size))
    mg_noises.append( maps.MapGen(shape,wcs,ps_noise))



for i in range(N):

    imap = mg.get_map()

    for beam,freq,mg_noise in zip(beams,freqs,mg_noises):
        sz = TCMB*y*f_nu(const,freq)
        imap_freq = imap+sz
        imap_freq = maps.filter_map(imap_freq,maps.gauss_beam(modlmap,beam))
        noise_map = mg_noise.get_map()
        imap_freq += noise_map
    #     io.plot_img(imap_freq,"map"+str(freq)+".png",lim=[-300,300])
    # sys.exit()

