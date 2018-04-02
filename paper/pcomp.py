from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats
from enlib import enmap
import numpy as np
import os,sys



from orphics.lensing import NlGenerator,getMax
from orphics import maps
deg = 5.
px = 1.0
dell = 10
kellmin = 10
shape,wcs = maps.rect_geometry(width_deg = deg, px_res_arcmin=px)

tellmax = 3000
pellmax = 5000
kellmin = 40
gradCut = None
bigell = 9000
beamY = 1.5
pellmin = 50
pols = ['TT']
dell = 10
plot = False

kellmax = max(tellmax,pellmax)

from orphics.cosmology import Cosmology
cc = Cosmology(lmax=6000,pickling=True)
theory = cc.theory
bin_edges = np.arange(kellmin,kellmax,dell)



myNls = NlGenerator(shape,wcs,theory,bin_edges,gradCut=gradCut,bigell=bigell,lensedEqualsUnlensed=False)



def noises():
    noiseTY = 10.0
    beamTX = beamY
    noiseTX = noiseTY
    tellminX = 50
    tellmaxX = 3000
    beamPX = beamY
    noisePX = np.sqrt(2.)*noiseTX
    pellminX = pellmin
    pellmaxX = pellmax
    beamTY = beamY
    beamPY = beamY
    noisePY = np.sqrt(2.)*noiseTY
    tellminY = 50
    tellmaxY = tellmax
    pellminY = pellmin
    pellmaxY = pellmax
    myNls.updateNoiseAdvanced(beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesX=[0,0],alphasX=[1,1],lkneesY=[0,0],alphasY=[1,1])
    lsmv,Nlmv = myNls.getNl(polComb='TT')

    beamTX = 5.0
    noiseTX = 45.0
    tellminX = 50

    from scipy.interpolate import interp1d
    proot = "/gpfs01/astro/workarea/msyriac/data/planck/"
    lsin,nlsin = np.loadtxt(proot+"lgmca_nls.txt",unpack=True)
    Nlfunc = interp1d(lsin,nlsin,bounds_error=False,fill_value=np.inf)
    TCMB = 2.7255e6
    # rfunc = lambda x: cosmology.white_noise_with_atm_func(x,noiseTX,0,1,dimensionless=False,TCMB=2.7255e6) / maps.gauss_beam(x,beamTX)**2. / TCMB**2.
    rfunc = lambda x: Nlfunc(x) / maps.gauss_beam(x,beamTX)**2. / TCMB**2.
    nfunc = cosmology.noise_pad_infinity(rfunc,tellminX,tellmaxX)
    
    nTX2,nPX,nTY,nPY = myNls.updateNoiseAdvanced(beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesX=[0,0],alphasX=[1,1],lkneesY=[0,0],alphasY=[1,1],noiseFuncTX=nfunc)

    lsmv2,Nlmv2 = myNls.getNl(polComb='TT')


    beamTX = 1.5
    noiseTX = 10.
    noisePX = 10.*np.sqrt(2.)
    tellminX = 50
    tellmaxX = 3000

    tellminY = 50
    tellmaxY = 3000

    noiseTY = np.inf
    beamTY = 5.0
    Nlfunc = interp1d(lsin,nlsin,bounds_error=False,fill_value=np.inf)
    TCMB = 2.7255e6
    # rfunc = lambda x: cosmology.white_noise_with_atm_func(x,noiseTX,0,1,dimensionless=False,TCMB=2.7255e6) / maps.gauss_beam(x,beamTX)**2. / TCMB**2.
    rfunc = lambda x: Nlfunc(x) / maps.gauss_beam(x,beamTY)**2. / TCMB**2.
    nfunc = cosmology.noise_pad_infinity(rfunc,tellminY,tellmaxY)
    
    nTX2,nPX,nTY,nPY = myNls.updateNoiseAdvanced(beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesY=[0,0],alphasY=[1,1],lkneesX=[0,0],alphasX=[1,1],noiseFuncTY=nfunc)

    lsmv3,Nlmv3 = myNls.getNl(polComb='TT')

    
    return lsmv,Nlmv,lsmv2,Nlmv2,lsmv3,Nlmv3



ells = np.arange(2,3000,1)
clkk = theory.gCl('kk',ells)

pl = io.Plotter(yscale='log',xlabel="$L$",ylabel="$C_L$")
lsmv,Nlmv,lsmv2,Nlmv2,lsmv3,Nlmv3 = noises()
pl.add(lsmv,Nlmv,color="C0",label="10 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{Hi-res}} , [T]_{\\mathrm{Hi-res}} )$")
pl.add(lsmv2,Nlmv2,color="C1",ls="--",label="10 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{LGMCA}} , [T]_{\\mathrm{Hi-res}} )$")
pl.add(lsmv3,Nlmv3,color="C2",ls="-.",label="Swap 10 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{LGMCA}} , [T]_{\\mathrm{Hi-res}} )$")

pl.add(ells,clkk,color="k",lw=2)
pl._ax.set_ylim(1e-8,1e-6)
pl._ax.set_xlim(2,2500)
pl.legend(loc='lower right',labsize=10)
pl.done(io.dout_dir+"pgradswap.png")
