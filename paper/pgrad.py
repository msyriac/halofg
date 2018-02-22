from __future__ import print_function
from orphics import maps,io,cosmology,catalogs,stats
from enlib import enmap
import numpy as np
import os,sys


import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])

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
pellmin = 500
pols = ['TT','ET','EB','EE','TB']
#pols = ['TT','TE','EB','EE','TB']
#pols = ['TT']
dell = 10
plot = False

kellmax = max(tellmax,pellmax)

from orphics.cosmology import LimberCosmology
cc = LimberCosmology(lmax=6000,pickling=True)
theory = cc.theory
bin_edges = np.arange(kellmin,kellmax,dell)


zedges = np.arange(0.,2.0,0.1)
zs = (zedges[1:]+zedges[:-1])/2.
dndz = catalogs.dndz(zs,z0=1./3.)
cc.addNz("g",zedges,dndz,bias=1.6,magbias=None,numzIntegral=300,ignore_exists=False)
cc.addNz("s",zedges,dndz,bias=None,magbias=None,numzIntegral=300,ignore_exists=False)

ellrange = np.arange(2,3000,1)
cc.generateCls(ellrange,autoOnly=False,zmin=0.)


clkg = cc.getCl("cmb","g")
clks = cc.getCl("cmb","s")
clgg = cc.getCl("g","g")
clss = cc.getCl("s","s")

myNls = NlGenerator(shape,wcs,theory,bin_edges,gradCut=gradCut,bigell=bigell,lensedEqualsUnlensed=False,unlensedEqualsLensed=True)



def noises(noiseTY):
    beamTX = beamY
    noiseTX = noiseTY
    tellminX = 500
    tellmaxX = 3000
    beamPX = beamY
    noisePX = np.sqrt(2.)*noiseTX
    pellminX = pellmin
    pellmaxX = pellmax
    beamTY = beamY
    beamPY = beamY
    noisePY = np.sqrt(2.)*noiseTY
    tellminY = 500
    tellmaxY = tellmax
    pellminY = pellmin
    pellmaxY = pellmax
    myNls.updateNoiseAdvanced(beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesX=[3000,300],alphasX=[-4,-4],lkneesY=[3000,300],alphasY=[-4,-4])
    lsmv,Nlmv,ells,dclbb,efficiency = myNls.getNlIterative(pols,kellmin,kellmax,tellmax,pellmin,pellmax,dell=dell,halo=True,plot=plot)

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
    
    # nTX1,nPX,nTY,nPY = myNls.updateNoiseAdvanced(beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesX=[0,300],alphasX=[1,-4],lkneesY=[3000,300],alphasY=[-4,-4])#,noiseFuncTX=nfunc)


    nTX2,nPX,nTY,nPY = myNls.updateNoiseAdvanced(beamTX,noiseTX,beamPX,noisePX,tellminX,tellmaxX,pellminX,pellmaxX,beamTY,noiseTY,beamPY,noisePY,tellminY,tellmaxY,pellminY,pellmaxY,lkneesX=[0,300],alphasX=[1,-4],lkneesY=[3000,300],alphasY=[-4,-4],noiseFuncTX=nfunc)

    # modlmap = myNls.N.modLMap
    # bin_edges = np.arange(100,3000,40)
    # binner = stats.bin2D(modlmap,bin_edges)
    # ells = np.arange(0,3000,1)
    # cltt = theory.lCl('TT',ells)
    # cents,n1d1 = binner.bin(nTX1)
    # cents,n1d2 = binner.bin(nTX2)
    # pl = io.Plotter(yscale='log')
    # pl.add(cents,n1d1*cents**2.)
    # pl.add(cents,n1d2*cents**2.)
    # pl.add(ells,cltt*ells**2.)
    # pl.done(io.dout_dir+"ncomp.png")
    # sys.exit()

    if pols==['TT']:
        lsmv2,Nlmv2 = myNls.getNl("TT",halo=True)
    else:        
        lsmv2,Nlmv2,ells,dclbb,efficiency = myNls.getNlIterative(pols,kellmin,kellmax,tellmax,pellmin,pellmax,dell=dell,halo=True,plot=plot)

    return lsmv,Nlmv,lsmv2,Nlmv2



ells = np.arange(2,3000,1)
clkk = theory.gCl('kk',ells)

LF = cosmology.LensForecast()
snbins = np.arange(100,2000,40)
LF.loadKS(ellrange,clks)
LF.loadKG(ellrange,clkg)

hsc_ngg = 20.0
des_ngg = 6.0
hsc_fsky = 1400./41250.
des_fsky = 5000./41250.

pl = io.Plotter(yscale='log',xlabel="$L$",ylabel="$C_L^{\\kappa\\kappa}$")
lsmv,Nlmv,lsmv2,Nlmv2 = noises(20.)
pl.add(lsmv,Nlmv,color="C1",label="20 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{Hi-res}} , [T]_{\\mathrm{Hi-res}} )$",lw=2)
pl.add(lsmv2,Nlmv2,color="C1",ls="--",label="20 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{LGMCA}} , [T]_{\\mathrm{Hi-res}} )$",lw=2)

print("20uK' HH")
LF.loadKK(ells,clkk,lsmv,Nlmv)
LF.loadSS(ells,clss,hsc_ngg)
LF.loadGG(ells,clgg,hsc_ngg)
hscks1 = LF.sn(snbins,hsc_fsky,"ks")[0]
hsckg1 = LF.sn(snbins,hsc_fsky,"kg")[0]
print("HSC ks: ",hscks1)
print("HSC kg: ",hsckg1)
LF.loadSS(ells,clss,des_ngg)
LF.loadGG(ells,clgg,des_ngg)
desks1 = LF.sn(snbins,des_fsky,"ks")[0]
deskg1 = LF.sn(snbins,des_fsky,"kg")[0]
print("DES ks: ",desks1)
print("DES kg: ",deskg1)

sn1 = LF.sn(snbins,0.4,"kk")[0]


print("20uK' PH")
LF.loadKK(ells,clkk,lsmv2,Nlmv2)
LF.loadSS(ells,clss,hsc_ngg)
LF.loadGG(ells,clgg,hsc_ngg)
hscks2 = LF.sn(snbins,hsc_fsky,"ks")[0]
hsckg2 = LF.sn(snbins,hsc_fsky,"kg")[0]
print("HSC ks: ",hscks2)
print("HSC kg: ",hsckg2)
LF.loadSS(ells,clss,des_ngg)
LF.loadGG(ells,clgg,des_ngg)
desks2 = LF.sn(snbins,des_fsky,"ks")[0]
deskg2 = LF.sn(snbins,des_fsky,"kg")[0]
print("DES ks: ",desks2)
print("DES kg: ",deskg2)

sn2 = LF.sn(snbins,0.4,"kk")[0]

print("Clkk diff: " , (sn2-sn1)*100./sn1)

print("DES ks diff: " , (desks2-desks1)*100./desks1)
print("DES kg diff: " , (deskg2-deskg1)*100./deskg1)
print("HSC ks diff: " , (hscks2-hscks1)*100./hscks1)
print("HSC kg diff: " , (hsckg2-hsckg1)*100./hsckg1)


lsmv,Nlmv,lsmv2,Nlmv2 = noises(10.)
pl.add(lsmv,Nlmv,color="C0",label="10 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{Hi-res}} , [T]_{\\mathrm{Hi-res}} )$",lw=2)
pl.add(lsmv2,Nlmv2,color="C0",ls="--",label="10 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{LGMCA}} , [T]_{\\mathrm{Hi-res}} )$",lw=2)

print("10uK' HH")
LF.loadKK(ells,clkk,lsmv,Nlmv)
LF.loadSS(ells,clss,hsc_ngg)
LF.loadGG(ells,clgg,hsc_ngg)
hscks1 = LF.sn(snbins,hsc_fsky,"ks")[0]
hsckg1 = LF.sn(snbins,hsc_fsky,"kg")[0]
print("HSC ks: ",hscks1)
print("HSC kg: ",hsckg1)
LF.loadSS(ells,clss,des_ngg)
LF.loadGG(ells,clgg,des_ngg)
desks1 = LF.sn(snbins,des_fsky,"ks")[0]
deskg1 = LF.sn(snbins,des_fsky,"kg")[0]
print("DES ks: ",desks1)
print("DES kg: ",deskg1)

sn1 = LF.sn(snbins,0.4,"kk")[0]

print("10uK' PH")
LF.loadKK(ells,clkk,lsmv2,Nlmv2)
LF.loadSS(ells,clss,hsc_ngg)
LF.loadGG(ells,clgg,hsc_ngg)
hscks2 = LF.sn(snbins,hsc_fsky,"ks")[0]
hsckg2 = LF.sn(snbins,hsc_fsky,"kg")[0]
print("HSC ks: ",hscks2)
print("HSC kg: ",hsckg2)
LF.loadSS(ells,clss,des_ngg)
LF.loadGG(ells,clgg,des_ngg)
desks2 = LF.sn(snbins,des_fsky,"ks")[0]
deskg2 = LF.sn(snbins,des_fsky,"kg")[0]
print("DES ks: ",desks2)
print("DES kg: ",deskg2)
sn2 = LF.sn(snbins,0.4,"kk")[0]
print("Clkk diff: " , (sn2-sn1)*100./sn1)


print("DES ks diff: " , (desks2-desks1)*100./desks1)
print("DES kg diff: " , (deskg2-deskg1)*100./deskg1)
print("HSC ks diff: " , (hscks2-hscks1)*100./hscks1)
print("HSC kg diff: " , (hsckg2-hsckg1)*100./hsckg1)


lsmv,Nlmv,lsmv2,Nlmv2 = noises(6.)


print("6uK' HH")
LF.loadKK(ells,clkk,lsmv,Nlmv)
LF.loadSS(ells,clss,hsc_ngg)
LF.loadGG(ells,clgg,hsc_ngg)
hscks1 = LF.sn(snbins,hsc_fsky,"ks")[0]
hsckg1 = LF.sn(snbins,hsc_fsky,"kg")[0]
print("HSC ks: ",hscks1)
print("HSC kg: ",hsckg1)
LF.loadSS(ells,clss,des_ngg)
LF.loadGG(ells,clgg,des_ngg)
desks1 = LF.sn(snbins,des_fsky,"ks")[0]
deskg1 = LF.sn(snbins,des_fsky,"kg")[0]
print("DES ks: ",desks1)
print("DES kg: ",deskg1)

sn1 = LF.sn(snbins,0.4,"kk")[0]

print("6uK' PH")
LF.loadKK(ells,clkk,lsmv2,Nlmv2)
LF.loadSS(ells,clss,hsc_ngg)
LF.loadGG(ells,clgg,hsc_ngg)
hscks2 = LF.sn(snbins,hsc_fsky,"ks")[0]
hsckg2 = LF.sn(snbins,hsc_fsky,"kg")[0]
print("HSC ks: ",hscks2)
print("HSC kg: ",hsckg2)
LF.loadSS(ells,clss,des_ngg)
LF.loadGG(ells,clgg,des_ngg)
desks2 = LF.sn(snbins,des_fsky,"ks")[0]
deskg2 = LF.sn(snbins,des_fsky,"kg")[0]
print("DES ks: ",desks2)
print("DES kg: ",deskg2)

sn2 = LF.sn(snbins,0.4,"kk")[0]
print("Clkk diff: " , (sn2-sn1)*100./sn1)

print("DES ks diff: " , (desks2-desks1)*100./desks1)
print("DES kg diff: " , (deskg2-deskg1)*100./deskg1)
print("HSC ks diff: " , (hscks2-hscks1)*100./hscks1)
print("HSC kg diff: " , (hsckg2-hsckg1)*100./hsckg1)


pl.add(lsmv,Nlmv,color="C2",label="6 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{Hi-res}} , [T]_{\\mathrm{Hi-res}} )$",lw=2)
pl.add(lsmv2,Nlmv2,color="C2",ls="--",label="6 $\\mu$K-arcmin $\\mathrm{QE}( [\\nabla T]_{\\mathrm{LGMCA}} , [T]_{\\mathrm{Hi-res}} )$",lw=2)
pl.add(ells,clkk,color="k",lw=3)
pl._ax.set_ylim(1e-8,1e-6)
pl._ax.set_xlim(2,2500)
pl.legend(loc='lower right',labsize=10)
pl.done(io.dout_dir+"pgrad.pdf")
