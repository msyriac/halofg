from __future__ import print_function
from orphics import maps,io,cosmology,stats,lensing
from enlib import enmap
import numpy as np
import os,sys
import cPickle as pickle
from szar.counts import ClusterCosmology
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

repeat = False

# massbins = ['M1','M2','M3','M4']
# x = range(len(massbins))
# y = [0.,0.,-1.,-2.]
# y2 = [0.,0.,0.,0.]
# yerr = [0.1,0.1,0.1,0.1]

# plt.errorbar(x, y,yerr,marker="o",ls="none")
# plt.errorbar(x, y2,yerr,marker="d",ls="none")
# plt.xticks(x, massbins, rotation='horizontal')
# plt.axhline(y=0.,ls="--",color="k",alpha=0.5)
# plt.savefig(io.dout_dir+"figtest.png")
# sys.exit()


out_dir = "/gpfs01/astro/workarea/msyriac/data/depot/halofg/paper/"

massbins = ['groups','low','medium','high']
#massbins = ['low']
medges = [1.e13,5e13,1e14,3e14,1.5e15]

mcents = []
xysigmas = []
ysigmas = []
xymeans = []
ymeans = []

maptype = "148_tsz"
mtype=maptype
gcut = 2000

arc = 100.
pix = 0.2
kellmax = 8096

shape,wcs = maps.rect_geometry(width_deg=arc/60.,px_res_arcmin=pix)
modrmapA = enmap.modrmap(shape,wcs)*180.*60./np.pi

rbin_edges = np.arange(0.,10.,0.4)
cents = (rbin_edges[1:]+rbin_edges[:-1])/2.
rbinner = stats.bin2D(modrmapA,rbin_edges)

cc = ClusterCosmology(lmax=6000,pickling=True,dimensionless=False)

for mindex,massbin in enumerate(massbins):

    # NO FG
    xfg = False
    yfg = False



    
    prefix = massbin+"_"+maptype+"_x_"+str(xfg)+"_y_"+str(yfg)+"_g_"+str(gcut)
    r = pickle.load(open(out_dir+"recon_stats_"+prefix+".pkl",'r'))

    m200s = r['clusters'][:,0]
    r200s = r['clusters'][:,1]
    zs = r['clusters'][:,2]

    histz, zbins = np.histogram(zs,bins=10)
    zcents = (zbins[1:]+zbins[:-1])/2.
    normz = np.trapz(histz,zcents)
    histz = histz / normz
    zrange = np.linspace(zcents.min(),zcents.max(),100)
    histz = interp1d(zcents,histz)(zrange)
    # pl = io.Plotter()
    # pl.add(zrange,histz)
    # pl.done(io.dout_dir+"histz.png")
    # sys.exit()
    
    avg_mass = np.mean(m200s)
    avg_z = zs.mean()
    print(m200s.min(),m200s.max(),avg_mass,avg_z,len(zs))
    # continue

    # avgk = 0.
    # for i,z in enumerate(zrange):
    #     avgk += lensing.nfw_kappa(avg_mass,modrmapA*np.pi/180./60.,cc,zL=z,concentration=3.2,overdensity=200.,critical=True,atClusterZ=True)*histz[i]*np.diff(zrange)[0]
    avgk = lensing.nfw_kappa(avg_mass,modrmapA*np.pi/180./60.,cc,zL=avg_z,concentration=3.2,overdensity=200.,critical=True,atClusterZ=True)


    def filter_and_bin(imap):
        lowpass_filter = maps.mask_kspace(shape,wcs, lmax = kellmax)
        inputk = maps.filter_map(imap,lowpass_filter)
        cents, i1d = rbinner.bin(inputk)
        return cents, i1d
    
    cents, i1d = filter_and_bin(r['input_stack'])
    cents, a1d = filter_and_bin(avgk)
    
    pl = io.Plotter()
    pl.add(cents,i1d,label='input')
    pl.add(cents,a1d,label='average')
    pl.add(cents,r['mean'],label='recon')
    pl.legend(loc='upper right')
    pl._ax.set_ylim(-0.01,0.85)
    pl.hline()
    pl.done(io.dout_dir+"postproc"+prefix+".png")
    
    # r['profiles'] 

    # r['N'] 
    # r['recon_stack'] 
    # r['input_stack']


    profile_correction = i1d - r['mean']

    def theory(mass):
        k = lensing.nfw_kappa(mass,modrmapA*np.pi/180./60.,cc,zL=avg_z,concentration=3.2,overdensity=200.,critical=True,atClusterZ=True)
        return filter_and_bin(k)[1]

    if massbin=="medium":
        mass_range = np.linspace(1e14,3e14,100)
    elif massbin=="high":
        mass_range = np.linspace(1e14,6e14,100)
    elif massbin=="low":
        mass_range = np.linspace(5e13,1e14,100)
    elif massbin=="groups":
        mass_range = np.linspace(1e13,5e13,100)

    def fit(xfg,yfg,maptype="148_tsz"):
        prefix = massbin+"_"+maptype+"_x_"+str(xfg)+"_y_"+str(yfg)+"_g_"+str(gcut)
        r = pickle.load(open(out_dir+"recon_stats_"+prefix+".pkl",'r'))

        fit_profile = r['mean'] + profile_correction
        cov = r['covmean']
        covinv = np.linalg.inv(cov)

        lnlikes = []
        for m in mass_range:

            diff = fit_profile-theory(m)
            chisquare = np.dot(np.dot(diff,covinv),diff)
            lnlikes.append(-0.5*chisquare)

        lnlikes = np.array(lnlikes)
        p = np.polyfit(mass_range,lnlikes,2)

        pl = io.Plotter()
        pl.add(mass_range,lnlikes)
        pl.add(mass_range,p[0]*mass_range**2.+p[1]*mass_range+p[2],ls="--")
        pl.done(io.dout_dir+"lnlikes"+prefix+".png")

        c,b,a = p
        mean = -b/2./c
        sigma = np.sqrt(-1./2./c)
        mean = mass_range[lnlikes==lnlikes.max()]
        print(mean,sigma)
        sn = (avg_mass/sigma)
        print ("S/N fit for 1000 : ",sn*np.sqrt(1000./r['N']))
        bias = (avg_mass - mean)/sigma
        print("Bias : ", bias ," sigma")
        biasp = (avg_mass - mean)*100./avg_mass
        print("Bias : ", biasp ," %")


        return mean[0],sigma,sn,bias,biasp


    # Fit False/False

    try:
        if repeat: raise
        refmean,refsigma,refsn,refbias,refbiasp = np.loadtxt("ff"+massbin+mtype+".txt",unpack=True)
    except:
        refmean,refsigma,refsn,refbias,refbiasp = fit(False,False,maptype=mtype)
        io.save_cols("ff"+massbin+mtype+".txt",(refmean,refsigma,refsn,refbias,refbiasp))
    
    # True / True

    try:
        if repeat: raise
        meanXY,sigmaXY,snXY,biasXY,biaspXY = np.loadtxt("tt"+massbin+mtype+".txt",unpack=True)
    except:
        meanXY,sigmaXY,snXY,biasXY,biaspXY = fit(True,True,maptype=mtype)
        io.save_cols("tt"+massbin+mtype+".txt",(meanXY,sigmaXY,snXY,biasXY,biaspXY))


    # False / True

    try:
        if repeat: raise
        meanY,sigmaY,snY,biasY,biaspY = np.loadtxt("ft"+massbin+mtype+".txt",unpack=True)
    except:
        meanY,sigmaY,snY,biasY,biaspY = fit(False,True,maptype=mtype)
        io.save_cols("ft"+massbin+mtype+".txt",(meanY,sigmaY,snY,biasY,biaspY))

    xymeans.append((meanXY-refmean)/refmean)
    ymeans.append((meanY-refmean)/refmean)
    xysigmas.append(sigmaXY/refmean)
    ysigmas.append(sigmaY/refmean)

    mcents.append((medges[mindex]+medges[mindex+1])/2.)
    
# sys.exit()
pl = io.Plotter(ylabel="$\\Delta M / M$",xlabel="$M^{200}_{\\mathrm{mean}} (M_{\\odot})$")
#massbins = ['groups','low','med','high']
#massbins = ['C1','C2','C3','C4']
massbins = ['$3\\times 10^{13}$','$7\\times 10^{13}$','$2\\times 10^{14}$','$4\\times 10^{14}$']
x = range(len(massbins))
pl.add_err(x,xymeans,yerr=xysigmas,ls="none",marker="o",mew=3,elinewidth=3,markersize=8,label="Contaminated gradient",color="crimson")
pl.add_err(np.array(x)+0.05,ymeans,yerr=ysigmas,ls="none",marker="d",mew=3,markersize=8,elinewidth=3,label="Clean gradient",color="darkorchid")
plt.xticks(x, massbins, rotation='horizontal')
pl.hline()
plt.tick_params(axis='x', which='major', labelsize=18,width=1,size=5)#,size=labsize)
plt.tick_params(axis='x', which='minor', labelsize=18,size=3)#,size=labsize)


pl.legend(loc='lower left',labsize=16)
pl.done(io.dout_dir+"Fig1"+mtype+".pdf")
    
