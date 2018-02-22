from __future__ import print_function
from orphics import maps,io,cosmology,stats
from enlib import enmap
import numpy as np
import os,sys,glob

import matplotlib as mpl
from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=['#2424f0','#df6f0e','#3cc03c','#d62728','#b467bd','#ac866b','#e397d9','#9f9f9f','#ecdd72','#77becf'])

orig_area = 100.
new_area = 1000.
scale = np.sqrt(orig_area/new_area)

out_dir = "/gpfs01/astro/workarea/msyriac/data/depot/halofg/covclkh/"
#prefix = "clkhs_bin_160_vrestricted_"
prefix = "clkhs_bin_160_restricted_"
N = len(glob.glob(out_dir+prefix+"*"))-1

st = stats.Stats()
for i in range(N):
    print(i+1, " / ", N)
    cents,clkhest,clkhyy,clkhxy = np.loadtxt(out_dir+prefix+str(i)+".txt",unpack=True)


    hdiffYY = (clkhyy-clkhest)/clkhest
    hdiffY = (clkhxy-clkhest)/clkhest

    st.add_to_stats("hkYYdiff",hdiffYY)
    st.add_to_stats("hkYdiff",hdiffY)
    
st.get_stats()

print(N)
pl = io.Plotter(ylabel="$\\Delta C^{\\kappa g}_L / C^{\\kappa g}_L$",xlabel="$L$")

# y,error = st.stats['hkYYdiff']['mean'],st.stats['hkYYdiff']['err']*scale #(p1dYY-p1d)/p1d
# pl._ax.fill_between(cents, y-error, y+error,alpha=0.2,color="C1")

# y,error = st.stats['hkYdiff']['mean'],st.stats['hkYdiff']['err']*scale #(p1dY-p1d)/p1d
# pl._ax.fill_between(cents, y-error, y+error,alpha=0.2,color="C0")

print((st.stats['hkYYdiff']['err']/st.stats['hkYYdiff']['errmean'])**2.)

diffYY,ediffYY = st.stats['hkYYdiff']['mean'],st.stats['hkYYdiff']['errmean'] #(p1dYY-p1d)/p1d
diffY,ediffY = st.stats['hkYdiff']['mean'],st.stats['hkYdiff']['errmean'] #(p1dY-p1d)/p1d
pl.add_err(cents,diffYY,yerr=ediffYY,label="tSZ contaminated gradient",marker="o",ls="-",color="C1",lw=2,elinewidth=2,mew=2,markersize=8)
pl.add_err(cents,diffY,yerr=ediffY,label="Clean gradient",marker="o",ls="-",color="C0",lw=2,elinewidth=2,mew=2,markersize=8)


pl.hline()
pl.legend(labsize=16)
pl._ax.set_ylim(-0.21,0.3)
#pl._ax.set_ylim(-0.6,0.4)
pl._ax.set_xlim(2,3000)
pl.done(io.dout_dir+"sehgalclkh_"+prefix+"cov.pdf")
print(diffY*100.)
print(diffYY*100.)

