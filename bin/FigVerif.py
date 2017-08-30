import numpy as np
import orphics.tools.io as io
import os, glob, sys
import ntpath

dirname = "cutouts_15041257462"
out_dir = os.environ['WWW']+"plots/"
map_root = os.environ['WORK2']+'/data/sehgal/'
save_dir = map_root + dirname


fnames = glob.glob(save_dir+"/profiles*")


pl = io.Plotter()

for k,fname in enumerate(fnames):

    lab = ntpath.basename(fname)[:-4]
    print lab

    cents,inp,inperr,rec,recerr = np.loadtxt(fname,unpack=True)

    diff = (rec-inp)*100./inp
    differr = 100.*recerr/inp    
    pl.addErr(cents+(k-2)*0.1,diff,yerr=differr,marker="o",label=lab)
pl.legendOn(labsize=10)
pl._ax.axhline(y=0.,ls="--")
pl._ax.set_ylim(-10,10)
pl._ax.set_xlim(0.,10.)
pl.done(out_dir+"diffper.png")


pl = io.Plotter()

for k,fname in enumerate(fnames):

    lab = ntpath.basename(fname)[:-4]
    print lab

    cents,inp,inperr,rec,recerr = np.loadtxt(fname,unpack=True)

    diff = (rec-inp)/recerr
    pl.add(cents+(k-2)*0.1,diff,marker="o",label=lab,ls="-")
pl.legendOn(labsize=10)
pl._ax.axhline(y=0.,ls="--")
#pl._ax.set_ylim(-40,40)
pl._ax.set_xlim(0.,10.)
pl.done(out_dir+"diffbias.png")




for k,fname in enumerate(fnames):
    pl = io.Plotter()

    lab = ntpath.basename(fname)[:-4]
    print lab

    cents,inp,inperr,rec,recerr = np.loadtxt(fname,unpack=True)

    
    pl.addErr(cents,cents*inp,yerr=cents*inperr,marker="x")
    pl.addErr(cents,cents*rec,yerr=cents*recerr,marker="o")
    #pl.legendOn(labsize=10)
    pl._ax.axhline(y=0.,ls="--")
    #pl._ax.set_ylim(-40,40)
    #pl._ax.set_xlim(0.,10.)
    pl.done(out_dir+"prof_"+lab+".png")

    


