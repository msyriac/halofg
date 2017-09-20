import os,sys
import numpy as np


out_dir = os.environ['WWW']+"plots/"

clean = "mass_sehgal_2000_postfilter_False_tszx_False_tszy_False_kszx_False_kszy_False_kappa1d.npy"
contam = "mass_sehgal_2000_postfilter_False_tszx_True_tszy_True_kszx_False_kszy_False_kappa1d.npy"
contam1k = "mass_sehgal_1000_postfilter_False_tszx_True_tszy_True_kszx_False_kszy_False_kappa1d.npy"
gradclean = "mass_sehgal_2000_postfilter_False_tszx_False_tszy_True_kszx_False_kszy_False_kappa1d.npy"
ksz = "mass_sehgal_2000_postfilter_False_tszx_False_tszy_False_kszx_True_kszy_True_kappa1d.npy"
ksz1k = "mass_sehgal_1000_postfilter_False_tszx_False_tszy_False_kszx_True_kszy_True_kappa1d.npy"


def load_file(filename):
    savemat = np.load(out_dir+filename)
    cents = savemat[:,0]
    mean = savemat[:,1]
    errs = savemat[:,2]
    
    return cents,mean,errs

def perdiff(x,y,xerr,yerr):

    rel = x/y
    err = xerr/y #rel*np.sqrt(x**2./xerr**2.+y**2./yerr**2.)
    
    return (x-y)/y,err

cents, clean_mean, clean_errs = load_file(clean)
cents, contam_mean, contam_errs = load_file(contam)
cents, contam1k_mean, contam1k_errs = load_file(contam1k)
cents, gradclean_mean, gradclean_errs = load_file(gradclean)
cents, ksz_mean, ksz_errs = load_file(ksz)
cents, ksz_mean1k, ksz_errs1k = load_file(ksz1k)

contamper,contamerr = perdiff(contam_mean,clean_mean,contam_errs,clean_errs)
contamper1k,contamerr1k = perdiff(contam1k_mean,clean_mean,contam1k_errs,clean_errs)
cleanper,cleanerr = perdiff(gradclean_mean,clean_mean,gradclean_errs,clean_errs)
kszper,kszerr = perdiff(ksz_mean,clean_mean,ksz_errs,clean_errs)
kszper1k,kszerr1k = perdiff(ksz_mean1k,clean_mean,ksz_errs1k,clean_errs)

import orphics.tools.io as io

pl = io.Plotter(labelX="$\\theta$ (arcmin)",labelY="$\\Delta \\kappa / \\kappa$")
pl.addErr(cents-0.1,contamper,yerr=contamerr,label="tSZ contaminated gradient",marker="o",ls="-")
#pl.addErr(cents,contamper1k,yerr=contamerr1k,label="tSZ contaminated gradient $\\ell_G<1000$",marker="o",ls="-")
#pl.addErr(cents+0.1,cleanper,yerr=cleanerr,label="Clean gradient",marker="o",ls="-")
pl.legendOn(loc="lower right",labsize=12)
pl.hline()
pl.done(out_dir+"clfgtsz.png")


pl = io.Plotter(labelX="$\\theta$ (arcmin)",labelY="$\\Delta \\kappa / \\kappa$")
pl.addErr(cents,kszper,yerr=kszerr,label="kSZ contaminated",marker="o",ls="-")
#pl.addErr(cents+0.1,kszper1k,yerr=kszerr1k,label="kSZ contaminated $\\ell_G<1000$",marker="o",ls="-")
pl._ax.set_ylim(-0.15,0.05)
pl.legendOn(loc="lower left",labsize=12)
pl.hline()
pl.done(out_dir+"clfgksz.png")


