from __future__ import print_function
from orphics import cosmology,maps,io
from szar import counts,foregrounds
import numpy as np

lmax = 8000
cc = counts.ClusterCosmology(lmax=lmax,pickling=True,dimensionless=False)


freqs = [90.,150.,220.]
fwhms = [1.4*150./90.,1.4,5.0]
ells = np.arange(0,lmax,1)
cmb_ps = cc.theory.lCl('TT',ells)
kbeams = [maps.gauss_beam(fwhm,ells) for fwhm in fwhms]
components = ['tsz','cibc','cibp']
fnoise =  foregrounds.fgNoises(cc.c,ksz_file='../szar/input/ksz_BBPS.txt',ksz_p_file='../szar/input/ksz_p_BBPS.txt',tsz_cib_file='../szar/input/sz_x_cib_template.dat',tsz_battaglia_template_csv='../szar/input/sz_template_battaglia.csv',components=components,lmax=lmax)


wnoises = [22.0,10.0,66.0]
noises = [ells*0.+(wnoise*np.pi/180./60.)**2. for wnoise in wnoises]

cinv = maps.ilc_cinv(ells,cmb_ps,kbeams,freqs,noises,components,fnoise)

fsz = foregrounds.f_nu(cc.c,freqs)
fdust = fnoise.f_nu_cib(np.asarray(freqs))

coadd_noise = maps.silc_noise(cinv,response=None)
deproj_sz = maps.cilc_noise(cinv,response_a=np.array(freqs)*0.+1.,response_b=fsz)
deproj_dust = maps.cilc_noise(cinv,response_a=np.array(freqs)*0.+1.,response_b=fdust)

pl = io.Plotter(yscale='log')
pl.add(ells[2:],cmb_ps[2:]*ells[2:]**2.,color="k",lw=2)
for noise,kbeam,freq in zip(noises,kbeams,freqs):
    fgcontrib = fnoise.get_tot_fg_noise(freq,ells)
    pl.add(ells,(noise+fgcontrib)*ells**2./kbeam**2.,label=str(freq),ls="-.",alpha=0.4)

io.plt.gca().set_color_cycle(None)
for noise,kbeam,freq in zip(noises,kbeams,freqs):
    pl.add(ells,noise*ells**2./kbeam**2.,label=str(freq),ls="--",alpha=0.2)
pl.add(ells,(coadd_noise-cmb_ps)*ells**2.,label="silc",lw=2)
pl.add(ells,(deproj_sz-cmb_ps)*ells**2.,label="cilc deproj sz",lw=2)
pl.add(ells,(deproj_dust-cmb_ps)*ells**2.,label="cilc deproj dust",lw=2)
pl._ax.set_ylim(1.,1.e5)
pl.legend()
pl.done()



# y_noise = maps.silc_noise(cinv,response=fsz)


# pl = io.Plotter(yscale='log')
# pl.add(ells[2:],cmb_ps[2:]*ells[2:]**2.,color="k",lw=2)
# for noise,kbeam,freq in zip(noises,kbeams,freqs):
#     fgcontrib = fnoise.get_tot_fg_noise(freq,ells)
#     pl.add(ells,(noise+fgcontrib)*ells**2./kbeam**2.,label=str(freq))

# io.plt.gca().set_color_cycle(None)
# for noise,kbeam,freq in zip(noises,kbeams,freqs):
#     pl.add(ells,noise*ells**2./kbeam**2.,label=str(freq),ls="--",alpha=0.5)
# pl.add(ells,(coadd_noise-cmb_ps)*ells**2.,label="silc")
# pl._ax.set_ylim(1.,1.e5)
# pl.legend()
# pl.done()

# #silc_noise(cinv,response=None)

