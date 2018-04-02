from __future__ import print_function
from orphics import io
import numpy as np
import os,sys
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

arc = 100.
f, axarr = plt.subplots(3,1)
dat = np.load("stack_143_random_False.npy")
img1 = axarr[0].imshow(dat-dat.mean(),extent=[-arc/2.,arc/2.,-arc/2.,arc/2.],vmin=-2.5,vmax=1.5)

axarr[0].set_ylabel("$\\theta$ (arcmin)")

divider = make_axes_locatable(axarr[0])
cax = divider.append_axes("right", "5%", pad="3%")
cbar = f.colorbar(img1, cax=cax)
cbar.set_label("$\\mu K$")#,rotation=0)


axarr[0].get_xaxis().set_visible(False)
dat = np.load("stack_smica_random_False.npy")
img2 = axarr[1].imshow(dat-dat.mean(),extent=[-arc/2.,arc/2.,-arc/2.,arc/2.],vmin=-2.5,vmax=1.5)
divider = make_axes_locatable(axarr[1])
cax = divider.append_axes("right", "5%", pad="3%")
cbar = f.colorbar(img2, cax=cax)
cbar.set_label("$\\mu K$")#,rotation=0)

axarr[1].get_xaxis().set_visible(False)
axarr[1].set_ylabel("$\\theta$ (arcmin)")
dat = np.load("stack_lgmca_random_False.npy")
img3 = axarr[2].imshow(dat-dat.mean(),extent=[-arc/2.,arc/2.,-arc/2.,arc/2.],vmin=-2.5,vmax=1.5)
divider = make_axes_locatable(axarr[2])
cax = divider.append_axes("right", "5%", pad="3%")
cbar = f.colorbar(img3, cax=cax)
cbar.set_label("$\\mu K$")#,rotation=0)

axarr[2].set_ylabel("$\\theta$ (arcmin)")
axarr[2].set_xlabel("$\\theta$ (arcmin)")

plt.tight_layout()
plt.savefig(io.dout_dir+"scomp.pdf",bbox_inches='tight')

sys.exit()

LL = np.load("stack_LL_random_False.npy")-np.load("stack_LL_random_True.npy")
io.plot_img(LL,io.dout_dir+"ll.png")

LL = np.load("stack_LF_random_False.npy")-np.load("stack_LF_random_True.npy")
io.plot_img(LL,io.dout_dir+"lf.png")

LL = np.load("stack_SS_random_False.npy")-np.load("stack_SS_random_True.npy")
io.plot_img(LL,io.dout_dir+"ss.png")

LL = np.load("stack_FF_random_False.npy")-np.load("stack_FF_random_True.npy")
io.plot_img(LL,io.dout_dir+"ff.png")

LL = np.load("stack_SF_random_False.npy")-np.load("stack_SF_random_True.npy")
io.plot_img(LL,io.dout_dir+"sf.png")

