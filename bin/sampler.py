import numpy as np
from enlib import enmap
from orphics import cosmology as cosmo, io, stats, maps as omaps


shape, wcs = omaps.rect_geometry(width_arcmin=20.,px_res_arcmin=0.5)

ellrange = np.arange(0,2000,1)
cltt = ellrange*1.e-6
ps = cltt.reshape((1,1,cltt.size))
mg = omaps.MapGen(shape,wcs,ps)

N = 10000
stamps = []
for i in range(N):
    imap = mg.get_map()
    stamps.append(imap)
stack = np.stack(stamps)

print(stack.shape)

bins = np.linspace(-200,200,81)
func = lambda x: np.histogram(x,bins)[0]

y = np.apply_along_axis(func,0,stack)
print(y.shape)
        
#io.plot_img(imap)
