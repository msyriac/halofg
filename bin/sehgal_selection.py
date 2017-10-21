import pandas as pd
import numpy as np
import halofg.sehgalInterface as si
import os,sys
import orphics.tools.io as io


SimConfig = io.config_from_file("input/sehgal.ini")
mmin = 1.e13
zmin = 0.
df,ra,dec,m200,z = si.select_from_halo_catalog(SimConfig,catalog_section='catalog_default',M200_min=mmin,M200_max=np.inf,z_min=zmin,z_max=np.inf,Nmax=None,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None)


mmbins = [1.e13,5e13,1e14,4e14,df['M200'].max()]
mlabels = ['groups','low','medium','high']

#zbins = [0.,0.5,1.0,2.0,df['Z'].max()]
#zlabels = ['z0','z1','z2','z3']
zbins = [0.5,df['Z'].max()]
zlabels = ['z0']

#hist,_,_ = np.histogram2d(df['M200'],df['Z'],bins=[mmbins,zbins])
#io.quickPlot2d(np.log10(hist),io.dout_dir+"sehgal_hist.png",extent=[mmbins[0],mmbins[-1],zbins[0],zbins[-1]],aspect="auto")

mb =  pd.cut(df['M200'],mmbins,labels=mlabels)
zb =  pd.cut(df['Z'],zbins,labels=zlabels)

for mlab in mlabels:
    print "======"
    print mlab
    print "======"
    for zlab in zlabels:
        

        sel = df[np.logical_and(mb==mlab , zb==zlab)]
        print zlab,":",sel.shape[0]

