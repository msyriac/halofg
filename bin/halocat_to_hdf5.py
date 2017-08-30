import numpy as np
import pandas as pd
import os,sys


map_root = os.environ['WORK2']+'/data/sehgal/'
halo_file = map_root + 'halo_nbody.ascii'


print "Loading halo catalog..."
halos = np.loadtxt(halo_file)


columns = ['Z','RA','DEC','POSX','POSY','POSZ','VX','VY','VZ','MFOF','MVIR','RVIR','M200','R200','M500','R500','M1500','R1500','M2500','R2500']
"""
"redshift"
"ra  in degrees"
"dec in degrees"
"comoving position of halo potential minimum in Mpc"
"proper peculiar velocity in km/s"
"Mfof in Msolar"
"Mvir in Msolar"
"Rvir in proper Mpc"
"M200,  R200"
"M500,  R500"
"M1500, R1500"
"M2500, R2500"
"""

print "Converting to dataframe..."
print len(columns)
print halos.shape

df = pd.DataFrame(halos,index=range(halos.shape[0]),columns=columns)
print "Saving to hd5..."
df.to_hdf(map_root+"halo_nbody.hd5","cat")

