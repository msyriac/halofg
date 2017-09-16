import halofg.sehgalInterface as si
import numpy as np
import os,sys
import orphics.tools.io as io

SimConfig = io.config_from_file("input/sehgal.ini")


z_edges = [0.,0.5,1.0,2.0,np.inf]
#m_edges = [8.e12,5.e13,1.e14,np.inf]
m_edges = [5.e13,1.e14,5.e14,np.inf]
ras,decs,m200s,zs = si.select_from_halo_catalog(SimConfig)
print "All: ", len(ras)

ras,decs,m200s,zs = si.select_from_halo_catalog(SimConfig,z_min=z_edges[0],z_max=z_edges[-1],M200_min=m_edges[0],M200_max=m_edges[-1])
print "Mass bounds: ", len(ras)

for zmin,zmax in zip(z_edges[:-1],z_edges[1:]):
    ras,decs,m200s,zs = si.select_from_halo_catalog(SimConfig,z_min=zmin,z_max=zmax,M200_min=m_edges[0],M200_max=m_edges[-1])
    print zmin,"< z <",zmax," : ", len(ras)


for mmin,mmax in zip(m_edges[:-1],m_edges[1:]):
    ras,decs,m200s,zs = si.select_from_halo_catalog(SimConfig,z_min=z_edges[0],z_max=z_edges[-1],M200_min=mmin,M200_max=mmax)
    print mmin,"< M200 <",mmax," : ", len(ras)



for mmin,mmax in zip(m_edges[:-1],m_edges[1:]):
    print mmin,"< M200 <",mmax
    for zmin,zmax in zip(z_edges[:-1],z_edges[1:]):
        ras,decs,m200s,zs = si.select_from_halo_catalog(SimConfig,z_min=zmin,z_max=zmax,M200_min=mmin,M200_max=mmax)
        print "\t",zmin,"< z <",zmax," : ", len(ras)
    
