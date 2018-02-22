from __future__ import print_function
from orphics import maps,io,cosmology,catalogs
from enlib import enmap,resample
import numpy as np
import os,sys
import healpy as hp

tap_width = 0.25
pad_width = 0.15
in_pix = 0.5
out_pix = 2.0

# mmin = 1.e13
# zmin = 0.2
# zmax = 0.8


mmin = 1.e13
zmin = 0.15
zmax = 0.3


#map_type = "kappa"
#map_type = "tsz"
map_type = "halo"

boxes = maps.split_sky(dec_width=10.,num_decs=6,ra_width=10.)

if map_type=="kappa":
    hp_map = hp.read_map("/gpfs01/astro/workarea/msyriac/data/sims/sehgal/healpix_4096_KappaeffLSStoCMBfullsky.fits")
    pix = in_pix
    pxstr = str(in_pix).replace(".","p")
    fstr = "kappa"
elif map_type=="tsz":
    hp_map = hp.read_map("/gpfs01/astro/workarea/msyriac/data/sims/sehgal/148_tsz_healpix.fits")
    pix = in_pix
    fstr = "tsz_148"
elif map_type=="halo":
    pix = out_pix
    #fstr = "halo_delta"
    fstr = "halo_delta_vrestricted"
    import halofg.sehgalInterface as si

    PathConfig = io.load_path_config()
    SimConfig = io.config_from_file("input/sehgal.ini")
    df,ras,decs,m200,z = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=mmin,M200_max=np.inf,z_min=zmin,z_max=zmax,Nmax=None,random_sampling=True,mirror=False)

pxstr = str(pix).replace(".","p")
    

for i,box in enumerate(boxes):
    print(box)
    shape,wcs = enmap.geometry(pos=box*np.pi/180.,res=pix*np.pi/180./60.)

    if map_type=="halo":
        cmapper = catalogs.CatMapper(ras,decs,shape=shape,wcs=wcs)
        taper,w2 = maps.get_taper_deg(shape,wcs,taper_width_degrees = tap_width,pad_width_degrees = pad_width,weight=None)
        dmap = cmapper.get_delta()*taper
        fname = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cutouts/taper_car_unrotated_"+str(i)+".hdf"
        enmap.write_map(fname,taper)
    else:
        omap = maps.enmap_from_healpix(shape,wcs,hp_map,hp_coords="j2000",interpolate=True)
        taper,w2 = maps.get_taper_deg(shape,wcs,taper_width_degrees = tap_width,pad_width_degrees = pad_width,weight=None)
        oshape,owcs = enmap.geometry(pos=box*np.pi/180.,res=out_pix*np.pi/180./60.)
        dmap = enmap.enmap(resample.resample_fft(omap*taper,oshape),owcs)
        
        
    io.plot_img(dmap,io.dout_dir+fstr+"_"+str(i)+".png",high_res=True)
    fname = "/gpfs01/astro/workarea/msyriac/data/sims/sehgal/cutouts/"+fstr+"_car_unrotated_tapered_"+str(i)+".hdf"
    enmap.write_map(fname,dmap)
