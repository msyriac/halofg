import healpy as hp
import numpy as np

def pix_from_config(SimConfig,cutout_section="cutout_default"):
    arc = SimConfig.getfloat(cutout_section,"arc")
    pix = SimConfig.getfloat(cutout_section,"px")
    Npix = int(arc/pix+0.5)
    return Npix,arc,pix

def get_components_map_from_config(SimConfig,freq,components,base_nside=None):
    """
    Returns sum of healpix maps from Sehgal et. al. / Hill sims
    in the same units with the right calibration
    specified in input/sehgal.ini.
    """
    if not(isinstance(components,list) or isinstance(components,tuple)): components = [components]
    Config = SimConfig
    map_root = Config.get("sims","map_root")

    totmap = 0.
    for component in components:
        sec_name = str(int(freq))+"_"+component
        map_name = map_root+Config.get(sec_name,"file")
        hpmap = hp.read_map(map_name)
        if base_nside is not None:
           nside = hp.get_nside(hpmap)
           if nside != base_nside:
               hpmap = hp.ud_grade(hpmap,nside_out = base_nside)
        calib = Config.getfloat(sec_name,"calibration")
        units = Config.getfloat(sec_name,"unit_conversion")
        totmap = totmap + hpmap*units*calib

    del hpmap
    return totmap
    

def get_kappa(SimConfig,section="kappa",base_nside=None):
    """
    """
    Config = SimConfig
    map_root = Config.get("sims","map_root")

    map_name = map_root+Config.get(section,"file")
    hpmap = hp.read_map(map_name)
    if base_nside is not None:
        nside = hp.get_nside(hpmap)
        if nside != base_nside:
            hpmap = hp.ud_grade(hpmap,nside_out = base_nside)
    return hpmap
    



def select_from_halo_catalog(SimConfig,catalog_section='catalog_default',M200_min=-np.inf,M200_max=np.inf,z_min=-np.inf,z_max=np.inf,Nmax=None,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None):
    import pandas as pd
    Config = SimConfig
    map_root = Config.get("sims","map_root")

    halo_file = map_root + Config.get(catalog_section,'file')
    df = pd.read_hdf(halo_file)
    sel = (df['Z']>z_min) & (df['Z']<z_max) & (df['M200']>M200_min) & (df['M200']<M200_max)
    assert random_sampling, "NotImplementedError: non-random sampling not implemented."
    df = df[sel].sample(Nmax) if ((Nmax is not None) and Nmax<df['Z'].size) else df[sel]

    
    halos_select_z = df['Z']
    halos_select_M200 = df['M200']
    halos_select_RA = df['RA']
    halos_select_DEC = df['DEC']
    halos_select_vz = df['VZ']


    if histogram_z_save_path is not None:
        # plot z and M200 histograms
        plt.clf()
        plt.hist(halos_select_z, bins=100, log=False, facecolor='blue')
        plt.xlabel(r'$z$', fontsize=18)
        plt.ylabel(r"$N_{halos}$",fontsize=18)
        plt.savefig(histogram_z_save_path)
    if histogram_M_save_path is not None:
        plt.clf()
        bins = np.logspace(np.log10(halos_select_M200.min()), np.log10(halos_select_M200.max()), 50)
        plt.hist(halos_select_M200, bins=bins, log=True, facecolor='blue')
        #plt.xlim( left=M200_min, right=1.0e15 )
        plt.xlabel(r'$M_{200} \, [{\rm M_{\odot}}]$', fontsize=18)
        plt.ylabel(r"$N_{halos}$",fontsize=18)
        plt.gca().set_xscale("log")
        plt.savefig(histogram_M_save_path)


    return halos_select_RA,halos_select_DEC,halos_select_M200,halos_select_z,halos_select_vz
