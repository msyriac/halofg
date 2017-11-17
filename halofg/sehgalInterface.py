import healpy as hp
import numpy as np

def pix_from_config(SimConfig,cutout_section="cutout_default"):
    arc = SimConfig.getfloat(cutout_section,"arc")
    pix = SimConfig.getfloat(cutout_section,"px")
    Npix = int(arc/pix+0.5)
    return Npix,arc,pix

def get_component_map_from_config(PathConfig,SimConfig,section,base_nside=None):
    Config = SimConfig
    sec_name = section
    map_root = PathConfig.get("paths","input_data")
    map_name = map_root+Config.get(sec_name,"file")
    hpmap = hp.read_map(map_name)
    if base_nside is not None:
       nside = hp.get_nside(hpmap)
       if nside != base_nside:
           hpmap = hp.ud_grade(hpmap,nside_out = base_nside)
    calib = Config.getfloat(sec_name,"calibration")
    units = Config.getfloat(sec_name,"unit_conversion")
    return hpmap*calib*units

def get_components_map_from_config(PathConfig,SimConfig,freq,components,base_nside=None):
    """
    Returns sum of healpix maps from Sehgal et. al. / Hill sims
    in the same units with the right calibration
    specified in input/sehgal.ini.
    """
    if not(isinstance(components,list) or isinstance(components,tuple)): components = [components]

    totmap = 0.
    for component in components:
        sec_name = str(int(freq))+"_"+component
        retmap = get_component_map_from_config(PathConfig,SimConfig,sec_name,base_nside=base_nside)
        totmap = totmap + retmap

    return totmap
    

def get_kappa(PathConfig,SimConfig,section="kappa",base_nside=None):
    """
    """
    Config = SimConfig
    map_root = PathConfig.get("paths","input_data")

    map_name = map_root+Config.get(section,"file")
    hpmap = hp.read_map(map_name)
    if base_nside is not None:
        nside = hp.get_nside(hpmap)
        if nside != base_nside:
            hpmap = hp.ud_grade(hpmap,nside_out = base_nside)
    return hpmap
    



def select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=-np.inf,M200_max=np.inf,z_min=-np.inf,z_max=np.inf,Nmax=None,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None,ra_min=None,ra_max=None,dec_min=None,dec_max=None,mirror=True):
    import pandas as pd
    Config = SimConfig
    map_root = PathConfig.get("paths","input_data")

    halo_file = map_root + Config.get(catalog_section,'file')
    df = pd.read_hdf(halo_file)

    ra = df['RA']*np.pi/180.
    dec = df['DEC']*np.pi/180.

    if mirror:
        new_dfs = []

        for n in range(0,4):
            new_df = df.copy()
            new_df['RA'] = (ra+n*np.pi/2.)*180./np.pi
            new_dfs.append(new_df.copy())

        for n in range(4):
            new_df = df.copy()
            new_df['RA'] = (np.pi/2.-ra+n*np.pi/2.)*180./np.pi
            new_df['DEC'] = -dec*180./np.pi
            new_dfs.append(new_df.copy())

        final_df = pd.concat(new_dfs)
        assert final_df['RA'].size==8*df['RA'].size
    else:
        final_df = df


    if ra_min is not None: assert ra_min<ra_max
    if ra_max is not None: assert dec_min<dec_max

    if ra_min is None:
        cond = np.abs(final_df['RA'])>=0.
    else:
        ra_left = 360.+ra_min if ra_min<0. else ra_min
        ra_right = 360.+ra_max if ra_max<0. else ra_max

        if ra_min*ra_max<0.:
            cond = (final_df['RA']>ra_left) | (final_df['RA']<ra_right)
        else:
            cond = (final_df['RA']>ra_left) & (final_df['RA']<ra_right)

    if dec_min is None:
        cond2 = np.abs(final_df['DEC'])>=0.
    else:
        cond2 = (final_df['DEC']>dec_min) & (final_df['DEC']<dec_max)
        
    sel = cond2 & cond
    final_df =final_df[sel]

    
    return select_from_halo_catalog_dataframe(final_df,M200_min,M200_max,z_min,z_max,Nmax,random_sampling,histogram_z_save_path,histogram_M_save_path)
    

def select_from_halo_catalog_dataframe(df_in,M200_min=-np.inf,M200_max=np.inf,z_min=-np.inf,z_max=np.inf,Nmax=None,random_sampling=True,histogram_z_save_path=None,histogram_M_save_path=None):

    df = df_in.copy()
    
    sel = (df['Z']>z_min) & (df['Z']<z_max) & (df['M200']>M200_min) & (df['M200']<M200_max)
    assert random_sampling, "NotImplementedError: non-random sampling not implemented."

    try:
        df = df[sel].sample(Nmax,replace=False) if ((Nmax is not None) and Nmax<df['Z'].size) else df[sel]
    except:
        print Nmax
        print df['Z'].size
        print df[sel]['Z'].size
        print M200_min,M200_max
        print z_min,z_max
        print "ERROR: NOT ENOUGH TO SAMPLE FROM"
        sys.exit()

    
    halos_select_z = df['Z']
    halos_select_M200 = df['M200']
    halos_select_RA = df['RA']
    halos_select_DEC = df['DEC']






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


    return df,halos_select_RA,halos_select_DEC,halos_select_M200,halos_select_z
