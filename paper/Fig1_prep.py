from __future__ import print_function
from orphics import maps,io,cosmology,mpi,stats,lensing
from enlib import enmap, lensing as enlensing
import numpy as np
import os,sys, pandas as pd
import halofg.sehgalInterface as si
import cPickle as pickle
from szar.counts import ClusterCosmology

import argparse
parser = argparse.ArgumentParser(description='Save maps from Sehgal et. al. sims')
parser.add_argument("MapType", type=str,help='tsz/ksz/radio/cib/none')
parser.add_argument("MassBin", type=str,help='groups/low/medium/high')
parser.add_argument("N", type=int,help='num clusters')
parser.add_argument("-x", "--xfg", action='store_true',help='Add foregrounds to X.')
parser.add_argument("-y", "--yfg", action='store_true',help='Add foregrounds to Y.')
parser.add_argument("-g", "--gradcut",     type=int,  default=2000,help="Gradcut.")

args = parser.parse_args()

np.random.seed(1)
arc = 100.
pix = 0.2
# beamY = 1.5
# noiseY = 1.5
# beamX = 5.0
# noiseX = 40.0
#noiseless = False

noiseless = False
noiseless_lab = True
beamY = 0.01
noiseY = 0.01
beamX = 0.01
noiseX = 0.01


lens_order = 5
trim_arc = 20.

PathConfig = io.load_path_config()

SimConfig = io.config_from_file("input/sehgal.ini")
mmin = 1.e13
zmin = 0.
df,ra,dec,m200,z = si.select_from_halo_catalog(PathConfig,SimConfig,catalog_section='catalog_default',M200_min=mmin,M200_max=np.inf,z_min=zmin,z_max=np.inf,Nmax=None,random_sampling=True,mirror=False)


mmbins = [1.e13,5e13,1e14,3e14,df['M200'].max()]
mlabels = ['groups','low','medium','high']


zbins = [0.3,df['Z'].max()]
zlabels = ['z0']

mb =  pd.cut(df['M200'],mmbins,labels=mlabels)
zb =  pd.cut(df['Z'],zbins,labels=zlabels)

for mlab in mlabels:
    print( "======")
    print (mlab)
    print( "======")
    for zlab in zlabels:
        sel = df[np.logical_and(mb==mlab , zb==zlab)]
        print (zlab,":",sel.shape[0])

        
sel = df[np.logical_and(mb==args.MassBin , zb==zlab)]

#kappa = si.get_kappa(PathConfig,SimConfig,section="kappa",base_nside=None)
comp = si.get_component_map_from_config(PathConfig,SimConfig,args.MapType,base_nside=None)


ras = sel['RA'].tolist()[:args.N]
decs = sel['DEC'].tolist()[:args.N]
m200s = sel['M200'].tolist()[:args.N]

# print(min(m200s),max(m200s))
# sys.exit()

r200s = sel['R200'].tolist()[:args.N]
zs = sel['Z'].tolist()[:args.N]


comm = mpi.MPI.COMM_WORLD
rank = comm.Get_rank()
numcores = comm.Get_size()
Nsims = len(ras)
Njobs = Nsims 
num_each,each_tasks = mpi.mpi_distribute(Njobs,numcores)
if rank==0: print ("At most ", max(num_each) , " tasks...")
my_tasks = each_tasks[rank]
st = stats.Stats(comm=comm)

shape,wcs = maps.rect_geometry(width_deg=arc/60.,px_res_arcmin=pix)
Npix = shape[0]
assert Npix == shape[1]
modlmap = enmap.modlmap(shape,wcs)
modrmapA = enmap.modrmap(shape,wcs)*180.*60./np.pi
cc = ClusterCosmology(lmax=6000,pickling=True,dimensionless=False)
theory = cc.theory
lmax = modlmap.max()
ells = np.arange(0,lmax,1)
ucltt = theory.uCl('TT',ells)
ps = ucltt.reshape((1,1,ucltt.size))
nY = (noiseY*np.pi/180./60.)**2.
nX = (noiseX*np.pi/180./60.)**2.
ps_noiseY = (ells*0.+nY).reshape((1,1,ells.size))
ps_noiseX = (ells*0.+nX).reshape((1,1,ells.size))
mgen = maps.MapGen(shape,wcs,ps)
ngenY = maps.MapGen(shape,wcs,ps_noiseY)
ngenX = maps.MapGen(shape,wcs,ps_noiseX)
kbeamY = maps.gauss_beam(beamY,modlmap)
kbeamX = maps.gauss_beam(beamX,modlmap)

if noiseless:
    nX=0.
    nY=0.
    kbeamX = kbeamX*0.+1.
    kbeamY = kbeamY*0.+1.

# QE
tellmin = modlmap[modlmap>2].min(); tellmax = 8000; kellmin = tellmin ; kellmax = 8096
tmask = maps.mask_kspace(shape,wcs,lmin=tellmin,lmax=tellmax)
kmask = maps.mask_kspace(shape,wcs,lmin=kellmin,lmax=kellmax)
qest =  lensing.Estimator(shape,wcs,
                  theory,
                  theorySpectraForNorm=theory,
                  noiseX2dTEB=[modlmap*0.+nX,modlmap*0.+nX,modlmap*0.+nX],
                  noiseY2dTEB=[modlmap*0.+nY,modlmap*0.+nY,modlmap*0.+nY],
                  noiseX_is_total = False,
                  noiseY_is_total = False,
                  fmaskX2dTEB=[tmask,tmask,tmask],
                  fmaskY2dTEB=[tmask,tmask,tmask],
                  fmaskKappa=kmask,
                  kBeamX = kbeamX,
                  kBeamY = kbeamY,
                  doCurl=False,
                  TOnly=True,
                  halo=True,
                  gradCut=args.gradcut,
                  verbose=False,
                  loadPickledNormAndFilters=None,
                  savePickledNormAndFilters=None,
                  uEqualsL=False,
                  bigell=9000,
                  mpi_comm=None,
                  lEqualsU=False)


rbin_edges = np.arange(0.,10.,pix*2.)
rbinner = stats.bin2D(modrmapA,rbin_edges)
posmap = enmap.posmap(shape,wcs)


for task in my_tasks:

    ra = ras[task]
    dec = decs[task]
    m200 = m200s[task]
    r200 = r200s[task]
    z = zs[task]

    st.add_to_stats("clusters",np.array([m200,r200,z]))
    
    cut_kappa = lensing.nfw_kappa(m200,modrmapA*np.pi/180./60.,cc,zL=z,concentration=3.2,overdensity=200.,critical=True,atClusterZ=True)
    # cut_kappa = maps.cutout_gnomonic(kappa, rot=(ra, dec), coord='C', xsize=Npix, ysize=Npix,reso=pix)
    # cut_kappa[modrmapA>trim_arc] = 0.
    st.add_to_stack("kappa",cut_kappa)

    cut_fg = maps.cutout_gnomonic(comp, rot=(ra, dec), coord='C', xsize=Npix, ysize=Npix,reso=pix)
    cut_fg[modrmapA>trim_arc] = 0.
    st.add_to_stack("fg",cut_fg)

    
    phi,_ = lensing.kappa_to_phi(enmap.enmap(cut_kappa,wcs),modlmap,return_fphi=True)
    grad_phi = enmap.grad(phi)
    pos = posmap + grad_phi
    alpha_pix = enmap.sky2pix(shape,wcs,pos, safe=False)

    
    ucmb = mgen.get_map()
    lensedY = enlensing.displace_map(ucmb, alpha_pix, order=lens_order)
    lensedX = lensedY.copy()

    if args.xfg:
        lensedX += cut_fg
    if args.yfg:
        lensedY += cut_fg
        
    obsX = maps.filter_map(lensedX,kbeamX)
    if not(noiseless): obsX += ngenX.get_map()
    obsY = maps.filter_map(lensedY,kbeamY)
    if not(noiseless): obsY += ngenY.get_map()

    recon = qest.kappa_from_map("TT",obsX,T2DDataY=obsY)
    st.add_to_stack("recon",recon)
    cents,r1d = rbinner.bin(recon)
    st.add_to_stats("r1d",r1d)

    if rank==0: print("Rank 0 doing task ", task , " / ", len(my_tasks))
    
st.get_stacks()
st.get_stats()


if rank==0:
    prefix = args.MassBin+"_"+args.MapType+"_x_"+str(args.xfg)+"_y_"+str(args.yfg)+"_g_"+str(args.gradcut)
    io.plot_img(st.stacks['kappa'],io.dout_dir+"kappastack_"+prefix+".png")
    io.plot_img(st.stacks['recon'],io.dout_dir+"reconstack_"+prefix+".png")

    r = st.stats['r1d']
    pl = io.Plotter()
    pl.add_err(cents,r['mean'],yerr=r['errmean'])
    pl.done(io.dout_dir+"r1d.png")

    r['clusters'] = st.vectors['clusters']
    r['profiles'] = st.vectors['r1d']
    r['N'] = args.N
    r['recon_stack'] = st.stacks['recon']
    r['input_stack'] = st.stacks['kappa']
    out_dir = "/gpfs01/astro/workarea/msyriac/data/depot/halofg/paper/"
    suff = "_noiseless" if noiseless_lab else ""
    pickle.dump(r,open(out_dir+"recon_stats"+suff+"_"+prefix+".pkl",'w'))

    
