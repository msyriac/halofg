from halofg.utils import HaloFgPipeline
import argparse
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    comm = None


# Parse command line
parser = argparse.ArgumentParser(description='Run halo foregrounds pipeline.')
parser.add_argument("InpDir", type=str,help='Input Directory Name (not path, that\'s specified in ini)')
parser.add_argument("OutDir", type=str,help='Output Directory Name')
parser.add_argument("analysis", type=str,help='Analysis section name')
parser.add_argument("sims", type=str,help='Sim section name')
parser.add_argument("recon", type=str,help='Recon section name')
parser.add_argument("cutout", type=str,help='Cutout section name')
parser.add_argument("catalog_bin", type=str,help='Name of catalog bin section')
parser.add_argument("experimentX", type=str,help='Name of experiment section')
parser.add_argument("experimentY", type=str,help='Name of experiment section')
parser.add_argument("components", type=str,help='Name of component section')
parser.add_argument("-N", "--nmax",     type=int,  default=None,help="Limit to nmax sims.")
parser.add_argument("-G", "--gradcut",     type=int,  default=2000)
parser.add_argument("-X", "--xclean", action='store_true',help='Do not add foregrounds to X leg.')
parser.add_argument("-v", "--verbose", action='store_true',help='Talk more.')
parser.add_argument("-c", "--cache", action='store_true',help='Cache lensed CMB.')
args = parser.parse_args()

# Initialize pipeline
pipe = HaloFgPipeline(args.InpDir,args.OutDir,args.nmax,args.analysis,args.sims,
                      args.recon,args.cutout,args.catalog_bin,args.experimentX,args.experimentY,
                      args.components,mpi_comm = comm,gradcut = args.gradcut,verbose = args.verbose)

# Define an observation
observe = lambda imap,XY,seed: pipe.beam(XY,imap)+pipe.get_noise(XY,seed=seed)

# Loop through clusters
for k,cluster_id in enumerate(pipe.clusters):
    
    try: # to see if we want to load cached files and if yes whether they can be found
        assert args.cache
        lensed = pipe.load_cached(cluster_id)
        if pipe.rank==0 and k==0: print "Sucessfully loaded cached lensed CMB."
    except: # do lensing if not
        if pipe.rank==0 and k==0: print "Did not load cached lensed CMB."
        unlensed = pipe.get_unlensed(seed=cluster_id)
        input_kappa = pipe.upsample(pipe.get_kappa(cluster_id,stack=True))
        lensed = pipe.downsample(pipe.get_lensed(unlensed,input_kappa))
    if args.cache:
        pipe.save_cache(lensed,cluster_id)

    fg = pipe.get_fg_single_band(cluster_id,stack=True)
    cmb = lensed+fg
    observedY = observe(cmb,"Y",cluster_id+int(1e9))
    if (args.experimentX==args.experimentY) and not(args.xclean):
            observedX = observedY.copy()
    else:
        xcmb = lensed if args.xclean else cmb # don't add foregrounds to X leg if xclean
        mul = 1 if args.experimentX==args.experimentY else 2 # reuse noise seed if it's the same experiment
        observedX = observe(xcmb,"X",cluster_id+mul*int(1e9))

    kappa = pipe.qest(observedX,observedY)

    pipe.mpibox.add_to_stack("reconstack",kappa)
    pipe.mpibox.add_to_stats("recon1d",pipe.profile(kappa))
    if pipe.rank==0 and (k+1)%10==0: print "Rank 0 done with ",k+1, " / ", len(pipe.clusters), " tasks."
    
pipe.mpibox.get_stacks()
pipe.mpibox.get_stats()

if pipe.rank==0:
    pipe.dump()
