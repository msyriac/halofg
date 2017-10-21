from halofg.utils import HaloFgPipeline
import argparse
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except:
    comm = None


# Parse command line
parser = argparse.ArgumentParser(description='Run halo foregrounds pipeline.')
parser.add_argument("InpDir", type=str,help='Input Directory Name (not path, that\'s specified in ini')
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
args = parser.parse_args()


pipe = HaloFgPipeline(args.InpDir,args.nmax,args.analysis,args.sims,
                      args.recon,args.cutout,args.catalog_bin,args.experimentX,args.experimentY,
                      args.components,mpi_comm = comm,gradcut = args.gradcut)

observe = lambda imap,XY,seed: pipe.beam(XY,imap)+pipe.get_noise(XY,seed=seed)

for k,cluster_id in enumerate(pipe.clusters):
    u = pipe.get_unlensed(seed=cluster_id)
    k = pipe.upsample(pipe.get_kappa(cluster_id,stack=True))
    l = pipe.downsample(pipe.get_lensed(u,k))

    fg = pipe.get_fg_single_band(cluster_id,stack=True)
    cmb = l+fg
    observedY = observe(cmb,"Y",cluster_id+int(1e9))
    if (args.experimentX==args.experimentY) and not(args.xclean):
            observedX = observedY.copy()
    else:
        xcmb = l if args.xclean else cmb
        mul = 1 if args.experimentX==args.experimentY else 2
        observedX = observe(xcmb,"X",cluster_id+mul*int(1e9))

    kappa = pipe.qest(observedX,observedY)

    pipe.mpibox.add_to_stack("reconstack",kappa)
    pipe.mpibox.add_to_stats("recon1d",pipe.profile(kappa))
    
    print kappa.shape


