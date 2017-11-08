from halofg.utils import HaloFgPipeline
import orphics.tools.io as io
import argparse
from enlib import enmap
import numpy as np


# Parse command line
parser = argparse.ArgumentParser(description='Post process halo foregrounds pipeline.')
parser.add_argument("InpDir", type=str,help='Input Directory Name (not path, that\'s specified in ini)')
parser.add_argument("OutDirs", type=str,help='Output Directory Names')
args = parser.parse_args()

shape,wcs = enmap.rect_geometry(100.,0.5)
modrmap = enmap.modrmap(shape,wcs)
modlmap = enmap.modlmap(shape,wcs)
kellmin = 200
kellmax = 8500
import orphics.tools.stats as stats
import orphics.analysis.flatMaps as fmaps
analysis_resolution =  np.min(enmap.extent(shape,wcs)/shape[-2:])*60.*180./np.pi
bin_edges = np.arange(0.,10.,2.*analysis_resolution)
binner = stats.bin2D(modrmap*60.*180./np.pi,bin_edges)

SimConfig = io.config_from_file("input/sehgal.ini")
PathConfig = io.load_path_config()
pout_dir = PathConfig.get("paths","plots")

for out_dir in args.OutDirs.split(','):
    for k in range(1,5):
        catalog_bin = "sehgal_bin_"+str(k)
        result_dir = PathConfig.get("paths","output_data")+args.InpDir+"/"+out_dir+"/"+catalog_bin+"/"
        cents, inp,recon = np.loadtxt(result_dir+"profile.txt",unpack=True)
        cov = np.load(result_dir+"covmean.npy")
        inpstack = np.load(result_dir+"inpstack.npy")
        reconstack = np.load(result_dir+"reconstack.npy")
        res_stack = inpstack - reconstack
        np.save(result_dir+"res_stack.npy",res_stack)

        io.quickPlot2d(inpstack,pout_dir+out_dir+"_inpstack.png",lim=[-0.01,0.05])
        io.quickPlot2d(reconstack,pout_dir+out_dir+"_reconstack.png",lim=[-0.01,0.05])
        #inpstack = fmaps.filter_map(inpstack,inpstack*0.+1.,modlmap,lowPass=kellmax,highPass=kellmin)
        #cents2,inp = binner.bin(inpstack)
        #assert np.all(np.isclose(cents,cents2))

        
        errs = np.sqrt(np.diagonal(cov))
        print errs
        diff = (recon-inp)/inp

        errrat = errs/inp
        
        pl = io.Plotter()
        pl.addErr(cents+k*0.1,diff,yerr=errrat,color="C"+str(k),label=str(k),marker="o")
        pl.hline()
        pl.hline(y=-0.05,ls="-.")
        pl._ax.set_ylim(-0.1,0.1)
        pl.legendOn()
        pl.done(pout_dir+out_dir+"_profdiff"+str(k)+".png")
        
        pl2 = io.Plotter()
        pl2.addErr(cents,recon,yerr=errs,color="C"+str(k),ls="-",marker="o")
        pl2.add(cents,inp,ls="--",color="C"+str(k),marker="x")


        pl2.hline()
        pl2.legendOn()
        pl2.done(pout_dir+out_dir+"_prof"+str(k)+".png")
