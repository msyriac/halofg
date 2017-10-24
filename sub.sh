#!/bin/bash

# i=1
# for k in 100 50 10 1
# do
#     smart_mpi $k "python -W ignore bin/recon.py run1_Oct19 fg_ksz_noiseless analysis_arc sims_arc reconstruction_cluster_lowell cutout_default sehgal_bin_${i} experiment_noiseless experiment_noiseless 148_ksz -v -c"
#     let i+=1
# done
    



smart_mpi 300 "python -W ignore bin/recon.py run1_Oct19 test analysis_arc sims_arc reconstruction_cluster_lowell cutout_default sehgal_bin_1 experiment_noiseless experiment_noiseless none -v -N 50000"


