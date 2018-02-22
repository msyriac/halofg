#!/bin/bash

Ngroups=30000
Nlow=10000
Nmedium=5000
Nhigh=3600


smart_mpi 8 "python -W ignore paper/Fig1_prep.py 148_tsz groups ${Ngroups}" -t 6 --include "gen4,gen5,gen6"
smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_tsz low ${Nlow}" -t 6  --include "gen4,gen5,gen6"
smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_tsz medium ${Nmedium}" -t 6  --include "gen4,gen5,gen6"
smart_mpi 2 "python -W ignore paper/Fig1_prep.py 148_tsz high ${Nhigh}" -t 6  --include "gen4,gen5,gen6"

# smart_mpi 8 "python -W ignore paper/Fig1_prep.py 148_tsz groups ${Ngroups} -x -y" -t 6  --include "gen4,gen5,gen6"
# smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_tsz low ${Nlow} -x -y" -t 6  --include "gen4,gen5,gen6"
# smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_tsz medium ${Nmedium} -x -y" -t 6  --include "gen4,gen5,gen6"
#smart_mpi 2 "python -W ignore paper/Fig1_prep.py 148_tsz high ${Nhigh} -x -y" -t 6  --include "gen4,gen5,gen6"

# smart_mpi 8 "python -W ignore paper/Fig1_prep.py 148_tsz groups ${Ngroups} -y" -t 6  --include "gen4,gen5,gen6"
# smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_tsz low ${Nlow} -y" -t 6  --include "gen4,gen5,gen6"
# smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_tsz medium ${Nmedium} -y" -t 6  --include "gen4,gen5,gen6"
#smart_mpi 2 "python -W ignore paper/Fig1_prep.py 148_tsz high ${Nhigh} -y" -t 6  --include "gen4,gen5,gen6"

smart_mpi 8 "python -W ignore paper/Fig1_prep.py 148_ksz groups ${Ngroups} -x -y" -t 8  --include "gen3,gen7"
smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_ksz low ${Nlow} -x -y" -t 8   --include "gen3,gen7"
smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_ksz medium ${Nmedium} -x -y" -t 8   --include "gen3,gen7"
smart_mpi 2 "python -W ignore paper/Fig1_prep.py 148_ksz high ${Nhigh} -x -y" -t 8   --include "gen3,gen7"

smart_mpi 8 "python -W ignore paper/Fig1_prep.py 148_ksz groups ${Ngroups} -x -y -g 1000" -t 8   --include "gen3,gen7"
smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_ksz low ${Nlow} -x -y -g 1000" -t 8   --include "gen3,gen7"
smart_mpi 4 "python -W ignore paper/Fig1_prep.py 148_ksz medium ${Nmedium} -x -y -g 1000" -t 8   --include "gen3,gen7"
smart_mpi 2 "python -W ignore paper/Fig1_prep.py 148_ksz high ${Nhigh} -x -y -g 1000" -t 8   --include "gen3,gen7"

