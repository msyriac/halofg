#!/bin/bash

N=200

for i in {0..41}
do
    smart_mpi 4 "python -W ignore paper/covclkh.py $i $N" -t 1
    echo $i
    sleep 0.5
done
