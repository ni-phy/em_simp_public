#!/bin/bash
#SBATCH --ntasks 80 
#SBATCH --time 10000
#SBATCH --qos bbdefault
#SBATCH --mail-type NONE 

set -e

module purge
module load bluebear
module load bear-apps/2022b

module load Meep/1.28.0-foss-2022b
module load NLopt/2.7.1-foss-2022b-Python
module load mpi4py/3.1.4-gompi-2022b
module load Gdspy/1.6.13-foss-2022b

#for i in {3..8}:
#do
mpirun -n 80 python3 3freq.py "2" 

#done
