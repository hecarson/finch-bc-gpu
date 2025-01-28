#!/bin/bash

#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH -n 1
#SBATCH --gres=gpu:2080ti:1
#SBATCH -o ad2d-prof-out.txt
#SBATCH -e ad2d-prof-err.txt

module load julia/1.9.2 cuda
cd ~/finch-bc-gpu
LD_LIBRARY_PATH=/uufs/chpc.utah.edu/sys/installdir/r8/julia/1.9.2/lib/julia nsys launch julia --project=. profile/ad2d/example-advection2d-fv-gpu.jl
