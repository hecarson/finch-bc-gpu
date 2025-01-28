#!/bin/bash

#SBATCH --account=owner-gpu-guest
#SBATCH --partition=notchpeak-gpu-guest
#SBATCH -n 1
#SBATCH --gres=gpu:2080ti:1
#SBATCH -o ad2d-out.txt
#SBATCH -e ad2d-err.txt

module load julia/1.9.2 cuda
cd ~/finch-bc-gpu
julia --project=. profile/ad2d/example-advection2d-fv-gpu.jl
