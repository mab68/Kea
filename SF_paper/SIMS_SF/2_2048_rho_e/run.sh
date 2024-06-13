#!/bin/sh
#SBATCH --job-name=fastSF_2_2048
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=bigmem
#SBATCH --time=06:00:00
#SBATCH -o raapoi.out
#SBATCH -e raapoi.err

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate fastSF
mpirun --oversubscribe -np 64 /nfs/home/bishopm1/ext_codes/fastSF/src/fastSF.out

