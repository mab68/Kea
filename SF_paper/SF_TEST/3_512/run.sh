#!/bin/sh
#SBATCH --job-name=fastSF_3_512
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=400G
#SBATCH --partition=bigmem
#SBATCH --time=02-00:00:00
#SBATCH -o raapoi.out
#SBATCH -e raapoi.err

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate fastSF
mpirun --oversubscribe -np 128 /nfs/home/bishopm1/ext_codes/fastSF/src/fastSF.out

