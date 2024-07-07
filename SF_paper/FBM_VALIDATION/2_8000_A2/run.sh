#!/bin/sh
#SBATCH --job-name=fastSF_FBM_2_8000
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem=200G
#SBATCH --partition=parallel
#SBATCH --time=4-00:00:00
#SBATCH -o raapoi.out
#SBATCH -e raapoi.err
#SBATCH --constraint="IB"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate fastSF
mpirun --oversubscribe -np 128 /nfs/home/bishopm1/ext_codes/fastSF/src/fastSF.out

