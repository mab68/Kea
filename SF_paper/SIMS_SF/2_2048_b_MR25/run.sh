#!/bin/sh
#SBATCH --job-name=fastSF_2_2048
#SBATCH --nodes=1
#SBATCH --ntasks=128
#SBATCH --mem-per-cpu=1G
#SBATCH --partition=parallel
#SBATCH --reservation=SpjReservation
#SBATCH --time=1-00:00:00
#SBATCH -o raapoi.out
#SBATCH -e raapoi.err
#SBATCH --constraint="IB"

module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate fastSF
mpirun --oversubscribe -np 128 /nfs/home/bishopm1/ext_codes/fastSF/src/fastSF.out

