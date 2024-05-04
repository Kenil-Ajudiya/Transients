#!/bin/bash -l
#SBATCH --account=pawsey0272
#SBATCH --partition=debugq
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --mem=16gb
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/pawsey0272/chorvath/out/%A

module load singularity/3.11.4-slurm
export export SINGULARITY_BINDPATH=`pwd`:$HOME,$HOME/.ssh
singularity exec python3 run_on_pawsey.py