#!/bin/bash -l
#SBATCH --account=pawsey0272
#SBATCH --partition=work
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --mem=16gb
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/pawsey0272/chorvath/out/%A

module load singularity/3.11.4-slurm
export export SINGULARITY_BINDPATH=`pwd`:$HOME
singularity exec container.sif python3 run_on_pawsey.py $1 $2
echo "Job took ${SECONDS} seconds"