#!/bin/bash -l
#SBATCH --account=pawsey0272
#SBATCH --partition=work
#SBATCH --array=0-950:50
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/pawsey0272/chorvath/out/%A

module load singularity/3.11.4-slurm
export export SINGULARITY_BINDPATH=`pwd`:$HOME,/tmp
singularity exec container.sif python3 run_on_pawsey.py ${SLURM_ARRAY_TASK_ID} $((SLURM_ARRAY_TASK_ID + 50))
echo "Job took ${SECONDS} seconds"