#!/bin/bash -l
#SBATCH --account=pawsey0272
#SBATCH --partition=work
#SBATCH --array=1-49
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=1
#SBATCH --output=/scratch/pawsey0272/chorvath/out/%A_%a

module load singularity/3.11.4-slurm
export export SINGULARITY_BINDPATH=`pwd`:$HOME,/tmp
singularity exec container.sif python3 run_on_pawsey.py $((20 * SLURM_ARRAY_TASK_ID)) $((20 * SLURM_ARRAY_TASK_ID + 20))
echo "Job took ${SECONDS} seconds"