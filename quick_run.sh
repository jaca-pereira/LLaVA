#!/bin/bash
#SBATCH --get-user-env=L
#SBATCH --job-name=quick_run_llava
# The line below writes to a logs dir inside the one where sbatch was called
# %x will be replaced by the job name, and %j by the job id
#SBATCH --output=logs/%j-%x.out
#SBATCH -e logs/%j-%x.err
#SBATCH --cpus-per-task 8 # Number of threads
#SBATCH --mem=32G # Memory - Use up to 2GB per requested CPU as a rule of thumb
#SBATCH --time=0 # No time limit
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1 # or shard:<nshards> or gpu:nvidia_a100-sxm4-40gb:<1-4> or nvidia_a100-pcie-40gb:<1-8>
#SBATCH --nodelist=nscluster # or nslab<0-3> or nscluster or novasearchdl
#SBATCH --partition=slurm_queue # also try slurm_queue; high_priority

eval "$(conda shell.bash hook)"
conda activate llava
cd ~/LLaVA/
python quick_run.py
