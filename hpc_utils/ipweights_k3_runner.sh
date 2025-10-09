#!/bin/bash
#SBATCH --job-name=dm_ipweights_k3
#SBATCH --output=results/output_%j.txt
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=tue.default.q


module purge
module load Python/3.11  


source $HOME/thesis311/bin/activate

python 3.dm_ipweights_k3.py
