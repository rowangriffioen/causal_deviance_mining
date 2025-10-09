#!/bin/bash
#SBATCH --job-name=dm_dt_ripperk_BPI15A_decl2
#SBATCH --output=results/output_%j_traffic_BPI15A_decl2.txt
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=tue.default.q


module purge
module load Python/3.11  


source $HOME/thesis311/bin/activate

python 3.dm_dt_ripperk.py