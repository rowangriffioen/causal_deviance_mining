#!/bin/bash
#SBATCH --job-name=dm_random_k2_traffic_payload_Pay36_dec_data
#SBATCH --output=results/output_%j_k2_traffic_payload_Pay36_dec_data.txt
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --partition=tue.default.q



module purge
module load Python/3.8.20  


source $HOME/thesis38/bin/activate

python 3.dm_random_k2.py
