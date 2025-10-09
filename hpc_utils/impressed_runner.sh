#!/bin/bash
#SBATCH --job-name=impressed_features
#SBATCH --output=results/output_%j.txt
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=32
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --partition=tue.default.q


module purge
module load Python/3.11  


source $HOME/thesis311/bin/activate

python feature_extraction_IMPresseD.py