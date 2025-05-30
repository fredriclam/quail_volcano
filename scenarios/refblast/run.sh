#!/bin/bash
#SBATCH --job-name=rb_H
#SBATCH --time=160:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=16GB
#SBATCH -p serc

module load python/3.9.0
module load py-numpy/1.24.2_py39
module load py-scipy/1.10.1_py39
module load viz
module load py-matplotlib/3.7.1_py39

python3 $HOME/quail_underwater/quail_volcano/src/quail vent_region.py
