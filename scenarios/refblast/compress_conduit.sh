#!/bin/bash
#SBATCH --job-name=compr1d
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64GB
#SBATCH -p serc

module load python/3.9.0
module load py-numpy/1.24.2_py39
module load py-scipy/1.10.1_py39
module load viz
module load py-matplotlib/3.7.1_py39

infile="$SCRATCH/refblast/H/refblastF_cond"
outfile="refblastH_cond"
order=1
scriptloc=$HOME/quail_underwater/quail_volcano/src/compress1D.py

export PATH="$PATH:$SCRATCH/refblast/H/"
export PYTHONPATH="$PYTHONPATH:$SCRATCH/refblast/H/"

echo $PYTHONPATH

python3 $scriptloc $infile 72000 $outfile

