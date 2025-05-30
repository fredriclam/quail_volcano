#!/bin/bash
#SBATCH --job-name=plt1x
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH -p serc

module load python/3.9.0
module load py-numpy/1.24.2_py39
module load py-scipy/1.10.1_py39
module load viz
module load py-matplotlib/3.7.1_py39

infile="$SCRATCH/refblast/H/refblastF_atm"
outfile="refblastHwcond_nomarker"
order=1
scriptloc=$HOME/quail_underwater/quail_volcano/src/jetcond_plot.py 
dpi=400

python3 $scriptloc $infile 3600 $outfile $order $dpi

