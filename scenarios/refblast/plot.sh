#!/bin/bash
#SBATCH --job-name=pltrbF
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH -p serc

module load python/3.9.0
module load py-numpy/1.24.2_py39
module load py-scipy/1.10.1_py39
module load viz
module load py-matplotlib/3.7.1_py39

infile="$SCRATCH/refblast/F/refblastF_atm"
outfile="refblastF"
order=1
scriptloc=$HOME/quail_underwater/quail_volcano/src/jet_plot_script.py 
dpi=200

for (( i=0; i <= 7200; i=i+80 ))
do
  python3 $scriptloc $infile $i $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+10)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+20)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+30)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+40)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+50)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+60)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+70)) $outfile $order $dpi &
  wait
done


