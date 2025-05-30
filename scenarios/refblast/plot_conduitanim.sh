#!/bin/bash
#SBATCH --job-name=drcoFB
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH -p serc

module load python/3.9.0
module load py-numpy/1.24.2_py39
module load py-scipy/1.10.1_py39
module load viz
module load py-matplotlib/3.7.1_py39

export PYTHONPATH="$PYTHONPATH:$SCRATCH/refblast/F/"
infile="$SCRATCH/refblast/F/refblastF_atm"
outfile="refblastFconduit"
order=1
scriptloc=$HOME/quail_underwater/quail_volcano/src/jet_plot_conduit.py 
dpi=400

for (( i=0; i <= 27000; i=i+800 ))
do
  python3 $scriptloc $infile $i $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+100)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+200)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+300)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+400)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+500)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+600)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+700)) $outfile $order $dpi &
  wait
done


