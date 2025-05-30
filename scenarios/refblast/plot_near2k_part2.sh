#!/bin/bash
#SBATCH --job-name=pltF2
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=60GB
#SBATCH -p serc

module load python/3.9.0
module load py-numpy/1.24.2_py39
module load py-scipy/1.10.1_py39
module load viz
module load py-matplotlib/3.7.1_py39

infile="$SCRATCH/refblast/F/refblastF_atm"
outfile="rb2kF"
order=1
scriptloc=$HOME/quail_underwater/quail_volcano/src/jet_plot_script_near2k.py 
dpi=200

# Draw final
# python3 $scriptloc $infile 72000 $outfile $order $dpi &
# Distribute 720 draws over 8 cpus
for (( i=1900; i < 9000; i=i+100 ))
do
  python3 $scriptloc $infile $i $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+9000)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+18000)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+27000)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+36000)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+45000)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+54000)) $outfile $order $dpi &
  python3 $scriptloc $infile $(($i+63000)) $outfile $order $dpi &
  wait
done


