#!/bin/bash 

# Run the script
declare -i batches=180

for (( i = 1; i <= $batches; i++ ))
do
    sbatch --cpus-per-task 4 -o sweepresults-$i-$batches.log run_sweep.sh $i $batches
done
