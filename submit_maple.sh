#!/bin/bash
#SBATCH --job-name=MapPT
#SBATCH --output=logs/mappt.%A_%a.out
#SBATCH --error=logs/mappt.%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --array=0-2

mkdir -p logs results

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

L_values=(
70
77
84
)

L=${L_values[$SLURM_ARRAY_TASK_ID]}

echo "Running L = $L"

start_time=$(date +%s)

python3 run_maple_mc.py \
    --L $L \
    --Tmin 0.01 \
    --Tmax 2.5 \
    --nrep 100 \
    --ntherm 25000 \
    --nmeas 10000 \
    --sweeps 20 \
    --out results/L${L}.npz

end_time=$(date +%s)

elapsed=$((end_time - start_time))

echo "Completed L=$L in ${elapsed}s"