#!/bin/bash
#PBS -N singlekosweep_256g
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=20gb
#PBS -l walltime=48:00:00
#PBS -m b
#PBS -M bruthans@pm.me



trap 'clean_scratch' EXIT


outd="/storage/brno2/home/jiribruthans/grn_files/sweep"

cd "$SCRATCHDIR" || exit 1

wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p "$SCRATCHDIR/miniforge"

source "$SCRATCHDIR/miniforge/bin/activate"

git clone https://github.com/JiriBruthans/grn-paper.git
cd grn-paper || exit 1

conda env create -f grn_env.yml
conda activate grn_env

nn=256
n_reps=5

config=0

for rr in 2 3 4 5 8 16; do
    for d_o in 1 3 10 30 100 300; do
        for di in 10 30 50 100 200 300; do

            config=$((config + 1))

            for rep in $(seq 1 "$n_reps"); do

                # Unique, reproducible seed
                seed=$((config * 100 + rep))

                echo "================================="
                echo "Configuration: $config / 216"
                echo "Replicate:     $rep / $n_reps"
                echo "r:             $rr"
                echo "delta_out:     $d_o"
                echo "delta_in:      $di"
                echo "seed:          $seed"
                echo "genes:         $nn"
                echo "================================="

                prefix="${outd}/graph.cfg${config}.rep${rep}.r${rr}.dout${d_o}.din${di}.seed${seed}"

                python3 src/grn.py \
                    --out "$prefix" \
                    --num-genes "$nn" \
                    --num-groups 1 \
                    --w 1 \
                    --r "$rr" \
                    --delta-in "$di" \
                    --delta-out "$d_o" \
                    --seed "$seed" \
                    --kos \
                    --cores 4

            done
        done
    done
done

