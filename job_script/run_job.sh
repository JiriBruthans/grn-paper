#!/bin/bash
#PBS -N doubleko_mod8_647
#PBS -l select=1:ncpus=96:mem=64gb:scratch_local=20gb
#PBS -l walltime=48:00:00
#PBS -m b
#PBS -M bruthans@pm.me

# 1. Ensure scratch cleanup on job exit or failure
trap 'clean_scratch' EXIT

# 2. Define path to permanent storage
PERM_DIR="/storage/brno2/home/jiribruthans/grn_files"

# 3. Navigate into local scratch directory
cd "$SCRATCHDIR" || exit 1

# 4. Install Miniforge into scratch space
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b -p "$SCRATCHDIR/miniforge"

# 5. Initialize Conda environment in subshell
source "$SCRATCHDIR/miniforge/bin/activate"

# 6. Clone repository
git clone https://github.com/JiriBruthans/grn-paper.git
cd grn-paper || exit 1

# 7. Create and activate Conda environment from yml file
conda env create -f grn_env.yml
conda activate grn_env

# 8. Copy input files to working directories
cp "$PERM_DIR/graph.647.gpickle" ../.
cp "$PERM_DIR/run_doubleko_mod8_GRN#647.py" .

# 9. Execute Python script
python "run_doubleko_mod8_GRN#647.py"

# 10. Copy output files back to permanent storage
cp ../*.npz "$PERM_DIR/"