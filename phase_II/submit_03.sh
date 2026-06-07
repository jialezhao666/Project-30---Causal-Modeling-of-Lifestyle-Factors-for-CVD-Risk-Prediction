#!/bin/bash -l
#$ -S /bin/bash
#$ -N nb3_joint
#$ -l h_rt=2:00:00
#$ -l mem=32G
#$ -pe smp 4
#$ -cwd
#$ -j y

# --- R module (multi_arm_causal_forest needs R + grf via rpy2) ---
module purge
module load r/4.4.2-openblas/gnu-10.2.0

# --- conda env (use absolute path; module purge wiped UCL_CONDA_PATH) ---
source /shared/ucl/apps/miniconda/24.3.0-0/etc/profile.d/conda.sh
conda activate ukb_env

# --- make sure rpy2 finds module R ---
export R_HOME=$(R RHOME)
export LD_LIBRARY_PATH=$R_HOME/lib:$LD_LIBRARY_PATH

cd ~/my_ukb_thesis/phase_II

# use the env's python explicitly, just to be safe
which python
python --version
python 03_multiarm_joint.py