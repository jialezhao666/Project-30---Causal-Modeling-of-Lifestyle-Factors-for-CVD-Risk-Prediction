#!/bin/bash -l
#$ -S /bin/bash
#$ -N phase2_03to11
#$ -l h_rt=8:00:00
#$ -l mem=32G
#$ -pe smp 4
#$ -cwd
#$ -j y

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on host: $(hostname)"

# --- R module: needed by 03_multiarm_joint.py via rpy2 + grf ---
module purge
module load r/4.4.2-openblas/gnu-10.2.0

# --- conda env ---
source /shared/ucl/apps/miniconda/24.3.0-0/etc/profile.d/conda.sh
conda activate ukb_env

# --- make sure rpy2 finds module R ---
export R_HOME=$(R RHOME)
export LD_LIBRARY_PATH=$R_HOME/lib:$LD_LIBRARY_PATH

cd ~/my_ukb_thesis/phase_II

echo "Python:"
which python
python --version

echo "R:"
which R
R --version | head -n 1

echo ""
echo "============================================================"
echo "STEP 03: multi-arm causal forest"
echo "============================================================"
python -u 03_multiarm_joint.py

echo ""
echo "============================================================"
echo "STEP 04: heterogeneity"
echo "============================================================"
python -u 04_heterogeneity.py

echo ""
echo "============================================================"
echo "STEP 05: transition matrix"
echo "============================================================"
python -u 05_transition_matrix.py

echo ""
echo "============================================================"
echo "STEP 06: MSM / IPTW"
echo "============================================================"
python -u 06_msm.py

echo ""
echo "============================================================"
echo "STEP 07: cohort characteristics"
echo "============================================================"
python -u 07_cohort_characteristics.py

echo ""
echo "============================================================"
echo "STEP 08: sensitivity E-value"
echo "============================================================"
python -u 08_sensitivity_evalue.py

echo ""
echo "============================================================"
echo "STEP 09: imaging date selection check"
echo "============================================================"
python -u 09_imaging_date_selection_check.py

echo ""
echo "============================================================"
echo "STEP 10: individual heterogeneity cross-check"
echo "============================================================"
python -u 10_individual_heterogeneity_cross_check.py

echo ""
echo "============================================================"
echo "STEP 11: single-treatment summary and figures"
echo "============================================================"
python -u 11_single_treatment_summary_and_figures.py

echo ""
echo "All Phase II scripts 03-11 completed successfully."
echo "Job finished at: $(date)"