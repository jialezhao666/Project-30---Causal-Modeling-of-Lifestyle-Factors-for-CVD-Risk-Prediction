#!/bin/bash -l
#$ -S /bin/bash
#$ -N phase2_04to07
#$ -l h_rt=2:00:00
#$ -l mem=32G
#$ -pe smp 4
#$ -cwd
#$ -j y

module purge
source /shared/ucl/apps/miniconda/24.3.0-0/etc/profile.d/conda.sh
conda activate ukb_env

cd ~/my_ukb_thesis/phase_II

echo "===== 04_heterogeneity.py ====="
python 04_heterogeneity.py
echo ""
echo "===== 05_transition_matrix.py ====="
python 05_transition_matrix.py
echo ""
echo "===== 06_msm.py ====="
python 06_msm.py
echo ""
echo "===== 07_cohort_characteristics.py ====="
python 07_cohort_characteristics.py
echo ""
echo "All done."