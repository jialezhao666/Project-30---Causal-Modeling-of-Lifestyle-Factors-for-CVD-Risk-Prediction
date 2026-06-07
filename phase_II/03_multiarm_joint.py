import os, sys

# --- R environment (safety fallback; terminal module load already sets these) ---
os.environ.setdefault('R_HOME', '/shared/ucl/apps/R/R-4.4.2-OpenBLAS/lib64/R')
_rlib = os.environ['R_HOME'] + '/lib'
if _rlib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _rlib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import numpy as np
import pandas as pd
import joblib

# paths 
PHASE2_DIR  = os.path.expanduser('~/my_ukb_thesis/phase_II')
OUTPUT_DIR  = os.path.join(PHASE2_DIR, 'outputs')
FIGURES_DIR = os.path.join(PHASE2_DIR, 'figures')

CONFOUNDERS = [
    'age_defined_baseline', 'genetic_sex', 'BMI', 'uni_degree',
    'FH_cvd_f', 'FH_cvd_m', 'FH_cvd_sib', 'mental_doctor', 'alc_curr',
]
TREATMENTS = ['smk_curr', 'PA_active', 'sleep_adequate']
OUTCOME = 'def_CVD_AF_HF_AFTER'

#  debug switch: set to an int for testing, None for full run 
N_TEST = None   # start small; set to None for full 298,245


# Load + replicate preprocessing (transform, NOT fit_transform)
df_B = pd.read_parquet(os.path.join(OUTPUT_DIR, 'split_B_phase2.parquet'))

w_imputer = joblib.load(os.path.join(OUTPUT_DIR, 'confounder_imputer.pkl'))
t_imputer = joblib.load(os.path.join(OUTPUT_DIR, 'treatment_imputer.pkl'))
scaler = joblib.load(os.path.join(OUTPUT_DIR, 'confounder_scaler.pkl'))

W_df = pd.DataFrame(w_imputer.transform(df_B[CONFOUNDERS]),
                    columns=CONFOUNDERS, index=df_B.index)
W_df['alc_curr'] = np.clip(np.round(W_df['alc_curr']), 0, 1)

T_df = pd.DataFrame(t_imputer.transform(df_B[TREATMENTS]),
                    columns=TREATMENTS, index=df_B.index)
for c in TREATMENTS:
    T_df[c] = np.clip(np.round(T_df[c]), 0, 1)

W_df[['age_defined_baseline', 'BMI']] = scaler.transform(
    W_df[['age_defined_baseline', 'BMI']])

W = W_df.values
Y = df_B[OUTCOME].values

print(f"Loaded: W{W.shape}, T{T_df.shape}, Y{Y.shape}")
print(f"NaN check: W={np.isnan(W).sum()}, T={T_df.isna().sum().sum()}, Y={np.isnan(Y).sum()}")


# 1. 8-class joint encoding 
no_smk = (1 - T_df['smk_curr'].values).astype(int)   # reverse to align healthy=1, unhealthy=0
pa  = T_df['PA_active'].values.astype(int)
sleep  = T_df['sleep_adequate'].values.astype(int)

T_combined = no_smk * 1 + pa * 2 + sleep * 4   

LABELS = {
    0: '000 no_smk=0,PA=0,sleep=0  (all-unhealthy)',
    1: '100 no_smk=1,PA=0,sleep=0',
    2: '010 no_smk=0,PA=1,sleep=0',
    3: '110 no_smk=1,PA=1,sleep=0',
    4: '001 no_smk=0,PA=0,sleep=1',
    5: '101 no_smk=1,PA=0,sleep=1',
    6: '011 no_smk=0,PA=1,sleep=1',
    7: '111 no_smk=1,PA=1,sleep=1  (all-healthy)'
}

print("\n=== 8-class joint distribution (unified healthy direction) ===")
vc = pd.Series(T_combined).value_counts().sort_index()
for code in range(8):
    n = int(vc.get(code, 0))
    pct = n / len(T_combined) * 100
    flag = "  <-- SMALL" if n < 2000 else ""
    print(f"  T={code} [{LABELS[code]}] : {n:>7,} ({pct:4.1f}%){flag}")

# outcome rate per arm — should DECREASE from T=0 (all-unhealthy) to T=7 (all-healthy)
print("\n=== CVD rate per arm (expect roughly monotonic decrease) ===")
prev = None
for code in range(8):
    mask = T_combined == code
    if mask.sum() > 0:
        rate = Y[mask].mean()
        print(f"  T={code} : CVD rate = {rate:.4f} (n={mask.sum():,})")

# sanity: count healthy behaviours per arm vs CVD rate (clearest signal)
print("\n=== CVD rate by NUMBER of healthy behaviours adopted ===")
n_healthy = no_smk + pa + sleep
for k in range(4):
    mask = n_healthy == k
    print(f"  {k} healthy behaviour(s): CVD rate = {Y[mask].mean():.4f} (n={mask.sum():,})")

print("\nPART 1 done. Verify CVD rate falls as healthy behaviours increase.")


# 2. multi_arm_causal_forest via rpy2  (test on N_TEST first)

import time, resource
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr

numpy2ri.activate()
pandas2ri.activate()

grf  = importr('grf', lib_loc='~/Rlibs')
base = importr('base')

#  subset for testing 
if N_TEST is not None:
    idx = np.arange(N_TEST)
    print(f"\n[TEST MODE] using first {N_TEST} rows")
else:
    idx = np.arange(len(Y))
    print(f"\n[FULL MODE] using all {len(Y):,} rows")

W_sub = W[idx]
T_sub = T_combined[idx].astype(int)
Y_sub = Y[idx].astype(float)

print(f"W_sub {W_sub.shape}, T_sub {T_sub.shape}, Y_sub {Y_sub.shape}")
print(f"T_sub arms present: {sorted(np.unique(T_sub))}")

#  push data into R (force pure R vector for Y and treatment) 
ro.globalenv['X_raw'] = numpy2ri.py2rpy(np.asarray(W_sub, dtype=float))
ro.globalenv['Y_raw'] = numpy2ri.py2rpy(np.asarray(Y_sub, dtype=float).ravel())
ro.globalenv['T_raw'] = numpy2ri.py2rpy(np.asarray(T_sub, dtype=float).ravel())

ro.r(f'''
X_r <- matrix(as.numeric(X_raw), nrow = {len(idx)}, ncol = {W_sub.shape[1]})
Y_r <- as.vector(as.numeric(Y_raw))          # strip array attr -> pure vector
W_factor <- factor(as.integer(as.vector(T_raw)), levels = 0:7)

cat("class(Y_r):", class(Y_r), "| is.vector:", is.vector(Y_r), "| length:", length(Y_r), "\\n")
cat("class(X_r):", class(X_r), "| dim:", dim(X_r), "\\n")
cat("class(W_factor):", class(W_factor), "| nlevels:", nlevels(W_factor), "\\n")
''')

#  fit multi_arm_causal_forest 
print("\nFitting multi_arm_causal_forest in R")
t0 = time.time()
ro.r('''
set.seed(42)
maf <- multi_arm_causal_forest(
    X = X_r,
    Y = Y_r,
    W = W_factor,
    num.trees = 500
)
cat("fit done\\n")
''')
fit_min = (time.time() - t0) / 60
print(f"fit time: {fit_min:.1f} min")

#  (a) individual CATEs via predict — usually robust to extreme propensity 
ro.r('preds <- predict(maf)$predictions')   # n x (n_arms-1) x n_outcomes
ro.r('cat("dim(preds):", dim(preds), "\\n")')
cate = np.asarray(ro.r('preds[,,1]'))        # n x 7  (arms 1..7 vs ref 0)
print(f"\nCATE shape: {cate.shape}  (rows x 7 arms)")
print("Mean CATE per arm (vs T=0 all-unhealthy), expect NEGATIVE:")
arm_labels = ['1-0','2-0','3-0','4-0','5-0','6-0','7-0']
for j, lab in enumerate(arm_labels):
    col = cate[:, j]
    print(f"  {lab}: mean={np.nanmean(col):+.4f}, "
        f"NaN={np.isnan(col).sum()}, n={len(col)}")


# Joint intervention summary: mean CATE + SE per arm (vs T=0)
# (AIPW ATE not used: limited propensity overlap, ref arm <1%. Forest CATEs do not rely on IPW and remain stable — see thesis text.)
print("\n=== Joint intervention effects: mean CATE per arm (vs T=0 all-unhealthy) ===")

ARM_LABELS = {
    1: 'no_smk only',
    2: 'PA only',
    3: 'no_smk + PA',
    4: 'sleep only',
    5: 'no_smk + sleep',
    6: 'PA + sleep',
    7: 'all three (no_smk+PA+sleep)',
}

rows = []
n = cate.shape[0]
for j in range(7):
    arm = j + 1                      # columns are arms 1..7
    col = cate[:, j]
    mean = float(np.nanmean(col)) 
    n_healthy_in_arm = bin(arm).count('1')
    rows.append({
        'arm'      : arm,
        'label'    : ARM_LABELS[arm],
        'n_healthy': n_healthy_in_arm,
        'mean_cate': mean,
    })
    print(f"  arm {arm} [{ARM_LABELS[arm]:30s}] "
        f"mean CATE = {mean:+.4f}")

summary = pd.DataFrame(rows)

# ── save: per-arm summary + full per-individual CATE matrix ──
summary.to_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_summary.parquet'), index=False)

cate_df = pd.DataFrame(cate, columns=[f'cate_arm{j+1}_vs0' for j in range(7)])
cate_df['eid'] = df_B['eid'].values[idx]
cate_df.to_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_full.parquet'), index=False)

print(f"\nSaved:")
print(f"  joint_cate_summary.parquet")
print(f"  joint_cate_full.parquet     ({cate_df.shape[0]:,} x 7 CATEs + eid)")


# Plot: joint intervention dose-response
import matplotlib
matplotlib.use('Agg')   # no display on compute node
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# build plot order: sort by n_healthy, then by effect within group
plot_df = summary.copy()
plot_df['short'] = plot_df['label'].replace({
    'no_smk only': 'no_smk\nonly',
    'PA only': 'PA\nonly',
    'sleep only': 'sleep\nonly',
    'no_smk + PA': 'no_smk\n+PA',
    'no_smk + sleep': 'no_smk\n+sleep',
    'PA + sleep': 'PA\n+sleep',
    'all three (no_smk+PA+sleep)': 'all\nthree',
})
plot_df = plot_df.sort_values(['n_healthy', 'mean_cate']).reset_index(drop=True)

cmap = {1: '#8AB6D6', 2: '#4C72B0', 3: '#2A4D69'}
colors = [cmap[n] for n in plot_df['n_healthy']]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(plot_df))
ax.bar(x, plot_df['mean_cate'], color=colors, edgecolor='white', width=0.7)

# value labels below each bar
for xi, eff in zip(x, plot_df['mean_cate']):
    ax.text(xi, eff - 0.003, f'{eff:+.4f}', ha='center', va='top', fontsize=9)

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(plot_df['short'], fontsize=9)
ax.set_ylabel('Mean CATE on CVD risk\n(vs all-unhealthy reference)', fontsize=10)
ax.set_title('Joint Intervention Effects on CVD Risk\n'
            'multi_arm_causal_forest (grf), n = 298,245', fontsize=12, pad=12)
ax.set_ylim(-0.10, 0.008)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, alpha=0.25, lw=0.5)
ax.set_axisbelow(True)

legend_elems = [
    Patch(facecolor=cmap[1], label='1 healthy behaviour'),
    Patch(facecolor=cmap[2], label='2 healthy behaviours'),
    Patch(facecolor=cmap[3], label='3 healthy behaviours'),
]
ax.legend(handles=legend_elems, loc='lower left', fontsize=9, frameon=False)

fig.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'joint_cate.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved figure: {fig_path}")

