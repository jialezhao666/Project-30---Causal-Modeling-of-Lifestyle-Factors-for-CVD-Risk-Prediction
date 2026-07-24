"""
Diagnostic: unreliable-rate audit for Phase III build_scenarios()
===================================================================
Draws a random subsample of n=5000 individuals from Split B, re-predicts
NB3 multi-arm causal forest CATEs + variance estimates for each of the 7
arms (vs T=0 reference), then runs the EXACT same unreliable / diverges
logic used in phase_III_app.py::build_scenarios for every (current_arm,
target_arm) pair that the Streamlit tool would ever construct, and reports:

  - overall % of (person, scenario) pairs flagged unreliable
  - breakdown by target arm / intervention type
  - breakdown by number of healthy behaviours already adopted (current_arm)
  - distribution of baseline_risk for unreliable vs reliable cases (to check
    whether the logit-scale fix actually removed the baseline-dependence)

Usage:
    module unload gcc-libs
    module load r/4.4.2-openblas/gnu-10.2.0
    conda activate ukb_env
    python diagnose_unreliable_rate.py
"""

import os
import numpy as np
import pandas as pd
import joblib

#  R environment 
os.environ.setdefault('R_HOME', '/shared/ucl/apps/R/R-4.4.2-OpenBLAS/lib64/R')
_rlib = os.environ['R_HOME'] + '/lib'
if _rlib not in os.environ.get('LD_LIBRARY_PATH', ''):
    os.environ['LD_LIBRARY_PATH'] = _rlib + ':' + os.environ.get('LD_LIBRARY_PATH', '')

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

_converter = ro.default_converter + numpy2ri.converter

# Configuration -- must match 03_multiarm_joint.py / phase_III_app.py
PHASE2_DIR = os.path.expanduser('~/my_ukb_thesis/phase_II')
OUTPUT_DIR = os.path.join(PHASE2_DIR, 'outputs')

N_SAMPLE = 5000
SEED = 42

CONFOUNDERS = [
    'age_defined_baseline', 'genetic_sex', 'BMI', 'uni_degree',
    'FH_cvd_f', 'FH_cvd_m', 'FH_cvd_sib', 'mental_doctor', 'alc_curr']
TREATMENTS = ['smk_curr', 'PA_active', 'sleep_adequate']

# NB3 population-average CATEs (from joint_cate_summary.parquet)
POP_AVG_CATE = {
    1: -0.070973,  # no_smk only
    2: -0.028031,  # PA only
    3: -0.080976,  # no_smk + PA
    4: -0.039017,  # sleep only
    5: -0.080518,  # no_smk + sleep
    6: -0.049289,  # PA + sleep
    7: -0.085386,  # all three
}
ARM_LABELS = {
    1: 'no_smk only', 2: 'PA only', 3: 'no_smk + PA', 4: 'sleep only',
    5: 'no_smk + sleep', 6: 'PA + sleep', 7: 'all three'}

# Fixed reference baseline for the probability<->logit yardstick conversion
# -- must match REF_CVD_RATE in phase_III_app.py exactly.
REF_CVD_RATE = 0.08790088685476706
REF_SLOPE = REF_CVD_RATE * (1 - REF_CVD_RATE)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# 1. Load Split B, draw random subsample, replicate NB1/NB3 preprocessing

print(f"Loading split_B_phase2.parquet ...")
df_B = pd.read_parquet(os.path.join(OUTPUT_DIR, 'split_B_phase2.parquet'))
print(f"  full Split B: {len(df_B):,} rows")

rng = np.random.default_rng(SEED)
sample_idx = rng.choice(len(df_B), size=min(N_SAMPLE, len(df_B)), replace=False)
df_s = df_B.iloc[sample_idx].reset_index(drop=True)
print(f"  drew random subsample: {len(df_s):,} rows (seed={SEED})")

w_imputer = joblib.load(os.path.join(OUTPUT_DIR, 'confounder_imputer.pkl'))
scaler = joblib.load(os.path.join(OUTPUT_DIR, 'confounder_scaler.pkl'))

W_df = pd.DataFrame(w_imputer.transform(df_s[CONFOUNDERS]),
                    columns=CONFOUNDERS, index=df_s.index)
W_df['alc_curr'] = np.clip(np.round(W_df['alc_curr']), 0, 1)
W_df[['age_defined_baseline', 'BMI']] = scaler.transform(
    W_df[['age_defined_baseline', 'BMI']])

W = W_df.values
print(f"  W shape: {W.shape}")

# need each person's CURRENT lifestyle arm (0..7), to mimic what
# current_arm would be set to in the Streamlit tool, and their baseline_risk
logit_model_path = os.path.join(OUTPUT_DIR, 'logistic_model.pkl')
HAVE_LOGIT_MODEL = os.path.exists(logit_model_path)

if HAVE_LOGIT_MODEL:
    import statsmodels.api as sm
    logit_model = joblib.load(logit_model_path)
    print("  loaded logistic_model.pkl for baseline_risk computation")
else:
    print("  WARNING: logistic_model.pkl not found -- baseline_risk will "
          "be approximated from a fixed value (REF_CVD_RATE) for all "
          "individuals. This will NOT reproduce the baseline-dependence "
          "check correctly. Please point LOGIT_MODEL_PATH if your file "
          "lives elsewhere.")

# Reconstruct current lifestyle arm from raw treatment columns 
t_imputer = joblib.load(os.path.join(OUTPUT_DIR, 'treatment_imputer.pkl'))
T_df = pd.DataFrame(t_imputer.transform(df_s[TREATMENTS]),
                    columns=TREATMENTS, index=df_s.index)
for c in TREATMENTS:
    T_df[c] = np.clip(np.round(T_df[c]), 0, 1).astype(int)

no_smk = 1 - T_df['smk_curr'].values
pa = T_df['PA_active'].values
sleep = T_df['sleep_adequate'].values
current_arm_arr = (no_smk * 1 + pa * 2 + sleep * 4).astype(int)

print(f"  current_arm distribution:\n{pd.Series(current_arm_arr).value_counts().sort_index()}")

# 2. Compute baseline_risk for each person (Phase I logistic regression)

# These constants are duplicated here (NOT re-derived) -- if Phase I's
# coefficients/scaler are ever refit, update both files together.
LR_INTERCEPT = -2.936917
LR_COEFS = {
    "age_defined_baseline": 0.702113, "BMI": 0.228672, "sleep_hrs": -0.028416,
    "genetic_sex": 0.757434, "mental_doctor": 0.209540, "uni_degree": -0.182411,
    "FH_cvd_f": 0.178086, "FH_cvd_m": 0.234762, "FH_cvd_sib": 0.245727,
    "smk_prev": 0.136889, "smk_curr": 0.582529, "alc_curr": -0.277854,
    "PA_active": -0.090007}
LR_SCALER = {
    "age_defined_baseline": {"mean": 56.207423, "std": 8.106073},
    "BMI": {"mean": 27.192648, "std": 4.646652},
    "sleep_hrs": {"mean": 7.146948, "std": 1.078493}}

needed_lr_cols = ["age_defined_baseline", "BMI", "sleep_hrs", "genetic_sex",
                "mental_doctor", "uni_degree", "FH_cvd_f", "FH_cvd_m",
                "FH_cvd_sib", "smk_prev", "smk_curr", "alc_curr", "PA_active"]
missing_lr_cols = [c for c in needed_lr_cols if c not in df_s.columns]
if missing_lr_cols:
    raise RuntimeError(
        f"df_s is missing columns required for the Phase I baseline_risk "
        f"calculation: {missing_lr_cols}. Check that split_B_phase2.parquet "
        f"retains these raw (pre-imputation) columns, or adjust this script "
        f"to source them from wherever Phase I's training frame lives.")

lr_x = df_s[needed_lr_cols].copy()
for col, p in LR_SCALER.items():
    lr_x[col] = (lr_x[col] - p["mean"]) / p["std"]

logit_vals = LR_INTERCEPT + sum(LR_COEFS[f] * lr_x[f].values for f in LR_COEFS)
baseline_risk_arr = _sigmoid(logit_vals)
print(f"  baseline_risk: min={baseline_risk_arr.min():.4f}, "
    f"median={np.median(baseline_risk_arr):.4f}, max={baseline_risk_arr.max():.4f}")


# 3. grf predict with variance for all 5000 x 7 arms
print("\nLoading maf_model.rds and predicting CATEs + variances for the sample ...")
numpy2ri.activate()
grf = importr('grf', lib_loc='~/Rlibs')

with localconverter(_converter):
    ro.r(f'maf <- readRDS("{os.path.join(OUTPUT_DIR, "maf_model.rds")}")')
    ro.r.assign("X_new_vec", ro.FloatVector(W.flatten().tolist()))
    ro.r(f"X_new <- matrix(X_new_vec, nrow={W.shape[0]}, ncol={W.shape[1]}, byrow=TRUE)")
    ro.r("pred_new <- predict(maf, newdata = X_new, estimate.variance = TRUE)")
    cates = np.asarray(ro.r("pred_new$predictions[,,1]"))       # n x 7
    var_est = np.asarray(ro.r("as.matrix(pred_new$variance.estimates)"))  # n x 7

cate_ses = np.sqrt(np.clip(var_est, 0, None))
print(f"  cates shape: {cates.shape}, cate_ses shape: {cate_ses.shape}")


# 4. Run build_scenarios logic for every person, every reachable scenario
def build_scenarios_logic(baseline_risk, cates_row, cate_ses_row, current_arm):
    """Mirrors phase_III_app.py::build_scenarios exactly."""
    cur_cate = 0.0 if current_arm == 0 else cates_row[current_arm - 1]
    cur_se = 0.0 if current_arm == 0 else cate_ses_row[current_arm - 1]
    cur_nosmk = current_arm & 1
    cur_pa = (current_arm >> 1) & 1
    cur_sleep = (current_arm >> 2) & 1

    out = []
    for tgt in range(1, 8):
        t_nosmk = tgt & 1
        t_pa = (tgt >> 1) & 1
        t_sleep = (tgt >> 2) & 1
        if t_nosmk < cur_nosmk or t_pa < cur_pa or t_sleep < cur_sleep:
            continue
        if tgt == current_arm:
            continue

        tgt_cate = cates_row[tgt - 1]
        tgt_se = cate_ses_row[tgt - 1]
        delta = tgt_cate - cur_cate
        delta_se = float(np.sqrt(tgt_se**2 + cur_se**2))

        p = baseline_risk
        slope = max(p * (1 - p), 1e-9)
        delta_logit = delta / slope
        delta_se_logit = delta_se / slope

        baseline_logit = _logit(baseline_risk)
        new_risk = float(_sigmoid(baseline_logit + delta_logit))

        ci_low_logit = delta_logit - 1.96 * delta_se_logit
        ci_high_logit = delta_logit + 1.96 * delta_se_logit

        pop_avg = POP_AVG_CATE[tgt]
        diverges = (np.sign(tgt_cate) != np.sign(pop_avg)) and abs(tgt_cate) > 1e-6

        ci_width_logit = ci_high_logit - ci_low_logit
        pop_avg_mag = max(abs(pop_avg), 0.01)
        pop_avg_mag_logit = pop_avg_mag / REF_SLOPE
        # Dual threshold, mirroring phase_III_app.py's three-tier classification:
        #   <= 4x  -> reliable (if direction also agrees)
        #   4x-8x  -> direction_only (or any width with diverges=True)
        #   > 8x   -> unreliable
        ci_moderately_wide = ci_width_logit > 4 * pop_avg_mag_logit
        ci_too_wide = ci_width_logit > 8 * pop_avg_mag_logit

        if ci_too_wide:
            tier = "unreliable"
        elif ci_moderately_wide or diverges:
            tier = "direction_only"
        else:
            tier = "reliable"

        out.append({
            "tgt": tgt,
            "n_healthy": bin(tgt).count('1'),
            "new_risk": new_risk,
            "diverges": diverges,
            "tier": tier,
            "unreliable": (tier == "unreliable"),  # kept for backward-compatible reporting below
            "baseline_risk": baseline_risk,
            "delta_se": delta_se,            # raw, probability-scale SE (pre-conversion)
            "delta_se_logit": delta_se_logit,  # converted via /slope, slope = baseline*(1-baseline)
            "slope": slope})
    return out


print("\nRunning build_scenarios logic for every person in the sample ...")
records = []
for i in range(len(df_s)):
    baseline_risk = float(baseline_risk_arr[i])
    current_arm = int(current_arm_arr[i])
    scens = build_scenarios_logic(baseline_risk, cates[i], cate_ses[i], current_arm)
    for s in scens:
        s["person_idx"] = i
        s["current_arm"] = current_arm
        records.append(s)

res = pd.DataFrame(records)
print(f"  total (person, scenario) pairs evaluated: {len(res):,}")


# 5. Report
print("\n" + "=" * 70)
print("OVERALL THREE-TIER BREAKDOWN (4x / 8x dual threshold)")
print("=" * 70)
tier_counts = res["tier"].value_counts()
tier_pcts = (res["tier"].value_counts(normalize=True) * 100).round(1)
for t in ["reliable", "direction_only", "unreliable"]:
    n_t = tier_counts.get(t, 0)
    pct_t = tier_pcts.get(t, 0.0)
    print(f"  {t:16s}: {pct_t:5.1f}%  (n={n_t:,})")
print(f"\n  (for comparison: the OLD single-threshold version routed "
    f"everything past 4x straight to 'unreliable', which is why the "
    f"previous audit reported 60.8% unreliable with almost nothing in "
    f"'direction_only')")

print("\n" + "=" * 70)
print("OVERALL UNRELIABLE RATE")
print("=" * 70)
overall_rate = res["unreliable"].mean()
print(f"  {overall_rate*100:.1f}% of all (person, scenario) pairs flagged unreliable "
      f"(n={len(res):,})")

print("\n" + "=" * 70)
print("BY TARGET ARM (intervention)")
print("=" * 70)
by_arm = res.groupby("tgt")["unreliable"].agg(["mean", "count"])
by_arm["label"] = by_arm.index.map(ARM_LABELS)
by_arm = by_arm[["label", "mean", "count"]]
by_arm["mean"] = (by_arm["mean"] * 100).round(1)
print(by_arm.to_string())

print("\n" + "=" * 70)
print("BY NUMBER OF HEALTHY BEHAVIOURS IN TARGET ARM")
print("=" * 70)
by_nh = res.groupby("n_healthy")["unreliable"].agg(["mean", "count"])
by_nh["mean"] = (by_nh["mean"] * 100).round(1)
print(by_nh.to_string())

print("\n" + "=" * 70)
print("BY CURRENT ARM (starting lifestyle)")
print("=" * 70)
by_cur = res.groupby("current_arm")["unreliable"].agg(["mean", "count"])
by_cur["mean"] = (by_cur["mean"] * 100).round(1)
print(by_cur.to_string())

print("\n" + "=" * 70)
print("PER-PERSON: how many of their available scenarios are unreliable?")
print("=" * 70)
per_person = res.groupby("person_idx").agg(
    n_scenarios=("unreliable", "size"),
    n_unreliable=("unreliable", "sum"),
)
per_person["frac_unreliable"] = per_person["n_unreliable"] / per_person["n_scenarios"]
print(f"  people with 0 available scenarios (already all-healthy): "
    f"{(current_arm_arr == 7).sum():,}")
print(f"  people with >=1 available scenario: {len(per_person):,}")
print(f"  of those, fraction with ALL scenarios unreliable: "
    f"{(per_person['frac_unreliable'] == 1.0).mean()*100:.1f}%")
print(f"  of those, fraction with 0 scenarios unreliable (fully clean): "
    f"{(per_person['frac_unreliable'] == 0.0).mean()*100:.1f}%")
print(f"  mean fraction of a person's scenarios flagged unreliable: "
    f"{per_person['frac_unreliable'].mean()*100:.1f}%")

print("\n" + "=" * 70)
print("BASELINE-DEPENDENCE CHECK (should be roughly flat across baseline_risk")
print("bins after the logit-scale fix -- this re-verifies image1-4 anecdotes)")
print("=" * 70)
res["baseline_bin"] = pd.cut(
    res["baseline_risk"], bins=[0, 0.03, 0.05, 0.08, 0.12, 0.20, 1.0],
    labels=["<3%", "3-5%", "5-8%", "8-12%", "12-20%", ">20%"])
by_baseline = res.groupby("baseline_bin")["unreliable"].agg(["mean", "count"])
by_baseline["mean"] = (by_baseline["mean"] * 100).round(1)
print(by_baseline.to_string())

print("\n" + "=" * 70)
print("MECHANISM CHECK: is the baseline gradient above real grf-variance")
print("behaviour, or an artefact of the /slope conversion to logit scale?")
print("=" * 70)
valid = res.dropna(subset=["baseline_risk", "delta_se", "delta_se_logit"])
print(f"  n valid rows for this check: {len(valid):,} (dropped "
    f"{len(res) - len(valid):,} rows with NaN baseline_risk/delta_se, "
    f"e.g. from missing sleep_hrs/alc_curr)")

corr_raw = np.corrcoef(valid["delta_se"], valid["baseline_risk"])[0, 1]
corr_logit = np.corrcoef(valid["delta_se_logit"], valid["baseline_risk"])[0, 1]
print(f"\n  Correlation(delta_se [RAW, probability-scale, pre-conversion], "
    f"baseline_risk) = {corr_raw:+.3f}")
print(f"  Correlation(delta_se_logit [CONVERTED via /slope], "
    f"baseline_risk)      = {corr_logit:+.3f}")


print("\n" + "=" * 70)
print("BASELINE_RISK DISTRIBUTION (how close to the 0%/100% boundary do")
print("people actually sit? -- the /slope artefact above matters most")
print("near the boundary, so this tells us how big a practical problem it is)")
print("=" * 70)
print(valid["baseline_risk"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_string())
n_below_2pct = (valid["baseline_risk"] < 0.02).sum()
n_below_5pct = (valid["baseline_risk"] < 0.05).sum()
print(f"\n  rows with baseline_risk < 2%: {n_below_2pct:,} "
    f"({n_below_2pct/len(valid)*100:.1f}% of valid rows)")
print(f"  rows with baseline_risk < 5%: {n_below_5pct:,} "
    f"({n_below_5pct/len(valid)*100:.1f}% of valid rows)")

print("\nSaving full per-(person,scenario) results to outputs/unreliable_diagnostic.parquet")
res.to_parquet(os.path.join(OUTPUT_DIR, "unreliable_diagnostic.parquet"), index=False)
print("Done.")