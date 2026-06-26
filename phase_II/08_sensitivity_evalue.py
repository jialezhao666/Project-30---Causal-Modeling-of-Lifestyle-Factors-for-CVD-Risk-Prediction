import os
import numpy as np
import pandas as pd
import joblib

PHASE2_DIR = os.path.expanduser('~/my_ukb_thesis/phase_II')
OUTPUT_DIR = os.path.join(PHASE2_DIR, 'outputs')

CONFOUNDERS = [
    'age_defined_baseline', 'genetic_sex', 'BMI', 'uni_degree',
    'FH_cvd_f', 'FH_cvd_m', 'FH_cvd_sib', 'mental_doctor', 'alc_curr',
]
TREATMENTS = ['smk_curr', 'PA_active', 'sleep_adequate']
OUTCOME = 'def_CVD_AF_HF_AFTER'

ARM_LABELS = {
    1: 'no_smk only',
    2: 'PA only',
    3: 'no_smk + PA',
    4: 'sleep only',
    5: 'no_smk + sleep',
    6: 'PA + sleep',
    7: 'all three (no_smk+PA+sleep)',
}


# ============================================================
# Step 1 — recover p0 (observed CVD rate in the all-unhealthy
# reference arm, T=0) directly from Split B — same logic as in
# 03_multiarm_joint.py's "CVD rate per arm" sanity check.
# ============================================================

print("=" * 68)
print("STEP 1: Recovering baseline event rate p0 (arm 0, all-unhealthy)")
print("=" * 68)

df_B = pd.read_parquet(os.path.join(OUTPUT_DIR, 'split_B_phase2.parquet'))

t_imputer = joblib.load(os.path.join(OUTPUT_DIR, 'treatment_imputer.pkl'))
T_df = pd.DataFrame(t_imputer.transform(df_B[TREATMENTS]),
                     columns=TREATMENTS, index=df_B.index)
for c in TREATMENTS:
    T_df[c] = np.clip(np.round(T_df[c]), 0, 1)

no_smk = (1 - T_df['smk_curr']).astype(int)
pa = T_df['PA_active'].astype(int)
sleep = T_df['sleep_adequate'].astype(int)
T_combined = no_smk * 1 + pa * 2 + sleep * 4

mask_arm0 = (T_combined == 0)
n_arm0 = int(mask_arm0.sum())
p0 = float(df_B.loc[mask_arm0, OUTCOME].mean())

print(f"  Arm 0 (all-unhealthy) n = {n_arm0:,}")
print(f"  Observed CVD rate p0 = {p0:.4f}")

if n_arm0 < 100:
    print("  WARNING: arm 0 sample size is small — p0 estimate may be unstable.")


# ============================================================
# Step 2 — load NB3 per-arm CATE summary (risk differences)
# ============================================================

print("\n" + "=" * 68)
print("STEP 2: Loading per-arm CATE summary")
print("=" * 68)

summary = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_summary.parquet'))
print(summary[['arm', 'label', 'mean_cate', 'ci_low', 'ci_high']])


# ============================================================
# Step 3 — convert risk difference (CATE) to approximate risk
# ratio, then compute E-value (VanderWeele & Ding 2017)
#
#   RR ≈ (p0 + RD) / p0
#
# This is the standard approximation used when only a risk
# difference and the reference-arm baseline risk are available
# (see e.g. VanderWeele 2020, "Optimal approximate conversions
# of odds ratios and hazard ratios to risk ratios").
# ============================================================

print("\n" + "=" * 68)
print("STEP 3: Converting RD -> approximate RR, then E-value")
print("=" * 68)


def evalue(rr):
    """E-value for a risk ratio (VanderWeele & Ding, 2017)."""
    if rr is None or np.isnan(rr) or rr <= 0:
        return np.nan
    if rr >= 1:
        return rr + np.sqrt(rr * (rr - 1))
    else:
        rr_inv = 1 / rr
        return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))


def rd_to_rr(rd, p0):
    """Approximate RR from a risk difference and reference-arm baseline risk."""
    treated_risk = p0 + rd
    if treated_risk <= 0:
        # RD more negative than p0 itself is not possible for a real
        # probability; clip to a tiny positive value to avoid RR <= 0.
        treated_risk = 1e-6
    return treated_risk / p0


rows = []
for _, r in summary.iterrows():
    arm = int(r['arm'])
    rd = r['mean_cate']
    ci_low = r['ci_low']   # more negative (further from 0)
    ci_high = r['ci_high']  # closer to 0 (or crossing it)

    rr_point = rd_to_rr(rd, p0)
    # E-value for the CI bound closer to the null (least protective effect)
    rr_ci_bound = rd_to_rr(ci_high, p0)

    ev_point = evalue(rr_point)
    ev_ci = evalue(rr_ci_bound)

    rows.append({
        'arm': arm,
        'label': ARM_LABELS[arm],
        'rd_point': rd,
        'rd_ci_bound_near_null': ci_high,
        'rr_point': rr_point,
        'rr_ci_bound_near_null': rr_ci_bound,
        'evalue_point': ev_point,
        'evalue_ci_bound': ev_ci,
    })

    print(f"\n  Arm {arm} [{ARM_LABELS[arm]}]")
    print(f"    RD (point)        = {rd:+.4f}  ->  RR ≈ {rr_point:.4f}  ->  E-value = {ev_point:.3f}")
    print(f"    RD (CI bound near null) = {ci_high:+.4f}  ->  RR ≈ {rr_ci_bound:.4f}  ->  E-value = {ev_ci:.3f}")

evalue_df = pd.DataFrame(rows)
evalue_df.to_csv(os.path.join(OUTPUT_DIR, 'nb3_evalues.csv'), index=False)

print("\n" + "=" * 68)
print("Saved: nb3_evalues.csv")
print("=" * 68)

print("""
Interpretation reminder:
- E-value (point) = minimum risk-ratio strength an unmeasured confounder
  would need with BOTH treatment and outcome to fully explain away the
  estimated effect for that arm.
- E-value (CI bound) = minimum strength needed to shift the CI bound
  closest to the null all the way to the null itself.
- Larger E-value = more robust to unmeasured confounding.
- This RD -> RR conversion is approximate (depends on the chosen
  reference arm's baseline risk, p0); report this as a limitation of
  the sensitivity analysis, consistent with the approximate nature of
  E-values generally.
""")