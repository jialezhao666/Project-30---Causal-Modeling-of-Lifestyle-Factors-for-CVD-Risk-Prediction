"""
10_individual_heterogeneity_cross_check.py
=============================================
Extends the original NB3 individual-level diagnostic (which reported
35.3% direction-reversal for PA-only) to all THREE single-intervention
arms, so the claim "PA shows the most individual-level heterogeneity"
can be checked against smoking and sleep on the same metric, rather
than asserted from a PA-only number.

Reads: outputs/joint_cate_summary.parquet (population-average CATE per arm)
       outputs/joint_cate_full.parquet     (n=298,245 individual CATEs, 7 arms)

Single-intervention arms (per the T = no_smk + PA*2 + sleep*4 encoding):
    arm 1 = no_smk only   (smoking cessation)
    arm 2 = PA only
    arm 4 = sleep only

These are the three "single intervention vs all-unhealthy" comparisons —
the directly comparable set for a cross-intervention heterogeneity check.
"""

import os
import numpy as np
import pandas as pd

PHASE2_DIR = os.path.expanduser('~/my_ukb_thesis/phase_II')
OUTPUT_DIR = os.path.join(PHASE2_DIR, 'outputs')

SINGLE_ARMS = {
    1: ('cate_arm1_vs0', 'Smoking cessation only'),
    2: ('cate_arm2_vs0', 'Physical activity only'),
    4: ('cate_arm4_vs0', 'Adequate sleep only'),
}


# ============================================================
# Step 1 — Load data
# ============================================================

print("=" * 68)
print("STEP 1: Loading NB3 individual-level CATE file")
print("=" * 68)

joint_full = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_full.parquet'))
summary = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_summary.parquet'))

print(f"  joint_cate_full.parquet: {joint_full.shape[0]:,} individuals, "
    f"{joint_full.shape[1]} columns")
print(f"  Columns: {joint_full.columns.tolist()}")


# ============================================================
# Step 2 — Population-average CATE per single-intervention arm
# (sanity check against the values already reported in the thesis)
# ============================================================

print("\n" + "=" * 68)
print("STEP 2: Population-average CATE per arm (sanity check)")
print("=" * 68)

pop_avg = {}
for arm, (col, label) in SINGLE_ARMS.items():
    avg = summary.loc[summary['arm'] == arm, 'mean_cate'].values[0]
    pop_avg[arm] = avg
    print(f"  Arm {arm} [{label}]: population-average CATE = {avg:+.4f}")


# ============================================================
# Step 3 — Direction-reversal rate per single-intervention arm
# (% of individuals whose own CATE sign differs from the
#  population-average sign for that same arm)
# ============================================================

print("\n" + "=" * 68)
print("STEP 3: Direction-reversal rate — cross-intervention comparison")
print("=" * 68)

rows = []
for arm, (col, label) in SINGLE_ARMS.items():
    vals = joint_full[col].values
    pop_sign = np.sign(pop_avg[arm])
    reversed_mask = (np.sign(vals) != pop_sign) & (np.abs(vals) > 1e-6)
    pct_reversed = reversed_mask.mean() * 100

    rows.append({
        'arm': arm,
        'intervention': label,
        'pop_avg_cate': pop_avg[arm],
        'pct_direction_reversed': pct_reversed,
        'n_reversed': int(reversed_mask.sum()),
        'n_total': len(vals),
    })
    print(f"\n  Arm {arm} [{label}]")
    print(f"    Population-average direction: "
        f"{'protective (negative)' if pop_sign < 0 else 'harmful (positive)'}")
    print(f"    % of individuals with REVERSED direction: {pct_reversed:.1f}% "
        f"({int(reversed_mask.sum()):,} / {len(vals):,})")

result_df = pd.DataFrame(rows).sort_values('pct_direction_reversed', ascending=False)
result_df.to_csv(os.path.join(OUTPUT_DIR, 'cross_intervention_heterogeneity.csv'), index=False)

print("\n" + "=" * 68)
print("SUMMARY TABLE — sorted by reversal rate (highest first)")
print("=" * 68)
print(f"\n{'Intervention':<28} {'Pop-avg CATE':>14} {'%% Reversed':>12}")
print("-" * 56)
for _, r in result_df.iterrows():
    print(f"{r['intervention']:<28} {r['pop_avg_cate']:>+14.4f} "
        f"{r['pct_direction_reversed']:>11.1f}%")

print(f"\nSaved: cross_intervention_heterogeneity.csv")


# ============================================================
# Step 4 — Interpretation guidance
# ============================================================

print("\n" + "=" * 68)
print("INTERPRETATION")
print("=" * 68)

top = result_df.iloc[0]
others = result_df.iloc[1:]
gap_to_next = top['pct_direction_reversed'] - others['pct_direction_reversed'].max()

print(f"""
Highest reversal rate: {top['intervention']} ({top['pct_direction_reversed']:.1f}%)
Gap to next-highest: {gap_to_next:.1f} percentage points

If this gap is large (e.g. >5pp) and {top['intervention']} is Physical
Activity, this provides a FOURTH independent line of evidence (alongside
NB3 E-value, MSM main-analysis E-value, and HRS replication fragility)
supporting the claim that PA shows the weakest/most heterogeneous causal
evidence among the three interventions — strengthening the cross-method
triangulation argument in the Discussion.

If the gap is small (e.g. <5pp) and reversal rates are broadly similar
across all three interventions, this indicates that ALL THREE
interventions show comparable individual-level heterogeneity, and the
claim "PA is uniquely the most heterogeneous" should NOT be made on this
basis. In that case, this diagnostic should be reported as a general
finding about individual-level prediction uncertainty in causal forests
(relevant context for Phase III's reliability flagging), not as
additional evidence specifically singling out PA.

Either way, report the actual numbers for all three interventions in the
Methods/Results — do not report only the PA figure in isolation, as this
selectively reports one number without the comparator needed to judge
whether it is actually distinctive.
""")