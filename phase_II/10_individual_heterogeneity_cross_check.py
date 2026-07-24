import os
import numpy as np
import pandas as pd

PHASE2_DIR = os.path.expanduser('~/my_ukb_thesis/phase_II')
OUTPUT_DIR = os.path.join(PHASE2_DIR, 'outputs')

SINGLE_ARMS = {
    1: ('cate_arm1_vs0', 'Smoking cessation only'),
    2: ('cate_arm2_vs0', 'Physical activity only'),
    4: ('cate_arm4_vs0', 'Adequate sleep only')}

# Step 1 — Load data

print("=" * 68)
print("STEP 1: Loading NB3 individual-level CATE file")
print("=" * 68)

joint_full = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_full.parquet'))
summary = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_summary.parquet'))

EXPECTED_N = 321188
assert joint_full.shape[0] == EXPECTED_N, (
    f"Expected joint_cate_full.parquet from 70% Phase II analysis set "
    f"({EXPECTED_N:,} rows), got {joint_full.shape[0]:,}. Re-run 03_multiarm_joint.")

if 'n' in summary.columns:
    n_unique = summary['n'].dropna().unique()
    print(f"  joint_cate_summary.parquet n values: {n_unique}")
    assert 321188 in n_unique, (
        "Expected joint_cate_summary.parquet from the 70% Phase II analysis set. "
        "Re-run 03_multiarm_joint.")

print(f"  joint_cate_full.parquet: {joint_full.shape[0]:,} individuals, "
    f"{joint_full.shape[1]} columns")
print(f"  Columns: {joint_full.columns.tolist()}")



# Step 2 — Population-average CATE per single-intervention arm
# (sanity check against the values already reported in the thesis)

print("\n" + "=" * 68)
print("STEP 2: Population-average CATE per arm (sanity check)")
print("=" * 68)

pop_avg = {}
for arm, (col, label) in SINGLE_ARMS.items():
    avg = summary.loc[summary['arm'] == arm, 'mean_cate'].values[0]
    pop_avg[arm] = avg
    print(f"  Arm {arm} [{label}]: population-average CATE = {avg:+.4f}")


# Step 3 — Direction-reversal rate per single-intervention arm
# (% of individuals whose own CATE sign differs from the population-average sign for that same arm)

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
        'n_total': len(vals)})
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
