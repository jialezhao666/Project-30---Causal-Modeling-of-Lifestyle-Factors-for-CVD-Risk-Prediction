import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/my_ukb_thesis'))
from const_paths import BASE_PATH, SAVE_DIR

OUTPUT_DIR = os.path.expanduser('~/my_ukb_thesis/phase_II/outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTCOME = 'def_CVD_AF_HF_AFTER'


def _clip_neg_na(x):
    c = x.astype(float)
    return c.where(c >= 0)


def smk_current(col):
    c = _clip_neg_na(col)
    return np.where(c.isna(), np.nan, (c == 2).astype(float))


def sleep_adequate(col):
    c = _clip_neg_na(col)
    return np.where(c.isna(), np.nan, (c >= 7).astype(float))


def smd(x1, x0):
    """Standardised mean difference — same convention as 06_msm.py."""
    m1, m0 = np.nanmean(x1), np.nanmean(x0)
    v1, v0 = np.nanvar(x1, ddof=1), np.nanvar(x0, ddof=1)
    pooled_sd = np.sqrt((v1 + v0) / 2)
    if pooled_sd == 0 or np.isnan(pooled_sd):
        return np.nan
    return (m1 - m0) / pooled_sd


# ============================================================
# Step 1 — Load + derive confounders/treatments, EXACT same logic
# as 07_cohort_characteristics.py, to guarantee a consistent
# has_imaging (n≈73,639) definition.
# ============================================================

print("=" * 68)
print("STEP 1: Loading full UKB cohort")
print("=" * 68)

file_exposure = os.path.join(BASE_PATH, 'group_1_clean_filtered_imputed_dataset_df.tsv')
file_outcome = os.path.join(BASE_PATH, 'group_1_outcomes_df_without_qc_df.tsv')
file_tab = os.path.join(BASE_PATH, 'ukb_tabular_data_causal_analysis.tsv')

exp_cols = ['eid', 'genetic_sex', 'age_defined_baseline',
            '21001-0.0', '6138.1', '2090-0.0',
            '20107.1', '20107.2', '20110.1', '20110.2', '20111.1', '20111.2']
exp_header = pd.read_csv(file_exposure, sep='\t', nrows=0).columns.tolist()
exp_cols_avail = [c for c in exp_cols if c in exp_header]
exp = pd.read_csv(file_exposure, sep='\t', usecols=exp_cols_avail)
out = pd.read_csv(file_outcome, sep='\t', usecols=['eid', OUTCOME])

tab_cols = ['eid', '20116-0.0', '20117-0.0',
            '884-0.0', '894-0.0', '904-0.0', '914-0.0', '1160-0.0',
            '20116-2.0', '884-2.0', '894-2.0', '904-2.0', '914-2.0', '1160-2.0']
tab = pd.read_csv(file_tab, sep='\t', usecols=tab_cols)

full = exp.merge(out, on='eid', how='inner')
full = full.merge(tab, on='eid', how='left')

# derive confounders (identical to 07)
full['BMI'] = _clip_neg_na(full['21001-0.0'])
full['uni_degree'] = (full['6138.1'] == 1).astype(float)
full.loc[full['6138.1'].isna(), 'uni_degree'] = np.nan
full['mental_doctor'] = _clip_neg_na(full['2090-0.0'])
full['mental_doctor'] = (full['mental_doctor'] == 1).astype(float)
full.loc[_clip_neg_na(full['2090-0.0']).isna(), 'mental_doctor'] = np.nan
full['FH_cvd_f'] = ((full['20107.1'] == 1) | (full['20107.2'] == 1)).astype(float)
full.loc[full['20107.1'].isna() & full['20107.2'].isna(), 'FH_cvd_f'] = np.nan
full['FH_cvd_m'] = ((full['20110.1'] == 1) | (full['20110.2'] == 1)).astype(float)
full.loc[full['20110.1'].isna() & full['20110.2'].isna(), 'FH_cvd_m'] = np.nan
full['FH_cvd_sib'] = ((full['20111.1'] == 1) | (full['20111.2'] == 1)).astype(float)
full.loc[full['20111.1'].isna() & full['20111.2'].isna(), 'FH_cvd_sib'] = np.nan
alc = _clip_neg_na(full['20117-0.0'])
full['alc_curr'] = (alc == 2).astype(float)
full.loc[alc.isna(), 'alc_curr'] = np.nan

# treatments
print("  Loading PA_active (field 22036) from Phase I split parquets...")
pa_parts = []
for f in ['split_A_explore.parquet', 'split_B_train.parquet', 'split_C_test.parquet']:
    p = os.path.join(SAVE_DIR, f)
    if os.path.exists(p):
        pa_parts.append(pd.read_parquet(p, columns=['eid', 'PA_active']))
pa_all = pd.concat(pa_parts, ignore_index=True).drop_duplicates('eid')
full = full.merge(pa_all, on='eid', how='left')
full['smk_curr'] = smk_current(full['20116-0.0'])
full['sleep_adequate'] = sleep_adequate(full['1160-0.0'])

# QC filter (same as 07: must be in one of the splits)
split_eids = set()
for f in ['split_A_explore.parquet', 'split_B_train.parquet', 'split_C_test.parquet']:
    p = os.path.join(SAVE_DIR, f)
    if os.path.exists(p):
        split_eids.update(pd.read_parquet(p, columns=['eid'])['eid'].tolist())
full = full[full['eid'].isin(split_eids)].copy()
print(f"  Full UKB after QC filter: {len(full):,}")

# has_imaging — EXACT same fields as 07
img_fields = ['20116-2.0', '884-2.0', '894-2.0', '904-2.0', '914-2.0', '1160-2.0']
full['has_imaging'] = full[img_fields].notna().any(axis=1)
imaging_cohort = full[full['has_imaging']].copy()
print(f"  Imaging behavioural cohort: {len(imaging_cohort):,} "
    f"(should match prior ~73,639 figure used elsewhere in thesis)")


# ============================================================
# Step 2 — Identify who HAS an imaging-visit date (53-2.0)
# ============================================================

print("\n" + "=" * 68)
print("STEP 2: Identifying imaging-date availability")
print("=" * 68)

img_dates = pd.read_csv(os.path.join(BASE_PATH, 'imaging_visit_date.tsv'), sep='\t')
img_dates = img_dates.rename(columns={'53-2.0': 'imaging_date'})
has_date_eids = set(img_dates.loc[img_dates['imaging_date'].notna(), 'eid'])

imaging_cohort['group'] = np.where(
    imaging_cohort['eid'].isin(has_date_eids), 'included', 'excluded')

print(imaging_cohort['group'].value_counts())
n_inc = (imaging_cohort['group'] == 'included').sum()
n_exc = (imaging_cohort['group'] == 'excluded').sum()
# Expected range updated for new source: file-level non-null count is
# ~80,767 (vs ~30,809 for the old 53-2.0 export); within the imaging
# behavioural cohort (has_imaging, n~73,639) the included count should
# track close to that file-level figure, not the old one.
if n_inc < 70000 or n_inc > 85000:
    print(f"  NOTE: included count ({n_inc:,}) differs from expected range for the "
        f"new imaging_visit_date.tsv source (~80,767 non-null file-wide) — "
        f"check whether has_imaging definition matches prior usage "
        f"before interpreting results.")


# ============================================================
# Step 3 — Table: included vs excluded, with SMD
# ============================================================

print("\n" + "=" * 68)
print("STEP 3: Baseline characteristics — included vs excluded (SMD)")
print("=" * 68)

imaging_cohort['FH_cvd_any'] = (
    (imaging_cohort['FH_cvd_f'] == 1) |
    (imaging_cohort['FH_cvd_m'] == 1) |
    (imaging_cohort['FH_cvd_sib'] == 1)
).astype(float)
imaging_cohort.loc[
    imaging_cohort['FH_cvd_f'].isna() &
    imaging_cohort['FH_cvd_m'].isna() &
    imaging_cohort['FH_cvd_sib'].isna(), 'FH_cvd_any'] = np.nan

compare_vars = [
    'age_defined_baseline', 'genetic_sex', 'BMI', 'uni_degree',
    'FH_cvd_any', 'mental_doctor', 'alc_curr',
    'smk_curr', 'PA_active', 'sleep_adequate', OUTCOME,
]

g_inc = imaging_cohort[imaging_cohort['group'] == 'included']
g_exc = imaging_cohort[imaging_cohort['group'] == 'excluded']

if len(g_exc) == 0:
    print(f"""
NOTE: excluded group is empty (included={len(g_inc):,}, excluded=0).
Skipping SMD computation; no output csv written for this run.
{'='*68}""")
    sys.exit(0)

rows = []
print(f"\n{'Variable':<25} {'Included':>15} {'Excluded':>15} {'SMD':>8}")
print("-" * 65)
for v in compare_vars:
    inc_mean = np.nanmean(g_inc[v].values)
    exc_mean = np.nanmean(g_exc[v].values)
    d = smd(g_inc[v].values, g_exc[v].values)
    flag = '  ← SMD>0.1' if abs(d) > 0.1 else ''
    rows.append({'variable': v, 'included_mean': inc_mean,
                'excluded_mean': exc_mean, 'smd': d})
    print(f"{v:<25} {inc_mean:>15.4f} {exc_mean:>15.4f} {d:>+8.3f}{flag}")

result_df = pd.DataFrame(rows)
result_df.to_csv(os.path.join(OUTPUT_DIR, 'imaging_date_selection_check.csv'), index=False)
print(f"\nSaved: imaging_date_selection_check.csv")

n_flagged = (result_df['smd'].abs() > 0.1).sum()
print(f"\n{'='*68}")
print(f"SUMMARY: {n_flagged} / {len(result_df)} variables show |SMD| > 0.1")
if n_flagged == 0:
    print("No evidence of meaningful selection bias on observed characteristics.")
elif n_flagged <= 2:
    print("Limited selection bias — report which variables differ in Limitations.")
else:
    print("Substantial selection bias — discuss explicitly as a limitation.")
print(f"{'='*68}")
