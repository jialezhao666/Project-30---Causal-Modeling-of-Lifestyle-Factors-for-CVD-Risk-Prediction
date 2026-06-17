import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/my_ukb_thesis'))
from const_paths import BASE_PATH, SAVE_DIR

OUTPUT_DIR  = os.path.expanduser('~/my_ukb_thesis/phase_II/outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTCOME = 'def_CVD_AF_HF_AFTER'

# --- helper functions for deriving treatments ---
def _clip_neg_na(x):
    """UKB negative codes (-1 don't know, -3 prefer not answer) to NaN."""
    c = x.astype(float)
    return c.where(c >= 0)

def smk_current(col):
    """Current smoker: 20116 == 2."""
    c = _clip_neg_na(col)
    return np.where(c.isna(), np.nan, (c == 2).astype(float))

def pa_active(mod_days, mod_mins, vig_days, vig_mins):
    """PA active: moderate >=150 min/wk OR vigorous >=75 min/wk."""
    idx = mod_days.index
    md = _clip_neg_na(mod_days)
    mm = _clip_neg_na(mod_mins)
    vd = _clip_neg_na(vig_days)
    vm = _clip_neg_na(vig_mins)
    mod_total = pd.Series(np.where(md == 0, 0.0, md * mm), index=idx)
    vig_total = pd.Series(np.where(vd == 0, 0.0, vd * vm), index=idx)
    active = pd.Series(
        ((mod_total >= 150) | (vig_total >= 75)).astype(float), index=idx)
    active[md.isna() & vd.isna()] = np.nan
    active[(((md > 0) & mm.isna()) | ((vd > 0) & vm.isna())) & (active != 1)] = np.nan
    return active

def sleep_adequate(col):
    """Adequate sleep: >= 7 hours."""
    c = _clip_neg_na(col)
    return np.where(c.isna(), np.nan, (c >= 7).astype(float))

# 1. Load data sources

print("=" * 70)
print("Cohort Comparison Table")
print("=" * 70)

file_exposure = os.path.join(BASE_PATH, 'group_1_clean_filtered_imputed_dataset_df.tsv')
file_outcome = os.path.join(BASE_PATH, 'group_1_outcomes_df_without_qc_df.tsv')
file_tab = os.path.join(BASE_PATH, 'ukb_tabular_data_causal_analysis.tsv')
file_splitB = os.path.join(OUTPUT_DIR, 'split_B_phase2.parquet')

# --- 1a. Full UKB: exposure + outcome + tabular ---
print("\n Loading Full UKB data")

exp_cols = ['eid', 'genetic_sex', 'age_defined_baseline',
            '21001-0.0',                        # BMI
            '6138.1',                            # uni_degree
            '2090-0.0',                          # mental_doctor
            '20107.1', '20107.2',                # FH father
            '20110.1', '20110.2',                # FH mother
            '20111.1', '20111.2']                # FH sibling

# check which columns actually exist
exp_header = pd.read_csv(file_exposure, sep='\t', nrows=0).columns.tolist()
exp_cols_avail = [c for c in exp_cols if c in exp_header]
exp_cols_miss = [c for c in exp_cols if c not in exp_header]
if exp_cols_miss:
    print(f"missing from exposure file: {exp_cols_miss}")

exp = pd.read_csv(file_exposure, sep='\t', usecols=exp_cols_avail)
print(f"  exposure rows: {len(exp):,}")

out = pd.read_csv(file_outcome, sep='\t', usecols=['eid', OUTCOME])
print(f"  outcome rows:  {len(out):,}")

tab_cols = ['eid', '20116-0.0', '20117-0.0',
            '884-0.0', '894-0.0', '904-0.0', '914-0.0',
            '1160-0.0',
            # imaging visit fields (for identifying imaging participants)
            '20116-2.0', '884-2.0', '894-2.0', '904-2.0', '914-2.0', '1160-2.0']
tab = pd.read_csv(file_tab, sep='\t', usecols=tab_cols)
print(f"  tabular rows:  {len(tab):,}")

# merge all three into one DataFrame
full = exp.merge(out, on='eid', how='inner')
full = full.merge(tab, on='eid', how='left')

# --- derive confounders ---
full['BMI'] = _clip_neg_na(full['21001-0.0'])  # clip negatives to NaN

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

# Combined FH: any first-degree relative with CVD (father OR mother OR sibling)
full['FH_cvd_any'] = ((full['FH_cvd_f'] == 1) |
                    (full['FH_cvd_m'] == 1) |
                    (full['FH_cvd_sib'] == 1)).astype(float)
full.loc[full['FH_cvd_f'].isna() &
        full['FH_cvd_m'].isna() &
        full['FH_cvd_sib'].isna(), 'FH_cvd_any'] = np.nan

alc = _clip_neg_na(full['20117-0.0'])
full['alc_curr'] = (alc == 2).astype(float)
full.loc[alc.isna(), 'alc_curr'] = np.nan

# --- derive treatment indicators (baseline) ---
# PA_active: load from Phase I split parquets (field 22036)
print("  Loading PA_active (field 22036) from Phase I split parquets...")
pa_parts = []
for f in ['split_A_explore.parquet', 'split_B_train.parquet', 'split_C_test.parquet']:
    p = os.path.join(SAVE_DIR, f)
    if os.path.exists(p):
        pa_parts.append(pd.read_parquet(p, columns=['eid', 'PA_active']))
pa_all = pd.concat(pa_parts, ignore_index=True).drop_duplicates('eid')
full = full.merge(pa_all, on='eid', how='left')
print(f"  PA_active coverage: {full['PA_active'].notna().sum():,} / {len(full):,}")

# smk_curr: 20116-0.0 == 2 is the exact Phase I definition
# sleep_adequate: 1160-0.0 >= 7 is the exact Phase I definition
full['smk_curr'] = smk_current(full['20116-0.0'])
full['sleep_adequate'] = sleep_adequate(full['1160-0.0'])

# --- identify imaging visit participants ---
# anyone with at least one -2.0 behaviour field non-null
img_fields = ['20116-2.0', '884-2.0', '894-2.0', '904-2.0', '914-2.0', '1160-2.0']
split_eids = set()
# check all splits in case some imaging participants were excluded from Split B QC filtering
for f in ['split_A_explore.parquet', 'split_B_train.parquet', 'split_C_test.parquet']:
    p = os.path.join(SAVE_DIR, f)
    if os.path.exists(p):
        split_eids.update(pd.read_parquet(p, columns=['eid'])['eid'].tolist())
full = full[full['eid'].isin(split_eids)].copy()
print(f"  Full UKB After QC filter: {len(full):,}")
full['has_imaging'] = full[img_fields].notna().any(axis=1)
print(f"\n  Imaging visit participants: {full['has_imaging'].sum():,}")


# --- 1b. Split B ---
print("\nLoading Split B Data")
splitB = pd.read_parquet(file_splitB)
print(f"  Split B rows: {len(splitB):,}")


# 2. Compute summary statistics

def cohort_summary(data, label, use_parquet_cols=False):
    """
    Compute cohort-level statistics.
    use_parquet_cols: if True, use Split B parquet column names directly.
    """
    n = len(data)
    stats = {'Cohort': label, 'N': n}

    # --- continuous ---
    stats['Age, mean'] = data['age_defined_baseline'].mean()
    stats['Age, SD']   = data['age_defined_baseline'].std()

    bmi_col = 'BMI'
    stats['BMI, mean'] = data[bmi_col].mean()
    stats['BMI, SD']   = data[bmi_col].std()

    # --- binary (%) ---
    stats['Male (%)'] = data['genetic_sex'].mean() * 100

    if use_parquet_cols:
        # Split B parquet: smk_curr=1 means current smoker
        stats['Current smoker (%)'] = data['smk_curr'].mean() * 100
        stats['PA active (%)'] = data['PA_active'].mean() * 100
        stats['Adequate sleep (%)'] = data['sleep_adequate'].mean() * 100
    else:
        stats['Current smoker (%)'] = pd.Series(data['smk_curr']).mean() * 100
        stats['PA active (%)'] = data['PA_active'].mean() * 100
        stats['Adequate sleep (%)'] = pd.Series(data['sleep_adequate']).mean() * 100

    # confounders
    stats['University degree (%)']        = data['uni_degree'].mean() * 100
    stats['Mental health visit (%)']      = data['mental_doctor'].mean() * 100
    stats['FH CVD any relative (%)']     = data['FH_cvd_any'].mean() * 100
    stats['Current drinker (%)']          = data['alc_curr'].mean() * 100

    # outcome
    stats['CVD events (%)'] = data[OUTCOME].mean() * 100

    return stats

# Full UKB data
s_full = cohort_summary(full, f'Full UKB (n={len(full):,})')

# Split B — derive FH_cvd_any from existing parquet columns
splitB['FH_cvd_any'] = ((splitB['FH_cvd_f'] == 1) |
                        (splitB['FH_cvd_m'] == 1) |
                        (splitB['FH_cvd_sib'] == 1)).astype(float)
splitB.loc[splitB['FH_cvd_f'].isna() &
        splitB['FH_cvd_m'].isna() &
        splitB['FH_cvd_sib'].isna(), 'FH_cvd_any'] = np.nan

# Split B — parquet already has all 9 confounders + treatments (raw, pre-imputation)
# sleep_adequate was derived from sleep_hrs in NB1 before saving
s_splitB = cohort_summary(splitB,
                        f'Split B (n={len(splitB):,})',
                        use_parquet_cols=True)

# Imaging visit subset of Full UKB
img = full[full['has_imaging']].copy()
s_img = cohort_summary(img, f'Imaging visit (n={len(img):,})')


# 3. Format and save
results = pd.DataFrame([s_full, s_splitB, s_img]).set_index('Cohort').T

# format for display
def fmt_row(row_name, row_data, fmt_str):
    return {col: fmt_str.format(v) for col, v in row_data.items()}

display_rows = []
for idx_name in results.index:
    if idx_name == 'N':
        display_rows.append(
            {col: f'{int(v):,}' for col, v in results.loc[idx_name].items()})
    elif 'SD' in idx_name:
        continue  # merged with mean row
    elif 'mean' in idx_name:
        base = idx_name.replace(', mean', '')
        sd_key = f'{base}, SD'
        row = {}
        for col in results.columns:
            m = results.loc[idx_name, col]
            s = results.loc[sd_key, col] if sd_key in results.index else np.nan
            if np.isnan(m):
                row[col] = '—'
            else:
                row[col] = f'{m:.1f} ± {s:.1f}'
        display_rows.append(row)
        # rename index
        idx_name = f'{base} (mean ± SD)'
    elif '(%)' in idx_name:
        row = {}
        for col in results.columns:
            v = results.loc[idx_name, col]
            row[col] = f'{v:.1f}' if not np.isnan(v) else '—'
        display_rows.append(row)
    else:
        continue

# build display index
disp_idx = []
for idx_name in results.index:
    if idx_name == 'N':
        disp_idx.append('N')
    elif 'SD' in idx_name:
        continue
    elif 'mean' in idx_name:
        disp_idx.append(idx_name.replace(', mean', ' (mean ± SD)'))
    elif '(%)' in idx_name:
        disp_idx.append(idx_name)
    else:
        continue

display_df = pd.DataFrame(display_rows, index=disp_idx)

# print
print("\n" + "=" * 90)
print("COHORT CHARACTERISTICS TABLE")
print("=" * 90)

col_w = max(len(c) for c in display_df.columns) + 2
row_w = max(len(r) for r in display_df.index) + 2

header = f'{"Characteristic":<{row_w}}'
for col in display_df.columns:
    header += f'{col:>{col_w}}'
print(header)
print('-' * len(header))

for idx_name, row in display_df.iterrows():
    line = f'{idx_name:<{row_w}}'
    for col in display_df.columns:
        line += f'{row[col]:>{col_w}}'
    print(line)

# save both formats
results.to_csv(os.path.join(OUTPUT_DIR, 'cohort_characteristics.csv'))
print(f"\nSaved: cohort_characteristics.csv  (machine-readable)")

# save formatted table as txt
txt_path = os.path.join(OUTPUT_DIR, 'cohort_characteristics.txt')
with open(txt_path, 'w') as f:
    f.write("COHORT CHARACTERISTICS TABLE\n")
    f.write("=" * 90 + "\n")
    f.write(header + "\n")
    f.write('-' * len(header) + "\n")
    for idx_name, row in display_df.iterrows():
        line = f'{idx_name:<{row_w}}'
        for col in display_df.columns:
            line += f'{row[col]:>{col_w}}'
        f.write(line + "\n")
print(f"Saved: cohort_characteristics.txt  (formatted)")


print("  - PA active: all three cohorts use field 22036 (self-reported guideline")
print("    attainment) at baseline, consistent with baseline causal forest analysis.")
print("    The minute-based approximation (884×894≥150 OR 904×914≥75) is used")
print("    only in the longitudinal MSM analysis where field 22036")
print("    has no imaging visit equivalent. Cohen's κ = 0.39 between definitions.")