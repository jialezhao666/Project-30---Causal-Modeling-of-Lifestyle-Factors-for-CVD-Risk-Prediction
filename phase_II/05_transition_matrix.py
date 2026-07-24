import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~/my_ukb_thesis'))
from const_paths import BASE_PATH, SAVE_DIR

OUTPUT_DIR  = os.path.expanduser('~/my_ukb_thesis/phase_II/outputs')
FIGURES_DIR = os.path.expanduser('~/my_ukb_thesis/phase_II/figures')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

file_tab      = os.path.join(BASE_PATH, 'ukb_tabular_data_causal_analysis.tsv')
file_outcome  = os.path.join(BASE_PATH, 'group_1_outcomes_df_without_qc_df.tsv')
file_exposure = os.path.join(BASE_PATH, 'group_1_clean_filtered_imputed_dataset_df.tsv')
file_pa22_img = os.path.join(BASE_PATH, 'output_22036_imaging.tsv')  # field 22036-2.0 (imaging visit PA)
OUTCOME = 'def_CVD_AF_HF_AFTER'

core_cols = ['eid',
            '20116-0.0', '20116-2.0',
            '884-0.0',   '884-2.0',   # instance 2 retained for concordance analysis only
            '894-0.0',   '894-2.0',
            '904-0.0',   '904-2.0',
            '914-0.0',   '914-2.0',
            '1160-0.0',  '1160-2.0']


# 1. longitudinal fields + outcome 
print("Loading tabular (longitudinal fields)")
tab = pd.read_csv(file_tab, sep='\t', usecols=core_cols)
print(f"  tabular rows: {len(tab):,}")
out = pd.read_csv(file_outcome, sep='\t', usecols=['eid', OUTCOME])
df = tab.merge(out, on='eid', how='inner')
print(f"  merged rows : {len(df):,}")

# Load field 22036-2.0 
# This replaces the proxy definition (884×894 + 904×914) for imaging-visit PA.
# Baseline PA continues to use field 22036-0.0 (via PA_active in Phase I parquets).
pa22_img = pd.read_csv(file_pa22_img, sep='\t', usecols=['eid', '22036-2.0'])
df = df.merge(pa22_img, on='eid', how='left')
n_pa22 = df['22036-2.0'].notna().sum()
print(f"  22036-2.0 (imaging PA, official): {n_pa22:,} non-null "
    f"({n_pa22/len(df)*100:.1f}% of merged rows)")

# 2. sex + age from exposure file (full population)
print("\nLoading sex + age from group_1_clean_filtered_imputed_dataset_df.tsv")
exp_header = pd.read_csv(file_exposure, sep='\t', nrows=0).columns.tolist()
exp_want = ['eid', 'genetic_sex', 'age_defined_baseline']
exp_have = [c for c in exp_want if c in exp_header]
exp_miss = [c for c in exp_want if c not in exp_header]
if exp_miss:
    print(f"  missing  : {exp_miss}")
if len(exp_have) > 1:
    exp = pd.read_csv(file_exposure, sep='\t', usecols=exp_have)
    df = df.merge(exp, on='eid', how='left')

#  3. BMI + PA_active from Phase I parquets
def load_bmi_pa():
    """Returns (DataFrame[eid,BMI,PA_active], coverage_tag)."""
    candidates = [('split_A_explore.parquet', os.path.join(SAVE_DIR, 'split_A_explore.parquet')),
                ('split_B_train.parquet', os.path.join(SAVE_DIR, 'split_B_train.parquet')),
                ('split_C_test.parquet', os.path.join(SAVE_DIR, 'split_C_test.parquet'))]
    found = [(n, p) for n, p in candidates if os.path.exists(p)]
    if found:
        parts = []
        for name, path in found:
            try:
                d = pd.read_parquet(path, columns=['eid', 'BMI', 'PA_active'])
                print(f"  {name}: {len(d):,}")
                parts.append(d)
            except Exception as e:
                print(f"  skip {name}: {e}")
        if parts:
            comb = pd.concat(parts, ignore_index=True).drop_duplicates('eid')
            tag = 'full' if len(parts) == 3 else f'partial({len(parts)}/3 splits)'
            return comb, tag
    # fallback
    alt = os.path.join(OUTPUT_DIR, 'split_B_phase2.parquet')
    if os.path.exists(alt):
        d = pd.read_parquet(alt, columns=['eid', 'BMI', 'PA_active'])
        print(f"  fallback: split_B_phase2.parquet: {len(d):,}")
        return d, 'split_B_only'
    return None, 'none'

print("\nLoading BMI + PA_active from Phase I parquets")
bmi_pa, coverage = load_bmi_pa()
if bmi_pa is not None:
    print(f"  BMI: mean={bmi_pa['BMI'].mean():.2f}, std={bmi_pa['BMI'].std():.2f}, "
        f"min={bmi_pa['BMI'].min():.2f}, max={bmi_pa['BMI'].max():.2f}  (raw values)")
    df = df.merge(bmi_pa, on='eid', how='left')
    print(f"  after merge: {df.shape[0]:,} rows * {df.shape[1]} cols")
    print(f"  imaging-visit participants with BMI: {df['BMI'].notna().sum():,}")
else:
    coverage = 'none'
    print(f"no Phase I parquet found, BMI/PA_active analyses will skip")


# healthy-behaviour indicators 
def _clip_neg_na(x):
    c = x.astype(float)
    return c.where(c >= 0) # treat negative values as NA (per UKB coding: -1 don't know, -3 prefer not answer)

def smk_healthy(col):
    c = _clip_neg_na(col)
    return np.where(c.isna(), np.nan, (c != 2).astype(float))  # not current smoker (0 never, 1 previous, 2 current)

def sleep_healthy(col):
    c = _clip_neg_na(col)
    return np.where(c.isna(), np.nan, (c >= 7).astype(float)) # 7 or more hours per night is healthy

# pa proxy indicator from minute-based fields (884×894 + 904×914), candidate
def pa_healthy_approx(mod_days, mod_mins, vig_days, vig_mins):
    """
    Proxy PA indicator from minute-based fields (884×894 + 904×914).
    Used for PA proxy construction in concordance analyses only.
    The main transition and longitudinal analyses use PA_active at baseline
    and field 22036-2.0 at imaging visit.
    """
    idx = mod_days.index
    md = _clip_neg_na(mod_days)
    mm = _clip_neg_na(mod_mins)
    vd = _clip_neg_na(vig_days)
    vm = _clip_neg_na(vig_mins)

    mod_total = pd.Series(np.where(md == 0, 0.0, md * mm), index=idx)
    vig_total = pd.Series(np.where(vd == 0, 0.0, vd * vm), index=idx)

    active = pd.Series(
        ((mod_total >= 150) | (vig_total >= 75)).astype(float), index=idx)

    both_nan = md.isna() & vd.isna()
    active[both_nan] = np.nan

    mod_uncertain = (md > 0) & mm.isna()
    vig_uncertain = (vd > 0) & vm.isna()
    uncertain = mod_uncertain | vig_uncertain
    active[uncertain & (active != 1)] = np.nan

    return active.values


def pa_official_imaging(col_22036_2):
    """
    Official imaging-visit PA indicator from field 22036-2.0.
    1 = meets PA guideline (self-reported), 0 = does not meet, NaN = missing.
    Definitionally consistent with the baseline indicator (field 22036-0.0
    via PA_active in Phase I parquets): both use the same self-report
    guideline-attainment question, making longitudinal comparison valid.
    """
    c = col_22036_2.astype(float)
    result = np.where(c.isna(), np.nan, (c == 1).astype(float))
    return result

pa_b_off = pd.Series(df['PA_active'], index=df.index).astype(float)
pa_i_off = pd.Series(pa_official_imaging(df['22036-2.0']), index=df.index)

H = {
    'Quit smoking': (smk_healthy(df['20116-0.0']),smk_healthy(df['20116-2.0'])),
    'Increase PA': (pa_b_off,pa_i_off),
    'Adequate sleep': (sleep_healthy(df['1160-0.0']),sleep_healthy(df['1160-2.0']))}


# SUB-ANALYSIS SETUP: exclude participants whose CVD/AF/HF event occurred before their imaging visit 
print("\n" + "=" * 78)
print("SUB-ANALYSIS SETUP: excluding participants with CVD event before imaging")
print("=" * 78)

events_f = os.path.join(BASE_PATH, 'bhf_all_individuals_plus_cvd_events.csv')
events = pd.read_csv(events_f, sep='\t', index_col=0,
                    usecols=['eid', 'defined_baseline_date',
                            'def_CVD_AF_HF_AFTER_days_from_baseline'])
events = events.reset_index()
events['defined_baseline_date'] = pd.to_datetime(events['defined_baseline_date'])
events['def_CVD_AF_HF_AFTER_date'] = (
    events['defined_baseline_date'] +
    pd.to_timedelta(events['def_CVD_AF_HF_AFTER_days_from_baseline'], unit='D'))

img_dates = pd.read_csv(os.path.join(BASE_PATH, 'imaging_visit_date.tsv'),
                        sep='\t').rename(columns={'53-2.0': 'imaging_date'})
img_dates['imaging_date'] = pd.to_datetime(img_dates['imaging_date'])

events = events.merge(img_dates, on='eid', how='left')
events['event_before_imaging'] = (
    events['def_CVD_AF_HF_AFTER_date'].notna() &
    events['imaging_date'].notna() &
    (events['def_CVD_AF_HF_AFTER_date'] < events['imaging_date']))
exclude_eids = set(events.loc[events['event_before_imaging'], 'eid'])
print(f"  Participants with imaging visit date  : {events['imaging_date'].notna().sum():,}")
print(f"  Participants with CVD event date      : {events['def_CVD_AF_HF_AFTER_date'].notna().sum():,}")
print(f"  Participants with event BEFORE imaging: {len(exclude_eids):,}  <- excluded from df_sub")

df_sub = df[~df['eid'].isin(exclude_eids)].copy()
print(f"  df (main, full imaging pool)          : {len(df):,}")
print(f"  df_sub (excl. pre-imaging CVD events) : {len(df_sub):,}  "
    f"({len(df) - len(df_sub):,} excluded)")

# 1. Full 2×2 transition matrix per behaviour
# Run on both df (main) and df_sub (sub-analysis) for comparison.

def build_transition_matrix(data, H_dict, label):
    print("\n" + "=" * 78)
    print(f"TRANSITION MATRIX [{label}]  (baseline -> imaging visit)")
    print("=" * 78)

    rows = []
    for name, (b_arr, i_arr) in H_dict.items():
        b = pd.Series(b_arr, index=data.index)
        i = pd.Series(i_arr, index=data.index)
        both = b.notna() & i.notna()
        b2, i2 = b[both], i[both]
        y2 = data.loc[both, OUTCOME]
        n = len(b2)

        cells = {
            'HH': (b2 == 1) & (i2 == 1),
            'HU': (b2 == 1) & (i2 == 0),
            'UH': (b2 == 0) & (i2 == 1),
            'UU': (b2 == 0) & (i2 == 0),
            }
        cell_n = {k: int(m.sum()) for k, m in cells.items()}
        cell_ev = {k: int(y2[m].sum()) for k, m in cells.items()}
        cell_rate = {k: y2[m].mean() if cell_n[k] else np.nan for k, m in cells.items()}

        print(f"\n### {name}  (both-visit n = {n:,})")
        print(f"                         imaging H            imaging U          row sum")
        print(f"  baseline HEALTHY      {cell_n['HH']:>7,} ({cell_n['HH']/n*100:4.1f}%)   "
            f"{cell_n['HU']:>7,} ({cell_n['HU']/n*100:4.1f}%)   "
            f"{cell_n['HH'] + cell_n['HU']:>7,}")
        print(f"  baseline UNHEALTHY    {cell_n['UH']:>7,} ({cell_n['UH']/n*100:4.1f}%)   "
            f"{cell_n['UU']:>7,} ({cell_n['UU']/n*100:4.1f}%)   "
            f"{cell_n['UH'] + cell_n['UU']:>7,}")
        print(f"  col sum               {cell_n['HH'] + cell_n['UH']:>7,}            "
            f"{cell_n['HU'] + cell_n['UU']:>7,}            {n:>7,}")
        print(f"  CVD events / rate per cell:")
        for k in ['HH', 'HU', 'UH', 'UU']:
            print(f"    {k}: events={cell_ev[k]:>5,}  "
                f"rate={cell_rate[k]:.4f}  (n={cell_n[k]:,})")

        for k in ['HH', 'HU', 'UH', 'UU']:
            rows.append({
                'cohort': label,
                'intervention' : name,
                'cell' : k,
                'baseline': k[0],
                'imaging' : k[1],
                'n' : cell_n[k],
                'pct_of_total' : cell_n[k] / n * 100,
                'cvd_events' : cell_ev[k],
                'cvd_rate' : cell_rate[k],})
    return pd.DataFrame(rows)

# main analysis (full imaging pool, as before)
mat_main = build_transition_matrix(df, H, 'main (full imaging pool)')

# sub-analysis (excludes pre-imaging CVD events)
pa_b_off_sub = pd.Series(df_sub['PA_active'], index=df_sub.index).astype(float)
pa_i_off_sub = pd.Series(pa_official_imaging(df_sub['22036-2.0']), index=df_sub.index)

H_sub = {
    'Quit smoking': (smk_healthy(df_sub['20116-0.0']),smk_healthy(df_sub['20116-2.0'])),
    'Increase PA': ( pa_b_off_sub,pa_i_off_sub),
    'Adequate sleep': (sleep_healthy(df_sub['1160-0.0']),sleep_healthy(df_sub['1160-2.0']))}

mat_sub = build_transition_matrix(df_sub, H_sub, 'sub (excl. pre-imaging CVD events)')

mat = pd.concat([mat_main, mat_sub], ignore_index=True)
mat.to_parquet(os.path.join(OUTPUT_DIR, 'transition_matrix.parquet'), index=False)
mat.to_csv(os.path.join(OUTPUT_DIR, 'transition_matrix.csv'), index=False)
print(f"\nSaved: transition_matrix.{{parquet,csv}}  ({len(mat)} rows, both cohorts)")

# comparison: UH cell CVD rate, main vs sub
print("\n" + "=" * 78)
print("KEY COMPARISON: UH (improved) cell CVD rate, main vs sub-analysis")
print("  If sub-analysis rate drops toward UU rate, this supports the")
print("  hypothesis that reverse-causation was partly driven by reactive")
print("  behaviour change following an undetected pre-imaging CVD event.")
print("=" * 78)
for name in H.keys():
    uh_main = mat_main[(mat_main['intervention'] == name) & (mat_main['cell'] == 'UH')]
    uh_sub  = mat_sub[(mat_sub['intervention'] == name) & (mat_sub['cell'] == 'UH')]
    uu_main = mat_main[(mat_main['intervention'] == name) & (mat_main['cell'] == 'UU')]
    if len(uh_main) and len(uh_sub):
        print(f"  {name:<16} UH rate: main={uh_main['cvd_rate'].values[0]:.4f} "
            f"-> sub={uh_sub['cvd_rate'].values[0]:.4f}   "
            f"(UU rate for reference: {uu_main['cvd_rate'].values[0]:.4f})")


# 2. PA definition concordance (three analyses)
#
#  (A) LONGITUDINAL — 22036-0.0 (baseline) vs 22036-2.0 (imaging)
#      Both official fields, same instrument. Reflects TEMPORAL STABILITY
#      of PA behaviour between visits. This is the primary concordance
#      reported in Methods since imaging-visit PA now uses 22036-2.0.
#
#  (B) BASELINE construct validity — 22036-0.0 vs proxy (884/894/904/914 inst-0)
#      Same time point, different instruments. Retained for reference only.
#
#  (C) IMAGING construct validity — 22036-2.0 vs proxy (884/894/904/914 inst-2)
#      Same time point, different instruments. Documents gap between official
#      and proxy at imaging visit; supports decision to use official field.

def _kappa(a, b):
    """Cohen's kappa for two binary integer Series."""
    agree = (a == b).mean()
    p1a, p1b = (a == 1).mean(), (b == 1).mean()
    pe = p1a * p1b + (1 - p1a) * (1 - p1b)
    return (agree - pe) / (1 - pe) if pe != 1 else float('nan'), agree

if 'PA_active' in df.columns:
    p_base_off = pd.Series(df['PA_active'].values, index=df.index).astype(float)
    p_img_off  = pd.Series(pa_official_imaging(df['22036-2.0']),  index=df.index)
    p_base_prx = pd.Series(pa_healthy_approx(df['884-0.0'], df['894-0.0'],
                                              df['904-0.0'], df['914-0.0']),
                            index=df.index)
    p_img_prx  = pd.Series(pa_healthy_approx(df['884-2.0'], df['894-2.0'],
                                              df['904-2.0'], df['914-2.0']),
                            index=df.index)

    # --- (A) Longitudinal: 22036-0.0 vs 22036-2.0 ---
    ok_A = p_base_off.notna() & p_img_off.notna()
    k_A, ag_A = _kappa(p_base_off[ok_A].astype(int), p_img_off[ok_A].astype(int))
    ct_A = pd.crosstab(p_base_off[ok_A].astype(int).rename('22036-0.0 baseline'),
                       p_img_off[ok_A].astype(int).rename('22036-2.0 imaging'),
                       margins=True)
    txt_A = [
        "PA concordance (A) — LONGITUDINAL: 22036-0.0 (baseline) vs 22036-2.0 (imaging)",
        "=" * 78,
        f"n compared    : {int(ok_A.sum()):,}  (participants with both official fields non-null)",
        f"raw agreement : {ag_A:.4f}",
        f"Cohen's kappa : {k_A:.4f}",
        "",
        "Cross-tab (rows = baseline 22036-0.0; cols = imaging 22036-2.0):",
        ct_A.to_string(), "",
        "Interpretation: kappa reflects TEMPORAL STABILITY of PA behaviour.",
        "  Low values may indicate genuine behaviour change, not measurement error.",
        "  Both instruments are definitionally identical (self-reported guideline",
        "  attainment), so this is the most valid longitudinal comparison available."]
    concord_A = "\n".join(txt_A)
    print("\n" + concord_A)
    with open(os.path.join(OUTPUT_DIR, 'pa_concordance_longitudinal.txt'), 'w') as f:
        f.write(concord_A)
    print("Saved: pa_concordance_longitudinal.txt")

    # --- (B) Baseline construct validity: 22036-0.0 vs proxy (inst-0) ---
    ok_B = p_base_off.notna() & p_base_prx.notna()
    k_B, ag_B = _kappa(p_base_off[ok_B].astype(int), p_base_prx[ok_B].astype(int))
    ct_B = pd.crosstab(p_base_off[ok_B].astype(int).rename('22036-0.0 (official)'),
                       p_base_prx[ok_B].astype(int).rename('proxy 884/894/904/914 inst-0'),
                       margins=True)
    txt_B = [
        "PA concordance (B) — BASELINE construct validity: 22036-0.0 vs proxy (inst-0)",
        "=" * 78,
        f"n compared    : {int(ok_B.sum()):,}",
        f"raw agreement : {ag_B:.4f}",
        f"Cohen's kappa : {k_B:.4f}",
        "",
        "Cross-tab:",
        ct_B.to_string(), "",
        "Interpretation: same time point, different instruments.",
        "  Baseline PA_active (22036-0.0) used in Phase I/NB3 is self-reported",
        "  guideline attainment; proxy uses objective minute counts. Moderate",
        "  agreement confirms partial equivalence but justifies keeping them",
        "  as separate constructs in the analysis.",
    ]
    concord_B = "\n".join(txt_B)
    print("\n" + concord_B)
    with open(os.path.join(OUTPUT_DIR, 'pa_concordance_baseline.txt'), 'w') as f:
        f.write(concord_B)
    print("Saved: pa_concordance_baseline.txt")

    # --- (C) Imaging construct validity: 22036-2.0 vs proxy (inst-2) ---
    ok_C = p_img_off.notna() & p_img_prx.notna()
    k_C, ag_C = _kappa(p_img_off[ok_C].astype(int), p_img_prx[ok_C].astype(int))
    ct_C = pd.crosstab(p_img_off[ok_C].astype(int).rename('22036-2.0 (official)'),
                       p_img_prx[ok_C].astype(int).rename('proxy 884/894/904/914 inst-2'),
                       margins=True)
    txt_C = [
        "PA concordance (C) — IMAGING construct validity: 22036-2.0 vs proxy (inst-2)",
        "=" * 78,
        f"n compared    : {int(ok_C.sum()):,}",
        f"raw agreement : {ag_C:.4f}",
        f"Cohen's kappa : {k_C:.4f}",
        "",
        "Cross-tab:",
        ct_C.to_string(), "",
        "Interpretation: same time point (imaging visit), different instruments.",
        "  Official 22036-2.0 attainment rate is substantially higher than proxy",
        "  (self-report tends to over-estimate guideline attainment relative to",
        "  objective minute counts). This supports using 22036-2.0 as primary",
        "  imaging-visit indicator for definitional consistency with baseline."]
    concord_C = "\n".join(txt_C)
    print("\n" + concord_C)
    with open(os.path.join(OUTPUT_DIR, 'pa_concordance_imaging.txt'), 'w') as f:
        f.write(concord_C)
    print("Saved: pa_concordance_imaging.txt")

else:
    print("\nSkipped PA concordance: PA_active not available.")


# 3. Stratification
#    sex + age : FULL imaging pool (from exposure tsv)
#    BMI : whichever coverage Phase I parquet gave

has_sex_age = all(c in df.columns for c in ['genetic_sex', 'age_defined_baseline'])
has_bmi = 'BMI' in df.columns

if has_sex_age:
    df['_sex'] = df['genetic_sex'].map({0: 'Female', 1: 'Male'}).fillna('NA')
    df['_age_band'] = pd.cut(df['age_defined_baseline'],
                            bins=[-np.inf, 55, 65, np.inf], right=False,
                            labels=['<55', '55-65', '>=65'])
if has_bmi:
    df['_bmi_band'] = pd.cut(df['BMI'],
                            bins=[-np.inf, 25, 30, np.inf], right=False,
                            labels=['<25', '25-30', '>=30'])

strat_vars = []
if has_sex_age:
    strat_vars += [('_sex', 'sex', 'full'), ('_age_band', 'age_band', 'full')]
if has_bmi:
    strat_vars += [('_bmi_band', 'bmi_band', coverage)]

if strat_vars:
    strat_rows = []
    for name, (b_arr, i_arr) in H.items():
        b = pd.Series(b_arr, index=df.index)
        i = pd.Series(i_arr, index=df.index)
        ok = b.notna() & i.notna()
        for sv_col, sv_name, basis in strat_vars:
            for level, sub in df[ok].groupby(sv_col, observed=True):
                if pd.isna(level) or str(level) == 'NA':
                    continue
                idx = sub.index
                b2 = b.loc[idx]; i2 = i.loc[idx]
                n = len(b2)
                if n == 0:
                    continue
                masks = {'HH': (b2 == 1) & (i2 == 1),
                        'HU': (b2 == 1) & (i2 == 0),
                        'UH': (b2 == 0) & (i2 == 1),
                        'UU': (b2 == 0) & (i2 == 0)}
                for k, mk in masks.items():
                    strat_rows.append({
                        'intervention': name,
                        'strat_var'   : sv_name,
                        'strat_level' : str(level),
                        'denom_basis' : basis,
                        'cell'        : k,
                        'n'           : int(mk.sum()),
                        'pct'         : mk.sum() / n * 100,
                    })
    strat_df = pd.DataFrame(strat_rows)
    strat_df.to_csv(
        os.path.join(OUTPUT_DIR, 'transition_matrix_strat.csv'), index=False)
    print(f"Saved: transition_matrix_strat.csv ({len(strat_df)} rows)")

    # improvement rate UH / (UH+UU) by strata
    print("\n--- Improvement rate (UH / baseline-unhealthy) by strata ---")
    for name in ['Quit smoking', 'Increase PA', 'Adequate sleep']:
        sub = strat_df[strat_df['intervention'] == name]
        for sv_name in ['sex', 'age_band', 'bmi_band']:
            ss = sub[sub['strat_var'] == sv_name]
            if len(ss) == 0:
                continue
            piv = ss.pivot(index='strat_level', columns='cell',
                        values='n').fillna(0)
            if {'UH', 'UU'}.issubset(piv.columns):
                denom = (piv['UH'] + piv['UU']).replace(0, np.nan)
                piv['rate'] = piv['UH'] / denom * 100
                basis = ss['denom_basis'].iloc[0]
                rates = ', '.join(f"{lvl}={r:.1f}%"
                                for lvl, r in piv['rate'].items())
                print(f"  {name} by {sv_name} [{basis}]: {rates}")
else:
    print("\nNo stratification variables available.")


# 4. Figure: 3-panel summary (unchanged)

fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
cell_colors = {'HH': '#4C8C5A','UH': '#3D5A80',
            'UU': '#9A9A9A','HU': '#E07A5F'}
cell_label  = {'HH': 'Always\nhealthy','UH': 'Improved\n(treated)',
            'UU': 'Persistent\nunhealthy (ctrl)','HU': 'Healthy →\nunhealthy'}

for ax, name in zip(axes, ['Quit smoking', 'Increase PA', 'Adequate sleep']):
    sub = mat[(mat['intervention'] == name) &
            (mat['cohort'] == 'main (full imaging pool)')].set_index('cell')
    order = ['HH', 'UH', 'UU', 'HU']
    sizes = [int(sub.loc[k, 'n'])          for k in order]
    rates = [float(sub.loc[k, 'cvd_rate']) for k in order]
    total = sum(sizes)
    pcts  = [s / total * 100 for s in sizes]

    ax.bar(range(4), pcts,
        color=[cell_colors[k] for k in order],
        edgecolor='white', width=0.78)
    for x, (n_, p_, r_) in enumerate(zip(sizes, pcts, rates)):
        ax.text(x, p_ + 1.2,
                f'n={n_:,}\n{p_:.1f}%\nCVD={r_:.3f}',
                ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([cell_label[k] for k in order], fontsize=8.5)
    ax.set_ylim(0, max(pcts) * 1.45)
    ax.set_title(name, fontsize=11, pad=8)
    if ax is axes[0]:
        ax.set_ylabel('% of both-visit sample')
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

fig.suptitle('Baseline → Imaging Visit Transitions  '
            '(per-cell n, % of both-visit pool, and CVD event rate)',
            fontsize=12, y=1.02)
fig.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'transition_panels.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Saved figure: transition_panels.png")


# 5. Joint longitudinal intervention analysis

print("\n" + "=" * 78)
print("JOINT LONGITUDINAL INTERVENTION  (Table 2 extension)")
print("  Reference = baseline all-unhealthy → imaging all-unhealthy (000→000)")
print("  Arms 1-7  = baseline all-unhealthy → imaging any other combination")
print("=" * 78)

# NB3 mean CATE results from 03_multiarm_joint.py output.
_nb3_summary = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_summary.parquet'))

if 'n' in _nb3_summary.columns:
    n_unique = _nb3_summary['n'].dropna().unique()
    print(f"  joint_cate_summary.parquet n values: {n_unique}")
    assert 321188 in n_unique, (
        "Expected joint_cate_summary.parquet from the 70% Phase II analysis set. "
        "Re-run 03_multiarm_joint."
    )

NB3_CATE = dict(zip(_nb3_summary['arm'].astype(int), _nb3_summary['mean_cate']))
print(f"  Loaded CATE from joint_cate_summary.parquet: {NB3_CATE}")
#NB3_CATE = {
#    1: -0.0710,   # no_smk only
#    2: -0.0280,   # PA only
#    3: -0.0810,   # no_smk + PA
#    4: -0.0390,   # sleep only
#    5: -0.0805,   # no_smk + sleep
#    6: -0.0493,   # PA + sleep
#    7: -0.0854,   # all three
#}

ARM_LABELS = {
    1: 'no_smk only',
    2: 'PA only',
    3: 'no_smk + PA',
    4: 'sleep only',
    5: 'no_smk + sleep',
    6: 'PA + sleep',
    7: 'all three',
}

# build imaging-visit healthy indicators
# Baseline PA: official PA_active from Phase I parquets, derived from field 22036-0.0.
# Imaging PA: official field 22036-2.0.
# Minute-based PA proxy fields are retained only for concordance checks.
smk_b  = pd.Series(smk_healthy(df['20116-0.0']),  index=df.index)
pa_b = pd.Series(df['PA_active'], index=df.index).astype(float)
slp_b  = pd.Series(sleep_healthy(df['1160-0.0']), index=df.index)
smk_i  = pd.Series(smk_healthy(df['20116-2.0']),  index=df.index)
pa_i = pd.Series(pa_official_imaging(df['22036-2.0']), index=df.index)
slp_i  = pd.Series(sleep_healthy(df['1160-2.0']), index=df.index)

# require all 6 indicators non-missing
all_obs = smk_b.notna() & pa_b.notna() & slp_b.notna() & \
        smk_i.notna() & pa_i.notna() & slp_i.notna()
print(f"\n  Participants with all 6 indicators observed: {all_obs.sum():,}")

smk_b2 = smk_b[all_obs]; pa_b2 = pa_b[all_obs]; slp_b2 = slp_b[all_obs]
smk_i2 = smk_i[all_obs]; pa_i2 = pa_i[all_obs]; slp_i2 = slp_i[all_obs]
y_all  = df.loc[all_obs, OUTCOME]

# arm code at imaging visit (no_smk = smk_healthy, same direction as NB3)
arm_imaging = (smk_i2 * 1 + pa_i2 * 2 + slp_i2 * 4).astype(int)
arm_baseline = (smk_b2 * 1 + pa_b2 * 2 + slp_b2 * 4).astype(int)

# reference = baseline arm 0 AND imaging arm 0 (all-unhealthy at both visits)
ref_mask = (arm_baseline == 0) & (arm_imaging == 0)
n_ref    = int(ref_mask.sum())
ev_ref   = int(y_all[ref_mask].sum())
rate_ref = y_all[ref_mask].mean()
print(f"\n  Reference group (000→000): n={n_ref:,}, events={ev_ref}, rate={rate_ref:.4f}")

# pool = baseline arm 0 only (all-unhealthy at baseline)
baseline_000 = (arm_baseline == 0)
n_pool = int(baseline_000.sum())
print(f"  Baseline all-unhealthy pool (000 at baseline): n={n_pool:,}")

joint_rows = []
print(f"\n{'Arm':<4} {'Label':<22} {'n':>7} {'CVD events':>10} "
    f"{'Crude rate':>11} {'Crude RD':>10} {'CATE':>10}")
print("-" * 78)

# reference row first
print(f"{'0':<4} {'(reference: 000→000)':<22} {n_ref:>7,} {ev_ref:>10,} "
    f"{rate_ref:>11.4f} {'0.0000':>10} {'(ref)':>10}")

for arm in range(1, 8):
    # treated = baseline 000 AND imaging = arm
    treated_mask = (arm_baseline == 0) & (arm_imaging == arm)
    n_t  = int(treated_mask.sum())
    ev_t = int(y_all[treated_mask].sum())
    rate_t = y_all[treated_mask].mean() if n_t > 0 else np.nan
    crude_rd = rate_t - rate_ref if n_t > 0 else np.nan
    nb3_cate = NB3_CATE.get(arm, np.nan)
    n_healthy = bin(arm).count('1')

    flag = '  ← LOW n' if n_t < 200 else ''
    print(f"{arm:<4} {ARM_LABELS[arm]:<22} {n_t:>7,} {ev_t:>10,} "
        f"{rate_t:>11.4f} {crude_rd:>+10.4f} {nb3_cate:>+10.4f}{flag}")

    joint_rows.append({
        'arm'              : arm,
        'label'            : ARM_LABELS[arm],
        'n_healthy_beh'    : n_healthy,
        'n_treated'        : n_t,
        'cvd_events'       : ev_t,
        'cvd_rate'         : rate_t,
        'crude_rd_vs_ref'  : crude_rd,
        'nb3_cate'         : nb3_cate,
        'n_ref'            : n_ref,
        'rate_ref'         : rate_ref})

joint_df = pd.DataFrame(joint_rows)
joint_df.to_csv(os.path.join(OUTPUT_DIR, 'joint_longitudinal.csv'), index=False)
print(f"\nSaved: joint_longitudinal.csv")

print("\nNote: Crude RD is NOT causal (reverse causality expected for smoking/sleep).")
print("      MSM/IPTW will adjust for confounding in the next analysis step.")
print("      CATE = cross-sectional causal estimate (baseline data only).")
print("      Direction mismatch between Crude RD and CATE = evidence of")
print("      time-varying confounding that MSM is designed to correct.")

# --- figure: crude RD vs CATE side by side ---
fig2, ax = plt.subplots(figsize=(11, 5))

x      = np.arange(len(joint_df))
width  = 0.38
colors_nb3   = ['#3D5A80'] * len(joint_df)
colors_crude = ['#E07A5F'] * len(joint_df)

bars1 = ax.bar(x - width/2, joint_df['nb3_cate'],    width,
            color=colors_nb3,   label='CATE (cross-sectional, causal)',
            edgecolor='white')
bars2 = ax.bar(x + width/2, joint_df['crude_rd_vs_ref'], width,
            color=colors_crude, label='Crude RD vs 000→000 (longitudinal, raw)',
            edgecolor='white')

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"arm {r['arm']}\n{r['label']}\n(n={r['n_treated']:,})"
                    for _, r in joint_df.iterrows()], fontsize=8)
ax.set_ylabel('Risk difference vs all-unhealthy reference')
ax.set_title('Joint Interventions: Cross-sectional CATE vs Longitudinal Crude RD\n'
            '(reference = all-unhealthy at both visits; crude RD not causal)',
            fontsize=11, pad=10)
ax.legend(fontsize=9, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, alpha=0.25, lw=0.5)
ax.set_axisbelow(True)

fig2.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'joint_longitudinal.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Saved figure: joint_longitudinal.png")

# 6. Pairwise joint intervention feasibility

print("\n" + "=" * 78)
print("PAIRWISE JOINT INTERVENTION FEASIBILITY")
print("  baseline pool = unhealthy on BOTH behaviours")
print("  treated = both improve; control = both stay unhealthy")
print("=" * 78)

beh_b = {'smk': smk_b, 'PA': pa_b, 'sleep': slp_b}
beh_i = {'smk': smk_i, 'PA': pa_i, 'sleep': slp_i}

#PAIRS = [
#    ('smk',  'PA',    'Smk+PA',    'arm 3: -0.0810'),
#    ('smk',  'sleep', 'Smk+Sleep', 'arm 5: -0.0805'),
#    ('PA',   'sleep', 'PA+Sleep',  'arm 6: -0.0493')]

PAIRS = [
    ('smk', 'PA', 'Smk+PA', 3),
    ('smk', 'sleep', 'Smk+Sleep', 5),
    ('PA', 'sleep', 'PA+Sleep', 6)]

pair_rows = []
print(f"\n{'Pair':<12} {'pool':>7} {'treated':>8} {'control':>8} "
    f"{'events T/C':>11} {'rate T/C':>14} {'feasible':>10}  NB3 ref")
print("-" * 85)

for bA, bB, label, arm_id in PAIRS:
    nb3_ref = NB3_CATE.get(arm_id, np.nan)
    bbA = beh_b[bA]; biA = beh_i[bA]
    bbB = beh_b[bB]; biB = beh_i[bB]

    ok = bbA.notna() & biA.notna() & bbB.notna() & biB.notna()
    pool_mask = ok & (bbA == 0) & (bbB == 0)
    n_pool = int(pool_mask.sum())

    treated = pool_mask & (biA == 1) & (biB == 1)
    ctrl    = pool_mask & (biA == 0) & (biB == 0)

    nt = int(treated.sum())
    nc = int(ctrl.sum())
    et = int(df.loc[treated, OUTCOME].sum()) if nt > 0 else 0
    ec = int(df.loc[ctrl, OUTCOME].sum()) if nc > 0 else 0
    rt = df.loc[treated, OUTCOME].mean() if nt > 0 else np.nan
    rc = df.loc[ctrl, OUTCOME].mean() if nc > 0 else np.nan

    feasible = 'YES' if (nt >= 500 and nc >= 500 and et >= 50 and ec >= 50) else 'LOW'
    rt_str = f'{rt:.3f}' if not np.isnan(rt) else 'NaN'
    rc_str = f'{rc:.3f}' if not np.isnan(rc) else 'NaN'

    print(f"{label:<12} {n_pool:>7,} {nt:>8,} {nc:>8,} "
        f"{et:>5}/{ec:<5} "
        f"{rt_str}/{rc_str}  {feasible:>10}  "
        f"arm {arm_id}: {nb3_ref:+.4f}")

    pair_rows.append({
        'pair'          : label,
        'behA'          : bA,  'behB'          : bB,
        'n_pool'        : n_pool,
        'n_treated'     : nt,  'n_control'     : nc,
        'events_treated': et,  'events_control': ec,
        'rate_treated'  : rt,  'rate_control'  : rc,
        'feasible'      : feasible,
        'nb3_arm'       : arm_id,
        'nb3_cate'      : nb3_ref})

pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_joint_feasibility.csv'), index=False)
print(f"\nSaved: pairwise_joint_feasibility.csv")
print("\nNote: 'treated' = both behaviours improved simultaneously.")
print("      'control' = both stayed unhealthy. Mixed changers excluded.")
print("      Feasibility: treated/control n>=500 and treated/control events>=50.")
print("      Feasible pairs → include in MSM alongside single interventions.")
print("      Infeasible pairs → limitation section.")

print(f"\n05_transition_matrix.py done.  BMI/PA coverage: {coverage}")