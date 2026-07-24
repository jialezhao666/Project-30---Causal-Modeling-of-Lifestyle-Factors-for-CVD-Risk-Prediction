import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser('~/my_ukb_thesis'))
from const_paths import BASE_PATH, SAVE_DIR

OUTPUT_DIR  = os.path.expanduser('~/my_ukb_thesis/phase_II/outputs')
FIGURES_DIR = os.path.expanduser('~/my_ukb_thesis/phase_II/figures')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

OUTCOME = 'def_CVD_AF_HF_AFTER'

CONFOUNDERS = [
    'age_defined_baseline', 'genetic_sex', 'BMI', 'uni_degree',
    'FH_cvd_f', 'FH_cvd_m', 'FH_cvd_sib', 'mental_doctor', 'alc_curr']

CONFOUNDER_LABELS = {
    'age_defined_baseline': 'Age at baseline',
    'genetic_sex': 'Male sex',
    'BMI': 'Body mass index',
    'uni_degree': 'University degree',
    'FH_cvd_f': 'Family history of CVD in father',
    'FH_cvd_m': 'Family history of CVD in mother',
    'FH_cvd_sib': 'Family history of CVD in sibling',
    'mental_doctor': 'Mental-health treatment history',
    'alc_curr': 'Current alcohol use'}

INTERVENTIONS = ['smk', 'pa', 'sleep', 'smk_sleep', 'pa_sleep', 'smk_pa']
PRIMARY_INTERVENTIONS = ['smk', 'pa', 'sleep']
EXPLORATORY_INTERVENTIONS = ['smk_sleep', 'pa_sleep', 'smk_pa']

INTERVENTION_LABELS = {
    'smk': 'Smoking cessation',
    'pa': 'Becoming physically active',
    'sleep': 'Achieving adequate sleep',
    'smk_sleep': 'Smoking cessation + adequate sleep',
    'pa_sleep': 'Physical activity + adequate sleep',
    'smk_pa': 'Smoking cessation + physical activity'}

MIN_ARM_N = 500
MIN_ARM_EVENTS = 50
N_BOOT = 500
TRUNCATE_PCT = (1, 99)
RANDOM_SEED = 42


# 1A. Load longitudinal behaviour fields + outcome

print("=" * 68)
print("Data loading and cohort construction")
print("=" * 68)

file_tab = os.path.join(BASE_PATH, 'ukb_tabular_data_causal_analysis.tsv')
tab_cols = [
    'eid',
    '20116-0.0', '20116-2.0',
    '1160-0.0', '1160-2.0',
    '20117-0.0']

print("\nLoading longitudinal fields + alcohol")
tab = pd.read_csv(file_tab, sep='\t', usecols=tab_cols)
print(f"  tabular rows: {len(tab):,}")

file_out = os.path.join(BASE_PATH, 'group_1_outcomes_df_without_qc_df.tsv')
out = pd.read_csv(file_out, sep='\t', usecols=['eid', OUTCOME])
df  = tab.merge(out, on='eid', how='inner')
print(f"  after outcome merge: {len(df):,}")

# Load official imaging-visit PA field (22036-2.0)
# Replaces the proxy measure (884/894/904/914 instance 2) for imaging-visit PA
pa22_img = pd.read_csv(os.path.join(BASE_PATH, 'output_22036_imaging.tsv'),
                    sep='\t', usecols=['eid', '22036-2.0'])
df = df.merge(pa22_img, on='eid', how='left')
print(f"  22036-2.0 (imaging PA official): "
    f"{df['22036-2.0'].notna().sum():,} non-null")

# 1B. Load confounders from exposure file (full population)

print("\nLoading confounders from exposure file (full population)")
file_exp  = os.path.join(BASE_PATH,
                        'group_1_clean_filtered_imputed_dataset_df.tsv')

exp_cols_want = ['eid',
                'genetic_sex',
                'age_defined_baseline',
                '21001-0.0',    # BMI
                '6138.1',       # university degree
                '2090-0.0',     # mental health doctor visit
                '20107.1', '20107.2',   # father CVD history
                '20110.1', '20110.2',   # mother CVD history
                '20111.1', '20111.2'    # sibling CVD history
                ]

exp_header = pd.read_csv(file_exp, sep='\t', nrows=0).columns.tolist()
exp_cols   = [c for c in exp_cols_want if c in exp_header]
missing    = [c for c in exp_cols_want if c not in exp_header]
if missing:
    print(f"missing from exposure file: {missing}")

exp = pd.read_csv(file_exp, sep='\t', usecols=exp_cols)
print(f"  exposure file rows: {len(exp):,}")
df  = df.merge(exp, on='eid', how='left')
print(f"  after merge: {df.shape}")

# Load baseline official PA_active from Phase I parquets
print("\nLoading baseline PA_active from Phase I parquets")
pa_parts = []
for fname in ['split_A_explore.parquet', 'split_B_train.parquet', 'split_C_test.parquet']:
    path = os.path.join(SAVE_DIR, fname)
    if os.path.exists(path):
        tmp = pd.read_parquet(path, columns=['eid', 'PA_active'])
        print(f"  {fname}: {len(tmp):,}")
        pa_parts.append(tmp)

if not pa_parts:
    raise FileNotFoundError("No Phase I parquet files found for PA_active.")

pa_all = (
    pd.concat(pa_parts, ignore_index=True)
    .drop_duplicates('eid'))

df = df.merge(pa_all, on='eid', how='left')
print(f"  PA_active coverage after merge: {df['PA_active'].notna().sum():,} / {len(df):,}")

# 1C.  Derive the 9 confounders in their final form

def _clip_neg_na(x):
    """UKB negative codes to NaN."""
    return x.astype(float).where(x.astype(float) >= 0)

# BMI (raw value, will be used unscaled in propensity model)
df['BMI'] = _clip_neg_na(df['21001-0.0'])

# university degree (6138.1 == 1 means "College or University degree")
df['uni_degree'] = (df['6138.1'] == 1).astype(float)
df.loc[df['6138.1'].isna(), 'uni_degree'] = np.nan

# mental health doctor visit (2090-0.0: 1=yes, 0=no)
df['mental_doctor'] = _clip_neg_na(df['2090-0.0'])
df['mental_doctor'] = (df['mental_doctor'] == 1).astype(float)
df.loc[_clip_neg_na(df['2090-0.0']).isna(), 'mental_doctor'] = np.nan

# family history CVD — father (20107.1=heart disease, 20107.2=stroke)
df['FH_cvd_f'] = (
    (df['20107.1'] == 1) | (df['20107.2'] == 1)).astype(float)
df.loc[df['20107.1'].isna() & df['20107.2'].isna(), 'FH_cvd_f'] = np.nan

# family history CVD — mother
df['FH_cvd_m'] = (
    (df['20110.1'] == 1) | (df['20110.2'] == 1)).astype(float)
df.loc[df['20110.1'].isna() & df['20110.2'].isna(), 'FH_cvd_m'] = np.nan

# family history CVD — sibling
df['FH_cvd_sib'] = (
    (df['20111.1'] == 1) | (df['20111.2'] == 1)).astype(float)
df.loc[df['20111.1'].isna() & df['20111.2'].isna(), 'FH_cvd_sib'] = np.nan

# alcohol current drinker (20117-0.0 == 2)
alc = _clip_neg_na(df['20117-0.0'])
df['alc_curr'] = (alc == 2).astype(float)
df.loc[alc.isna(), 'alc_curr'] = np.nan

# genetic_sex and age_defined_baseline already named correctly in exposure file
# report confounder availability
print("\nConfounder availability (should be ~458k for all):")
for c in CONFOUNDERS:
    n_obs = df[c].notna().sum()
    pct   = n_obs / len(df) * 100
    print(f"  {c:25s}: {n_obs:>7,}  ({pct:.1f}% non-missing)")


# 1D. Behaviour indicator functions

def smk_healthy(col): 
    c = _clip_neg_na(col)
    return np.where(c.isna(), np.nan, (c != 2).astype(float))

def sleep_healthy(col):
    c = _clip_neg_na(col)
    return np.where(c.isna(), np.nan, (c >= 7).astype(float))


def pa_official_imaging(col_22036_2):
    """
    Official imaging-visit PA indicator from field 22036-2.0.
    1 = meets PA guideline (self-reported), 0 = does not meet, NaN = missing.
    Definitionally consistent with the baseline indicator (field 22036-0.0):
    both use the same self-report guideline-attainment question.
    """
    c = col_22036_2.astype(float)
    return np.where(c.isna(), np.nan, (c == 1).astype(float))

smk_b = pd.Series(smk_healthy(df['20116-0.0']), index=df.index)
smk_i = pd.Series(smk_healthy(df['20116-2.0']), index=df.index)
pa_b = pd.Series(df['PA_active'], index=df.index).astype(float)
pa_i = pd.Series(pa_official_imaging(df['22036-2.0']), index=df.index)
slp_b = pd.Series(sleep_healthy(df['1160-0.0']), index=df.index)
slp_i = pd.Series(sleep_healthy(df['1160-2.0']), index=df.index)


# 1D-2. Sub-analysis flag: exclude participants whose CVD/AF/HF event occurred before their imaging visit.

print("\n" + "=" * 68)
print("SUB-ANALYSIS SETUP: flagging participants with CVD event before imaging")
print("=" * 68)

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

# Source-level flags include participants who may not be present in the
# final merged analytic dataset. Report both values explicitly.
exclude_mask = df['eid'].isin(exclude_eids)
n_flagged_source = len(exclude_eids)
n_excluded_analytic = int(exclude_mask.sum())

print(f"  Flagged in event-date source: {n_flagged_source:,}")
print(f"  Present and excluded from analytic data: {n_excluded_analytic:,}")
print(f"  df (full merged dataset): {len(df):,}")
print(f"  df_sub after exclusion: {(~exclude_mask).sum():,}")


# 1E. Cohort construction

def build_cohort(label, base_mask, treat_mask, ctrl_mask, restrict_mask=None):
    """
    Construct treated/control cohort with complete W.
    base_mask : eligible at baseline (unhealthy + both visits observed)
    treat_mask : imaging = healthy (treated)
    ctrl_mask : imaging = unhealthy (control)
    restrict_mask : optional additional eligibility filter (exclude
        participants with a pre-imaging CVD event for the sub-analysis).
        None = no extra restriction (main analysis).
    """
    w_complete = pd.Series(True, index=df.index)
    for c in CONFOUNDERS:
        w_complete = w_complete & df[c].notna()

    eligible = base_mask & w_complete
    if restrict_mask is not None:
        eligible = eligible & restrict_mask
    treated  = eligible & treat_mask
    control  = eligible & ctrl_mask

    treated_n = int(treated.sum())
    control_n = int(control.sum())
    treated_events = int(df.loc[treated, OUTCOME].sum())
    control_events = int(df.loc[control, OUTCOME].sum())
    treated_event_rate = df.loc[treated, OUTCOME].mean() if treated_n > 0 else np.nan
    control_event_rate = df.loc[control, OUTCOME].mean() if control_n > 0 else np.nan

    cohort = df.loc[treated | control,
                    ['eid', OUTCOME] + CONFOUNDERS].copy()
    cohort['treatment'] = 0
    cohort.loc[treated[treated | control].values.astype(bool),
            'treatment'] = 1

    feasible = 'OK' if (treated_n >= MIN_ARM_N and control_n >= MIN_ARM_N and treated_events >= MIN_ARM_EVENTS and control_events >= MIN_ARM_EVENTS) else 'LOW'

    stats = {
        'label': label,
        'treated_n': treated_n, 'treated_events': treated_events,
        'treated_event_rate': treated_event_rate,
        'control_n': control_n, 'control_events': control_events,
        'control_event_rate': control_event_rate,
        'feasible': feasible}
    return cohort, stats


print("\n" + "=" * 68)
print("Cohort construction (main = full imaging pool, "
    "sub = excl. pre-imaging CVD events)")
print("=" * 68)

not_excluded = ~exclude_mask


_mask_specs = {
    'smk': dict(
        base_mask=smk_b.notna() & smk_i.notna() & (smk_b == 0),
        treat_mask=smk_i == 1, ctrl_mask=smk_i == 0),
    'pa': dict(
        base_mask=pa_b.notna() & pa_i.notna() & (pa_b == 0),
        treat_mask=pa_i == 1, ctrl_mask=pa_i == 0),
    'sleep': dict(
        base_mask=slp_b.notna() & slp_i.notna() & (slp_b == 0),
        treat_mask=slp_i == 1, ctrl_mask=slp_i == 0),
    'smk_sleep': dict(
    base_mask=(smk_b.notna() & smk_i.notna() &
            slp_b.notna() & slp_i.notna() &
            (smk_b == 0) & (slp_b == 0)),
    treat_mask=(smk_i == 1) & (slp_i == 1),
    ctrl_mask=(smk_i == 0) & (slp_i == 0)),
    'pa_sleep': dict(
        base_mask=(pa_b.notna() & pa_i.notna() &
                    slp_b.notna() & slp_i.notna() &
                    (pa_b == 0) & (slp_b == 0)),
        treat_mask=(pa_i == 1) & (slp_i == 1),
        ctrl_mask=(pa_i == 0) & (slp_i == 0)),
    'smk_pa': dict(
        # baseline unhealthy on BOTH; imaging improves BOTH (treated)
        # or stays unhealthy on BOTH (control). Mixed changers excluded.
        # Feasibility is assessed below using treated/control sample size and event counts.
        base_mask=(smk_b.notna() & smk_i.notna() &
                pa_b.notna()  & pa_i.notna()  &
                (smk_b == 0)  & (pa_b == 0)),
        treat_mask=(smk_i == 1) & (pa_i == 1),
        ctrl_mask= (smk_i == 0) & (pa_i == 0))}

cohorts_main, cohorts_sub = {}, {}
stats_main, stats_sub = {}, {}

for name, spec in _mask_specs.items():
    label = INTERVENTION_LABELS[name]
    cohorts_main[name], stats_main[name] = build_cohort(
        f'{label} [main]', **spec)
    cohorts_sub[name], stats_sub[name] = build_cohort(
        f'{label} [sub]', **spec, restrict_mask=not_excluded)

# Reverse-causality contamination check 
# How much of the main analysis's outcome-positive signal comes from
# participants whose CVD/AF/HF event occurred BEFORE their imaging visit

print("\n" + "=" * 68)
print("Reverse-causality contamination check (main analysis cohorts)")
print("  % of outcome-positive participants in each MAIN cohort whose "
    "event occurred\n  before their imaging visit (and were therefore "
    "removed in the SUB cohort)")
print("=" * 68)
contam_rows = []
for name in INTERVENTIONS:
    base_mask = _mask_specs[name]['base_mask']
    treat_mask = _mask_specs[name]['treat_mask']
    ctrl_mask = _mask_specs[name]['ctrl_mask']
    w_complete = pd.Series(True, index=df.index)
    for c in CONFOUNDERS:
        w_complete = w_complete & df[c].notna()
    in_cohort = base_mask & w_complete & (treat_mask | ctrl_mask)
    event_pos_eids = set(df.loc[in_cohort & (df[OUTCOME] == 1), 'eid'])
    contaminated = event_pos_eids & exclude_eids
    pct = len(contaminated) / max(len(event_pos_eids), 1) * 100
    contam_rows.append({
        'intervention': INTERVENTION_LABELS[name],
        'main_event_positive_n': len(event_pos_eids),
        'pre_imaging_event_n': len(contaminated),
        'pre_imaging_event_pct': pct})
    print(f"  {INTERVENTION_LABELS[name]:<18} "
        f"main outcome-positive={len(event_pos_eids):>4}  "
        f"pre-imaging (removed in sub)={len(contaminated):>4}  "
        f"({pct:.1f}%)")
contam_df = pd.DataFrame(contam_rows)
contam_df.to_csv(os.path.join(OUTPUT_DIR, 'reverse_causality_contamination.csv'),
                index=False)
print(f"\nSaved: reverse_causality_contamination.csv")
print("Interpretation: a high percentage means the full-cohort diagnostic's outcome "
    "signal for that\nintervention is dominated by participants who were "
    "likely diagnosed BEFORE their\nimaging-visit behaviour was measured "
    "-- i.e. reverse causality, not a causal effect\nof the behaviour. "
    "The SUB-ANALYSIS below removes these participants, which is why "
    "\nits event counts are much lower despite only a small drop in "
    "total sample size.")

# --- Unified cohort-construction summary table (main vs sub) ---------
print("\n" + "=" * 68)
print("COHORT SUMMARY — full-cohort diagnostic vs post-imaging sub-analysis")
print("  n        = number of people in that arm")
print("  events   = number of those people with OUTCOME=1")
print("  rate     = events / n")
print("=" * 68)
hdr = (f"{'Intervention':<36} {'Arm':<9} "
    f"{'n (full)':>9} {'ev (full)':>10} {'rate (full)':>12} | "
    f"{'n (sub)':>9} {'ev (sub)':>9} {'rate (sub)':>11}")
print(hdr)
print("-" * len(hdr))
for name in INTERVENTIONS:
    sm, ss = stats_main[name], stats_sub[name]
    label = INTERVENTION_LABELS[name]
    print(f"{label:<36} {'treated':<9} "
        f"{sm['treated_n']:>9,} {sm['treated_events']:>10,} "
        f"{sm['treated_event_rate']:>12.4f} | "
        f"{ss['treated_n']:>9,} {ss['treated_events']:>9,} "
        f"{ss['treated_event_rate']:>11.4f}")
    print(f"{'':<36} {'control':<9} "
        f"{sm['control_n']:>9,} {sm['control_events']:>10,} "
        f"{sm['control_event_rate']:>12.4f} | "
        f"{ss['control_n']:>9,} {ss['control_events']:>9,} "
        f"{ss['control_event_rate']:>11.4f}")
    feas_flag = '' if ss['feasible'] == 'OK' else ('  <- LOW feasibility: requires treated/control n>=500 and events>=50 in both arms')
    print(f"{'':<36} {'feasible':<9} {sm['feasible']:>9} {'':>10} {'':>12} | "
        f"{ss['feasible']:>9}{feas_flag}")
    print()


# 1F. Save cohorts (both main and sub)

for name, coh in cohorts_main.items():
    coh.to_parquet(os.path.join(OUTPUT_DIR, f'msm_cohort_{name}.parquet'),
                index=False)
for name, coh in cohorts_sub.items():
    coh.to_parquet(os.path.join(OUTPUT_DIR, f'msm_cohort_{name}_sub.parquet'),
                index=False)
print(f"Saved {len(cohorts_main) + len(cohorts_sub)} cohort parquet files "
      f"({len(cohorts_main)} interventions x main/sub)")

# Steps 2-4 (IPTW weighting, bootstrap CI, E-value) are wrapped into
# run_pipeline() below and called once for cohorts_main and once for
# cohorts_sub, so both the main analysis and the sub-analysis excluding
# pre-imaging CVD events go through the identical pipeline.

print("(Sanity check: cohort sizes above should be within ~5% of "
    "05_transition_matrix.py's pre-W-filtering counts; "
    "if not, check confounder missingness.)")


def run_pipeline(cohorts, suffix, run_label):
    """
    Runs Steps 2-4 (IPTW weighting, weighted RD/RR + bootstrap CI,
    E-value sensitivity) for a given cohorts dict.

    suffix    : '' for main analysis, '_sub' for sub-analysis
                (used in output filenames so main outputs are never
                overwritten by the sub-analysis run)
    run_label : human-readable label for print headers, e.g.
                'FULL-COHORT DIAGNOSTIC' or 'POST-IMAGING OUTCOME SUB-ANALYSIS'

    Returns results_df, evalue_df for the caller to combine across runs.
    """

    print("\n" + "=" * 68)
    print(f"STEP 2 [{run_label}]: IPTW weight estimation + diagnostics")
    print("=" * 68)

    def estimate_weights(cohort, label, truncate_pct=(1, 99)):
        """
        Estimate stabilised IPTW weights for a binary treatment cohort.
        """
        coh = cohort.copy()
        T = coh['treatment'].values
        W = coh[CONFOUNDERS].values

        scaler = StandardScaler()
        W_std  = scaler.fit_transform(W)

        lr = LogisticRegression(max_iter=1000, random_state=42)
        lr.fit(W_std, T)
        ps = np.clip(lr.predict_proba(W_std)[:, 1], 1e-6, 1 - 1e-6)
        coh['ps'] = ps

        p1 = T.mean()
        p0 = 1 - p1

        w_raw = np.where(T == 1, p1 / ps, p0 / (1 - ps))
        coh['weight_raw'] = w_raw

        lo = np.percentile(w_raw, truncate_pct[0])
        hi = np.percentile(w_raw, truncate_pct[1])
        w_trunc = np.clip(w_raw, lo, hi)
        coh['weight'] = w_trunc

        print(f"\n  {label}")
        print(f"    Propensity score  : mean={ps.mean():.3f}  "
            f"min={ps.min():.3f}  max={ps.max():.3f}")
        print(f"    Weights (raw)     : mean={w_raw.mean():.3f}  "
            f"min={w_raw.min():.3f}  max={w_raw.max():.3f}  "
            f"std={w_raw.std():.3f}")
        print(f"    Truncation bounds : [{lo:.3f}, {hi:.3f}]")
        print(f"    Weights (trunc)   : mean={w_trunc.mean():.3f}  "
            f"max={w_trunc.max():.3f}  std={w_trunc.std():.3f}")

        print(f"    SMD (before -> after IPTW, threshold < 0.1):")
        smd_rows = []
        for c in CONFOUNDERS:
            x = coh[c].values.astype(float)
            t_mask = T == 1
            c_mask = T == 0

            m1u = x[t_mask].mean();  m0u = x[c_mask].mean()
            s1u = x[t_mask].std();   s0u = x[c_mask].std()
            pool_std = np.sqrt((s1u**2 + s0u**2) / 2)
            smd_before = abs(m1u - m0u) / pool_std if pool_std > 0 else 0

            w1 = w_trunc[t_mask]; w0 = w_trunc[c_mask]
            m1w = np.average(x[t_mask], weights=w1)
            m0w = np.average(x[c_mask], weights=w0)
            v1w = np.average((x[t_mask] - m1w)**2, weights=w1)
            v0w = np.average((x[c_mask] - m0w)**2, weights=w0)
            pool_std_w = np.sqrt((v1w + v0w) / 2)
            smd_after = abs(m1w - m0w) / pool_std_w if pool_std_w > 0 else 0

            flag = '' if smd_after < 0.1 else '  <- IMBALANCED'
            print(f"      {c:25s}: {smd_before:.3f} -> {smd_after:.3f}{flag}")
            smd_rows.append({'confounder': c,
                            'smd_before': smd_before,
                            'smd_after': smd_after})

        smd_df = pd.DataFrame(smd_rows)
        smd_df['label'] = smd_df['confounder'].map(CONFOUNDER_LABELS).fillna(smd_df['confounder'])
        return coh, smd_df

    def estimate_weights_silent(cohort, truncate_pct=TRUNCATE_PCT):
        """Estimate stabilised IPTW without printing, for bootstrap replicates."""
        coh = cohort.copy()
        T = coh['treatment'].to_numpy()
        if len(np.unique(T)) < 2:
            return None

        W = coh[CONFOUNDERS].to_numpy()
        scaler = StandardScaler()
        W_std = scaler.fit_transform(W)

        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
        lr.fit(W_std, T)
        ps = np.clip(lr.predict_proba(W_std)[:, 1], 1e-6, 1 - 1e-6)

        p1 = T.mean()
        p0 = 1 - p1
        w_raw = np.where(T == 1, p1 / ps, p0 / (1 - ps))
        lo, hi = np.percentile(w_raw, truncate_pct)

        coh['ps'] = ps
        coh['weight_raw'] = w_raw
        coh['weight'] = np.clip(w_raw, lo, hi)
        return coh

    weighted_cohorts = {}
    smd_dfs = {}

    for name, coh in cohorts.items():
        wcoh, smd = estimate_weights(coh, name, truncate_pct=TRUNCATE_PCT)
        weighted_cohorts[name] = wcoh
        smd_dfs[name] = smd

    print(f"\nSaving weighted cohorts [{run_label}]")
    for name, wcoh in weighted_cohorts.items():
        save_cols = ['eid', OUTCOME, 'treatment', 'ps',
                    'weight_raw', 'weight'] + CONFOUNDERS
        save_cols = [c for c in save_cols if c in wcoh.columns]
        path = os.path.join(OUTPUT_DIR, f'msm_weighted_{name}{suffix}.parquet')
        wcoh[save_cols].to_parquet(path, index=False)
        print(f"  msm_weighted_{name}{suffix}.parquet")

    n_plots = len(smd_dfs)
    fig, axes = plt.subplots(1, n_plots, figsize=(4 * n_plots, 5), sharey=True)

    if n_plots == 1:
        axes = [axes]

    titles = INTERVENTION_LABELS

    for ax, (name, smd) in zip(axes, smd_dfs.items()):
        ax.scatter(smd['smd_before'], smd['label'],
                color='#E07A5F', label='Before weighting', zorder=3)
        ax.scatter(smd['smd_after'],  smd['label'],
                color='#3D5A80', label='After weighting',  zorder=3)
        ax.axvline(0.1, color='grey', lw=1, ls='--', alpha=0.7)
        ax.set_xlabel('Absolute standardised mean difference')
        ax.set_title(titles[name], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    love_path = os.path.join(FIGURES_DIR, f'msm_love_plot{suffix}.png')
    fig.savefig(love_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: msm_love_plot{suffix}.png")

    print(f"\n{'='*68}")
    print("Check SMD table: all 'after' values should be < 0.1.")

    print("\n" + "=" * 68)
    print(f"STEP 3 [{run_label}]: Weighted outcome model + bootstrap CI")
    print("=" * 68)

    RNG = np.random.default_rng(RANDOM_SEED)

    def weighted_rd_rr(cohort_w):
        T = cohort_w['treatment'].values
        Y = cohort_w[OUTCOME].values
        W = cohort_w['weight'].values
        r1 = np.average(Y[T == 1], weights=W[T == 1])
        r0 = np.average(Y[T == 0], weights=W[T == 0])
        rd = r1 - r0
        rr = r1 / r0 if r0 > 0 else np.nan
        return rd, rr, r1, r0

    def bootstrap_ci(cohort, n_boot=N_BOOT, alpha=0.05):
        """
        Patient-level non-parametric bootstrap.

        Each replicate resamples participants, refits the propensity-score
        model, recalculates and truncates stabilised weights, and re-estimates
        the weighted marginal risks, risk difference, and risk ratio.
        """
        n = len(cohort)
        all_idx = np.arange(n)
        rd_boots, rr_boots = [], []

        for _ in range(n_boot):
            boot_idx = RNG.choice(all_idx, size=n, replace=True)
            boot = cohort.iloc[boot_idx].copy()

            if boot['treatment'].nunique() < 2:
                continue

            boot_w = estimate_weights_silent(boot)
            if boot_w is None:
                continue

            T = boot_w['treatment'].to_numpy()
            Y = boot_w[OUTCOME].to_numpy()
            W = boot_w['weight'].to_numpy()

            t_mask = T == 1
            c_mask = T == 0

            r1 = np.average(Y[t_mask], weights=W[t_mask])
            r0 = np.average(Y[c_mask], weights=W[c_mask])

            # RD remains defined even when one arm has zero events.
            rd_boots.append(r1 - r0)

            # RR is defined whenever the weighted control risk is greater than zero.
            if r0 > 0:
                rr_boots.append(r1 / r0)

        min_valid = max(10, int(n_boot * 0.5))

        if len(rd_boots) < min_valid:
            rd_ci = (np.nan, np.nan)
        else:
            rd_ci = tuple(np.percentile(
                rd_boots,
                [100 * alpha / 2, 100 * (1 - alpha / 2)]))

        if len(rr_boots) < min_valid:
            rr_ci = (np.nan, np.nan)
        else:
            rr_ci = tuple(np.percentile(
                rr_boots,
                [100 * alpha / 2, 100 * (1 - alpha / 2)]))

        return rd_ci, rr_ci, len(rd_boots), len(rr_boots)

    result_rows = []

    ANALYSIS_LABELS = INTERVENTION_LABELS

    _nb3_summary = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_summary.parquet'))
    _nb3_cate = dict(zip(_nb3_summary['arm'].astype(int), _nb3_summary['mean_cate']))
    NB3_SINGLE = {
        'smk'       : _nb3_cate.get(1, np.nan),   # arm 1: no_smk only
        'pa'        : _nb3_cate.get(2, np.nan),   # arm 2: PA only
        'sleep'     : _nb3_cate.get(4, np.nan),   # arm 4: sleep only
        'smk_sleep' : _nb3_cate.get(5, np.nan),   # arm 5: no_smk + sleep
        'pa_sleep'  : _nb3_cate.get(6, np.nan),   # arm 6: PA + sleep
        'smk_pa'    : _nb3_cate.get(3, np.nan)    # arm 3: no_smk + PA
    }

    print(f"\n  Running weighted RD/RR + {N_BOOT}-sample bootstrap CI "
        f"for {len(cohorts)} interventions...")

    for name, wcoh in weighted_cohorts.items():
        label = ANALYSIS_LABELS[name]
        treated_events = int(wcoh.loc[wcoh['treatment'] == 1, OUTCOME].sum())
        control_events = int(wcoh.loc[wcoh['treatment'] == 0, OUTCOME].sum())
        treated_n = int((wcoh['treatment'] == 1).sum())
        control_n = int((wcoh['treatment'] == 0).sum())
        feasible = 'OK' if (treated_n >= MIN_ARM_N and control_n >= MIN_ARM_N and treated_events >= MIN_ARM_EVENTS 
                            and control_events >= MIN_ARM_EVENTS) else 'LOW'

        print(f"    {label}...", end=' ', flush=True)
        rd, rr, r1, r0 = weighted_rd_rr(wcoh)
        rd_ci, rr_ci, n_boot_rd_valid, n_boot_rr_valid = bootstrap_ci(cohorts[name])
        print(f"done (valid RD boots={n_boot_rd_valid}, valid RR boots={n_boot_rr_valid})")

        nb3 = NB3_SINGLE[name]
        direction_ok = (rd < 0) == (nb3 < 0)

        result_rows.append({
            'analysis'     : label,
            'cohort'       : run_label,
            'treated_n'    : treated_n,
            'control_n'    : control_n,
            'treated_events': treated_events,
            'control_events': control_events,
            'feasible'     : feasible,
            'r_treated'    : r1,
            'r_control'    : r0,
            'RD'           : rd,
            'RD_ci_low'    : rd_ci[0],
            'RD_ci_high'   : rd_ci[1],
            'RR'           : rr,
            'RR_ci_low'    : rr_ci[0],
            'RR_ci_high'   : rr_ci[1],
            'nb3_cate'     : nb3,
            'direction_consistent': direction_ok,
            'n_boot_rd_valid': n_boot_rd_valid,
            'n_boot_rr_valid': n_boot_rr_valid,
            'analysis_role': ('single-behaviour contrast'
                            if name in PRIMARY_INTERVENTIONS
                            else 'exploratory combined contrast')})

    results_df = pd.DataFrame(result_rows)
    results_path = os.path.join(OUTPUT_DIR, f'msm_results{suffix}.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\n  {'Analysis':<36} {'treated_n':>9} {'treated_ev':>10} "
        f"{'RD (95% CI)':^26} {'CATE':>9}  {'Feas':>4}  {'Dir':>8}")
    print("  " + "-" * 95)
    for _, row in results_df.iterrows():
        def _fmt(v, fmt='+.4f'):
            return f'{v:{fmt}}' if not (isinstance(v, float) and np.isnan(v)) else '   nan'
        ci_str = f"({_fmt(row['RD_ci_low'])},{_fmt(row['RD_ci_high'])})"
        flag   = 'match' if row['direction_consistent'] else 'MISMATCH'
        print(f"  {row['analysis']:<36} {row['treated_n']:>9,} "
            f"{row['treated_events']:>10,} {row['RD']:>+8.4f} {ci_str:>17} "
            f"{row['nb3_cate']:>+9.4f}  {row['feasible']:>4}  {flag:>8}")
    print(f"  Saved: msm_results{suffix}.csv")

    print("\n" + "=" * 68)
    print(f"STEP 4 [{run_label}]: Sensitivity analysis — E-value")
    print("=" * 68)

    def evalue(rr):
        if pd.isna(rr) or rr <= 0:
            return np.nan
        if np.isclose(rr, 1.0):
            return 1.0
        if rr > 1:
            return rr + np.sqrt(rr * (rr - 1))
        rr_inv = 1 / rr
        return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

    def evalue_ci_bound(rr, rr_lo, rr_hi):
        """Return the RR confidence bound closest to 1 and its E-value."""
        if pd.isna(rr) or pd.isna(rr_lo) or pd.isna(rr_hi):
            return np.nan, np.nan
        if rr_lo <= 1 <= rr_hi:
            return 1.0, 1.0
        ci_bound = rr_hi if rr < 1 else rr_lo
        return ci_bound, evalue(ci_bound)

    print(f"\n  {'Analysis':<36} {'RR':>7} {'E-val(point)':>13} "
        f"{'CI-bound RR':>11} {'E-val(CI)':>10}  Note")
    print("  " + "-" * 110)

    evalue_rows = []
    for _, row in results_df.iterrows():
        rr = row['RR']
        rr_lo = row['RR_ci_low']
        rr_hi = row['RR_ci_high']

        if row['feasible'] != 'OK':
            ev_point = np.nan
            ci_bound = np.nan
            ev_ci = np.nan
            note = "not reported: insufficient event support"
        else:
            ev_point = evalue(rr)
            ci_bound, ev_ci = evalue_ci_bound(rr, rr_lo, rr_hi)

            if pd.isna(ev_ci):
                note = "confidence interval unavailable"
            elif ev_ci == 1.0:
                note = "confidence interval includes the null"
            else:
                note = "confidence interval excludes the null"

        rr_text = f"{rr:.4f}" if pd.notna(rr) else "nan"
        evp_text = f"{ev_point:.3f}" if pd.notna(ev_point) else "nan"
        cib_text = f"{ci_bound:.4f}" if pd.notna(ci_bound) else "nan"
        evc_text = f"{ev_ci:.3f}" if pd.notna(ev_ci) else "nan"

        print(f"  {row['analysis']:<36} {rr_text:>7} {evp_text:>13} "
            f"{cib_text:>11} {evc_text:>10}  {note}")

        evalue_rows.append({
            'analysis': row['analysis'],
            'cohort': run_label,
            'analysis_role': row['analysis_role'],
            'feasible': row['feasible'],
            'RR': rr,
            'RR_ci_low': rr_lo,
            'RR_ci_high': rr_hi,
            'evalue_point': ev_point,
            'ci_bound_used': ci_bound,
            'evalue_ci': ev_ci,
            'note': note})

    evalue_df = pd.DataFrame(evalue_rows)
    evalue_path = os.path.join(OUTPUT_DIR, f'msm_evalues{suffix}.csv')
    evalue_df.to_csv(evalue_path, index=False)
    print(f"  Saved: msm_evalues{suffix}.csv")

    return results_df, evalue_df


# Run pipeline twice: MAIN analysis, then SUB-ANALYSIS
results_main, evalues_main = run_pipeline(
    cohorts_main, suffix='', run_label='FULL-COHORT DIAGNOSTIC')

results_sub, evalues_sub = run_pipeline(
    cohorts_sub, suffix='_sub', run_label='POST-IMAGING OUTCOME SUB-ANALYSIS')

# Combined comparison table: main vs sub, side by side
print("\n" + "=" * 100)

def _f(v):
    """Format float for comparison table; returns nan string if NaN."""
    return f'{v:>+9.4f}' if not (isinstance(v, float) and np.isnan(v)) else '      nan'

print("FINAL COMPARISON — FULL-COHORT DIAGNOSTIC VS POST-IMAGING SUB-ANALYSIS")
print("  events     = treated-arm OUTCOME=1 count (post-imaging events are low")
print("               because pre-imaging-event participants are removed --")
print("               see 'pre-imaging event %' column and the reverse-")
print("               causality contamination check earlier in this run)")
print("  Dir        = does RD's sign match the causal-forest CATE's sign?")
print("=" * 100)
print(f"{'Intervention':<36} {'RD full':>9} {'ev':>4} {'Dir':>5}  | "
    f"{'RD sub':>9} {'ev':>4} {'Dir':>5} {'Feas':>5}  | "
    f"{'CATE':>9}  {'pre-img ev %':>12}")
print("-" * 100)
for name in INTERVENTIONS:
    label = INTERVENTION_LABELS[name]
    rm = results_main[results_main['analysis'] == label].iloc[0]
    rs = results_sub[results_sub['analysis'] == label].iloc[0]
    contam_pct = contam_df.loc[contam_df['intervention'] == label,
                                'pre_imaging_event_pct'].iloc[0]
    print(f"{label:<36} {rm['RD']:>+9.4f} {rm['treated_events']:>4} "
        f"{'Y' if rm['direction_consistent'] else 'N':>5}  | "
        f"{rs['RD']:>+9.4f} {rs['treated_events']:>4} "
        f"{'Y' if rs['direction_consistent'] else 'N':>5} {rs['feasible']:>5}  | "
        f"{_f(rm['nb3_cate'])}  {contam_pct:>11.1f}%")

combined_results = pd.concat([results_main, results_sub], ignore_index=True)
combined_results.to_csv(os.path.join(OUTPUT_DIR, 'msm_results_combined.csv'), index=False)

combined_evalues = pd.concat([evalues_main, evalues_sub], ignore_index=True)
combined_evalues.to_csv(os.path.join(OUTPUT_DIR, 'msm_evalues_combined.csv'), index=False)


print("  reverse_causality_contamination.csv               (Step 1)")
print("  msm_cohort_{name}.parquet / _sub.parquet           (Step 1)")
print("  msm_weighted_{name}.parquet / _sub.parquet          (Step 2)")
print("  msm_love_plot.png / msm_love_plot_sub.png           (Step 2)")
print("  msm_results.csv / msm_results_sub.csv / _combined   (Step 3)")
print("  msm_evalues.csv / msm_evalues_sub.csv / _combined   (Step 4)")