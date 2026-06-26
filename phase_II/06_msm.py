import os, sys
import numpy as np
import pandas as pd

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


# 1A. Load longitudinal behaviour fields + outcome

print("=" * 68)
print("Data loading and cohort construction")
print("=" * 68)

file_tab = os.path.join(BASE_PATH, 'ukb_tabular_data_causal_analysis.tsv')
tab_cols = ['eid',
            '20116-0.0', '20116-2.0',
            '884-0.0',   '884-2.0',
            '894-0.0',   '894-2.0',
            '904-0.0',   '904-2.0',
            '914-0.0',   '914-2.0',
            '1160-0.0',  '1160-2.0',
            '20117-0.0']            # alcohol status (for alc_curr)

print("\nLoading longitudinal fields + alcohol")
tab = pd.read_csv(file_tab, sep='\t', usecols=tab_cols)
print(f"  tabular rows: {len(tab):,}")

file_out = os.path.join(BASE_PATH, 'group_1_outcomes_df_without_qc_df.tsv')
out = pd.read_csv(file_out, sep='\t', usecols=['eid', OUTCOME])
df  = tab.merge(out, on='eid', how='inner')
print(f"  after outcome merge: {len(df):,}")

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
                '20111.1', '20111.2',   # sibling CVD history
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
    (df['20107.1'] == 1) | (df['20107.2'] == 1)
).astype(float)
df.loc[df['20107.1'].isna() & df['20107.2'].isna(), 'FH_cvd_f'] = np.nan

# family history CVD — mother
df['FH_cvd_m'] = (
    (df['20110.1'] == 1) | (df['20110.2'] == 1)
).astype(float)
df.loc[df['20110.1'].isna() & df['20110.2'].isna(), 'FH_cvd_m'] = np.nan

# family history CVD — sibling
df['FH_cvd_sib'] = (
    (df['20111.1'] == 1) | (df['20111.2'] == 1)
).astype(float)
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

def pa_healthy(mod_days, mod_mins, vig_days, vig_mins):
    idx = mod_days.index
    md  = _clip_neg_na(mod_days)
    mm  = _clip_neg_na(mod_mins)
    vd  = _clip_neg_na(vig_days)
    vm  = _clip_neg_na(vig_mins)
    mod_total = pd.Series(np.where(md == 0, 0.0, md * mm), index=idx)
    vig_total = pd.Series(np.where(vd == 0, 0.0, vd * vm), index=idx)
    active = pd.Series(
        ((mod_total >= 150) | (vig_total >= 75)).astype(float), index=idx)
    active[md.isna() & vd.isna()] = np.nan
    active[(((md > 0) & mm.isna()) | ((vd > 0) & vm.isna())) & (active != 1)] = np.nan
    return active.values

smk_b = pd.Series(smk_healthy(df['20116-0.0']), index=df.index)
smk_i = pd.Series(smk_healthy(df['20116-2.0']), index=df.index)
pa_b  = pd.Series(pa_healthy(df['884-0.0'], df['894-0.0'],
                            df['904-0.0'], df['914-0.0']), index=df.index)
pa_i  = pd.Series(pa_healthy(df['884-2.0'], df['894-2.0'],
                            df['904-2.0'], df['914-2.0']), index=df.index)
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
    pd.to_timedelta(events['def_CVD_AF_HF_AFTER_days_from_baseline'], unit='D')
)

img_dates = pd.read_csv(os.path.join(BASE_PATH, 'imaging_visit_date.tsv'),
                        sep='\t').rename(columns={'53-2.0': 'imaging_date'})
img_dates['imaging_date'] = pd.to_datetime(img_dates['imaging_date'])

events = events.merge(img_dates, on='eid', how='left')
events['event_before_imaging'] = (
    events['def_CVD_AF_HF_AFTER_date'].notna() &
    events['imaging_date'].notna() &
    (events['def_CVD_AF_HF_AFTER_date'] < events['imaging_date'])
)
exclude_eids = set(events.loc[events['event_before_imaging'], 'eid'])
print(f"  Participants with CVD event BEFORE imaging: {len(exclude_eids):,}")

# boolean mask aligned to df.index: True = exclude from sub-analysis
exclude_mask = df['eid'].isin(exclude_eids)
print(f"  df (main)            : {len(df):,}")
print(f"  df_sub (excl. pre-imaging events): {(~exclude_mask).sum():,}  "
    f"({exclude_mask.sum():,} excluded)")


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

    feasible = 'OK' if (treated_n >= 500 and treated_events >= 50) else 'LOW'

    stats = {
        'label': label,
        'treated_n': treated_n, 'treated_events': treated_events,
        'treated_event_rate': treated_event_rate,
        'control_n': control_n, 'control_events': control_events,
        'control_event_rate': control_event_rate,
        'feasible': feasible,
    }
    return cohort, stats


print("\n" + "=" * 68)
print("Cohort construction (main = full imaging pool, "
    "sub = excl. pre-imaging CVD events)")
print("=" * 68)

not_excluded = ~exclude_mask

INTERVENTIONS = ['smk', 'pa', 'sleep', 'pa_sleep']
INTERVENTION_LABELS = {
    'smk': 'Quit smoking', 'pa': 'Increase PA',
    'sleep': 'Adequate sleep', 'pa_sleep': 'PA + Sleep'}

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
    'pa_sleep': dict(
        base_mask=(pa_b.notna() & pa_i.notna() &
                    slp_b.notna() & slp_i.notna() &
                    (pa_b == 0) & (slp_b == 0)),
        treat_mask=(pa_i == 1) & (slp_i == 1),
        ctrl_mask=(pa_i == 0) & (slp_i == 0)),
}

cohorts_main, cohorts_sub = {}, {}
stats_main, stats_sub = {}, {}

for name, spec in _mask_specs.items():
    label = INTERVENTION_LABELS[name]
    cohorts_main[name], stats_main[name] = build_cohort(
        f'{label} [main]', **spec)
    cohorts_sub[name], stats_sub[name] = build_cohort(
        f'{label} [sub]', **spec, restrict_mask=not_excluded)

# --- Reverse-causality contamination check ---------------------------
# How much of the main analysis's outcome-positive signal comes from
# participants whose CVD/AF/HF event occurred BEFORE their imaging visit
# (i.e. their imaging-time behaviour may be a *reaction* to an event that
# already happened, not a cause of a future one -- "sick quitter/sleeper"
# bias). This was previously checked in an ad-hoc diagnostic script; it
# is folded in here permanently since it's central to interpreting why
# the sub-analysis event counts collapse relative to the main analysis.
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
print("Interpretation: a high % here means the MAIN analysis's outcome "
    "signal for that\nintervention is dominated by participants who were "
    "likely diagnosed BEFORE their\nimaging-visit behaviour was measured "
    "-- i.e. reverse causality, not a causal effect\nof the behaviour. "
    "The SUB-ANALYSIS below removes these participants, which is why "
    "\nits event counts are much lower despite only a small drop in "
    "total sample size.")

# --- Unified cohort-construction summary table (main vs sub) ---------
print("\n" + "=" * 68)
print("COHORT SUMMARY — main vs sub, side by side")
print("  n        = number of people in that arm")
print("  events   = number of those people with OUTCOME=1")
print("  rate     = events / n")
print("=" * 68)
hdr = (f"{'Intervention':<16} {'Arm':<9} "
    f"{'n (main)':>9} {'ev (main)':>10} {'rate (main)':>12} | "
    f"{'n (sub)':>9} {'ev (sub)':>9} {'rate (sub)':>11}")
print(hdr)
print("-" * len(hdr))
for name in INTERVENTIONS:
    sm, ss = stats_main[name], stats_sub[name]
    label = INTERVENTION_LABELS[name]
    print(f"{label:<16} {'treated':<9} "
        f"{sm['treated_n']:>9,} {sm['treated_events']:>10,} "
        f"{sm['treated_event_rate']:>12.4f} | "
        f"{ss['treated_n']:>9,} {ss['treated_events']:>9,} "
        f"{ss['treated_event_rate']:>11.4f}")
    print(f"{'':<16} {'control':<9} "
        f"{sm['control_n']:>9,} {sm['control_events']:>10,} "
        f"{sm['control_event_rate']:>12.4f} | "
        f"{ss['control_n']:>9,} {ss['control_events']:>9,} "
        f"{ss['control_event_rate']:>11.4f}")
    feas_flag = '' if ss['feasible'] == 'OK' else '  <- sub LOW feasibility (Peduzzi 1996: need treated events>=50)'
    print(f"{'':<16} {'feasible':<9} {sm['feasible']:>9} {'':>10} {'':>12} | "
        f"{ss['feasible']:>9}{feas_flag}")
    print()


# 1F. Save cohorts (both main and sub)

for name, coh in cohorts_main.items():
    coh.to_parquet(os.path.join(OUTPUT_DIR, f'msm_cohort_{name}.parquet'),
                index=False)
for name, coh in cohorts_sub.items():
    coh.to_parquet(os.path.join(OUTPUT_DIR, f'msm_cohort_{name}_sub.parquet'),
                index=False)
print(f"Saved 8 cohort parquet files (4 interventions x main/sub) to "
    f"{OUTPUT_DIR}")

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
                'MAIN' or 'SUB-ANALYSIS (excl. pre-imaging CVD events)'

    Returns results_df, evalue_df for the caller to combine across runs.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

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
        ps = lr.predict_proba(W_std)[:, 1]
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

        return coh, pd.DataFrame(smd_rows)

    weighted_cohorts = {}
    smd_dfs = {}

    for name, coh in cohorts.items():
        wcoh, smd = estimate_weights(coh, name)
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

    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    titles = {'smk': 'Quit smoking', 'pa': 'Increase PA',
            'sleep': 'Adequate sleep', 'pa_sleep': 'PA + Sleep'}

    for ax, (name, smd) in zip(axes, smd_dfs.items()):
        ax.scatter(smd['smd_before'], smd['confounder'],
                color='#E07A5F', label='Before IPTW', zorder=3)
        ax.scatter(smd['smd_after'],  smd['confounder'],
                color='#3D5A80', label='After IPTW',  zorder=3)
        ax.axvline(0.1, color='grey', lw=1, ls='--', alpha=0.7)
        ax.set_xlabel('Standardised Mean Difference')
        ax.set_title(titles[name], fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
        if ax is axes[0]:
            ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f'Love Plot: Covariate Balance Before and After IPTW [{run_label}]\n'
                '(dashed line = SMD 0.1 threshold)', fontsize=12)
    fig.tight_layout()
    love_path = os.path.join(FIGURES_DIR, f'msm_love_plot{suffix}.png')
    fig.savefig(love_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved figure: msm_love_plot{suffix}.png")

    print(f"\n{'='*68}")
    print("Check SMD table: all 'after' values should be < 0.1.")

    print("\n" + "=" * 68)
    print(f"STEP 3 [{run_label}]: Weighted outcome model + bootstrap CI")
    print("=" * 68)

    N_BOOT = 500
    RNG = np.random.default_rng(42)

    def weighted_rd_rr(cohort_w):
        T = cohort_w['treatment'].values
        Y = cohort_w[OUTCOME].values
        W = cohort_w['weight'].values
        r1 = np.average(Y[T == 1], weights=W[T == 1])
        r0 = np.average(Y[T == 0], weights=W[T == 0])
        rd = r1 - r0
        rr = r1 / r0 if r0 > 0 else np.nan
        return rd, rr, r1, r0

    def bootstrap_ci(cohort_w, n_boot=N_BOOT, alpha=0.05):
        T = cohort_w['treatment'].values
        Y = cohort_w[OUTCOME].values
        W = cohort_w['weight'].values
        idx_t = np.where(T == 1)[0]
        idx_c = np.where(T == 0)[0]
        rd_boots, rr_boots = [], []
        for _ in range(n_boot):
            bt = RNG.choice(idx_t, size=len(idx_t), replace=True)
            bc = RNG.choice(idx_c, size=len(idx_c), replace=True)
            r1b = np.average(Y[bt], weights=W[bt])
            r0b = np.average(Y[bc], weights=W[bc])
            rd_boots.append(r1b - r0b)
            rr_boots.append(r1b / r0b if r0b > 0 else np.nan)
        rd_arr = np.array(rd_boots)
        rr_arr = np.array([x for x in rr_boots if not np.isnan(x)])
        rd_ci = (np.percentile(rd_arr, 100 * alpha / 2),
                 np.percentile(rd_arr, 100 * (1 - alpha / 2)))
        rr_ci = (np.percentile(rr_arr, 100 * alpha / 2),
                 np.percentile(rr_arr, 100 * (1 - alpha / 2)))
        return rd_ci, rr_ci

    result_rows = []

    ANALYSIS_LABELS = {
        'smk'      : 'Quit smoking',
        'pa'       : 'Increase PA',
        'sleep'    : 'Adequate sleep',
        'pa_sleep' : 'PA + Sleep'}

    _nb3_summary = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_summary.parquet'))
    _nb3_cate = dict(zip(_nb3_summary['arm'].astype(int), _nb3_summary['mean_cate']))
    NB3_SINGLE = {
        'smk'      : _nb3_cate.get(1, np.nan),
        'pa'       : _nb3_cate.get(2, np.nan),
        'sleep'    : _nb3_cate.get(4, np.nan),
        'pa_sleep' : _nb3_cate.get(6, np.nan),
    }

    print(f"\n  Running weighted RD/RR + {N_BOOT}-sample bootstrap CI "
        f"for 4 interventions...")

    for name, wcoh in weighted_cohorts.items():
        label = ANALYSIS_LABELS[name]
        treated_events = int(wcoh.loc[wcoh['treatment'] == 1, OUTCOME].sum())
        control_events = int(wcoh.loc[wcoh['treatment'] == 0, OUTCOME].sum())
        treated_n = int((wcoh['treatment'] == 1).sum())
        control_n = int((wcoh['treatment'] == 0).sum())
        feasible = 'OK' if (treated_n >= 500 and treated_events >= 50) else 'LOW'

        print(f"    {label}...", end=' ', flush=True)
        rd, rr, r1, r0 = weighted_rd_rr(wcoh)
        rd_ci, rr_ci   = bootstrap_ci(wcoh)
        print("done")

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
        })

    results_df = pd.DataFrame(result_rows)
    results_path = os.path.join(OUTPUT_DIR, f'msm_results{suffix}.csv')
    results_df.to_csv(results_path, index=False)

    print(f"\n  {'Analysis':<16} {'treated_n':>9} {'treated_ev':>10} "
        f"{'RD (95% CI)':^26} {'CATE':>9}  {'Feas':>4}  {'Dir':>8}")
    print("  " + "-" * 95)
    for _, row in results_df.iterrows():
        ci_str = f"({row['RD_ci_low']:+.4f},{row['RD_ci_high']:+.4f})"
        flag   = 'match' if row['direction_consistent'] else 'MISMATCH'
        print(f"  {row['analysis']:<16} {row['treated_n']:>9,} "
            f"{row['treated_events']:>10,} {row['RD']:>+8.4f} {ci_str:>17} "
            f"{row['nb3_cate']:>+9.4f}  {row['feasible']:>4}  {flag:>8}")
    print(f"  Saved: msm_results{suffix}.csv")

    print("\n" + "=" * 68)
    print(f"STEP 4 [{run_label}]: Sensitivity analysis — E-value")
    print("=" * 68)

    def evalue(rr):
        if rr <= 0 or np.isnan(rr):
            return np.nan
        if rr >= 1:
            return rr + np.sqrt(rr * (rr - 1))
        else:
            rr_inv = 1 / rr
            return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))

    print(f"\n  {'Analysis':<16} {'RR':>7} {'E-val(point)':>13} "
        f"{'CI-bound RR':>11} {'E-val(CI)':>10}  Note")
    print("  " + "-" * 80)

    evalue_rows = []
    for _, row in results_df.iterrows():
        rr    = row['RR']
        rr_lo = row['RR_ci_low']
        rr_hi = row['RR_ci_high']

        ev_point = evalue(rr)
        ci_bound = rr_hi if rr < 1 else rr_lo
        ev_ci = evalue(ci_bound)

        direction_ok = row['direction_consistent']
        note = ("consistent with causal forest" if direction_ok
                else "caution: reverse causality")

        print(f"  {row['analysis']:<16} {rr:>7.4f} {ev_point:>13.3f} "
            f"{ci_bound:>11.4f} {ev_ci:>10.3f}  {note}")

        evalue_rows.append({
            'analysis'       : row['analysis'],
            'cohort'         : run_label,
            'RR'             : rr,
            'evalue_point'   : ev_point,
            'ci_bound_used'  : ci_bound,
            'evalue_ci'      : ev_ci,
            'direction_ok'   : direction_ok
        })

    evalue_df = pd.DataFrame(evalue_rows)
    evalue_path = os.path.join(OUTPUT_DIR, f'msm_evalues{suffix}.csv')
    evalue_df.to_csv(evalue_path, index=False)
    print(f"  Saved: msm_evalues{suffix}.csv")

    return results_df, evalue_df


# ============================================================
# Run pipeline twice: MAIN analysis, then SUB-ANALYSIS
# ============================================================
results_main, evalues_main = run_pipeline(
    cohorts_main, suffix='', run_label='MAIN (full imaging pool)')

results_sub, evalues_sub = run_pipeline(
    cohorts_sub, suffix='_sub', run_label='SUB-ANALYSIS (excl. pre-imaging CVD events)')

# ============================================================
# Combined comparison table: main vs sub, side by side
# ============================================================
print("\n" + "=" * 100)
print("FINAL COMPARISON — MAIN vs SUB-ANALYSIS, all interventions")
print("  events     = treated-arm OUTCOME=1 count (sub-analysis events are low")
print("               because pre-imaging-event participants are removed --")
print("               see 'pre-imaging event %' column and the reverse-")
print("               causality contamination check earlier in this run)")
print("  Dir        = does RD's sign match the causal-forest CATE's sign?")
print("=" * 100)
print(f"{'Intervention':<16} {'RD main':>9} {'ev':>4} {'Dir':>5}  | "
    f"{'RD sub':>9} {'ev':>4} {'Dir':>5} {'Feas':>5}  | "
    f"{'CATE':>9}  {'pre-img ev %':>12}")
print("-" * 100)
for name in INTERVENTIONS:
    label = INTERVENTION_LABELS[name]
    rm = results_main[results_main['analysis'] == label].iloc[0]
    rs = results_sub[results_sub['analysis'] == label].iloc[0]
    contam_pct = contam_df.loc[contam_df['intervention'] == label,
                                'pre_imaging_event_pct'].iloc[0]
    print(f"{label:<16} {rm['RD']:>+9.4f} {rm['treated_events']:>4} "
        f"{'Y' if rm['direction_consistent'] else 'N':>5}  | "
        f"{rs['RD']:>+9.4f} {rs['treated_events']:>4} "
        f"{'Y' if rs['direction_consistent'] else 'N':>5} {rs['feasible']:>5}  | "
        f"{rm['cate']:>+9.4f}  {contam_pct:>11.1f}%")

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