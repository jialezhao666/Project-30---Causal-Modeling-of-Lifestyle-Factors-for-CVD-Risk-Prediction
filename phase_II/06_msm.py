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


# 1. Load longitudinal behaviour fields + outcome

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



# 1b. Load confounders from exposure file (full population)

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



# 1E. Cohort construction

def build_cohort(label, base_mask, treat_mask, ctrl_mask):
    """
    Construct treated/control cohort with complete W.
    base_mask : eligible at baseline (unhealthy + both visits observed)
    treat_mask : imaging = healthy (treated)
    ctrl_mask : imaging = unhealthy (control)
    """
    w_complete = pd.Series(True, index=df.index)
    for c in CONFOUNDERS:
        w_complete = w_complete & df[c].notna()

    eligible = base_mask & w_complete
    treated  = eligible & treat_mask
    control  = eligible & ctrl_mask

    nt = int(treated.sum())
    nc = int(control.sum())
    et = int(df.loc[treated, OUTCOME].sum())
    ec = int(df.loc[control, OUTCOME].sum())
    rt = df.loc[treated, OUTCOME].mean() if nt > 0 else np.nan
    rc = df.loc[control, OUTCOME].mean() if nc > 0 else np.nan

    cohort = df.loc[treated | control,
                    ['eid', OUTCOME] + CONFOUNDERS].copy()
    cohort['treatment'] = 0
    cohort.loc[treated[treated | control].values.astype(bool),
            'treatment'] = 1

    feasible = 'OK' if (nt >= 500 and et >= 50) else 'LOW'
    print(f"\n  {label}")
    print(f"    treated  : {nt:>7,}  events={et:>4}  rate={rt:.4f}")
    print(f"    control  : {nc:>7,}  events={ec:>4}  rate={rc:.4f}")
    print(f"    total    : {nt+nc:>7,}  feasible={feasible}")
    return cohort


print("\n" + "=" * 68)
print("Cohort construction")
print("=" * 68)

# For each behaviour, we define the baseline eligibility mask as those
cohort_smk = build_cohort(
    'Quit smoking',
    base_mask = smk_b.notna() & smk_i.notna() & (smk_b == 0),
    treat_mask = smk_i == 1,
    ctrl_mask = smk_i == 0,
)

cohort_pa = build_cohort(
    'Increase PA',
    base_mask = pa_b.notna() & pa_i.notna() & (pa_b == 0),
    treat_mask = pa_i == 1,
    ctrl_mask = pa_i == 0,
)

cohort_slp = build_cohort(
    'Adequate sleep',
    base_mask = slp_b.notna() & slp_i.notna() & (slp_b == 0),
    treat_mask = slp_i == 1,
    ctrl_mask  = slp_i == 0,
)

cohort_pair = build_cohort(
    'PA + Sleep (pairwise joint)',
    base_mask = (pa_b.notna() & pa_i.notna() &
                slp_b.notna() & slp_i.notna() &
                (pa_b == 0) & (slp_b == 0)),
    treat_mask = (pa_i == 1) & (slp_i == 1),
    ctrl_mask = (pa_i == 0) & (slp_i == 0))


# 1F. Save cohorts

cohorts = {
    'smk': cohort_smk,
    'pa' : cohort_pa,
    'sleep' : cohort_slp,
    'pa_sleep' : cohort_pair}

print(f"\nSaving cohorts")
for name, coh in cohorts.items():
    path = os.path.join(OUTPUT_DIR, f'msm_cohort_{name}.parquet')
    coh.to_parquet(path, index=False)
    nt = int((coh['treatment'] == 1).sum())
    nc = int((coh['treatment'] == 0).sum())
    print(f"  msm_cohort_{name}.parquet  "
        f"(n={len(coh):,}, treated={nt:,}, control={nc:,})")

print(f"\n{'='*68}")
print("STEP 1 complete.")
print("Expected vs 05_transition_matrix.py (before W filtering):")
print("  smk    : treated≈2,750  control≈1,945")
print("  pa     : treated≈13,198 control≈15,282")
print("  sleep  : treated≈6,751  control≈9,246")
print("  pa_slp : treated≈1,291  control≈2,014")
print("Numbers will be slightly lower due to W completeness filtering.")
print("If loss > 5%, check confounder missingness above.")
print("Proceed to Step 2 (IPTW weights) once numbers look reasonable.")


# 2. IPTW weight estimation + diagnostics

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("\n" + "=" * 68)
print("STEP 2: IPTW weight estimation + diagnostics")
print("=" * 68)

def estimate_weights(cohort, label, truncate_pct=(1, 99)):
    """
    Estimate stabilised IPTW weights for a binary treatment cohort.
    """
    coh = cohort.copy()
    T = coh['treatment'].values
    W = coh[CONFOUNDERS].values

    # standardise W for logistic regression stability
    scaler = StandardScaler()
    W_std  = scaler.fit_transform(W)

    # propensity score P(T=1 | W)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(W_std, T)
    ps = lr.predict_proba(W_std)[:, 1]   # P(T=1|W)
    coh['ps'] = ps

    # marginal treatment probability P(T=1)
    p1 = T.mean()
    p0 = 1 - p1

    # stabilised weights
    w_raw = np.where(T == 1, p1 / ps, p0 / (1 - ps))
    coh['weight_raw'] = w_raw

    # truncate at percentiles
    lo = np.percentile(w_raw, truncate_pct[0])
    hi = np.percentile(w_raw, truncate_pct[1])
    w_trunc = np.clip(w_raw, lo, hi)
    coh['weight'] = w_trunc

    # diagnostics 
    print(f"\n  {label}")
    print(f"    Propensity score  : mean={ps.mean():.3f}  "
        f"min={ps.min():.3f}  max={ps.max():.3f}")
    print(f"    Weights (raw)     : mean={w_raw.mean():.3f}  "
        f"min={w_raw.min():.3f}  max={w_raw.max():.3f}  "
        f"std={w_raw.std():.3f}")
    print(f"    Truncation bounds : [{lo:.3f}, {hi:.3f}]")
    print(f"    Weights (trunc)   : mean={w_trunc.mean():.3f}  "
        f"max={w_trunc.max():.3f}  std={w_trunc.std():.3f}")

    #  SMD before and after weighting 
    print(f"    SMD (before → after IPTW, threshold < 0.1):")
    smd_rows = []
    for c in CONFOUNDERS:
        x = coh[c].values.astype(float)
        t_mask = T == 1
        c_mask = T == 0

        # unweighted SMD
        m1u = x[t_mask].mean();  m0u = x[c_mask].mean()
        s1u = x[t_mask].std();   s0u = x[c_mask].std()
        pool_std = np.sqrt((s1u**2 + s0u**2) / 2)
        smd_before = abs(m1u - m0u) / pool_std if pool_std > 0 else 0

        # weighted SMD
        w1 = w_trunc[t_mask]; w0 = w_trunc[c_mask]
        m1w = np.average(x[t_mask], weights=w1)
        m0w = np.average(x[c_mask], weights=w0)
        v1w = np.average((x[t_mask] - m1w)**2, weights=w1)
        v0w = np.average((x[c_mask] - m0w)**2, weights=w0)
        pool_std_w = np.sqrt((v1w + v0w) / 2)
        smd_after = abs(m1w - m0w) / pool_std_w if pool_std_w > 0 else 0

        flag = '' if smd_after < 0.1 else '  ← IMBALANCED'
        print(f"      {c:25s}: {smd_before:.3f} → {smd_after:.3f}{flag}")
        smd_rows.append({'confounder': c,
                        'smd_before': smd_before,
                        'smd_after': smd_after})

    #coh['smd_df'] = None   # placeholder; return separately
    return coh, pd.DataFrame(smd_rows)


# run for each cohort
weighted_cohorts = {}
smd_dfs = {}

for name, coh in cohorts.items():
    wcoh, smd = estimate_weights(coh, name)
    weighted_cohorts[name] = wcoh
    smd_dfs[name] = smd

#  save weighted cohorts 
print(f"\nSaving weighted cohorts")
for name, wcoh in weighted_cohorts.items():
    save_cols = ['eid', OUTCOME, 'treatment', 'ps',
                'weight_raw', 'weight'] + CONFOUNDERS
    save_cols = [c for c in save_cols if c in wcoh.columns]
    path = os.path.join(OUTPUT_DIR, f'msm_weighted_{name}.parquet')
    wcoh[save_cols].to_parquet(path, index=False)
    print(f"  msm_weighted_{name}.parquet")

#  love plot: SMD before vs after for all cohorts 
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

fig.suptitle('Love Plot: Covariate Balance Before and After IPTW\n'
            '(dashed line = SMD 0.1 threshold)', fontsize=12)
fig.tight_layout()
love_path = os.path.join(FIGURES_DIR, 'msm_love_plot.png')
fig.savefig(love_path, dpi=150, bbox_inches='tight')
print(f"\nSaved figure: {love_path}")

print(f"\n{'='*68}")
print("Check SMD table: all 'after' values should be < 0.1.")
print("If any SMD after > 0.1, consider adding interaction terms to")
print("the propensity model before proceeding to Step 3.")



# 3. Weighted outcome model + bootstrap CI
# Estimand: marginal risk difference (RD) and risk ratio (RR) treated vs control, after IPTW adjustment

from scipy import stats

print("\n" + "=" * 68)
print("STEP 3: Weighted outcome model + bootstrap CI")
print("=" * 68)

N_BOOT = 500
RNG = np.random.default_rng(42)

def weighted_rd_rr(cohort_w):
    """
    Compute weighted risk difference (RD) and risk ratio (RR).
    cohort_w must have columns: treatment, OUTCOME, weight
    """
    T = cohort_w['treatment'].values
    Y = cohort_w[OUTCOME].values
    W = cohort_w['weight'].values

    # weighted risk in each arm
    r1 = np.average(Y[T == 1], weights=W[T == 1])
    r0 = np.average(Y[T == 0], weights=W[T == 0])

    rd = r1 - r0
    rr = r1 / r0 if r0 > 0 else np.nan
    return rd, rr, r1, r0


def bootstrap_ci(cohort_w, n_boot=N_BOOT, alpha=0.05):
    """
    Bootstrap percentile CI for RD and RR.
    Resamples within treated and control separately to preserve balance.
    """
    T = cohort_w['treatment'].values
    Y = cohort_w[OUTCOME].values
    W = cohort_w['weight'].values

    idx_t = np.where(T == 1)[0]
    idx_c = np.where(T == 0)[0]

    rd_boots = []
    rr_boots = []

    for _ in range(n_boot):
        # resample within each arm
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
    'smk'      : _nb3_cate.get(1, np.nan),   # arm 1: no_smk only
    'pa'       : _nb3_cate.get(2, np.nan),   # arm 2: PA only
    'sleep'    : _nb3_cate.get(4, np.nan),   # arm 4: sleep only
    'pa_sleep' : _nb3_cate.get(6, np.nan),   # arm 6: PA + sleep
}
print(f"  Loaded NB3 CATE: {NB3_SINGLE}")

#NB3_SINGLE = {
#    'smk'      : -0.0710,   # arm 1 no_smk only
#    'pa'       : -0.0280,   # arm 2 PA only
#    'sleep'    : -0.0390,   # arm 4 sleep only
#    'pa_sleep' : -0.0493}   # arm 6 PA + sleep


print(f"\n  {'Analysis':<22} {'r_treated':>10} {'r_control':>10} "
    f"{'RD':>8} {'95% CI':>18} {'RR':>6} {'95% CI':>16}  NB3 CATE")
print("  " + "-" * 105)

for name, wcoh in weighted_cohorts.items():
    label = ANALYSIS_LABELS[name]
    print(f"\n  Running bootstrap for {label} (n_boot={N_BOOT})...",
        end=' ', flush=True)

    rd, rr, r1, r0 = weighted_rd_rr(wcoh)
    rd_ci, rr_ci   = bootstrap_ci(wcoh)

    nb3 = NB3_SINGLE[name]
    direction_ok = (rd < 0) == (nb3 < 0)
    direction_flag = '' if direction_ok else '  ← DIRECTION MISMATCH'

    print("done")
    print(f"    r_treated={r1:.4f}  r_control={r0:.4f}")
    print(f"    RD = {rd:+.4f}  95% CI ({rd_ci[0]:+.4f}, {rd_ci[1]:+.4f})")
    print(f"    RR = {rr:.4f}   95% CI ({rr_ci[0]:.4f}, {rr_ci[1]:.4f})")
    print(f"    NB3 CATE = {nb3:+.4f}  "
        f"direction consistent: {direction_ok}{direction_flag}")

    result_rows.append({
        'analysis'     : label,
        'n_treated'    : int((wcoh['treatment'] == 1).sum()),
        'n_control'    : int((wcoh['treatment'] == 0).sum()),
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
results_path = os.path.join(OUTPUT_DIR, 'msm_results.csv')
results_df.to_csv(results_path, index=False)
print(f"\nSaved: {results_path}")

#  summary table 
print(f"\n{'='*68}")
print("SUMMARY TABLE — MSM Results vs Cross-sectional CATE")
print(f"{'='*68}")
print(f"{'Analysis':<22} {'RD (95% CI)':^28} {'CATE':>10}  Direction")
print("-" * 75)
for _, row in results_df.iterrows():
    ci_str = f"({row['RD_ci_low']:+.4f}, {row['RD_ci_high']:+.4f})"
    rd_str = f"{row['RD']:+.4f}"
    flag   = '✓' if row['direction_consistent'] else '✗ MISMATCH'
    print(f"{row['analysis']:<22} {rd_str:>8}  {ci_str:<24} "
        f"{row['nb3_cate']:>+10.4f}  {flag}")

print(f"\n{'='*68}")
print("Interpretation guide:")
print("  RD < 0 + CI entirely < 0 → significant protective effect")
print("  RD < 0 + CI crosses 0   → protective trend, not significant")
print("  RD > 0                   → reverse causality not fully adjusted")
print("                             (expected for smk/sleep, check magnitude)")
print("  Direction consistent     → MSM and NB3 agree on sign of effect")



# 4. Sensitivity analysis: E-value
# E-value is a measure of how strong an unmeasured confounder would need to be
# We report E-values for PA only (the one consistent result)


print("\n" + "=" * 68)
print("STEP 4: Sensitivity analysis — E-value")
print("=" * 68)

def evalue(rr):
    """
    Compute E-value for a given risk ratio.
    Returns NaN if rr <= 0.
    """
    if rr <= 0 or np.isnan(rr):
        return np.nan
    if rr >= 1:
        return rr + np.sqrt(rr * (rr - 1))
    else:  # rr < 1 (protective)
        rr_inv = 1 / rr
        return rr_inv + np.sqrt(rr_inv * (rr_inv - 1))


print("\nE-values")
print("Interpretation: minimum association strength an unmeasured")
print("confounder would need with BOTH treatment and outcome to")
print("fully explain away the observed effect.")
print()
print(f"{'Analysis':<22} {'RR':>6} {'E-val (point)':>15} "
    f"{'CI bound RR':>12} {'E-val (CI)':>12}  Note")
print("-" * 85)

evalue_rows = []

for _, row in results_df.iterrows():
    rr      = row['RR']
    rr_lo   = row['RR_ci_low']
    rr_hi   = row['RR_ci_high']
    rd      = row['RD']

    ev_point = evalue(rr)

    # E-value for the CI bound closer to null
    # For protective effects (RR<1): upper CI bound is closer to null
    # For harmful effects  (RR>1): lower CI bound is closer to null
    if rr < 1:
        ci_bound = rr_hi   # upper bound, closer to 1
    else:
        ci_bound = rr_lo   # lower bound, closer to 1

    ev_ci = evalue(ci_bound)

    # decide which analyses to report
    direction_ok = row['direction_consistent']
    if direction_ok:
        note = "report — direction consistent with NB3"
    else:
        note = "caution — reverse causality present"

    print(f"{row['analysis']:<22} {rr:>6.4f} {ev_point:>15.3f} "
        f"{ci_bound:>12.4f} {ev_ci:>12.3f}  {note}")

    evalue_rows.append({
        'analysis'       : row['analysis'],
        'RR'             : rr,
        'evalue_point'   : ev_point,
        'ci_bound_used'  : ci_bound,
        'evalue_ci'      : ev_ci,
        'direction_ok'   : direction_ok
    })

evalue_df = pd.DataFrame(evalue_rows)
evalue_df.to_csv(os.path.join(OUTPUT_DIR, 'msm_evalues.csv'), index=False)
print(f"\nSaved: outputs/msm_evalues.csv")

# ── focused interpretation for PA (the primary consistent result) ──
pa_row = evalue_df[evalue_df['analysis'] == 'Increase PA'].iloc[0]
print(f"\nFocused interpretation — Increase PA:")
print(f"  RR = {pa_row['RR']:.4f}")
print(f"  E-value (point estimate) = {pa_row['evalue_point']:.3f}")
print(f"  E-value (CI bound)       = {pa_row['evalue_ci']:.3f}")
print()
print("  To explain away the PA effect entirely, an unmeasured confounder")
print(f"  would need to be associated with both PA behaviour change AND")
print(f"  CVD outcome by a factor of ≥{pa_row['evalue_point']:.2f}-fold on the risk ratio scale.")
print(f"  To shift the CI to cross null, the confounder association")
print(f"  would need to be ≥{pa_row['evalue_ci']:.2f}-fold.")
if pa_row['evalue_ci'] > 1.5:
    print("  This suggests the PA result is moderately robust to unmeasured")
    print("  confounding — a relatively strong confounder would be needed.")
else:
    print("  This suggests the PA result is sensitive to even modest")
    print("  unmeasured confounding — interpret with caution.")

#  final results summary 
print(f"\n{'='*68}")
print("FINAL MSM RESULTS SUMMARY")
print(f"{'='*68}")
print(f"\n{'Analysis':<22} {'n_treated':>10} {'RD':>8} "
    f"{'95% CI':>22} {'NB3 CATE':>10}  {'Direction':>10}")
print("-" * 85)
for _, row in results_df.iterrows():
    ci_str = f"({row['RD_ci_low']:+.4f}, {row['RD_ci_high']:+.4f})"
    flag   = '✓' if row['direction_consistent'] else '✗'
    print(f"{row['analysis']:<22} {row['n_treated']:>10,} "
        f"{row['RD']:>+8.4f} {ci_str:>22} "
        f"{row['nb3_cate']:>+10.4f}  {flag:>10}")

print(f"\n{'='*68}")
print("Outputs saved:")
print("  msm_cohort_{{smk,pa,sleep,pa_sleep}}.parquet   (Step 1)")
print("  msm_weighted_{{smk,pa,sleep,pa_sleep}}.parquet (Step 2)")
print("  msm_love_plot.png                              (Step 2)")
print("  msm_results.csv                                (Step 3)")
print("  msm_evalues.csv                                (Step 4)")
