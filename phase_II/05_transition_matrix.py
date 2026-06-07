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
OUTCOME = 'def_CVD_AF_HF_AFTER'

core_cols = ['eid',
            '20116-0.0', '20116-2.0',
            '884-0.0',   '884-2.0', 
            '894-0.0',   '894-2.0',
            '904-0.0',   '904-2.0', # 22036 has no imaging visit, so we use similar PA definition for imaging visit
            '914-0.0',   '914-2.0',
            '1160-0.0',  '1160-2.0']

# 1. longitudinal fields + outcome 
print("Loading tabular (longitudinal fields)")
tab = pd.read_csv(file_tab, sep='\t', usecols=core_cols)
print(f"  tabular rows: {len(tab):,}")
out = pd.read_csv(file_outcome, sep='\t', usecols=['eid', OUTCOME])
df = tab.merge(out, on='eid', how='inner')
print(f"  merged rows : {len(df):,}")

# 2. sex + age from exposure file (full population)
print("\nLoading sex + age from group_1_clean_filtered_imputed_dataset_df.tsv")
exp_header = pd.read_csv(file_exposure, sep='\t', nrows=0).columns.tolist()
exp_want = ['eid', 'genetic_sex', 'age_defined_baseline']
exp_have = [c for c in exp_want if c in exp_header]
exp_miss = [c for c in exp_want if c not in exp_header]
print(f"  available: {exp_have}")
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
                print(f"  loaded {name}: {len(d):,}")
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
    print(f"  coverage: {coverage}  unique eids: {len(bmi_pa):,}")
    print(f"  BMI: mean={bmi_pa['BMI'].mean():.2f}, std={bmi_pa['BMI'].std():.2f}, "
        f"min={bmi_pa['BMI'].min():.2f}, max={bmi_pa['BMI'].max():.2f}  (raw values)")
    df = df.merge(bmi_pa, on='eid', how='left')
    print(f"  after merge: {df.shape[0]:,} rows × {df.shape[1]} cols")
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

def pa_healthy_approx(mod_days, mod_mins, vig_days, vig_mins):
    """
    Approximates UK PA guideline (field 22036) using:
    moderate >= 150 min/week  (884 days * 894 mins/day)
    vigorous >= 75  min/week (904 days * 914 mins/day)
    Edge cases (preserves original Series index throughout):
    days=0 -> total=0  (not NaN; person confirmed no activity)
    days>0, mins=NaN ->total=NaN (conservative; can't confirm duration)
    both days NaN -> NaN
    """
    idx = mod_days.index  # preserve original df index
    md = _clip_neg_na(mod_days)   # Series with original index
    mm = _clip_neg_na(mod_mins)
    vd = _clip_neg_na(vig_days)
    vm = _clip_neg_na(vig_mins)

    # minutes/week — use index=idx so boolean ops stay aligned
    mod_total = pd.Series(np.where(md == 0, 0.0, md * mm), index=idx)
    vig_total = pd.Series(np.where(vd == 0, 0.0, vd * vm), index=idx)

    active = pd.Series(
        ((mod_total >= 150) | (vig_total >= 75)).astype(float), index=idx)

    # if BOTH day-fields are NaN → truly no data → NaN
    both_nan = md.isna() & vd.isna()
    active[both_nan] = np.nan

    # days>0 but mins missing → uncertain; set NaN unless other arm already
    # pushes active=1
    mod_uncertain = (md > 0) & mm.isna()
    vig_uncertain = (vd > 0) & vm.isna()
    uncertain = mod_uncertain | vig_uncertain
    active[uncertain & (active != 1)] = np.nan

    return active.values   # return plain array (same as smk_healthy / sleep_healthy)

H = {
    'Quit smoking'  : (smk_healthy(df['20116-0.0']),
                    smk_healthy(df['20116-2.0'])),
    'Increase PA'   : (pa_healthy_approx(df['884-0.0'], df['894-0.0'],
                                        df['904-0.0'], df['914-0.0']),
                    pa_healthy_approx(df['884-2.0'], df['894-2.0'],
                                        df['904-2.0'], df['914-2.0'])),
    'Adequate sleep': (sleep_healthy(df['1160-0.0']),
                    sleep_healthy(df['1160-2.0'])),}


# 1. Full 2×2 transition matrix per behaviour (FULL IMAGING)
print("\n" + "=" * 78)
print("FULL TRANSITION MATRIX  (baseline → imaging visit, full imaging pool)")
print("=" * 78)

rows = []
for name, (b_arr, i_arr) in H.items():
    b = pd.Series(b_arr, index=df.index)
    i = pd.Series(i_arr, index=df.index)
    both = b.notna() & i.notna() # require both visits observed for this behaviour
    b2, i2 = b[both], i[both]
    y2 = df.loc[both, OUTCOME]
    n = len(b2)

    cells = {
        'HH': (b2 == 1) & (i2 == 1), # healthy at baseline AND imaging
        'HU': (b2 == 1) & (i2 == 0), # healthy at baseline, unhealthy at imaging (worsened)
        'UH': (b2 == 0) & (i2 == 1), # unhealthy at baseline, healthy at imaging (improved)
        'UU': (b2 == 0) & (i2 == 0), # unhealthy at baseline AND imaging
        }
    cell_n = {k: int(m.sum()) for k, m in cells.items()} # count of participants in each cell
    cell_ev = {k: int(y2[m].sum()) for k, m in cells.items()} # count of CVD events in each cell
    cell_rate = {k: y2[m].mean() if cell_n[k] else np.nan  for k, m in cells.items()}  # CVD event rate in each cell

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
            'intervention' : name,
            'cell' : k,
            'baseline': k[0],
            'imaging' : k[1],
            'n' : cell_n[k],
            'pct_of_total' : cell_n[k] / n * 100,
            'cvd_events' : cell_ev[k],
            'cvd_rate' : cell_rate[k],})

mat = pd.DataFrame(rows)
mat.to_parquet(os.path.join(OUTPUT_DIR, 'transition_matrix.parquet'), index=False)
mat.to_csv(os.path.join(OUTPUT_DIR, 'transition_matrix.csv'), index=False)
print(f"\nSaved: outputs/transition_matrix.{{parquet,csv}}  ({len(mat)} rows)")


# 2. PA definition concordance: PA_active (22036) vs approximation using 884/894/904/914 (full imaging pool)
#   Uses whichever coverage we got from Phase I parquets.
if 'PA_active' in df.columns:
    p_nb1 = pd.Series(df['PA_active'].values, index=df.index).astype(float)
    p_new = pd.Series(
        pa_healthy_approx(df['884-0.0'], df['894-0.0'],
                        df['904-0.0'], df['914-0.0']),index=df.index)
    ok = p_nb1.notna() & p_new.notna()
    p_nb1_2 = p_nb1[ok].astype(int)
    p_new_2 = p_new[ok].astype(int)

    ct = pd.crosstab(p_nb1_2.rename('PA_active (22036)'), 
                    p_new_2.rename('884+904 approx'), margins=True) # cross-tabulation with totals
    agree = (p_nb1_2 == p_new_2).mean() # overall agreement
    p_nb1_1 = (p_nb1_2 == 1).mean()
    p_new_1 = (p_new_2 == 1).mean()
    pe = p_nb1_1 * p_new_1 + (1 - p_nb1_1) * (1 - p_new_1)
    kappa = (agree - pe) / (1 - pe) if pe != 1 else float('nan')

    txt = []
    txt.append("PA definition concordance at BASELINE")
    txt.append("=" * 60)
    txt.append(f"data basis : {coverage}")
    txt.append(f"n compared : {int(ok.sum()):,}  "
            f"(imaging-visit participants with PA_active available)")
    txt.append(f"raw agreement : {agree:.4f}") # 
    txt.append(f"Cohen's kappa : {kappa:.4f}") # cohen's kappa 
    txt.append("")
    txt.append("Cross-tab (rows = PA_active （22036）; cols = 884+904 approx):")
    txt.append(ct.to_string())
    txt.append("")
    txt.append("Interpretation:")
    txt.append("  kappa > 0.80  : excellent — definitions interchangeable")
    txt.append("  kappa 0.6-0.8 : substantial — note in limitations")
    txt.append("  kappa 0.4-0.6 : moderate — methods must separate the two")
    txt.append("                  PA constructs (cross-sectional vs longitudinal)")
    txt.append("  kappa < 0.4   : poor — NB2/NB3 PA results may not extend to")
    txt.append("                  longitudinal analysis; re-think PA definition")
    concord_text = "\n".join(txt)
    print("\n" + concord_text)
    with open(os.path.join(OUTPUT_DIR, 'pa_concordance.txt'), 'w') as f:
        f.write(concord_text)
    print(f"\nSaved: outputs/pa_concordance.txt")
else:
    print("\nSkipped PA concordance: PA_active not available.")


# 3. Stratification
#    sex + age : FULL imaging pool (from exposure tsv)
#    BMI : whichever coverage Phase I parquet gave

has_sex_age = all(c in df.columns for c in ['genetic_sex', 'age_defined_baseline'])
has_bmi = 'BMI' in df.columns

if has_sex_age:
    df['_sex'] = df['genetic_sex'].map({0: 'F', 1: 'M'}).fillna('NA')
    df['_age_band'] = pd.cut(df['age_defined_baseline'],
                            bins=[0, 55, 65, 200],
                            labels=['<55', '55-65', '>=65'])
if has_bmi:
    df['_bmi_band'] = pd.cut(df['BMI'],
                            bins=[0, 25, 30, 200],
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
    print(f"Saved: outputs/transition_matrix_strat.csv ({len(strat_df)} rows)")

    # quick view: improvement rate UH / (UH+UU) by strata
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
    sub = mat[mat['intervention'] == name].set_index('cell')
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
print(f"Saved figure: {fig_path}")


# 5. Joint longitudinal intervention analysis

print("\n" + "=" * 78)
print("JOINT LONGITUDINAL INTERVENTION  (Table 2 extension)")
print("  Reference = baseline all-unhealthy → imaging all-unhealthy (000→000)")
print("  Arms 1-7  = baseline all-unhealthy → imaging any other combination")
print("=" * 78)

# NB3 mean CATE results (hardcoded from 03_multiarm_joint.py output)
# Update these if you re-run NB3 with different parameters
_nb3_summary = pd.read_parquet(os.path.join(OUTPUT_DIR, 'joint_cate_summary.parquet'))
NB3_CATE = dict(zip(_nb3_summary['arm'].astype(int), _nb3_summary['mean_cate']))
print(f"  Loaded NB3 CATE from joint_cate_summary.parquet: {NB3_CATE}")
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

# build imaging-visit healthy indicators (same functions as above)
smk_b  = pd.Series(smk_healthy(df['20116-0.0']),  index=df.index)
pa_b = pd.Series(pa_healthy_approx(df['884-0.0'], df['894-0.0'],
                                    df['904-0.0'], df['914-0.0']), index=df.index)
slp_b  = pd.Series(sleep_healthy(df['1160-0.0']), index=df.index)
smk_i  = pd.Series(smk_healthy(df['20116-2.0']),  index=df.index)
pa_i = pd.Series(pa_healthy_approx(df['884-2.0'], df['894-2.0'],
                                    df['904-2.0'], df['914-2.0']), index=df.index)
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
    f"{'Crude rate':>11} {'Crude RD':>10} {'NB3 CATE':>10}")
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
        'rate_ref'         : rate_ref,
    })

joint_df = pd.DataFrame(joint_rows)
joint_df.to_csv(os.path.join(OUTPUT_DIR, 'joint_longitudinal.csv'), index=False)
print(f"\nSaved: outputs/joint_longitudinal.csv")

print("\nNote: Crude RD is NOT causal (reverse causality expected for smoking/sleep).")
print("      MSM/IPTW will adjust for confounding in the next analysis step.")
print("      NB3 CATE = cross-sectional causal estimate (baseline data only).")
print("      Direction mismatch between Crude RD and NB3 CATE = evidence of")
print("      time-varying confounding that MSM is designed to correct.")

# --- figure: crude RD vs NB3 CATE side by side ---
fig2, ax = plt.subplots(figsize=(11, 5))

x      = np.arange(len(joint_df))
width  = 0.38
colors_nb3   = ['#3D5A80'] * len(joint_df)
colors_crude = ['#E07A5F'] * len(joint_df)

bars1 = ax.bar(x - width/2, joint_df['nb3_cate'],    width,
            color=colors_nb3,   label='NB3 CATE (cross-sectional, causal)',
            edgecolor='white')
bars2 = ax.bar(x + width/2, joint_df['crude_rd_vs_ref'], width,
            color=colors_crude, label='Crude RD vs 000→000 (longitudinal, raw)',
            edgecolor='white')

ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels([f"arm {r['arm']}\n{r['label']}\n(n={r['n_treated']:,})"
                    for _, r in joint_df.iterrows()], fontsize=8)
ax.set_ylabel('Risk difference vs all-unhealthy reference')
ax.set_title('Joint Interventions: NB3 Cross-sectional CATE vs Longitudinal Crude RD\n'
            '(reference = all-unhealthy at both visits; crude RD not causal)',
            fontsize=11, pad=10)
ax.legend(fontsize=9, frameon=False)
ax.spines[['top', 'right']].set_visible(False)
ax.yaxis.grid(True, alpha=0.25, lw=0.5)
ax.set_axisbelow(True)

fig2.tight_layout()
fig2_path = os.path.join(FIGURES_DIR, 'joint_longitudinal.png')
fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
print(f"Saved figure: {fig2_path}")

print(f"\n05_transition_matrix.py (v4) done.  BMI/PA coverage: {coverage}")


# 6. Pairwise joint intervention feasibility

print("\n" + "=" * 78)
print("PAIRWISE JOINT INTERVENTION FEASIBILITY")
print("  baseline pool = unhealthy on BOTH behaviours")
print("  treated = both improve; control = both stay unhealthy")
print("=" * 78)

beh_b = {'smk': smk_b, 'PA': pa_b, 'sleep': slp_b}
beh_i = {'smk': smk_i, 'PA': pa_i, 'sleep': slp_i}

PAIRS = [
    ('smk',  'PA',    'Smk+PA',    'arm 3 (NB3): -0.0810'),
    ('smk',  'sleep', 'Smk+Sleep', 'arm 5 (NB3): -0.0805'),
    ('PA',   'sleep', 'PA+Sleep',  'arm 6 (NB3): -0.0493'),
]

pair_rows = []
print(f"\n{'Pair':<12} {'pool':>7} {'treated':>8} {'control':>8} "
    f"{'events T/C':>11} {'rate T/C':>14} {'feasible':>10}  NB3 ref")
print("-" * 85)

for bA, bB, label, nb3_ref in PAIRS:
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

    feasible = 'YES' if (nt >= 500 and et >= 50) else 'LOW'
    rt_str = f'{rt:.3f}' if not np.isnan(rt) else 'NaN'
    rc_str = f'{rc:.3f}' if not np.isnan(rc) else 'NaN'

    print(f"{label:<12} {n_pool:>7,} {nt:>8,} {nc:>8,} "
        f"{et:>5}/{ec:<5} "
        f"{rt_str}/{rc_str}  {feasible:>10}  {nb3_ref}")

    pair_rows.append({
        'pair'          : label,
        'behA'          : bA,  'behB'          : bB,
        'n_pool'        : n_pool,
        'n_treated'     : nt,  'n_control'     : nc,
        'events_treated': et,  'events_control': ec,
        'rate_treated'  : rt,  'rate_control'  : rc,
        'feasible'      : feasible,
        'nb3_ref'       : nb3_ref,
    })

pair_df = pd.DataFrame(pair_rows)
pair_df.to_csv(os.path.join(OUTPUT_DIR, 'pairwise_joint_feasibility.csv'), index=False)
print(f"\nSaved: outputs/pairwise_joint_feasibility.csv")
print("\nNote: 'treated' = both behaviours improved simultaneously.")
print("      'control' = both stayed unhealthy. Mixed changers excluded.")
print("      Feasibility: n>=500 AND events>=50.")
print("      Feasible pairs → include in MSM alongside single interventions.")
print("      Infeasible pairs → limitation section.")

print(f"\n05_transition_matrix.py (v5) done.  BMI/PA coverage: {coverage}")