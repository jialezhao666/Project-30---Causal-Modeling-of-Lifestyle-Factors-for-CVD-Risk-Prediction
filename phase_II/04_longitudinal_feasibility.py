
# Phase II — Longitudinal feasibility diagnostic
# Treated = baseline UNHEALTHY -> imaging HEALTHY
# Control = baseline UNHEALTHY -> imaging STILL UNHEALTHY

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/my_ukb_thesis'))
from const_paths import BASE_PATH

file_tab     = os.path.join(BASE_PATH, 'ukb_tabular_data_causal_analysis.tsv')
file_outcome = os.path.join(BASE_PATH, 'group_1_outcomes_df_without_qc_df.tsv')
OUTCOME = 'def_CVD_AF_HF_AFTER'

tab_cols = ['eid',
            '20116-0.0', '20116-2.0',          # smoking status
            '884-0.0', '884-2.0',              # moderate activity days
            '904-0.0', '904-2.0',              # vigorous activity days
            '1160-0.0', '1160-2.0']            # sleep hours

# load tabular data (only relevant columns)
tab = pd.read_csv(file_tab, sep='\t', usecols=tab_cols)
print(f"tabular rows: {len(tab):,}")

# load outcomes and merge
out = pd.read_csv(file_outcome, sep='\t', usecols=['eid', OUTCOME])
df = tab.merge(out, on='eid', how='inner')
print(f"merged rows: {len(df):,}")

# --- healthy-behaviour indicators (1 = healthy state) ---
def smk_healthy(col):
    c = col.astype(float)
    return np.where(c.isna(), np.nan, (c != 2).astype(float))   # not current smoker

def sleep_healthy(col):
    c = col.astype(float)
    return np.where(c.isna(), np.nan, (c >= 7).astype(float))

def pa_healthy(mod_days, vig_days):
    # APPROX definition for longitudinal (22036 has no imaging visit):
    # active if moderate >=5 days/wk OR vigorous >=1 day/wk
    m = mod_days.astype(float).copy()
    v = vig_days.astype(float).copy()
    m[m < 0] = np.nan        # -1 don't know, -3 prefer not answer
    v[v < 0] = np.nan
    active = ((m >= 5) | (v >= 1)).astype(float)
    active[m.isna() & v.isna()] = np.nan
    return active

beh_simple = {
    'Quit smoking': (smk_healthy(df['20116-0.0']), smk_healthy(df['20116-2.0'])),
    'Adequate sleep': (sleep_healthy(df['1160-0.0']), sleep_healthy(df['1160-2.0']))}
beh_simple['Increase PA'] = (
    pa_healthy(df['884-0.0'], df['904-0.0']),
    pa_healthy(df['884-2.0'], df['904-2.0']))

print("\n" + "="*64)
print("LONGITUDINAL FEASIBILITY DIAGNOSTIC")
print("  PA uses APPROX definition (moderate>=5d/wk or vigorous>=1d/wk)")
print("="*64)

summary_rows = []
for name in ['Quit smoking', 'Increase PA', 'Adequate sleep']:
    hb_arr, hi_arr = beh_simple[name]
    hb = pd.Series(hb_arr, index=df.index)
    hi = pd.Series(hi_arr, index=df.index)

    both = hb.notna() & hi.notna()
    n_both = int(both.sum())
    hb2, hi2 = hb[both], hi[both]
    y2 = df.loc[both, OUTCOME]

    base_unhealthy = (hb2 == 0)
    n_pool = int(base_unhealthy.sum())
    treated = base_unhealthy & (hi2 == 1)
    control = base_unhealthy & (hi2 == 0)
    n_treated, n_control = int(treated.sum()), int(control.sum())
    prop_changed = n_treated / n_pool if n_pool > 0 else np.nan

    ev_treated = int(y2[treated].sum())
    ev_control = int(y2[control].sum())
    rate_t = y2[treated].mean() if n_treated > 0 else np.nan
    rate_c = y2[control].mean() if n_control > 0 else np.nan

    print(f"\n### {name}")
    print(f"  Q1 observed at BOTH visits      : {n_both:,}")
    print(f"     baseline-unhealthy pool      : {n_pool:,}")
    print(f"  Q2 changed to healthy (treated) : {n_treated:,} ({prop_changed*100:.1f}% of pool)")
    print(f"  Q3 treated / control            : {n_treated:,} / {n_control:,}")
    print(f"  Q4 CVD events  treated/control  : {ev_treated:,} / {ev_control:,}")
    if n_treated > 0:
        print(f"     event rate treated/control   : {rate_t:.4f} / {rate_c:.4f}")
    flag = "OK" if (ev_treated >= 50 and n_treated >= 500) else "LOW - may be underpowered"
    print(f"  feasibility: {flag}")

    summary_rows.append({
        'intervention': name, 'n_both_visits': n_both,
        'baseline_unhealthy_pool': n_pool,
        'n_treated': n_treated, 'n_control': n_control,
        'prop_changed': prop_changed,
        'events_treated': ev_treated, 'events_control': ev_control,
        'rate_treated': rate_t, 'rate_control': rate_c,
    })

summary = pd.DataFrame(summary_rows)
os.makedirs(os.path.expanduser('~/my_ukb_thesis/phase_II/outputs'), exist_ok=True)
out_path = os.path.expanduser('~/my_ukb_thesis/phase_II/outputs/longitudinal_feasibility.csv')
summary.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}") 