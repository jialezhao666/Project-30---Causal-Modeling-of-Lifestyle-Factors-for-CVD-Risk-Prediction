import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.expanduser('~/my_ukb_thesis'))

OUTPUT_DIR  = os.path.expanduser('~/my_ukb_thesis/phase_II/outputs')
FIGURES_DIR = os.path.expanduser('~/my_ukb_thesis/phase_II/figures')
os.makedirs(OUTPUT_DIR,  exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
# Load ITE and joint CATE results, merge if both available
ite = pd.read_parquet(os.path.join(OUTPUT_DIR, 'ite_results.parquet'))
print(f"  ite_results.parquet    : {ite.shape}  columns: {ite.columns.tolist()}")

joint_path = os.path.join(OUTPUT_DIR, 'joint_cate_full.parquet')
if os.path.exists(joint_path):
    joint = pd.read_parquet(joint_path)
    print(f"  joint_cate_full.parquet: {joint.shape}  columns: {joint.columns.tolist()}")
    df = ite.merge(joint, on='eid', how='inner')
    print(f"  after merge: {df.shape}")
    HAS_JOINT = True
else:
    df = ite.copy()
    HAS_JOINT = False
    print("  joint_cate_full.parquet not found — heterogeneity skipped")
    
# create subgroup bins
# age bands: <55, 55-65, >=65 
df['age_band'] = pd.cut(df['age_defined_baseline'],
                        bins=[0, 55, 65, 200],
                        labels=['<55', '55-65', '>=65'])
# BMI bands: <25, 25-30, >=30
df['bmi_band'] = pd.cut(df['BMI'],
                        bins=[0, 25, 30, 200],
                        labels=['<25', '25-30', '>=30'])
# sex labels
df['sex_label'] = df['genetic_sex'].map({0: 'Female', 1: 'Male'})

# BLP-style OLS: CATE ~ age + sex + BMI (HC3 robust SE)
import statsmodels.formula.api as smf

# define which outcomes to analyze
single_targets = {
    'ite_smk': 'Quit smoking',
    'ite_pa': 'Increase PA',
    'ite_sleep': 'Adequate sleep',
}
joint_targets = {}
if HAS_JOINT:
    joint_targets = {
        'cate_arm1_vs0': 'no_smk only (arm 1)',
        'cate_arm2_vs0': 'PA only (arm 2)',
        'cate_arm4_vs0': 'sleep only (arm 4)',
        'cate_arm7_vs0': 'all three (arm 7)',
    }
all_targets = {**single_targets, **joint_targets}
# Run BLP-style OLS for each outcome, save results in a list of dicts for export
blp_rows = []
for col, label in all_targets.items():
    if col not in df.columns:
        continue
    formula = f'{col} ~ age_defined_baseline + genetic_sex + BMI'
    res = smf.ols(formula, data=df).fit(cov_type='HC3')
    print(f"\n  {label}")
    print(f"  {'Variable':<25} {'coef':>9} {'SE':>9} {'t':>7} {'p':>8}")
    print(f"  {'-'*60}")
    for vname in ['age_defined_baseline', 'genetic_sex', 'BMI']:
        c = res.params[vname]
        se = res.bse[vname]
        t  = res.tvalues[vname]
        p  = res.pvalues[vname]
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f"  {vname:<25} {c:>+9.5f} {se:>9.5f} {t:>7.2f} {p:>8.4f} {sig}")
        blp_rows.append({
            'outcome': label, 'variable': vname,
            'coef': c, 'se': se, 'tstat': t, 'pval': p,
        })
    print(f"  R² = {res.rsquared:.4f}")

blp_df = pd.DataFrame(blp_rows)
blp_df.to_csv(os.path.join(OUTPUT_DIR, 'heterogeneity_blp.csv'), index=False)
print(f"\nSaved: outputs/heterogeneity_blp.csv")

# subgroup mean CATEs and 95% CIs
print("Subgroup mean CATE (mean ± 95% CI)")

def subgroup_stats(data, cate_col, group_col):
    rows = []
    for level, sub in data.groupby(group_col, observed=True):
        vals = sub[cate_col].dropna()
        n = len(vals)
        m = vals.mean()
        se = vals.std(ddof=1) / np.sqrt(n)
        rows.append({'group': str(level), 'n': n,
                    'mean': m, 'ci_low': m - 1.96*se, 'ci_high': m + 1.96*se})
    return pd.DataFrame(rows)

subgroup_rows = []
for col, label in all_targets.items():
    if col not in df.columns:
        continue
    print(f"\n  {label}")
    for gvar, gname in [('sex_label', 'sex'),
                        ('age_band',  'age band'),
                        ('bmi_band',  'BMI band')]:
        stats = subgroup_stats(df, col, gvar)
        print(f"    by {gname}:")
        for _, row in stats.iterrows():
            print(f"{row['group']:>8}  n={row['n']:>7,}  "
                f"mean={row['mean']:+.4f}  "
                f"95%CI ({row['ci_low']:+.4f}, {row['ci_high']:+.4f})")
            subgroup_rows.append({
                'outcome': label, 'strat_var': gname,
                'group': row['group'], 'n': row['n'],
                'mean_cate': row['mean'],
                'ci_low': row['ci_low'], 'ci_high': row['ci_high'],
            })

subgroup_df = pd.DataFrame(subgroup_rows)
subgroup_df.to_csv(os.path.join(OUTPUT_DIR, 'heterogeneity_subgroup.csv'), index=False)
print(f"\nSaved: outputs/heterogeneity_subgroup.csv")

# figure 1: single intervention ite by subgroup
ITE_COLS   = ['ite_smk', 'ite_pa', 'ite_sleep']
ITE_LABELS = ['Quit smoking', 'Increase PA', 'Adequate sleep']
ITE_COLORS = ['#C44E52', '#4C72B0', '#55A868']

STRAT_VARS = [
    ('sex_label', 'Sex', ['Female', 'Male']),
    ('age_band', 'Age band', ['<55', '55-65', '>=65']),
    ('bmi_band', 'BMI band', ['<25', '25-30', '>=30']),
]

fig, axes = plt.subplots(3, 3, figsize=(13, 10))

for row_i, (col, lbl, color) in enumerate(zip(ITE_COLS, ITE_LABELS, ITE_COLORS)):
    for col_j, (gvar, gname, levels) in enumerate(STRAT_VARS):
        ax = axes[row_i][col_j]
        
        data_per_group = []
        for lvl in levels:
            sub = df[df[gvar] == lvl][col].dropna().values
            data_per_group.append(sub)
            
        parts = ax.violinplot(data_per_group, positions=range(len(levels)),
                            showmedians=True, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(1.5)
        
        # add mean dots
        for i, vals in enumerate(data_per_group):
            ax.scatter([i], [vals.mean()], color='black', s=20, zorder=5)
            
        ax.axhline(0, color='grey', lw=0.8, ls='--', alpha=0.6)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels(levels, fontsize=9)
        ax.set_title(f'{lbl}\nby {gname}', fontsize=9, pad=4)
        if col_j == 0:
            ax.set_ylabel('Individual treatment effect\n(risk difference)', fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)

fig.suptitle('Heterogeneity of Treatment Effects by Subgroup\n'
            '(dot = mean, line = median; positive ite_smk = smoking increases risk)',
            fontsize=11, y=1.01)
fig.tight_layout()
violin_path = os.path.join(FIGURES_DIR, 'heterogeneity_violin.png')
fig.savefig(violin_path, dpi=150, bbox_inches='tight')
print(f"\nSaved figure: {violin_path}")
plt.close()

# figure 2: mena cate per subgroup with * intervention

# single intervention 
from matplotlib.transforms import blended_transform_factory

fig2, axes2 = plt.subplots(1, 3, figsize=(16, 5.5))
fig2.subplots_adjust(left=0.22, right=0.90, wspace=0.42)

STRAT_ORDER2 = [
    ('sex',      ['Female', 'Male']),
    ('age band', ['<55', '55-65', '>=65']),
    ('BMI band', ['<25', '25-30', '>=30']),
]

for ax_idx, (ax, col, lbl, color) in enumerate(
        zip(axes2, ITE_COLS, ITE_LABELS, ITE_COLORS)):

    sub_df = subgroup_df[subgroup_df['outcome'] == lbl].copy()
    overall_mean = df[col].mean()

    y_pos_list = []
    ylabels_list = []
    means_list = []
    pos = 0

    for strat_name, levels in STRAT_ORDER2:
        grp = sub_df[sub_df['strat_var'] == strat_name]
        for lvl in levels:
            row = grp[grp['group'] == lvl]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            m = r['mean_cate']

            ax.scatter(m, pos, color=color, s=60, zorder=4)
            ax.plot([overall_mean, m], [pos, pos],
                    color=color, lw=1.0, alpha=0.35, zorder=3)

            y_pos_list.append(pos)
            ylabels_list.append(
                f"{strat_name}: {lvl}  (n={int(r['n']):,})")
            means_list.append(m)
            pos += 1
        pos += 0.7   # gap between strat groups

    # reference lines
    ax.axvline(overall_mean, color='dimgrey', lw=1.2, ls='-', alpha=0.45,
            label=f'Overall mean ({overall_mean:+.4f})')
    ax.axvline(0, color='silver', lw=0.8, ls='--', alpha=0.55)

    # y-axis: labels on leftmost panel only
    ax.set_yticks(y_pos_list)
    ax.set_yticklabels(
        ylabels_list if ax_idx == 0 else ['' for _ in ylabels_list],
        fontsize=8.5)
    ax.set_ylim(max(y_pos_list) + 0.8, -0.5)
    ax.invert_yaxis()

    # x-axis: max 5 ticks to avoid crowding
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(axis='x', labelsize=8)

    ax.set_xlabel('Mean CATE (risk difference)', fontsize=9)
    ax.set_title(lbl, fontsize=10, color=color, pad=8, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc='lower right',
            handlelength=1.2)

    # right-margin value column
    # blended transform: x = axes fraction (fixed), y = data coords
    # all values align at x=1.03 regardless of x-axis scale;
    # vertical spacing follows row positions so no overlap.
    trans = blended_transform_factory(ax.transAxes, ax.transData)
    # column header
    ax.text(1.02, -0.4, 'CATE', transform=trans,
            va='center', ha='left', fontsize=8,
            color='dimgrey', fontweight='bold')
    for y, m in zip(y_pos_list, means_list):
        ax.text(1.02, y, f'{m:+.4f}', transform=trans,
                va='center', ha='left', fontsize=8.5,
                color=color, family='monospace')

fig2.suptitle(
    'Heterogeneity of Treatment Effects: Mean CATE by Subgroup\n'
    '(NB2 causal forest; no CI shown — n > 100,000 per subgroup)',
    fontsize=10, y=1.02)
forest_path = os.path.join(FIGURES_DIR, 'heterogeneity_forest.png')
fig2.savefig(forest_path, dpi=150, bbox_inches='tight')
print(f"Saved figure: {forest_path}")
plt.close()

print("\n" + "=" * 68)
print("Outputs:")
print("  outputs/heterogeneity_subgroup.csv")
print("  outputs/heterogeneity_blp.csv")
print("  figures/heterogeneity_violin.png")
print("  figures/heterogeneity_forest.png")