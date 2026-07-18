import pandas as pd
import numpy as np
import os

HRS_BASE = os.path.expanduser("~/my_ukb_thesis/external_validation_hrs")
RAND_PATH = os.path.join(HRS_BASE, "data", "randhrs1992_2022v1.dta")

needed_cols = [
    "hhid", "pn",
    "r13agey_b",
    "ragender",
    "r13bmi",
    "raedyrs",
    "r13smokev", "r13smoken",
    "r13vgactx",
    "r13sleep",
    "r13cesd", "r13depres",
    "r13drinkn",
    "r13heart", "r14heart", "r15heart", "r16heart",
    "r13strok", "r14strok", "r15strok", "r16strok",
]

df = pd.read_stata(RAND_PATH, columns=needed_cols, convert_categoricals=False)

print(f"Total respondents in raw file: {len(df)}")


# Step 1: Identify who has complete baseline exposure data

baseline_required = [
    "r13smokev", "r13smoken", "r13vgactx", "r13sleep",
    "r13heart", "r13strok", "r13bmi", "r13agey_b", "ragender"
]

df["has_complete_baseline"] = df[baseline_required].notna().all(axis=1)

n_complete = df["has_complete_baseline"].sum()
n_incomplete = (~df["has_complete_baseline"]).sum()
print(f"\nComplete baseline data: {n_complete}")
print(f"Incomplete baseline data (excluded by complete-case approach): {n_incomplete}")


# Step 2: Compare baseline characteristics between the two groups
# check whether complete-case exclusion is associated with
# systematic differences 

compare_vars = ["r13agey_b", "ragender", "r13bmi", "raedyrs"]

print("\n=== Comparison: Complete vs Incomplete baseline data ===")
comparison = df.groupby("has_complete_baseline")[compare_vars].agg(['mean', 'count'])
print(comparison)


# Step 3: Standardised Mean Difference (SMD) for each variable
# SMD > 0.1 is the conventional threshold for a "meaningful" imbalance
# (same convention used in your UKB Task B imaging-date selection check)

def smd(group1, group2):
    mean1, mean2 = group1.mean(), group2.mean()
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt((var1 + var2) / 2)
    if pooled_std == 0:
        return np.nan
    return (mean1 - mean2) / pooled_std

print("\n=== Standardised Mean Differences (Complete vs Incomplete) ===")
complete_group = df[df["has_complete_baseline"]]
incomplete_group = df[~df["has_complete_baseline"]]

smd_results = {}
for var in compare_vars:
    smd_results[var] = smd(complete_group[var].dropna(), incomplete_group[var].dropna())

smd_df = pd.Series(smd_results, name="SMD").to_frame()
smd_df["meaningful_imbalance (SMD>0.1)"] = smd_df["SMD"].abs() > 0.1
print(smd_df)

# Step 4: Among those with complete baseline data, check whether the
# further exclusions (not CVD-free, or no valid follow-up) are also
# associated with systematic differences

complete_group = complete_group.copy()
complete_group["cvd_free_at_baseline"] = (
    (complete_group["r13heart"] == 0) & (complete_group["r13strok"] == 0))

followup_cols = ["r14heart", "r15heart", "r16heart", "r14strok", "r15strok", "r16strok"]
complete_group["has_followup"] = complete_group[followup_cols].notna().any(axis=1)

complete_group["in_final_cohort"] = (
    complete_group["cvd_free_at_baseline"] & complete_group["has_followup"])

print(f"\n=== Among complete-baseline respondents ===")
print(f"In final analytic cohort: {complete_group['in_final_cohort'].sum()}")
print(f"Excluded at this stage (had baseline CVD, or no follow-up): "
    f"{(~complete_group['in_final_cohort']).sum()}")

print("\n=== Comparison: Final cohort vs Excluded-at-this-stage ===")
comparison2 = complete_group.groupby("in_final_cohort")[compare_vars].agg(['mean', 'count'])
print(comparison2)

print("\n=== SMD: Final cohort vs Excluded-at-this-stage ===")
final_group = complete_group[complete_group["in_final_cohort"]]
excluded_group = complete_group[~complete_group["in_final_cohort"]]

smd_results_2 = {}
for var in compare_vars:
    smd_results_2[var] = smd(final_group[var].dropna(), excluded_group[var].dropna())

smd_df_2 = pd.Series(smd_results_2, name="SMD").to_frame()
smd_df_2["meaningful_imbalance (SMD>0.1)"] = smd_df_2["SMD"].abs() > 0.1
print(smd_df_2)
