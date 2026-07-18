import pandas as pd
import numpy as np
import os

# Config

HRS_BASE = os.path.expanduser("~/my_ukb_thesis/external_validation_hrs")
RAND_PATH = os.path.join(HRS_BASE, "data", "randhrs1992_2022v1.dta")
OUTPUT_PATH = os.path.join(HRS_BASE, "outputs", "hrs_cohort_processed.csv")

NEEDED_COLS = [
    "hhid", "pn",
    "r13agey_b", "ragender", "r13bmi", "raedyrs",
    "r13smokev", "r13smoken",
    "r13vgactx",
    "r13sleep",
    "r13cesd", "r13depres",
    "r13drinkn",
    "r13heart", "r14heart", "r15heart", "r16heart",
    "r13strok", "r14strok", "r15strok", "r16strok",
]

BASELINE_REQUIRED = [
    "r13smokev", "r13smoken", "r13vgactx", "r13sleep",
    "r13heart", "r13strok", "r13bmi", "r13agey_b", "ragender"
]

FOLLOWUP_COLS = ["r14heart", "r15heart", "r16heart", "r14strok", "r15strok", "r16strok"]


def any_incident(row, cols):
    """Return 1 if any non-missing follow-up wave shows the condition newly
    present, 0 if all non-missing waves show absence, None if all missing."""
    vals = row[cols].dropna()
    if len(vals) == 0:
        return None
    return int((vals == 1).any())


def build_cohort():
    df = pd.read_stata(RAND_PATH, columns=NEEDED_COLS, convert_categoricals=False)
    print(f"Total respondents in raw file: {len(df)}")

    
    # Step 1: require complete Wave 13 baseline exposure + outcome data
    baseline_complete = df.dropna(subset=BASELINE_REQUIRED).copy()
    print(f"Complete Wave 13 baseline data: {len(baseline_complete)}")
    
    # Step 2: require CVD-free at baseline (no prior heart disease or stroke)
    cohort = baseline_complete[
        (baseline_complete["r13heart"] == 0) & (baseline_complete["r13strok"] == 0)
    ].copy()
    print(f"CVD-free at baseline: {len(cohort)}")

    # Step 3: require at least one valid follow-up wave (14/15/16)
    cohort = cohort.dropna(subset=FOLLOWUP_COLS, how="all").copy()
    print(f"Final analytic cohort (has follow-up): {len(cohort)}")

    # Step 4: construct exposure variables aligned to UKB definitions
    # Smoking: direct mapping, already 0/1 coded consistently with UKB
    cohort["smk_curr"] = cohort["r13smoken"]
    cohort["smk_prev"] = ((cohort["r13smokev"] == 1) & (cohort["r13smoken"] == 0)).astype(int)

    # Physical activity: r13vgactx is reverse-coded
    # (1=every day ... 5=never), so active = categories 1 or 2
    cohort["PA_active"] = cohort["r13vgactx"].isin([1, 2]).astype(int)

    # Sleep: HRS has no sleep-duration variable, only a sleep-disorder
    # indicator. This is a distinct construct from UKB's sleep_hrs and is
    # kept under its own name to avoid implying equivalence.
    cohort["sleep_disorder"] = cohort["r13sleep"]

    # Education: HRS sex coding is 1=male, 2=female; UKB genetic_sex=1 is male
    cohort["female"] = (cohort["ragender"] == 2).astype(int)
    cohort["uni_degree"] = (cohort["raedyrs"] >= 16).astype(int)

    # Alcohol
    cohort["alc_curr"] = (cohort["r13drinkn"] > 0).astype(int)

    # Step 5: construct incident CVD outcome (heart disease OR stroke)
    cohort["incident_heart"] = cohort.apply(
        lambda r: any_incident(r, ["r14heart", "r15heart", "r16heart"]), axis=1)
    cohort["incident_strok"] = cohort.apply(
        lambda r: any_incident(r, ["r14strok", "r15strok", "r16strok"]), axis=1)
    cohort["incident_cvd"] = (
        (cohort["incident_heart"] == 1) | (cohort["incident_strok"] == 1)
    ).astype(int)

    cohort.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved processed cohort to hrs_cohort_processed.csv")
    print(f"Incident CVD rate: {cohort['incident_cvd'].mean()*100:.2f}%")

    return cohort


if __name__ == "__main__":
    build_cohort()