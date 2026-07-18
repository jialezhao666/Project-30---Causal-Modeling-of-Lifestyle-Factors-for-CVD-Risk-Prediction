import pandas as pd
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
import os

HRS_BASE = os.path.expanduser("~/my_ukb_thesis/external_validation_hrs")
INPUT_PATH = os.path.join(HRS_BASE, "outputs", "hrs_cohort_processed.csv")
OUTPUT_PATH = os.path.join(HRS_BASE, "outputs", "hrs_causal_forest_results.csv")
SUMMARY_OUTPUT_PATH = os.path.join(HRS_BASE, "outputs", "hrs_causal_forest_summary.csv")

COVARIATES = ["r13agey_b", "female", "r13bmi", "uni_degree", "r13cesd"]

# run causal forest for a given treatment and outcome
def run_causal_forest(cohort_model, treatment_col, label):
    Y = cohort_model["incident_cvd"].values
    T = cohort_model[treatment_col].values
    X = cohort_model[COVARIATES].values
    # Fit the causal forest model
    est = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42),
        model_t=GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        discrete_treatment=True,
        n_estimators=1000,
        min_samples_leaf=10,
        random_state=42,
        cv=5)
    est.fit(Y, T, X=X, W=X)
    # Estimate ATE and CATE
    ate = est.ate(X)
    ate_inference = est.ate_inference(X)
    ci_lower, ci_upper = ate_inference.conf_int_mean()
    cate = est.effect(X)

    expected_direction = (
        "positive_harmful" if treatment_col in ["smk_curr", "sleep_disorder"]
        else "negative_protective")
    direction_consistent = (
            ate > 0 if treatment_col in ["smk_curr", "sleep_disorder"]
    else ate < 0)

    summary_row = {
        "treatment": treatment_col,
        "label": label,
        "n": int(len(Y)),
        "events": int(np.sum(Y)),
        "event_rate": float(np.mean(Y)),
        "ate": float(ate),
        "ate_ci_low": float(ci_lower),
        "ate_ci_high": float(ci_upper),
        "ate_ci_crosses_zero": bool(ci_lower <= 0 <= ci_upper),
        "expected_direction": expected_direction,
        "direction_consistent": bool(direction_consistent),
        "cate_mean": float(np.mean(cate)),
        "cate_sd": float(np.std(cate)),
        "cate_median": float(np.median(cate)),
        "cate_q25": float(np.percentile(cate, 25)),
        "cate_q75": float(np.percentile(cate, 75)),
        "cate_min": float(np.min(cate)),
        "cate_max": float(np.max(cate)),
        "pct_positive": float(np.mean(cate > 0) * 100),
        "pct_negative": float(np.mean(cate < 0) * 100),}

    print(f"\n=== {label} ===")
    print(f"ATE: {ate:.4f}  (95% CI: {ci_lower:.4f} to {ci_upper:.4f})")
    print(
        f"CATE distribution: min={cate.min():.4f}, "
        f"median={np.median(cate):.4f}, max={cate.max():.4f}")
    print(f"Direction consistent: {direction_consistent}")

    return cate, summary_row

# main function to run causal forest for smoking, physical activity, and sleep disorder
def main():
    cohort = pd.read_csv(INPUT_PATH)
    cohort_model = cohort.dropna(subset=[
        "incident_cvd", "smk_curr", "PA_active", "sleep_disorder",
        "r13agey_b", "female", "r13bmi", "uni_degree", "r13cesd"
    ]).copy()
    print(f"Sample size for causal forest: {len(cohort_model)}")

    summary_rows = []

    cate, row = run_causal_forest(
        cohort_model, "smk_curr", "Smoking -> Incident CVD"
    )
    cohort_model["cate_smk"] = cate
    summary_rows.append(row)

    cate, row = run_causal_forest(
        cohort_model, "PA_active", "Physical Activity -> Incident CVD")
    cohort_model["cate_pa"] = cate
    summary_rows.append(row)

    cate, row = run_causal_forest(
        cohort_model, "sleep_disorder", "Sleep Disorder -> Incident CVD")
    cohort_model["cate_sleep"] = cate
    summary_rows.append(row)

    cohort_model.to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(summary_rows).to_csv(SUMMARY_OUTPUT_PATH, index=False)

    print(f"\nSaved CATE results to hrs_causal_forest_results.csv")
    print(f"Saved causal forest summary to hrs_causal_forest_summary.csv")


if __name__ == "__main__":
    main()