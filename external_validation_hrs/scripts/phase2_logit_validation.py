import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np
import os
from sklearn.metrics import roc_auc_score, brier_score_loss

HRS_BASE = os.path.expanduser("~/my_ukb_thesis/external_validation_hrs")
INPUT_PATH = os.path.join(HRS_BASE, "outputs", "hrs_cohort_processed.csv")
OUTPUT_DIR = os.path.join(HRS_BASE, "outputs")

#  bootstrap confidence interval for a given metric function 
def bootstrap_metric_ci(y_true, pred_prob, metric_func, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_prob = np.asarray(pred_prob)
    n = len(y_true)

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_func(y_true[idx], pred_prob[idx]))

    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(len(vals))

# logit transformation of predicted probabilities
def logit_probability(prob, eps=1e-8):
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, eps, 1 - eps)
    return np.log(prob / (1 - prob))

# calibration intercept and slope using logistic regression
def calibration_intercept_slope(y_true, pred_prob):
    y = np.asarray(y_true, dtype=float)
    lp = logit_probability(pred_prob)

    intercept_model = sm.GLM(
        y, np.ones((len(y), 1)),
        family=sm.families.Binomial(),
        offset=lp).fit()
    intercept_ci = np.asarray(intercept_model.conf_int())[0]

    slope_model = sm.GLM(
        y, sm.add_constant(lp),
        family=sm.families.Binomial()).fit()
    slope_ci = np.asarray(slope_model.conf_int())[1]

    return {
        "calibration_intercept": float(intercept_model.params[0]),
        "calibration_intercept_ci_low": float(intercept_ci[0]),
        "calibration_intercept_ci_high": float(intercept_ci[1]),
        "calibration_slope": float(slope_model.params[1]),
        "calibration_slope_ci_low": float(slope_ci[0]),
        "calibration_slope_ci_high": float(slope_ci[1])}

def main():
    cohort = pd.read_csv(INPUT_PATH)

    cohort_model = cohort.dropna(subset=[
        "incident_cvd", "smk_curr", "PA_active", "sleep_disorder",
        "r13agey_b", "female", "r13bmi", "uni_degree", "r13cesd"]).copy()
    print(f"Sample size for adjusted regression: {len(cohort_model)}")

    formula = (
        "incident_cvd ~ smk_curr + PA_active + sleep_disorder "
        "+ r13agey_b + female + r13bmi + uni_degree + r13cesd")
    model = smf.logit(formula, data=cohort_model).fit()
    print(model.summary())

    print("\n=== Odds Ratios (exp(coef)) ===")
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues

    or_table = pd.DataFrame({
        "predictor": params.index,
        "coef": params.values,
        "OR": np.exp(params.values),
        "ci_low": np.exp(conf[0].values),
        "ci_high": np.exp(conf[1].values),
        "p_value": pvals.values,
        "n": len(cohort_model)})
    print(or_table)

    or_path = os.path.join(OUTPUT_DIR, "hrs_logistic_or_table.csv")
    or_table.to_csv(or_path, index=False)
    print(f"Saved OR table: {or_path}")

    y_true = cohort_model["incident_cvd"].values
    pred_prob = model.predict(cohort_model)

    auc = roc_auc_score(y_true, pred_prob)
    brier = brier_score_loss(y_true, pred_prob)

    auc_ci_low, auc_ci_high, auc_boot_n = bootstrap_metric_ci(
        y_true, pred_prob, roc_auc_score, seed=42)
    brier_ci_low, brier_ci_high, brier_boot_n = bootstrap_metric_ci(
        y_true, pred_prob, brier_score_loss, seed=43)
    cal = calibration_intercept_slope(y_true, pred_prob)

    perf = pd.DataFrame([{
        "dataset": "HRS refitted logistic",
        "n": len(y_true),
        "events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "auc": float(auc),
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
        "auc_boot_n": auc_boot_n,
        "brier": float(brier),
        "brier_ci_low": brier_ci_low,
        "brier_ci_high": brier_ci_high,
        "brier_boot_n": brier_boot_n,
        "mean_predicted": float(np.mean(pred_prob)),
        "observed_rate": float(np.mean(y_true)),
        **cal}])

    perf_path = os.path.join(OUTPUT_DIR, "hrs_refit_logistic_performance.csv")
    perf.to_csv(perf_path, index=False)
    print(f"Saved performance table: {perf_path}")
    print(perf)


if __name__ == "__main__":
    main()