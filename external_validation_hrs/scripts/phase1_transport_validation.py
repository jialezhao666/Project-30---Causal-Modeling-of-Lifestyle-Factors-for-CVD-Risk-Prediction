import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os

HRS_BASE = os.path.expanduser("~/my_ukb_thesis/external_validation_hrs")
INPUT_PATH = os.path.join(HRS_BASE, "outputs", "hrs_cohort_processed.csv")
OUTPUT_DIR = os.path.join(HRS_BASE, "outputs")
UKB_OUTPUT_DIR = os.path.expanduser("~/my_ukb_thesis/phase_II/outputs")

# UKB Phase I trained coefficients from baseline_model_training
UKB_COEFS = {
    "const": -2.9369,
    "age_defined_baseline": 0.7021,
    "BMI": 0.2287,
    "sleep_hrs": -0.0284,
    "genetic_sex": 0.7574,
    "mental_doctor": 0.2095,
    "uni_degree": -0.1824,
    "FH_cvd_f": 0.1781,
    "FH_cvd_m": 0.2348,
    "FH_cvd_sib": 0.2457,
    "smk_prev": 0.1369,
    "smk_curr": 0.5825,
    "alc_curr": -0.2779,
    "PA_active": -0.0900,
}

# StandardScaler parameters fitted on UKB Split B (continuous features only)
UKB_SCALER_PARAMS = {
    "age_defined_baseline": {"mean": 56.207423427048234, "std": 8.10607250287873},
    "BMI": {"mean": 27.19264814598736, "std": 4.646651987383076},
    "sleep_hrs": {"mean": 7.147008420143575, "std": 1.0817655324133455},
}

# True Split B means for family history variables 
UKB_FH_MEANS = {
    "FH_cvd_f": 0.382434,
    "FH_cvd_m": 0.278677,
    "FH_cvd_sib": 0.121977,
}

# UKB Split C internal test performance 
#UKB_SPLITC_AUC = 0.7271
#UKB_SPLITC_BRIER = 0.0757

FEATURE_ORDER = [
    "age_defined_baseline", "BMI", "sleep_hrs", "genetic_sex",
    "mental_doctor", "uni_degree", "FH_cvd_f", "FH_cvd_m", "FH_cvd_sib",
    "smk_prev", "smk_curr", "alc_curr", "PA_active"
]


def build_features(cohort_model, mental_doctor_threshold=4):
    X_hrs = pd.DataFrame(index=cohort_model.index)

    X_hrs["age_defined_baseline"] = (
        cohort_model["r13agey_b"] - UKB_SCALER_PARAMS["age_defined_baseline"]["mean"]
    ) / UKB_SCALER_PARAMS["age_defined_baseline"]["std"]

    X_hrs["BMI"] = (
        cohort_model["r13bmi"] - UKB_SCALER_PARAMS["BMI"]["mean"]
    ) / UKB_SCALER_PARAMS["BMI"]["std"]

    # sleep_hrs: no HRS duration variable; neutralized to UKB training mean
    X_hrs["sleep_hrs"] = 0.0

    # genetic_sex: HRS ragender 1=male, 2=female; UKB genetic_sex=1 is male
    X_hrs["genetic_sex"] = (cohort_model["ragender"] == 1).astype(int)

    # mental_doctor: CESD score used as proxy; threshold sensitivity-tested
    X_hrs["mental_doctor"] = (cohort_model["r13cesd"] >= mental_doctor_threshold).astype(int)

    X_hrs["uni_degree"] = cohort_model["uni_degree"]

    X_hrs["FH_cvd_f"] = UKB_FH_MEANS["FH_cvd_f"]
    X_hrs["FH_cvd_m"] = UKB_FH_MEANS["FH_cvd_m"]
    X_hrs["FH_cvd_sib"] = UKB_FH_MEANS["FH_cvd_sib"]

    X_hrs["smk_prev"] = cohort_model["smk_prev"]
    X_hrs["smk_curr"] = cohort_model["smk_curr"]
    X_hrs["alc_curr"] = cohort_model["alc_curr"]
    X_hrs["PA_active"] = cohort_model["PA_active"]

    return X_hrs


def get_pred_prob(X_hrs):
    linear_pred = UKB_COEFS["const"] + sum(
        X_hrs[feat] * UKB_COEFS[feat] for feat in FEATURE_ORDER)
    return 1 / (1 + np.exp(-linear_pred))

def bootstrap_metric_ci(y_true, pred_prob, metric_func, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    pred_prob = np.asarray(pred_prob)
    n = len(y_true)

    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        # AUC needs both outcome classes in the bootstrap sample.
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_func(y_true[idx], pred_prob[idx]))

    vals = np.asarray(vals)
    return {
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "boot_n": int(len(vals)),
    }

def logit_probability(prob, eps=1e-8):
    prob = np.asarray(prob, dtype=float)
    prob = np.clip(prob, eps, 1 - eps)
    return np.log(prob / (1 - prob))

def calibration_metrics(y_true, pred_prob):
    y = np.asarray(y_true, dtype=float)
    pred_prob = np.asarray(pred_prob, dtype=float)
    lp = logit_probability(pred_prob)

    intercept_model = sm.GLM(
        y, np.ones((len(y), 1)),
        family=sm.families.Binomial(),
        offset=lp
    ).fit()
    intercept_ci = np.asarray(intercept_model.conf_int())[0]

    slope_model = sm.GLM(
        y, sm.add_constant(lp),
        family=sm.families.Binomial()
    ).fit()
    slope_ci = np.asarray(slope_model.conf_int())[1]

    auc_ci = bootstrap_metric_ci(y, pred_prob, roc_auc_score, seed=42)
    brier_ci = bootstrap_metric_ci(y, pred_prob, brier_score_loss, seed=43)

    return {
        "n": int(len(y)),
        "events": int(np.sum(y)),
        "event_rate": float(np.mean(y)),
        "auc": float(roc_auc_score(y, pred_prob)),
        "auc_ci_low": auc_ci["ci_low"],
        "auc_ci_high": auc_ci["ci_high"],
        "auc_boot_n": auc_ci["boot_n"],
        "brier": float(brier_score_loss(y, pred_prob)),
        "brier_ci_low": brier_ci["ci_low"],
        "brier_ci_high": brier_ci["ci_high"],
        "brier_boot_n": brier_ci["boot_n"],
        "mean_predicted": float(np.mean(pred_prob)),
        "observed_rate": float(np.mean(y)),
        "intercept": float(intercept_model.params[0]),
        "intercept_ci_low": float(intercept_ci[0]),
        "intercept_ci_high": float(intercept_ci[1]),
        "slope": float(slope_model.params[1]),
        "slope_ci_low": float(slope_ci[0]),
        "slope_ci_high": float(slope_ci[1])
    }


def predict_and_evaluate(X_hrs, y_true, label=""):
    pred_prob = get_pred_prob(X_hrs)
    metrics = calibration_metrics(y_true, pred_prob)

    print(f"\n=== {label} ===")
    print(f"AUC: {metrics['auc']:.4f} "
        f"(95% CI {metrics['auc_ci_low']:.4f} to {metrics['auc_ci_high']:.4f})")
    print(f"Brier score: {metrics['brier']:.4f} "
        f"(95% CI {metrics['brier_ci_low']:.4f} to {metrics['brier_ci_high']:.4f})")
    print(f"Mean predicted probability: {metrics['mean_predicted']:.4f}")
    print(f"Observed event rate: {metrics['observed_rate']:.4f}")
    print(f"Calibration intercept: {metrics['intercept']:.4f} "
        f"(95% CI {metrics['intercept_ci_low']:.4f} to {metrics['intercept_ci_high']:.4f})")
    print(f"Calibration slope: {metrics['slope']:.4f} "
        f"(95% CI {metrics['slope_ci_low']:.4f} to {metrics['slope_ci_high']:.4f})")
    return pred_prob, metrics


def plot_roc_and_calibration(y_true_hrs, pred_prob_hrs, save_path=None):
    y_true_ukb = np.load(os.path.join(UKB_OUTPUT_DIR, "splitC_y_true.npy"))
    pred_prob_ukb = np.load(os.path.join(UKB_OUTPUT_DIR, "splitC_pred_prob.npy"))

    ukb_metrics = calibration_metrics(y_true_ukb, pred_prob_ukb)
    hrs_metrics = calibration_metrics(y_true_hrs, pred_prob_hrs)

    fpr_ukb, tpr_ukb, _ = roc_curve(y_true_ukb, pred_prob_ukb)
    fpr_hrs, tpr_hrs, _ = roc_curve(y_true_hrs, pred_prob_hrs)

    frac_pos_ukb, mean_pred_ukb = calibration_curve(y_true_ukb, pred_prob_ukb, n_bins=10, strategy="quantile")
    frac_pos_hrs, mean_pred_hrs = calibration_curve(y_true_hrs, pred_prob_hrs, n_bins=10, strategy="quantile")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.plot(fpr_ukb, tpr_ukb, label=f"UKB Test Set (AUC = {ukb_metrics['auc']:.3f})",
            color="#4EACC5", lw=2)
    ax1.plot(fpr_hrs, tpr_hrs, label=f"HRS transport (AUC = {hrs_metrics['auc']:.3f})",
            color="#E76254", lw=2, linestyle="--")
    ax1.plot([0, 1], [0, 1], color="grey", linestyle="--", lw=1,
            label="No discrimination")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("False-positive rate")
    ax1.set_ylabel("True-positive rate")
    ax1.text(-0.08, 1.03, "a", transform=ax1.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(mean_pred_ukb, frac_pos_ukb, "o-", label="UKB Test Set",
            color="#4EACC5", lw=2)
    ax2.plot(mean_pred_hrs, frac_pos_hrs, "s--", label="HRS transport",
            color="#E76254", lw=2)
    ax2.plot([0, 0.30], [0, 0.30], color="grey", linestyle="--", lw=1,
            label="Perfect calibration")

    ax2.set_xlim(0, 0.30)
    ax2.set_ylim(0, 0.30)
    ax2.set_xticks(np.arange(0, 0.31, 0.05))
    ax2.set_yticks(np.arange(0, 0.31, 0.05))
    ax2.set_xlabel("Mean predicted risk")
    ax2.set_ylabel("Observed event proportion")
    ax2.text(-0.08, 1.03, "b", transform=ax2.transAxes,
            fontsize=14, fontweight="bold", va="bottom", ha="left")
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)

    metric_text = (
        f"UKB: Brier = {ukb_metrics['brier']:.4f}, intercept = {ukb_metrics['intercept']:.3f}, "
        f"slope = {ukb_metrics['slope']:.3f}\n"
        f"HRS: Brier = {hrs_metrics['brier']:.4f}, intercept = {hrs_metrics['intercept']:.3f}, "
        f"slope = {hrs_metrics['slope']:.3f}"
    )

    ax2.text(0.97, 0.04, metric_text, transform=ax2.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                    edgecolor="grey", alpha=0.9))

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(OUTPUT_DIR, "ukb_hrs_roc_calibration.png")

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    metrics_df = pd.DataFrame([
        {"dataset": "UKB Split C", **ukb_metrics},
        {"dataset": "HRS transported model", **hrs_metrics}
    ])

    metrics_path = os.path.join(OUTPUT_DIR, "ukb_hrs_validation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved figure: ukb_hrs_roc_calibration.png ")
    print(f"Saved metrics: ukb_hrs_validation_metrics.csv")
    print(metrics_df)

    return ukb_metrics, hrs_metrics

def main():
    cohort = pd.read_csv(INPUT_PATH)
    cohort_model = cohort.dropna(subset=[
        "incident_cvd", "smk_curr", "smk_prev", "PA_active", "r13agey_b",
        "ragender", "r13bmi", "uni_degree", "r13cesd", "alc_curr"
    ]).copy()
    print(f"Sample size for Phase I transport validation: {len(cohort_model)}")

    y_true = cohort_model["incident_cvd"]

    # Main result
    X_hrs_main = build_features(cohort_model, mental_doctor_threshold=4)
    pred_main, hrs_metrics = predict_and_evaluate(
    X_hrs_main, y_true, "HRS transport validation (CESD >= 4)")

    # Sensitivity analysis: mental_doctor threshold 
    print("\n" + "=" * 50)
    print("Sensitivity analysis: effect of mental_doctor threshold")
    print("=" * 50)
    sensitivity_rows = []

    for thresh in [2, 3, 4, 5, 6]:
        X_sens = build_features(cohort_model, mental_doctor_threshold=thresh)
        _, sens_metrics = predict_and_evaluate(X_sens, y_true, f"CESD >= {thresh}")
        sensitivity_rows.append({"cesd_threshold": thresh, **sens_metrics})

    sensitivity_path = os.path.join(OUTPUT_DIR, "hrs_transport_cesd_sensitivity.csv")
    pd.DataFrame(sensitivity_rows).to_csv(sensitivity_path, index=False)
    print(f"Saved sensitivity results: hrs_transport_cesd_sensitivity.csv")

    plot_roc_and_calibration(y_true, pred_main)


if __name__ == "__main__":
    main()