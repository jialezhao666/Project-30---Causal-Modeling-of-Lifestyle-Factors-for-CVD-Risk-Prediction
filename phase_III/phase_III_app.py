import streamlit as st
import numpy as np
import pandas as pd
import os
import joblib
import statsmodels.api as sm
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter

_converter = ro.default_converter + numpy2ri.converter

# ============================================================
# Configuration
# ============================================================

# Paths — adjust if running outside HPC
BASE_DIR = os.path.expanduser("~/my_ukb_thesis")
OUTPUTS_DIR = os.path.join(BASE_DIR, "phase_II/outputs")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Phase I: logistic regression coefficients (statsmodels Logit, Split B)
# Continuous vars (age, BMI, sleep_hrs) standardised before prediction
LR_INTERCEPT = -2.936917
LR_COEFS = {
    "age_defined_baseline": 0.702113,
    "BMI": 0.228672,
    "sleep_hrs": -0.028416,
    "genetic_sex": 0.757434,
    "mental_doctor": 0.209540,
    "uni_degree": -0.182411,
    "FH_cvd_f": 0.178086,
    "FH_cvd_m": 0.234762,
    "FH_cvd_sib": 0.245727,
    "smk_prev": 0.136889,
    "smk_curr": 0.582529,
    "alc_curr": -0.277854,
    "PA_active": -0.090007,
}

# Phase I scaler params (StandardScaler fitted on Split B continuous cols)
LR_SCALER = {
    "age_defined_baseline": {"mean": 56.207423, "std": 8.106073},
    "BMI": {"mean": 27.192648, "std": 4.646652},
    "sleep_hrs": {"mean": 7.146948, "std": 1.078493},
}

# Phase II : confounder column order (must match confounder_scaler.pkl)
NB3_CONFOUNDER_COLS = [
    "age_defined_baseline", "genetic_sex", "BMI", "uni_degree",
    "FH_cvd_f", "FH_cvd_m", "FH_cvd_sib", "mental_doctor", "alc_curr",
]

# population-average CATEs (from joint_cate_summary.parquet, n=298,245)
# Used to flag when an individual's predicted direction diverges from the
# population-level finding reported in the thesis.
POP_AVG_CATE = {
    1: -0.070973,  # no_smk only
    2: -0.028031,  # PA only
    3: -0.080976,  # no_smk + PA
    4: -0.039017,  # sleep only
    5: -0.080518,  # no_smk + sleep
    6: -0.049289,  # PA + sleep
    7: -0.085386,  # all three
}

REF_CVD_RATE = 0.08790088685476706
REF_SLOPE = REF_CVD_RATE * (1 - REF_CVD_RATE)

# arm encoding: T = no_smk + PA*2 + sleep*4
# Arms 1-7 are estimated relative to arm 0 (all-unhealthy reference)
ARM_DESCRIPTIONS = {
    1: "Quit smoking",
    2: "Become physically active",
    3: "Quit smoking + become active",
    4: "Improve sleep (≥ 7 h)",
    5: "Quit smoking + improve sleep",
    6: "Become active + improve sleep",
    7: "All three lifestyle changes",
}


# ============================================================
# Model Loading (cached so only loaded once)
# ============================================================

@st.cache_resource
def load_grf_model():
    """Load multi-arm causal forest into R global env via rpy2."""
    with localconverter(_converter):
        ro.r('.libPaths(c(path.expand("~/Rlibs"), .libPaths()))')
        ro.r("suppressPackageStartupMessages(library(grf))")
        model_path = os.path.join(OUTPUTS_DIR, "maf_model.rds")
        ro.r(f'maf <- readRDS("{model_path}")')
    return True


@st.cache_resource
def load_confounder_scaler():
    """Load Phase II confounder StandardScaler (fitted in NB1 on raw W)."""
    return joblib.load(os.path.join(OUTPUTS_DIR, "confounder_scaler.pkl"))


@st.cache_resource
def load_logit_model():
    """Load Phase I statsmodels Logit model (for cov_params / delta method CI)."""
    return sm.load(os.path.join(OUTPUTS_DIR, "logistic_model.pkl"))


@st.cache_data
def load_sample_support_bounds():
    """Compute 1st / 99th percentiles from Split B for sample-support check."""
    cols = ["age_defined_baseline", "BMI", "sleep_hrs"]
    df = pd.read_parquet(os.path.join(DATA_DIR, "split_B_train.parquet"), columns=cols)
    return {
        col: {"p1": float(df[col].quantile(0.01)),
               "p99": float(df[col].quantile(0.99))}
        for col in cols
    }


# ============================================================
# Prediction helpers
# ============================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

# Order must exactly match statsmodels Logit fit: ['const'] + final_features
LR_FEATURE_ORDER = [
    "age_defined_baseline", "BMI", "sleep_hrs", "genetic_sex", "mental_doctor",
    "uni_degree", "FH_cvd_f", "FH_cvd_m", "FH_cvd_sib", "smk_prev",
    "smk_curr", "alc_curr", "PA_active",
]


def predict_baseline_risk(inputs: dict) -> tuple[float, float, float]:
    """
    Phase I logistic regression: P(CVD) from 13 features (raw values).
    Returns (point_estimate, ci_low, ci_high) using the delta method:
    Var(logit) = x^T * cov_params * x ; Var(p) = Var(logit) * [p(1-p)]^2
    """
    x = inputs.copy()
    for col, p in LR_SCALER.items():
        x[col] = (x[col] - p["mean"]) / p["std"]

    logit = LR_INTERCEPT + sum(LR_COEFS[f] * x[f] for f in LR_COEFS)
    p_hat = float(_sigmoid(logit))

    model = load_logit_model()
    cov = np.asarray(model.cov_params())  # shape (14, 14): const + 13 features

    x_vec = np.array([1.0] + [x[f] for f in LR_FEATURE_ORDER])  # const first
    logit_var = float(x_vec @ cov @ x_vec)
    logit_se = np.sqrt(max(logit_var, 0))

    logit_ci_low = logit - 1.96 * logit_se
    logit_ci_high = logit + 1.96 * logit_se
    p_ci_low = float(_sigmoid(logit_ci_low))
    p_ci_high = float(_sigmoid(logit_ci_high))

    return p_hat, p_ci_low, p_ci_high


def predict_cates(W_raw: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    grf prediction for one new observation.
    confounder_scaler.pkl only scales [age, BMI] (2 continuous vars);
    the 7 binary confounders are passed through unstandardised.
    Returns (cates, cate_ses) each shape (7,).
    """
    scaler = load_confounder_scaler()  # fitted on [age, BMI] only

    # Build the 9-column W row in the same order as NB1
    W_row = []
    continuous_idx = []  # positions of age and BMI in the row
    for i, col in enumerate(NB3_CONFOUNDER_COLS):
        W_row.append(W_raw[col])
        if col in ("age_defined_baseline", "BMI"):
            continuous_idx.append(i)

    W = np.array([W_row])

    # Scale only the 2 continuous columns in-place
    W[:, continuous_idx] = scaler.transform(W[:, continuous_idx])

    with localconverter(_converter):
        ro.r.assign("X_new_vec", ro.FloatVector(W.flatten().tolist()))
        ro.r(f"X_new <- matrix(X_new_vec, nrow=1, ncol={W.shape[1]})")
        ro.r("pred_new <- predict(maf, newdata = X_new, estimate.variance = TRUE)")
        cates = np.array(list(ro.r("as.numeric(pred_new$predictions)")))[:7]
        var_est = np.array(list(ro.r("as.numeric(pred_new$variance.estimates)")))[:7]

    cate_ses = np.sqrt(np.clip(var_est, 0, None))
    return cates, cate_ses


def arm_index(no_smk: int, pa: int, sleep: int) -> int:
    """T = no_smk + PA*2 + sleep*4 (matches 03_multiarm_joint.py arm coding)."""
    return no_smk + pa * 2 + sleep * 4


def risk_category(risk_pct: float) -> str:
    """
    Maps a risk percentage to the same four-bucket label used for the
    baseline risk display (Low / Moderate / Elevated / High), so the same
    classification can be reused to describe a "New Risk" category change
    after a simulation, without duplicating the cutoffs in two places.
    """
    if risk_pct < 5:
        return "Low"
    elif risk_pct < 10:
        return "Moderate"
    elif risk_pct < 15:
        return "Elevated"
    else:
        return "High"


def compute_scenario(baseline_risk, cates, cate_ses, current_arm, tgt):
    """
    Compute the full result dict for ONE specific (current_arm -> tgt) change.
    This is the single-scenario building block used both by the user-driven
    "Simulate" button (one tgt at a time, chosen via the checkboxes) and,
    internally, by build_scenarios() below if every reachable scenario is
    ever needed again (e.g. for the n=5,000 diagnostic audit script).
    """
    cur_cate = 0.0 if current_arm == 0 else cates[current_arm - 1]
    cur_se = 0.0 if current_arm == 0 else cate_ses[current_arm - 1]
    cur_nosmk = (current_arm) & 1
    cur_pa = (current_arm >> 1) & 1
    cur_sleep = (current_arm >> 2) & 1

    t_nosmk = (tgt) & 1
    t_pa = (tgt >> 1) & 1
    t_sleep = (tgt >> 2) & 1

    tgt_cate = cates[tgt - 1]
    tgt_se = cate_ses[tgt - 1]
    delta = tgt_cate - cur_cate
    delta_se = float(np.sqrt(tgt_se**2 + cur_se**2))

    # --- Logit-scale combination ---
    p = baseline_risk
    slope = max(p * (1 - p), 1e-9)  # avoid div-by-zero as p -> 0 or 1
    delta_logit = delta / slope
    delta_se_logit = delta_se / slope

    baseline_logit = _logit(baseline_risk)
    new_risk = float(_sigmoid(baseline_logit + delta_logit))

    ci_low_logit = delta_logit - 1.96 * delta_se_logit
    ci_high_logit = delta_logit + 1.96 * delta_se_logit
    new_risk_ci_low = float(_sigmoid(baseline_logit + ci_low_logit))
    new_risk_ci_high = float(_sigmoid(baseline_logit + ci_high_logit))

    # The risk change is the difference in probability scale
    risk_change = new_risk - baseline_risk
    # The 95% CI for the risk difference (delta) 
    ci_low = delta - 1.96 * delta_se
    ci_high = delta + 1.96 * delta_se

    changes = []
    if t_nosmk > cur_nosmk:
        changes.append("Quit smoking")
    if t_pa > cur_pa:
        changes.append("Become physically active")
    if t_sleep > cur_sleep:
        changes.append("Improve sleep (≥ 7 h)")
    # `pop_avg` is the population-average effect of the same net change
    pop_avg = POP_AVG_CATE[tgt] - POP_AVG_CATE.get(current_arm, 0.0)
    # Flag when the individual's predicted direction of effect diverges from
    diverges = (np.sign(delta) != np.sign(pop_avg)) and abs(delta) > 1e-6

    ci_width_logit = ci_high_logit - ci_low_logit
    pop_avg_mag = max(abs(pop_avg), 0.01)  # floor at 1pp to avoid extreme ratios
    pop_avg_mag_logit = pop_avg_mag / REF_SLOPE
    # Two thresholds, not one -- see the tier-classification comment below
    # for why a single cutoff collapses the middle tier.
    ci_moderately_wide = ci_width_logit > 4 * pop_avg_mag_logit
    ci_too_wide = ci_width_logit > 8 * pop_avg_mag_logit

    if ci_too_wide:
        tier = "unreliable"
    elif ci_moderately_wide or diverges:
        tier = "direction_only"
    else:
        tier = "reliable"

    # Coarse magnitude label for tier 2, based on how the point estimate's
    # SIZE compares to the population-average effect size for this arm --
    # NOT a claim about precision, just a rough "small vs large" descriptor
    # so "direction only" doesn't feel completely contentless.
    if abs(risk_change) >= abs(pop_avg) * 0.75:
        magnitude_label = "a larger-than-typical"
    elif abs(risk_change) >= abs(pop_avg) * 0.25:
        magnitude_label = "a moderate"
    else:
        magnitude_label = "a small"

    return {
        "arm": tgt,
        "label": " + ".join(changes),
        "new_risk": new_risk,
        "new_risk_ci_low": new_risk_ci_low,
        "new_risk_ci_high": new_risk_ci_high,
        "risk_change": risk_change,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "diverges": diverges,
        "unreliable": (tier == "unreliable"),  # kept for backward compatibility
        "tier": tier,
        "direction": "decrease" if risk_change < 0 else "increase",
        "magnitude_label": magnitude_label,
        "pop_avg": pop_avg,
    }


def build_scenarios(baseline_risk, cates, cate_ses, current_arm):
    """
    Return every reachable strictly-healthier scenario from current_arm,
    sorted by largest risk reduction. Not used by the main UI any more (the
    UI now calls compute_scenario() once per user-selected change via the
    checkboxes + Simulate button), but kept for the diagnostic audit script
    and any future use that needs the full set of 7 arms at once.
    """
    cur_nosmk = (current_arm) & 1
    cur_pa = (current_arm >> 1) & 1
    cur_sleep = (current_arm >> 2) & 1

    scenarios = []
    for tgt in range(1, 8):
        t_nosmk = (tgt) & 1
        t_pa = (tgt >> 1) & 1
        t_sleep = (tgt >> 2) & 1
        if t_nosmk < cur_nosmk or t_pa < cur_pa or t_sleep < cur_sleep:
            continue
        if tgt == current_arm:
            continue
        scenarios.append(compute_scenario(baseline_risk, cates, cate_ses, current_arm, tgt))

    scenarios.sort(key=lambda s: s["risk_change"])
    return scenarios


def check_support(inputs, bounds):
    """Return warning strings for out-of-range continuous inputs."""
    labels = {"age_defined_baseline": "Age", "BMI": "BMI", "sleep_hrs": "Sleep hours"}
    warnings = []
    for col, nice in labels.items():
        v = inputs[col]
        b = bounds[col]
        if v < b["p1"] or v > b["p99"]:
            warnings.append(f"{nice} = {v:.1f}  (training range: {b['p1']:.1f} – {b['p99']:.1f})")
    return warnings


# ============================================================
# Streamlit App
# ============================================================

st.set_page_config(page_title="CVD Risk Simulation", layout="wide")

st.title("10-Year CVD Risk Simulation Tool")

st.error(
    "⚠️ **For research illustration only.** This tool is part of an MSc dissertation "
    "and is **not a validated clinical risk calculator**. Estimates rely on "
    "assumptions (e.g. no unmeasured confounding) that cannot be fully verified. "
    "Do not use this tool to make medical decisions — speak to a doctor about your "
    "individual CVD risk and prevention options."
)

st.markdown(
    "This tool estimates your 10-year risk of cardiovascular disease (CVD, including "
    "heart failure and atrial fibrillation) and shows how lifestyle changes could "
    "reduce that risk, based on causal effect estimates from UK Biobank data "
    "(n = 298,245)."
)

# ---- Load models ----
with st.spinner("Loading models (first run only)…"):
    load_grf_model()
    load_logit_model()
    bounds = load_sample_support_bounds()

# ---- Sidebar inputs ----
with st.sidebar:
    st.header("Your Profile")

    with st.form("profile_form"):
        st.subheader("Demographics")
        age = st.slider("Age (years)", 40, 70, 56)
        sex = st.radio("Sex", ["Female", "Male"], horizontal=True)
        bmi = st.slider("BMI (kg/m²)", 15.0, 50.0, 27.0, 0.1)

        st.subheader("Background")
        uni = st.radio("University degree?", ["No", "Yes"], horizontal=True)
        mental = st.radio(
            "Seen a doctor for anxiety or depression?", ["No", "Yes"], horizontal=True,
            help="Have you ever consulted a GP or other doctor specifically "
                 "about feelings of anxiety or depression?",
        )
        alc = st.radio(
            "Current alcohol drinker?", ["No", "Yes"], horizontal=True,
            help="Drinks alcohol at least occasionally, as opposed to "
                 "never or having stopped entirely.",
        )

        st.subheader("Family History of Heart Disease / Stroke")
        fh_f = st.radio("Father", ["No", "Yes"], horizontal=True, key="fh_f")
        fh_m = st.radio("Mother", ["No", "Yes"], horizontal=True, key="fh_m")
        fh_sib = st.radio("Sibling", ["No", "Yes"], horizontal=True, key="fh_sib")

        st.subheader("Current Lifestyle")
        smoking_status = st.radio(
            "Smoking status", ["Never", "Previous", "Current"], horizontal=True,
            help="'Previous' = used to smoke regularly but has now stopped; "
                 "'Current' = currently smokes (any frequency).",
        )
        pa = st.radio(
            "Physically active?", ["No", "Yes"], horizontal=True,
            help="≥ 150 min moderate or ≥ 75 min vigorous activity per week",
        )
        sleep_hrs = st.slider(
            "Average sleep (hours / night)", 3.0, 12.0, 7.0, 0.5,
            help="This tool treats ≥ 7 hours/night as 'adequate' sleep, "
                 "the threshold used in the underlying causal analysis.",
        )

        submitted = st.form_submit_button("Calculate my risk", type="primary", use_container_width=True)

# Stop here until the user has submitted the form at least once in this session
if "has_run" not in st.session_state:
    st.session_state.has_run = False
if submitted:
    st.session_state.has_run = True
    # A new profile submission invalidates any previously simulated
    # scenario (it was computed against the OLD baseline_risk / cates) --
    # clear it so a stale result can't be shown next to a new baseline.
    st.session_state.sim_result = None

if not st.session_state.has_run:
    st.info("👈 Fill in your profile in the sidebar and click **Calculate my risk** to begin.")
    st.stop()

# ---- Derive model inputs ----
inputs = {
    "age_defined_baseline": float(age),
    "genetic_sex": 1.0 if sex == "Male" else 0.0,
    "BMI": float(bmi),
    "uni_degree": 1.0 if uni == "Yes" else 0.0,
    "FH_cvd_f": 1.0 if fh_f == "Yes" else 0.0,
    "FH_cvd_m": 1.0 if fh_m == "Yes" else 0.0,
    "FH_cvd_sib": 1.0 if fh_sib == "Yes" else 0.0,
    "mental_doctor": 1.0 if mental == "Yes" else 0.0,
    "alc_curr": 1.0 if alc == "Yes" else 0.0,
    "smk_prev": 1.0 if smoking_status == "Previous" else 0.0,
    "smk_curr": 1.0 if smoking_status == "Current" else 0.0,
    "PA_active": 1.0 if pa == "Yes" else 0.0,
    "sleep_hrs": float(sleep_hrs),
}
sleep_adequate = 1.0 if sleep_hrs >= 7.0 else 0.0

# ---- Compute ----
baseline_risk, baseline_ci_low, baseline_ci_high = predict_baseline_risk(inputs)

W_raw = {c: inputs[c] for c in NB3_CONFOUNDER_COLS}
cates, cate_ses = predict_cates(W_raw)

no_smk = 1.0 - inputs["smk_curr"]
current_arm_idx = arm_index(int(no_smk), int(inputs["PA_active"]), int(sleep_adequate))

# ---- Display: current risk ----
col_left, col_right = st.columns([1, 2])

with col_left:
    risk_pct = baseline_risk * 100
    st.metric("Your Estimated 10-Year CVD Risk", f"{risk_pct:.1f} %")
    st.caption(f"95% CI: [{baseline_ci_low * 100:.1f} %, {baseline_ci_high * 100:.1f} %]")
    baseline_category = risk_category(risk_pct)
    _category_widget = {"Low": st.success, "Moderate": st.info,
                         "Elevated": st.warning, "High": st.error}[baseline_category]
    _category_widget(f"{baseline_category} risk")
 
with col_right:
    # Current behaviour summary
    behaviours = []
    if inputs["smk_curr"]:
        behaviours.append("Current smoker")
    elif inputs["smk_prev"]:
        behaviours.append("Previous smoker")
    else:
        behaviours.append("Never smoked")
    behaviours.append("Physically active" if inputs["PA_active"] else "Physically inactive")
    behaviours.append(f"Sleep {sleep_hrs:.1f} h / night ({'adequate' if sleep_adequate else 'insufficient'})")
    st.markdown("**Current lifestyle:** " + "  ·  ".join(behaviours))

# ---- Sample-support warning ----
sw = check_support(inputs, bounds)
if sw:
    st.warning(
        "**Sample support warning** — some inputs fall outside the 1st–99th "
        "percentile of the training data. Estimates may be less reliable:\n\n"
        + "\n".join(f"- {w}" for w in sw)
    )
 
# ---- Display: user-driven scenario simulation ----
st.divider()

cur_nosmk_flag = int(no_smk)
cur_pa_flag = int(inputs["PA_active"])
cur_sleep_flag = int(sleep_adequate)

if current_arm_idx == 7:
    st.success(
        "You are already following all three healthy behaviours — "
        "no further lifestyle improvements to simulate."
    )
else:
    st.subheader("What if you changed your lifestyle?")
    st.caption(
        "Tick the changes you'd like to simulate, then press **Simulate**. "
        "You can select one change or combine several. Behaviours you "
        "already follow are disabled below, since there's nothing to "
        "change there."
    )

    chk_col1, chk_col2, chk_col3 = st.columns(3)
    with chk_col1:
        want_quit_smoking = st.checkbox(
            "Quit smoking", value=False, disabled=bool(cur_nosmk_flag),
            help=("Already part of your current lifestyle." if cur_nosmk_flag
                  else "Simulate the effect of quitting smoking."),
        )
    with chk_col2:
        want_pa = st.checkbox(
            "Become physically active", value=False, disabled=bool(cur_pa_flag),
            help=("Already part of your current lifestyle." if cur_pa_flag
                  else "≥ 150 min moderate or ≥ 75 min vigorous activity per week."),
        )
    with chk_col3:
        want_sleep = st.checkbox(
            "Improve sleep (≥ 7 h)", value=False, disabled=bool(cur_sleep_flag),
            help=("Already part of your current lifestyle." if cur_sleep_flag
                  else "Simulate increasing average sleep to 7 hours or more per night."),
        )

    simulate_clicked = st.button("Simulate", type="primary")

    # Persist the simulated result across reruns (e.g. when Streamlit
    # reruns the script for unrelated widget interactions) so the result
    # doesn't disappear until the user runs a new simulation.
    if simulate_clicked:
        chosen_nosmk = 1 if (cur_nosmk_flag or want_quit_smoking) else 0
        chosen_pa = 1 if (cur_pa_flag or want_pa) else 0
        chosen_sleep = 1 if (cur_sleep_flag or want_sleep) else 0
        tgt_arm = arm_index(chosen_nosmk, chosen_pa, chosen_sleep)

        if tgt_arm == current_arm_idx:
            st.session_state.sim_result = None
            st.warning("Tick at least one behaviour change above, then press **Simulate**.")
        else:
            st.session_state.sim_result = compute_scenario(
                baseline_risk, cates, cate_ses, current_arm_idx, tgt_arm
            )

    sim = st.session_state.get("sim_result")
    if sim is not None:
        st.markdown(f"#### Simulating: {sim['label']}")

        if sim["tier"] == "reliable":
            m1, m2, m3 = st.columns(3)
            m1.metric("New Risk", f"{sim['new_risk'] * 100:.1f} %",
                      delta=f"{sim['risk_change'] * 100:+.1f} pp",
                      delta_color="inverse")  
            m2.metric("New Risk 95% CI",
                      f"[{sim['new_risk_ci_low'] * 100:.1f}, {sim['new_risk_ci_high'] * 100:.1f}]")
            m3.metric("From baseline", f"{baseline_risk * 100:.1f} %")

            new_category = risk_category(sim["new_risk"] * 100)
            if new_category != baseline_category:
                st.success(
                    f"This change is predicted to move you from "
                    f"**{baseline_category} risk** to **{new_category} risk**."
                )

            st.caption(
                "The 95% CI reflects estimation uncertainty for someone with "
                "**your specific profile** — it is much wider than a typical "
                "group-average range, because predicting one person's outcome "
                "is inherently less certain than estimating an average effect "
                "across a large group."
            )

        elif sim["tier"] == "direction_only":
            # The direction is the single most important piece of
            # information in this tier, so it gets its own prominent
            # metric/banner rather than being buried inside a sentence.
            if sim["direction"] == "decrease":
                st.success(f"⬇️ Likely to **decrease** your risk, by {sim['magnitude_label']} amount")
            else:
                st.warning(f"⬆️ Likely to **increase** your risk, by {sim['magnitude_label']} amount")

            if sim["diverges"]:
                st.info(
                    "ℹ️ **Not precise enough for a specific number.** Note: "
                    "for most people, this change tends to have the opposite "
                    "effect (the typical finding across the wider study group "
                    "goes the other way). Individual results can genuinely "
                    "differ from the typical pattern — this isn't a tool "
                    "error — but it's worth knowing your result here runs "
                    "against the general trend. The exact size of the effect "
                    "for you is too uncertain to show as a specific number."
                )
                st.caption(
                    "Why might this happen? One possible explanation: the "
                    "model can only learn from the information it's given, "
                    "and it doesn't have access to clinical markers like "
                    "blood pressure or blood sugar. So if someone's health "
                    "profile carries hidden risk that isn't captured by the "
                    "factors in this tool, the model may not fully credit "
                    "a lifestyle change for them — even though the change "
                    "is still likely beneficial in general. This is one "
                    "plausible reason among others, not a confirmed cause "
                    "for your specific result."
                )
            else:
                st.info(
                    "ℹ️ **Not precise enough for a specific number.** The "
                    "direction above is a reasonably confident estimate, but "
                    "there isn't enough precision in the data to give an "
                    "exact percentage for your specific profile."
                )

        else:  # unreliable
            st.warning(
                "⚠️ **Not enough precision to estimate this — not even the "
                "direction.** The uncertainty for your specific profile is "
                "too large to confidently say whether this change would "
                "increase or decrease your risk. This doesn't mean the "
                "change has no effect — it means this tool can't pin down "
                "the effect precisely enough for you individually."
            )

# ---- Methodology ----
with st.expander("Methodology & Limitations"):
    st.markdown("""
**Baseline risk** is estimated by a logistic regression model trained on
298,245 UK Biobank participants (AUC = 0.73 on held-out Split C).

**Causal effects** are estimated by a multi-arm causal forest (Athey, Tibshirani
& Wager, 2019) that jointly models seven combinations of three lifestyle
interventions — smoking cessation, physical activity, and adequate sleep —
relative to an all-unhealthy reference group.

**Why the numbers might look surprising:** Your *baseline risk* comes from a
logistic regression — an associational model. Your *risk change* under each
scenario comes from a causal forest that adjusts for confounding. These two
models can disagree on the size of a lifestyle effect (e.g. physical
activity), because one reflects raw association and the other an estimate of
the causal effect. This divergence is itself a finding of this dissertation,
not a bug in the tool.

**Key assumptions and limitations:**
- Conditional exchangeability (no unmeasured confounding) is assumed. Sensitivity
  analysis via E-values suggests moderate robustness.
- The model is trained on UK Biobank participants (aged 40–70, predominantly
  White British), limiting generalisability to other populations.
- The narrow confidence intervals around causal effects reflect high
  statistical precision from the large sample, not certainty about causation.
- Clinical markers (blood pressure, blood glucose) are excluded from
  confounders to avoid mediator bias; this is a deliberate design choice.
- This tool is for **research illustration only** and must not be used for
  clinical decision-making.
""")