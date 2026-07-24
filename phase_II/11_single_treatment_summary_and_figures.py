import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PHASE2_DIR = os.path.expanduser("~/my_ukb_thesis/phase_II")
OUTPUT_DIR = os.path.join(PHASE2_DIR, "outputs")
EXPECTED_N = 321188
FIGURE_DIR = os.path.join(PHASE2_DIR, "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# Part 1: Supplementary single-treatment CATE summary

specs = [
    {
        "label": "Smoking cessation",
        "treatment": "no_current_smoking",
        "ite_file": "ite_smk_curr.npy",
        "se_file": "ite_smk_curr_se.npy",
        "orientation": -1,
        "model_source": "EconML single-treatment CausalForestDML",
        "estimand_type": "marginal single-behaviour contrast",
        "note": (
            "Original model estimates current-smoking effect; "
            "multiplied by -1 for no-current-smoking / smoking-cessation effect."
        ),
    },
    {
        "label": "Increase physical activity",
        "treatment": "PA_active",
        "ite_file": "ite_PA_active.npy",
        "se_file": "ite_PA_active_se.npy",
        "orientation": 1,
        "model_source": "EconML single-treatment CausalForestDML",
        "estimand_type": "marginal single-behaviour contrast",
        "note": "Original model estimates active vs inactive physical activity.",
    },
    {
        "label": "Adequate sleep",
        "treatment": "sleep_adequate",
        "ite_file": "ite_sleep_adequate.npy",
        "se_file": "ite_sleep_adequate_se.npy",
        "orientation": 1,
        "model_source": "EconML single-treatment CausalForestDML",
        "estimand_type": "marginal single-behaviour contrast",
        "note": "Original model estimates adequate vs inadequate sleep.",
    },
]

rows = []

for s in specs:
    ite_raw = np.load(os.path.join(OUTPUT_DIR, s["ite_file"]))
    se_raw = np.load(os.path.join(OUTPUT_DIR, s["se_file"]))

    ite = s["orientation"] * ite_raw
    se = se_raw
    n = len(ite)

    assert n == EXPECTED_N, (
        f"{s['ite_file']} has {n:,} rows, expected {EXPECTED_N:,}. "
        "Re-run 02_causalforest on the 70% Phase II analysis set.")
    assert len(se) == EXPECTED_N, (
        f"{s['se_file']} has {len(se):,} rows, expected {EXPECTED_N:,}. "
        "Re-run 02_causalforest on the 70% Phase II analysis set.")
    
    mean_cate = float(np.mean(ite))
    sd_cate = float(np.std(ite, ddof=1))
    se_mean = sd_cate / np.sqrt(n)

    rows.append({
        "treatment": s["treatment"],
        "label": s["label"],
        "model_source": s["model_source"],
        "estimand_type": s["estimand_type"],
        "n": int(n),
        "mean_cate": mean_cate,
        "ci_low": mean_cate - 1.96 * se_mean,
        "ci_high": mean_cate + 1.96 * se_mean,
        "median_cate": float(np.median(ite)),
        "q25": float(np.percentile(ite, 25)),
        "q75": float(np.percentile(ite, 75)),
        "sd_cate": sd_cate,
        "mean_individual_se": float(np.mean(se)),
        "median_individual_se": float(np.median(se)),
        "se_min": float(np.min(se)),
        "se_max": float(np.max(se)),
        "pct_protective": float(np.mean(ite < 0) * 100),
        "pct_harmful": float(np.mean(ite > 0) * 100),
        "note": s["note"]})

summary = pd.DataFrame(rows)
summary_path = os.path.join(OUTPUT_DIR, "single_treatment_cate_summary.csv")
summary.to_csv(summary_path, index=False)

print("\n=== Supplementary single-treatment CATE summary ===")
print(summary)
print(f"Saved: single_treatment_cate_summary.csv")


# Part 2: Dissertation-ready joint CATE bar chart with CI

joint_path = os.path.join(OUTPUT_DIR, "joint_cate_summary.parquet")

if os.path.exists(joint_path):
    joint = pd.read_parquet(joint_path)
    joint = joint.sort_values("arm").copy()
    
    if 'n' in joint.columns:
        n_unique = joint['n'].dropna().unique()
        print(f"  joint_cate_summary.parquet n values: {n_unique}")
        assert 321188 in n_unique, (
            "Expected joint_cate_summary.parquet from the 70% Phase II analysis set. "
            "Re-run 03_multiarm_joint.")

    label_map = {
        1: "Current non-smoker\nonly",
        2: "Increase physical\nactivity only",
        3: "Current non-smoker +\nincrease activity",
        4: "Improve sleep\nonly",
        5: "Current non-smoker +\nimprove sleep",
        6: "Increase activity +\nimprove sleep",
        7: "All three\nchanges"}

    group_map = {
        1: "1 lifestyle change",
        2: "1 lifestyle change",
        4: "1 lifestyle change",
        3: "2 lifestyle changes",
        5: "2 lifestyle changes",
        6: "2 lifestyle changes",
        7: "3 lifestyle changes"}

    color_map = {
        "1 lifestyle change": "#8DB9D6",
        "2 lifestyle changes": "#4F76B5",
        "3 lifestyle changes": "#2C5573"}

    # Logical display order: singles, pairs, all three
    display_order = [1, 4, 2, 3, 5, 6, 7]
    plot_df = joint.set_index("arm").loc[display_order].reset_index()
    plot_df["plot_label"] = plot_df["arm"].map(label_map)
    plot_df["group"] = plot_df["arm"].map(group_map)
    plot_df["color"] = plot_df["group"].map(color_map)

    x = np.arange(len(plot_df))
    y = plot_df["mean_cate"].values
    yerr_low = y - plot_df["ci_low"].values
    yerr_high = plot_df["ci_high"].values - y

    fig, ax = plt.subplots(figsize=(14, 6.0))

    bars = ax.bar(x, y, color=plot_df["color"],
        width=0.6, edgecolor="white", linewidth=0.8)

    ax.errorbar(x, y, yerr=[yerr_low, yerr_high],
        fmt="none", ecolor="black", elinewidth=1.2,
        capsize=3, capthick=1.2, zorder=3)

    ax.axhline(0, color="black", linewidth=1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["plot_label"], fontsize=10)
    ax.set_ylabel("Risk difference vs all-unhealthy reference")
    ax.set_xlabel("Lifestyle change scenario")
    ax.set_ylim(-0.105, 0.005)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    # Mean and 95% CI labels under each bar.
    for i, row in enumerate(plot_df.itertuples()):
        label = (
            f"{row.mean_cate:.4f}\n"
            f"[{row.ci_low:.4f}, {row.ci_high:.4f}]")
        ax.text(i, row.mean_cate - 0.004, label,
            ha="center", va="top", fontsize=8.5,
            color="black", linespacing=1.05)

    # Legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=color_map["1 lifestyle change"], label="1 lifestyle change"),
        Patch(facecolor=color_map["2 lifestyle changes"], label="2 lifestyle changes"),
        Patch(facecolor=color_map["3 lifestyle changes"], label="3 lifestyle changes")]
    ax.legend(handles=handles, loc="lower left", frameon=False)

    plt.tight_layout()

    fig_path = os.path.join(FIGURE_DIR, "joint_cate_bar_dissertation.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved: joint_cate_bar_dissertation.png")
else:
    print(f"Skipped joint CATE figure; file not found")