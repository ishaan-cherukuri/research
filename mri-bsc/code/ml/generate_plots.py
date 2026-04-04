
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

RC = {
    "font.family":       "DejaVu Sans",
    "font.size":         10.5,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

KM_COLORS = {
    "high": "#D32F2F",
    "mid":  "#E87722",
    "low":  "#2E7D32",
}

def create_km_curves(train_df, test_df, out_path):
    plt.rcParams.update(RC)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.32)

    for ax, df, title in [
        (axes[0], train_df, "Training Set"),
        (axes[1], test_df,  "Test Set"),
    ]:
        risk  = df["predicted_risk"].values
        time  = df["true_time"].values
        event = df["event"].values

        p33, p67  = np.percentile(risk, [33.33, 66.67])
        high_mask = risk >= p67
        low_mask  = risk <= p33
        mid_mask  = ~(high_mask | low_mask)

        h_n, h_e = high_mask.sum(), event[high_mask].sum()
        m_n, m_e = mid_mask.sum(),  event[mid_mask].sum()
        l_n, l_e = low_mask.sum(),  event[low_mask].sum()

        for mask, color, label in [
            (high_mask, KM_COLORS["high"],
             f"High Risk (n={h_n}, events={h_e})"),
            (mid_mask,  KM_COLORS["mid"],
             f"Medium Risk (n={m_n}, events={m_e})"),
            (low_mask,  KM_COLORS["low"],
             f"Low Risk (n={l_n}, events={l_e})"),
        ]:
            kmf = KaplanMeierFitter()
            kmf.fit(time[mask], event[mask], label=label)
            kmf.plot_survival_function(
                ax=ax,
                color=color,
                linewidth=2.2,
                ci_show=True,
                ci_alpha=0.20,
            )

        lr = logrank_test(
            time[high_mask], time[low_mask],
            event[high_mask], event[low_mask],
        )

        ax.set_title(
            f"{title}\nKaplan-Meier Survival Curves by Predicted Risk",
            fontsize=12.5, fontweight="bold", pad=9,
        )
        ax.set_xlabel("Time (years)", fontsize=11, fontweight="bold", labelpad=5)
        ax.set_ylabel(
            "Probability of Remaining MCI\n(Not Converting to AD)",
            fontsize=10.5, fontweight="bold", labelpad=5,
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlim(left=0)
        ax.grid(True, linestyle="--", linewidth=0.5, color="#cccccc", alpha=0.85,
                zorder=0)
        ax.set_axisbelow(True)

        ax.legend(
            loc="lower left",
            fontsize=9.5,
            frameon=True,
            framealpha=0.93,
            edgecolor="#bbbbbb",
            handlelength=1.6,
        )

        ax.text(
            0.97, 0.03,
            f"Log-rank p={lr.p_value:.4f}",
            transform=ax.transAxes,
            fontsize=10, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="#F5DEB3", edgecolor="#c8a86b",
                      alpha=0.95, linewidth=0.8),
        )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close()
    print(f"  Saved KM curves → {out_path}")

def create_boxplot(cv_data, out_path):
    plt.rcParams.update(RC)

    PANEL_BG  = "#e8e8e8"
    BOX_COLOR = "#5b8db8"
    MED_COLOR = "#1a1a1a"
    MEAN_COLOR = "#d94f00"

    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    fig.suptitle(
        "Model Performance Comparison (5-Fold Cross-Validation)",
        fontsize=13, fontweight="bold", y=1.01,
    )

    panels = [
        ("train", "Train C-Index",             "Random (0.5)",  0.5),
        ("test",  "Test C-Index",              "Random (0.5)",  0.5),
        ("gap",   "Overfitting Gap\n(Train − Test)", "No overfitting", 0.0),
    ]

    for ax, (key, ylabel, ref_label, ref_val) in zip(axes, panels):
        data = cv_data[key]

        ax.set_facecolor(PANEL_BG)
        ax.grid(True, linestyle="--", linewidth=0.55, color="white", zorder=0)
        ax.set_axisbelow(True)

        bp = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.42,
            showfliers=True,
            flierprops=dict(marker="o", markersize=5, linewidth=0.8,
                            markerfacecolor="white", markeredgecolor="#666666"),
            medianprops=dict(color=MED_COLOR, linewidth=2.2),
            whiskerprops=dict(color="#444444", linewidth=1.1),
            capprops=dict(color="#444444", linewidth=1.1),
            boxprops=dict(linewidth=0.8),
        )
        bp["boxes"][0].set_facecolor(BOX_COLOR)
        bp["boxes"][0].set_alpha(0.82)

        mean_val = np.mean(data)
        ax.scatter([1], [mean_val], marker="D", color=MEAN_COLOR,
                   s=60, zorder=6, linewidths=0.6, edgecolors="#aa3300")

        ax.axhline(ref_val, color="#888888", linestyle="--",
                   linewidth=1.2, alpha=0.85, zorder=1)
        ax.annotate(
            ref_label,
            xy=(1, ref_val), xycoords=("axes fraction", "data"),
            xytext=(-4, 3), textcoords="offset points",
            fontsize=8.5, color="#666666", ha="right", va="bottom",
        )

        ax.set_ylabel(ylabel, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Model Configuration", fontsize=10.5, fontweight="bold")
        ax.set_xticks([1])
        ax.set_xticklabels(["RSF-59\n(BSC+T1)"], fontsize=9.5)

        offset = (max(data) - min(data)) * 0.06 if max(data) != min(data) else 0.01
        ax.text(1.18, mean_val, f"μ={mean_val:.3f}",
                va="center", ha="left", fontsize=9.5, color="#333333",
                transform=ax.get_yaxis_transform())

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved boxplot → {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--train_preds", required=True,
                        help="CSV: rsf_combined_train_predictions.csv")
    parser.add_argument("--test_preds",  required=True,
                        help="CSV: rsf_combined_predictions.csv")
    parser.add_argument("--cv_json",     required=True,
                        help="JSON: rsf_combined_cv.json")
    parser.add_argument("--figs_dir",    default="figs")
    args = parser.parse_args()

    figs_dir = Path(args.figs_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    print("Generating styled KM curves ...")
    train_preds = pd.read_csv(args.train_preds)
    test_preds  = pd.read_csv(args.test_preds)
    create_km_curves(train_preds, test_preds, figs_dir / "kaplan_meier_curves.png")

    print("Generating styled boxplot ...")
    with open(args.cv_json) as f:
        cv_data = json.load(f)
    create_boxplot(cv_data, figs_dir / "model_comparison_boxplot.png")

    print("\nDone. Figures written to:", figs_dir)

if __name__ == "__main__":
    main()
