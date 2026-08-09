"""
Build all paper figures for the ADNI-Trillium (657-subject) analysis with OASIS-3
external validation. Everything is written to a single directory (default research/figs).

Figures produced:
    sample_overview_trillium.png          cohort age distribution + longitudinal spans
    model_comparison_boxplot_trillium.png 7-configuration 5-fold CV comparison
    kaplan_meier_curves_trillium.png      KM by XGBoost predicted-risk tertile
    external_validation_oasis.png         OASIS-3 KM + internal-vs-external C-index

Risk direction: the survival:aft objective predicts log survival TIME, so risk is
taken as -prediction everywhere below.
"""

from __future__ import annotations

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
from sksurv.metrics import concordance_index_censored

RC = {"font.family": "DejaVu Sans", "font.size": 10.5,
      "axes.titlesize": 12, "axes.labelsize": 11, "figure.facecolor": "white"}

MODEL_COLORS = {
    "Weibull-BL\n(baseline BSC)": "#8f8f8f",
    "Weibull-20":                 "#a87090",
    "LogLogistic-20":             "#c07060",
    "LogNormal-20":               "#c4a030",
    "RSF-20\n(BSC only)":         "#7090b8",
    "RSF-57\n(BSC+T1)":           "#568a6a",
    "XGB-57\n(BSC+T1)":           "#2f6f4f",
}
KM_COLORS = {"high": "#d62728", "mid": "#ff9e2c", "low": "#2ca02c"}
F_COLOR, M_COLOR = "#F8766D", "#00BFC4"
PANEL_BG = "#eaeaee"


# ------------------------------------------------------------------ fig 1
def fig_sample_overview(cohort, manifest, out_path):
    plt.rcParams.update(RC)
    c = cohort.dropna(subset=["age"]).copy()
    c["sex"] = np.where(c.female, "F", "M")

    span = (manifest.assign(acq=pd.to_datetime(manifest.acq_date))
            .groupby("subject").acq.agg(["min", "max"]))
    span["span_yr"] = (span["max"] - span["min"]).dt.days / 365.25
    c = c.merge(span[["span_yr"]], left_on="subject", right_index=True, how="left")
    c["span_yr"] = c.span_yr.fillna(c.time_years)
    c["age_end"] = c.age + c.span_yr

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    ax = axes[0]
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color="white", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    # Stacked by sex, with converters shown as the hatched portion of each stack.
    bins = np.arange(np.floor(c.age.min()), np.ceil(c.age.max()) + 2, 2)
    centers, width = (bins[:-1] + bins[1:]) / 2, (bins[1] - bins[0]) * 0.92
    bottom = np.zeros(len(bins) - 1)
    for sex, col in [("F", F_COLOR), ("M", M_COLOR)]:
        for conv, hatch in [(0, None), (1, "///")]:
            sub = c[(c.sex == sex) & (c.event == conv)]
            h, _ = np.histogram(sub.age, bins=bins)
            ax.bar(centers, h, width=width, bottom=bottom, color=col,
                   edgecolor="#3a3a3a" if conv else "white",
                   linewidth=0.7 if conv else 0.4, hatch=hatch, alpha=0.9, zorder=3)
            bottom += h
    for lab, col in [("F", F_COLOR), ("M", M_COLOR)]:
        ax.bar(0, 0, color=col, label=lab)
    ax.bar(0, 0, facecolor="white", edgecolor="#3a3a3a", hatch="///", label="Converter")
    ax.set_xlim(bins[0] - 1, bins[-1] + 1)
    ax.set_xlabel("Age at baseline (years)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title("Cross-sectional sample", fontweight="bold")
    ax.legend(title="Sex / Group", fontsize=9, title_fontsize=9)

    ax = axes[1]
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, color="white", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    s = c.sort_values("age").reset_index(drop=True)
    for i, r in s.iterrows():
        col = F_COLOR if r.sex == "F" else M_COLOR
        conv = r.event == 1
        ax.plot([r.age, r.age_end], [i, i], color=col, linewidth=0.9 if conv else 0.7,
                linestyle="--" if conv else "-", alpha=0.85, zorder=2)
        ax.plot([r.age_end], [i], marker="x" if conv else "o",
                markersize=4.5 if conv else 2.0, color=col, zorder=3)
    for lbl, col, ls, mk in [("F - Stable", F_COLOR, "-", "o"), ("M - Stable", M_COLOR, "-", "o"),
                             ("F - Converter", F_COLOR, "--", "x"), ("M - Converter", M_COLOR, "--", "x")]:
        ax.plot([], [], color=col, linestyle=ls, marker=mk, markersize=4, label=lbl)
    ax.set_xlabel("Age (years)", fontweight="bold")
    ax.set_ylabel("Subject", fontweight="bold")
    ax.set_title("Longitudinal sample", fontweight="bold")
    ax.legend(title="Sex / Group", fontsize=8, title_fontsize=8, loc="upper left")

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_path).with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  sample overview -> {out_path}  (n={len(c)})")


# ------------------------------------------------------------------ fig 2
def fig_boxplot(all_cv, out_path):
    plt.rcParams.update(RC)
    labels = list(all_cv.keys())
    pos = list(range(1, len(labels) + 1))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    fig.suptitle("Model Performance Comparison (5-Fold Cross-Validation, ADNI n=657)",
                 fontsize=13, fontweight="bold", y=1.01)
    panels = [("train", "Train C-Index", "Random (0.5)", 0.5),
              ("test", "Test C-Index", "Random (0.5)", 0.5),
              ("gap", "Overfitting Gap\n(Train − Test)", "No overfitting", 0.0)]

    for ax, (key, ylabel, ref_label, ref) in zip(axes, panels):
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, linestyle="--", linewidth=0.6, color="white", alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
        for p, lab in zip(pos, labels):
            data = all_cv[lab][key]
            bp = ax.boxplot(data, positions=[p], patch_artist=True, widths=0.58,
                            medianprops=dict(color="#111111", linewidth=2.0),
                            whiskerprops=dict(color="#555555", linewidth=1.0),
                            capprops=dict(color="#555555", linewidth=1.0),
                            boxprops=dict(linewidth=0.7, edgecolor="#555555"),
                            flierprops=dict(marker="o", markersize=5, linewidth=0,
                                            markerfacecolor="none",
                                            markeredgecolor="#666666"))
            bp["boxes"][0].set_facecolor(MODEL_COLORS.get(lab, "#7090b8"))
            bp["boxes"][0].set_alpha(0.82)
            ax.scatter([p], [np.mean(data)], marker="D", color="#d85820", s=50,
                       zorder=6, linewidths=0.5, edgecolors="#993300")
        ax.axhline(ref, color="#999999", linestyle="--", linewidth=1.1, zorder=1)
        ax.text(0.02, ref, f" {ref_label}", transform=ax.get_yaxis_transform(),
                fontsize=8, color="#777777", va="bottom")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xlabel("Model Configuration", fontweight="bold")
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, fontsize=7.2, rotation=30, ha="right")
        ax.set_xlim(0.3, len(labels) + 0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_path).with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  boxplot -> {out_path}")


# ------------------------------------------------------------------ KM helper
def km_panel(ax, time, event, risk, title):
    p33, p67 = np.percentile(risk, [33.33, 66.67])
    high, low = risk >= p67, risk <= p33
    mid = ~(high | low)
    meds = {}
    for mask, key, stub in [(high, "high", "High Risk"), (mid, "mid", "Medium Risk"),
                            (low, "low", "Low Risk")]:
        n, e = int(mask.sum()), int(event[mask].sum())
        k = KaplanMeierFitter().fit(time[mask], event[mask],
                                    label=f"{stub} (n={n}, events={e})")
        k.plot_survival_function(ax=ax, color=KM_COLORS[key], linewidth=2.5, ci_show=True)
        meds[key] = (n, e, float(k.median_survival_time_))
    lr = logrank_test(time[high], time[low], event[high], event[low])
    txt = "Log-rank p<0.0001" if lr.p_value < 1e-4 else f"Log-rank p={lr.p_value:.4f}"
    ax.set_xlabel("Time (years)", fontweight="bold")
    ax.set_ylabel("P(Remain MCI)", fontweight="bold")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.text(0.98, 0.02, txt, transform=ax.transAxes, fontsize=10, ha="right",
            va="bottom", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
    ax.set_ylim(0, 1.05)
    return {"medians": meds, "p": float(lr.p_value), "chi2": float(lr.test_statistic)}


# ------------------------------------------------------------------ fig 3
def fig_km(train_df, test_df, out_path):
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.subplots_adjust(wspace=0.32)
    stats = {}
    for ax, d, title in [(axes[0], train_df, "Training Set"), (axes[1], test_df, "Test Set")]:
        stats[title] = km_panel(ax, d.true_time.values, d.event.values,
                                -d.predicted_risk.values,
                                f"{title}\nKaplan-Meier by Predicted Risk")
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(Path(out_path).with_suffix("." + ext), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  KM curves -> {out_path}")
    return stats


# ------------------------------------------------------------------ fig 4
def fig_external(ext_df, internal_cv, out_path, n_boot=4000):
    plt.rcParams.update(RC)
    t, e = ext_df.true_time.values, ext_df.event.values.astype(bool)
    risk = -ext_df.predicted_risk.values
    ci = float(concordance_index_censored(e, t, risk)[0])

    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        i = rng.integers(0, len(ext_df), len(ext_df))
        if e[i].sum() < 2:
            continue
        try:
            boot.append(concordance_index_censored(e[i], t[i], risk[i])[0])
        except Exception:
            pass
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    stats = km_panel(axes[0], t, ext_df.event.values, risk,
                     f"OASIS-3 External Cohort (n={len(ext_df)})\n"
                     "Kaplan-Meier by Predicted Risk")

    ax = axes[1]
    ax.set_facecolor(PANEL_BG)
    ax.grid(True, axis="x", linestyle="--", color="white", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    itest = np.array(internal_cv["XGB-57\n(BSC+T1)"]["test"])
    rows = [("ADNI internal\n(5-fold CV)", itest.mean(),
             itest.mean() - itest.std() * 1.96 / np.sqrt(len(itest)),
             itest.mean() + itest.std() * 1.96 / np.sqrt(len(itest)), "#2f6f4f"),
            ("OASIS-3 external\n(frozen model)", ci, lo, hi, "#b5453a")]
    for i, (lab, m, l, h, col) in enumerate(rows):
        ax.errorbar(m, i, xerr=[[m - l], [h - m]], fmt="o", color=col, markersize=11,
                    capsize=6, linewidth=2.2, zorder=5)
        ax.text(m, i + 0.16, f"{m:.3f}  [{l:.2f}, {h:.2f}]", ha="center",
                fontsize=9.5, color=col, fontweight="bold")
    ax.axvline(0.5, color="#999999", linestyle="--", linewidth=1.2)
    ax.text(0.5, -0.42, " chance", color="#777777", fontsize=8.5, va="bottom")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_ylim(-0.5, len(rows) - 0.3)
    ax.set_xlim(0.40, 1.0)
    ax.set_xlabel("Concordance index", fontweight="bold")
    ax.set_title("Internal vs. external discrimination\n(95% CI)", fontweight="bold")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    plt.tight_layout()
    for ext_ in ("png", "pdf"):
        fig.savefig(Path(out_path).with_suffix("." + ext_), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  external validation -> {out_path}")
    return {"c_index": ci, "ci95": [float(lo), float(hi)],
            "n": int(len(ext_df)), "n_events": int(e.sum()),
            "p_below_chance": float((boot <= 0.5).mean()), "km": stats}


# ------------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="../from_cluster")
    p.add_argument("--results_dir", default="results")
    p.add_argument("--ext_pred",
                   default="../from_cluster/results_external_oasis/external_eval_predictions.csv")
    p.add_argument("--figs_dir", default="/Users/ishu/research/figs")
    a = p.parse_args()

    D, R, F = Path(a.data_dir), Path(a.results_dir), Path(a.figs_dir)
    F.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting figures to {F.resolve()}\n")

    cohort = pd.read_csv(R / "cohort_table.csv", parse_dates=["baseline_date"])
    manifest = pd.read_csv(D / "manifest.csv")
    all_cv = json.load(open(R / "all_models_cv.json"))
    tr = pd.read_csv(R / "xgb_train_predictions.csv")
    te = pd.read_csv(R / "xgb_predictions.csv")
    ext = pd.read_csv(a.ext_pred)

    fig_sample_overview(cohort, manifest, F / "sample_overview_trillium.png")
    fig_boxplot(all_cv, F / "model_comparison_boxplot_trillium.png")
    km = fig_km(tr, te, F / "kaplan_meier_curves_trillium.png")
    ext_stats = fig_external(ext, all_cv, F / "external_validation_oasis.png")

    stats = {"km_internal": km, "external": ext_stats,
             "cv": {k.replace("\n", " "): {"test_mean": float(np.mean(v["test"])),
                                           "test_sd": float(np.std(v["test"])),
                                           "train_mean": float(np.mean(v["train"])),
                                           "gap_mean": float(np.mean(v["gap"]))}
                    for k, v in all_cv.items()}}
    json.dump(stats, open(F / "figure_stats_trillium.json", "w"), indent=2, default=str)
    print(f"\n  caption numbers -> {F / 'figure_stats_trillium.json'}")

    print("\nExternal: C={c_index:.4f} 95% CI [{a:.3f}, {b:.3f}]  n={n} events={n_events}"
          .format(c_index=ext_stats["c_index"], a=ext_stats["ci95"][0],
                  b=ext_stats["ci95"][1], n=ext_stats["n"], n_events=ext_stats["n_events"]))
    for k, v in km.items():
        h = v["medians"]["high"]
        print(f"  {k}: high-risk n={h[0]} events={h[1]} median={h[2]}")
    print()


if __name__ == "__main__":
    main()
