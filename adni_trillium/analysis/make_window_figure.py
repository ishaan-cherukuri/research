"""Observation-window figure for the Brain Communications submission.

fig_window_sweep.png  left panel: fold-matched increment of the BSC block over
                      the clinical covariate model at each landmark, with 95%
                      bootstrap intervals. Right panel: raw cross-validated
                      C-index for each feature block at each landmark.

The point of the pair is that the observation window quadruples across the
x-axis while the increment stays on zero.
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

BLUE, ORANGE, PURPLE = "#2f6f9f", "#c8642a", "#7a5aa8"
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d8d8d4"

LANDMARKS = [12, 24, 36, 48]
PRIMARY_MODEL = "weibull"


def style(ax) -> None:
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def collect(results_root: Path) -> pd.DataFrame:
    rows = []
    for m in LANDMARKS:
        d = results_root / f"landmark{m}"
        cohort = pd.read_csv(d / "landmark_cohort.csv")
        supp = json.load(open(d / "supplementary.json"))
        cv = pd.read_csv(d / "cv_model_comparison.csv")

        inc = next(i for i in supp["incremental"]
                   if i["model"] == PRIMARY_MODEL and i["base"] == "cov"
                   and i["full"] == "cov+bsc")

        def best(block: str) -> float:
            s = cv[cv["features"] == block]
            return float(s["test_mean"].max())

        rows.append({
            "landmark": m,
            "n": len(cohort),
            "events": int(cohort["event"].sum()),
            "scans": float(cohort["n_scans_window"].mean()),
            "span": float(cohort["window_span_years"].mean()),
            "inc": inc["mean_diff"],
            "lo": inc["ci95"][0],
            "hi": inc["ci95"][1],
            "cov": best("cov"),
            "bsc": best("bsc"),
            "t1": best("t1x+t1long"),
            "lr_p": supp["cox"]["lr_p_bsc_block"],
        })
    return pd.DataFrame(rows)


def make(df: pd.DataFrame, out: Path) -> None:
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.2, 4.0), dpi=300)
    x = df["span"].to_numpy()

    axl.axhline(0.0, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    axl.fill_between(x, df["lo"], df["hi"], color=BLUE, alpha=0.16, zorder=2)
    axl.plot(x, df["inc"], "-o", color=BLUE, lw=1.6, ms=5, zorder=4)
    for xi, r in zip(x, df.itertuples()):
        axl.annotate(f"{r.landmark} mo\n{r.events} events", (xi, r.inc),
                     textcoords="offset points", xytext=(0, -26),
                     ha="center", fontsize=7.2, color=MUTED)
    axl.set_ylim(-0.11, 0.06)
    axl.set_xlabel("Mean observation window before the landmark (years)", fontsize=9)
    axl.set_ylabel("Fold-matched C-index gain from BSC slopes\nover clinical covariates",
                   fontsize=9)
    axl.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    axl.set_axisbelow(True)
    style(axl)

    for col, colour, label in (("cov", BLUE, "Clinical covariates"),
                               ("t1", PURPLE, "T1 morphometry"),
                               ("bsc", ORANGE, "BSC slopes")):
        axr.plot(x, df[col], "-o", color=colour, lw=1.6, ms=5, label=label, zorder=4)
    axr.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    axr.text(x[0], 0.508, "chance", color=MUTED, fontsize=7.5)
    axr.set_ylim(0.45, 0.87)
    axr.set_xlabel("Mean observation window before the landmark (years)", fontsize=9)
    axr.set_ylabel("5-fold cross-validated test C-index", fontsize=9)
    axr.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    axr.set_axisbelow(True)
    axr.legend(frameon=False, fontsize=8, loc="upper left", handlelength=1.4)
    style(axr)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_root", default="analysis/results")
    p.add_argument("--out", default="../mri-bsc/figs/revision/fig_window_sweep.png")
    p.add_argument("--table_csv", default="analysis/results/window_sweep.csv")
    a = p.parse_args()

    df = collect(Path(a.results_root))
    df.to_csv(a.table_csv, index=False)
    print(df.to_string(index=False))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    make(df, out)


if __name__ == "__main__":
    main()
