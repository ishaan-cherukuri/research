"""Which batch definition does scanner harmonization actually need?

ComBat needs a batch variable, and the obvious choice for MRI is field strength.
This sweeps candidate definitions and scores each by how much it reduces the
within-subject step in BSC observed when a subject changes scanner mid-study.
That step is a within-subject quantity, so it is not confounded by differences
between the people scanned on each machine, which makes it a fairer target than
a group-level comparison of 1.5T against 3T.

Also writes the trajectory of one subject who changed scanner, used as the
illustrative panel in the scanner figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import longcombat as lc

ROOT = Path(__file__).resolve().parent
CLUSTER = ROOT.parent / "from_cluster"


def load(bsc_path, rd_path):
    raw = pd.read_csv(bsc_path)
    rd = pd.read_csv(rd_path)
    rd["image_id"] = (rd.subject + "_" + rd.visit_code + "_"
                      + pd.to_datetime(rd.acq_date).dt.strftime("%Y-%m-%d"))
    fs = pd.to_numeric(rd.field_strength_ida, errors="coerce")
    rd["fsb"] = np.where(fs < 2.25, "1.5T", "3T")
    rd["fsnum"] = np.where(fs < 2.25, 1.5, 3.0)
    rd.loc[fs.isna(), "fsnum"] = np.nan
    rd["vendor"] = rd.manufacturer.fillna("unk").astype(str).str.split().str[0]
    rd["model"] = rd.scanner_model.fillna("unk").astype(str)
    rd["site"] = rd.subject.str.slice(0, 3)
    keep = ["image_id", "fsb", "fsnum", "vendor", "model", "site", "real_date"]
    d = raw.merge(rd[keep], on="image_id", how="left")
    d["dt"] = pd.to_datetime(d.real_date.fillna(d.acq_date), errors="coerce")
    d["dx"] = pd.to_numeric(d.diagnosis, errors="coerce").fillna(2.0)
    d["age_proxy"] = (d.dt - d.groupby("subject").dt.transform("min")).dt.days / 365.25
    return raw, d.dropna(subset=["dt"])


def mean_abs_step(df, col="bsc_dir_mean"):
    """Mean absolute 3T-minus-1.5T difference within subjects who switched."""
    out = []
    for _, g in df.sort_values(["subject", "dt"]).groupby("subject"):
        g = g.dropna(subset=["fsnum"])
        if g.fsnum.nunique() < 2:
            continue
        a = g[g.fsnum == 1.5][col].mean()
        b = g[g.fsnum == 3.0][col].mean()
        if np.isfinite(a) and np.isfinite(b):
            out.append(b - a)
    return float(np.abs(np.array(out)).mean()), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bsc", default=str(CLUSTER / "bsc_simple_features_merged.csv"))
    ap.add_argument("--realdates", default=str(CLUSTER / "manifest_realdates.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "results/spec_v3_harmonized"))
    ap.add_argument("--example_subject", default="002_S_0729")
    args = ap.parse_args()

    raw, d = load(args.bsc, args.realdates)
    feats = [c for c in raw.columns if c.startswith("bsc_") or c == "Nboundary"]
    base, n_sw = mean_abs_step(d)
    print(f"raw mean |step| = {base:.4f} over {n_sw} subjects who switched\n")

    defs = {
        "field strength": d.fsb,
        "vendor x field strength": d.vendor + "_" + d.fsb,
        "model x field strength": d.model + "_" + d.fsb,
        "site x vendor x field strength": d.site + "_" + d.vendor + "_" + d.fsb,
        "site x model x field strength": d.site + "_" + d.model + "_" + d.fsb,
    }
    rows = []
    for name, col in defs.items():
        dd = d.copy()
        dd["batch"] = col.astype(str)
        # Batches too small to support a variance estimate fall back to the
        # vendor-by-field-strength pool rather than being dropped.
        vc = dd.batch.value_counts()
        small = dd.batch.map(vc) < 15
        dd.loc[small, "batch"] = (dd.vendor + "_" + dd.fsb)[small]
        params = lc.fit(dd, feats, batch_col="batch", covars=["age_proxy", "dx"],
                        subject_col="subject")
        h = lc.apply(dd, params)
        h["dt"], h["fsnum"] = dd.dt, dd.fsnum
        step, _ = mean_abs_step(h)
        pct = 100 * (1 - step / base)
        rows.append({"batch_definition": name, "n_batches": int(dd.batch.nunique()),
                     "mean_abs_step": step, "pct_reduction": pct})
        print(f"{name:32s} batches={dd.batch.nunique():4d}  "
              f"|step|={step:.4f}  ({pct:+.1f}%)")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.attrs["raw_mean_abs_step"] = base
    df.to_csv(out / "batch_sweep.csv", index=False)

    s = d[d.subject == args.example_subject].sort_values("dt")
    if len(s):
        s = s.assign(years=(s.dt - s.dt.min()).dt.days / 365.25)
        s[["subject", "visit_code", "dt", "years", "fsnum", "vendor", "model",
           "bsc_dir_mean"]].rename(columns={"fsnum": "field_strength"}).to_csv(
            out / "example_subject_trajectory.csv", index=False)
        print(f"\nwrote example trajectory for {args.example_subject} "
              f"({len(s)} scans)")
    print(f"wrote {out/'batch_sweep.csv'}")


if __name__ == "__main__":
    main()
