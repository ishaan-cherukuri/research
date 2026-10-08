"""Write the scan and subject lists the vertex-wise pipeline runs on.

The frozen cohort (analysis/results/spec_v3_final) stores one row per subject
with the landmark date, which is the date of the last scan acquired while the
subject was still MCI. The pre-outcome window is every scan of that subject at
or before the landmark that does not carry an AD diagnosis code, on the real
IDA acquisition date axis. Rebuilding that here rather than reading a stored
list keeps the vertex-wise cohort tied to the same definition as the paper's
Section 3 analyses; the script asserts that it recovers the same 2,252 scans.

Outputs (in --out_dir):
  vertex_scans.csv     one row per scan: image_id, subject, t1_path (Trillium
                       BIDS path), bsc_dir (Trillium derivatives path), date,
                       years since the subject's first window scan
  vertex_subjects.csv  one row per subject: event and the Model A covariates

Usage:
    python3 build_vertex_cohort.py [--cohort ...] [--manifest ...] [--out_dir ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AD = 3

# Trillium locations; the manifest already stores the BIDS path on /project.
BSC_ROOT = "/scratch/ishaan/adni_trillium/derivatives/bsc"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort",
                    default=str(ROOT / "analysis/results/spec_v3_final/spec_cohort.csv"))
    ap.add_argument("--manifest",
                    default=str(ROOT / "from_cluster/manifest_realdates.csv"))
    ap.add_argument("--out_dir", default=str(HERE / "cohort"))
    ap.add_argument("--expect_scans", type=int, default=2252)
    args = ap.parse_args()

    cohort = pd.read_csv(args.cohort)
    cohort["landmark_date"] = pd.to_datetime(cohort["landmark_date"])
    man = pd.read_csv(args.manifest)

    # Real IDA date where matched, synthetic date otherwise, as in
    # analysis/build_spec_cohort.py::apply_real_dates.
    man["date"] = pd.to_datetime(man["real_date"].fillna(man["acq_date"]))
    scans = man[man["subject"].isin(cohort["subject"])].merge(
        cohort[["subject", "landmark_date", "n_scans_window"]], on="subject")
    in_window = (scans["date"] <= scans["landmark_date"]) & (scans["diagnosis"] != AD)
    scans = scans[in_window].sort_values(["subject", "date"]).reset_index(drop=True)

    per_subj = scans.groupby("subject").size().reindex(cohort["subject"])
    mismatch = per_subj.values != cohort["n_scans_window"].values
    if mismatch.any() or len(scans) != args.expect_scans:
        bad = cohort.loc[mismatch, "subject"].tolist()
        raise SystemExit(f"window reconstruction disagrees with frozen cohort: "
                         f"{len(scans)} scans, mismatched subjects {bad[:10]}")

    first = scans.groupby("subject")["date"].transform("min")
    scans["years"] = (scans["date"] - first).dt.days / 365.25
    scans["bsc_dir"] = BSC_ROOT + "/" + scans["image_id"]
    scans["date"] = scans["date"].dt.date.astype(str)
    scan_cols = ["image_id", "subject", "path", "bsc_dir", "date", "years",
                 "field_strength_ida"]
    scans = scans[scan_cols].rename(columns={"path": "t1_path"})

    subj_cols = ["subject", "event", "time_years", "n_scans_window",
                 "age_at_landmark", "female", "apoe4", "field_strength_bl",
                 "field_strength_changed"]
    subjects = cohort[subj_cols].copy()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scans.to_csv(out / "vertex_scans.csv", index=False)
    subjects.to_csv(out / "vertex_subjects.csv", index=False)
    print(f"{len(scans)} scans from {scans['subject'].nunique()} subjects "
          f"({int(subjects['event'].sum())} converters) -> {out}")


if __name__ == "__main__":
    main()
