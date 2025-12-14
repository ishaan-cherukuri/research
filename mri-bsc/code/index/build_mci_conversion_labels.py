"""code.index.build_mci_conversion_labels

Build subject-level MCI->AD conversion labels from a longitudinal ADNI manifest.

Labeling (default):
  - Include subjects with baseline (visit_code == 'bl') diagnosis == 'MCI'
  - label=1 (converter) if ANY later visit has diagnosis == 'AD' within horizon (optional)
  - label=0 (non-converter) otherwise

Outputs:
  - labels CSV: subject,label,bl_date,first_ad_date,conversion_months,n_visits,notes
  - unmatched CSV (optional): subjects/rows skipped and why
"""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _months_between(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    return float((b - a).days / 30.44)


def build_labels(
    manifest_csv: str,
    out_csv: str,
    baseline_code: str = "bl",
    baseline_dx: str = "MCI",
    converter_dx: str = "AD",
    horizon_months: float | None = 36.0,
    out_unmatched_csv: str | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(manifest_csv)

    required = {"subject", "visit_code", "acq_date", "diagnosis"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    df = df.copy()
    df["acq_date_dt"] = _to_dt(df["acq_date"])
    df = df.sort_values(["subject", "acq_date_dt"])

    rows = []
    unmatched = []

    for subject, g in df.groupby("subject", sort=False):
        g = g.sort_values("acq_date_dt")
        bl = g[g["visit_code"] == baseline_code]

        if bl.empty:
            unmatched.append({"subject": subject, "reason": "missing_baseline"})
            continue

        bl_row = bl.iloc[0]
        bl_dx = str(bl_row["diagnosis"])
        bl_date = bl_row["acq_date_dt"]

        if bl_dx != baseline_dx:
            unmatched.append({"subject": subject, "reason": f"baseline_dx_not_{baseline_dx}", "baseline_dx": bl_dx})
            continue

        # Identify first AD after baseline (by date)
        later = g[g["acq_date_dt"] >= bl_date]
        ad_rows = later[later["diagnosis"] == converter_dx].sort_values("acq_date_dt")

        first_ad_date = pd.NaT
        conversion_months = None
        label = 0
        notes = ""

        if not ad_rows.empty:
            first_ad_date = ad_rows.iloc[0]["acq_date_dt"]
            conversion_months = _months_between(bl_date, first_ad_date)

            if horizon_months is None:
                label = 1
            else:
                label = int(conversion_months is not None and conversion_months <= float(horizon_months))

            if label == 0 and conversion_months is not None:
                notes = f"AD found but after horizon ({conversion_months:.2f}mo > {horizon_months}mo)"

        rows.append({
            "subject": subject,
            "label": int(label),
            "bl_date": bl_date.date().isoformat() if not pd.isna(bl_date) else "",
            "first_ad_date": first_ad_date.date().isoformat() if not pd.isna(first_ad_date) else "",
            "conversion_months": round(conversion_months, 3) if conversion_months is not None else "",
            "n_visits": int(len(g)),
            "notes": notes,
        })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(rows).sort_values(["subject"])
    out_df.to_csv(out_path, index=False)

    if out_unmatched_csv:
        um_path = Path(out_unmatched_csv)
        um_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(unmatched).to_csv(um_path, index=False)

    print("[OK] Wrote labels CSV:", out_path)
    print("  labeled subjects:", len(out_df))
    if unmatched:
        msg = f"[WARNING] skipped subjects: {len(unmatched)}"
        if out_unmatched_csv:
            msg += f" (see {out_unmatched_csv})"
        print(msg)

    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--baseline_code", default="bl")
    ap.add_argument("--baseline_dx", default="MCI")
    ap.add_argument("--converter_dx", default="AD")
    ap.add_argument("--horizon_months", type=float, default=36.0, help="Set <=0 to disable horizon.")
    ap.add_argument("--out_unmatched_csv", default="")
    args = ap.parse_args()

    horizon = None
    if args.horizon_months is not None and args.horizon_months > 0:
        horizon = float(args.horizon_months)

    build_labels(
        manifest_csv=args.manifest,
        out_csv=args.out_csv,
        baseline_code=args.baseline_code,
        baseline_dx=args.baseline_dx,
        converter_dx=args.converter_dx,
        horizon_months=horizon,
        out_unmatched_csv=(args.out_unmatched_csv or None),
    )


if __name__ == "__main__":
    main()
