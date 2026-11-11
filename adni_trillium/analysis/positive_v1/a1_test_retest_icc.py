#!/usr/bin/env python3
"""A1, the gating analysis of PRESPEC_positive_v1.md: scan-rescan reliability.

Every association this folder tests is bounded by how well a measure agrees
with itself. This script estimates that bound.

Two pair definitions, because ADNI does not give one clean test-retest sample
and conflating them would be wrong:

  cross_field    The screening and baseline T1s. 81 subjects have both, 67
                 with real IDA dates, median 23 days apart. ADNI1 deliberately
                 scanned subjects at 1.5T and 3T, so 65 of 67 of these pairs
                 straddle a field strength change. Over 23 days the biological
                 change in a boundary measure is negligible, so disagreement
                 here is measurement variance PLUS field strength. This is a
                 cross-scanner reproducibility estimate, not a noise floor.

  same_scanner   All within-subject scan pairs at the same field strength and
                 the same scanner model, at most --max_gap_days apart, keeping
                 the shortest-gap pair per subject so that each subject
                 contributes once and the ICC independence assumption holds.
                 Field strength and scanner model are held constant, so this is
                 much closer to a true reliability estimate. At a median gap
                 near 127 days it carries a small amount of genuine biological
                 change, which biases the ICC down, so it is a lower bound.

Both are reported. The difference between them is the field strength effect,
which is the measurement result this project already cares about.

Positive controls are included and are not optional. If a quantity that must
agree across two scans of the same subject does not agree, the pairing or the
upstream pipeline is at fault and no ICC in the table means anything. Read the
control rows before the measure rows.

Estimand is ICC(1,1), absolute agreement, single measurement, one-way random
effects: the two scans of a pair are interchangeable, there is no systematic
rater distinguishing them, and a measure that is consistently offset between
sessions is unusable for longitudinal slopes, so absolute agreement rather
than consistency is the right target.

Reads only files already on this machine. Writes a1_icc.csv.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# (label, file, column, is_control)
MEASURES = [
    ("gBSC directional (cortex mean)", "bsc_simple_features_merged.csv", "bsc_dir_mean", False),
    ("gBSC magnitude (cortex mean)", "bsc_simple_features_merged.csv", "bsc_mag_mean", False),
    ("cosBSC (cortex mean)", "bsc_simple_features_cos.csv", "bsc_cos_mean", False),
    ("gBSC AD-signature composite", "per_scan_regional.csv", "bscdir_adsig", False),
    ("cosBSC AD-signature composite", "per_scan_cos.csv", "cos_adsig", False),
    # Controls. TIV is the strongest: it is fixed in adulthood, so anything
    # below about 0.95 on same-scanner pairs indicts the pipeline or the join.
    ("CONTROL intracranial volume", "t1_scan_features.csv", "seg_tiv_mm3", True),
    ("CONTROL brain volume", "t1_scan_features.csv", "seg_brain_mm3", True),
    ("CONTROL brain parenchymal fraction", "t1_scan_features.csv", "seg_bpf", True),
    ("CONTROL SNR", "t1_scan_features.csv", "qc_snr", True),
    ("CONTROL boundary voxel count", "bsc_simple_features_merged.csv", "Nboundary", True),
]

DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d/%m/%Y")


def parse_date(text: str):
    if not text:
        return None
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(text[:10], fmt)
        except ValueError:
            continue
    return None


def icc11(pairs: np.ndarray, alpha: float = 0.05):
    """ICC(1,1) with an exact F-based confidence interval. pairs is (n, 2)."""
    n, k = pairs.shape
    grand = pairs.mean()
    subject_means = pairs.mean(axis=1)
    ms_between = k * ((subject_means - grand) ** 2).sum() / (n - 1)
    ms_within = ((pairs - subject_means[:, None]) ** 2).sum() / (n * (k - 1))
    icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)

    f_obs = ms_between / ms_within if ms_within > 0 else np.inf
    df1, df2 = n - 1, n * (k - 1)
    f_upper = stats.f.ppf(1 - alpha / 2, df1, df2)
    f_lower = stats.f.ppf(1 - alpha / 2, df2, df1)
    lo = (f_obs / f_upper - 1) / (k - 1 + f_obs / f_upper)
    hi = (f_obs * f_lower - 1) / (k - 1 + f_obs * f_lower)
    return icc, lo, hi, ms_within


def load_column(root: Path, filename: str, column: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with open(root / filename, newline="") as handle:
        for row in csv.DictReader(handle):
            raw = row.get(column, "")
            if raw in ("", "nan", "NaN", "NA"):
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
            if np.isfinite(value):
                out[row["image_id"]] = value
    return out


def read_manifest(root: Path) -> list[dict]:
    with open(root / "manifest_realdates.csv", newline="") as handle:
        return list(csv.DictReader(handle))


def pairs_cross_field(rows: list[dict], max_gap_days: int) -> list[dict]:
    """Screening versus baseline, which in ADNI1 straddles a field change."""
    by_subject: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["visit_code"] in ("sc", "bl"):
            by_subject[row["subject"]][row["visit_code"]] = row

    out = []
    for subject, visits in by_subject.items():
        if len(visits) != 2:
            continue
        first, second = visits["sc"], visits["bl"]
        date_a, date_b = parse_date(first["real_date"]), parse_date(second["real_date"])
        if date_a is None or date_b is None:
            continue
        gap = abs((date_b - date_a).days)
        if gap > max_gap_days:
            continue
        out.append(
            {
                "subject": subject,
                "image_id_a": first["image_id"],
                "image_id_b": second["image_id"],
                "gap_days": gap,
                "same_field": first["field_strength_ida"].strip() == second["field_strength_ida"].strip()
                and bool(first["field_strength_ida"].strip()),
            }
        )
    return out


def pairs_same_scanner(rows: list[dict], max_gap_days: int) -> list[dict]:
    """Same field strength and scanner model, one shortest-gap pair per subject."""
    by_subject: dict[str, list[tuple[dt.datetime, dict]]] = defaultdict(list)
    for row in rows:
        date = parse_date(row["real_date"])
        if date is not None:
            by_subject[row["subject"]].append((date, row))

    best: dict[str, dict] = {}
    for subject, entries in by_subject.items():
        entries.sort(key=lambda item: item[0])
        for (date_a, row_a), (date_b, row_b) in itertools.combinations(entries, 2):
            gap = abs((date_b - date_a).days)
            if gap > max_gap_days:
                continue
            field_a = row_a["field_strength_ida"].strip()
            field_b = row_b["field_strength_ida"].strip()
            model_a = row_a["scanner_model"].strip()
            model_b = row_b["scanner_model"].strip()
            if not (field_a and field_b and field_a == field_b):
                continue
            if not (model_a and model_b and model_a == model_b):
                continue
            candidate = {
                "subject": subject,
                "image_id_a": row_a["image_id"],
                "image_id_b": row_b["image_id"],
                "gap_days": gap,
                "same_field": True,
            }
            if subject not in best or gap < best[subject]["gap_days"]:
                best[subject] = candidate
    return list(best.values())


def describe(name: str, pairs: list[dict]) -> None:
    gaps = sorted(p["gap_days"] for p in pairs)
    same = sum(1 for p in pairs if p["same_field"])
    print(f"{name}: {len(pairs)} pairs, {len(pairs)} subjects")
    if gaps:
        print(
            f"  gap days: min {gaps[0]}, median {gaps[len(gaps) // 2]}, max {gaps[-1]}"
            f"  |  same field strength: {same}/{len(pairs)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/Users/ishu/research/adni_trillium/from_cluster")
    parser.add_argument("--out_dir", default=str(Path(__file__).parent))
    parser.add_argument("--max_gap_days", type=int, default=180)
    args = parser.parse_args()

    root = Path(args.data_root)
    rows = read_manifest(root)

    pair_sets = {
        "cross_field": pairs_cross_field(rows, args.max_gap_days),
        "same_scanner": pairs_same_scanner(rows, args.max_gap_days),
    }
    # A tighter window on the same-scanner set, where biological drift is
    # smaller and the estimate is closer to pure measurement variance.
    pair_sets["same_scanner_90d"] = [
        p for p in pairs_same_scanner(rows, 90)
    ]

    for name, pairs in pair_sets.items():
        describe(name, pairs)
    print()

    results = []
    for label, filename, column, is_control in MEASURES:
        values = load_column(root, filename, column)
        for subset_name, pairs in pair_sets.items():
            usable = [
                (values[p["image_id_a"]], values[p["image_id_b"]])
                for p in pairs
                if p["image_id_a"] in values and p["image_id_b"] in values
            ]
            base = {
                "measure": label,
                "column": column,
                "is_control": int(is_control),
                "pair_set": subset_name,
                "n_pairs": len(usable),
            }
            if len(usable) < 10:
                results.append({**base, "icc": "", "icc_lo": "", "icc_hi": "",
                                "within_sd": "", "between_sd": "", "wscv_pct": "",
                                "sdc": "", "note": "fewer than 10 usable pairs"})
                continue

            arr = np.asarray(usable, dtype=float)
            icc, lo, hi, ms_within = icc11(arr)
            within_sd = float(np.sqrt(ms_within))
            between_sd = float(arr.mean(axis=1).std(ddof=1))
            mean_level = float(arr.mean())
            wscv = 100.0 * within_sd / abs(mean_level) if mean_level else float("nan")
            results.append(
                {
                    **base,
                    "icc": round(icc, 4),
                    "icc_lo": round(lo, 4),
                    "icc_hi": round(hi, 4),
                    "within_sd": round(within_sd, 6),
                    "between_sd": round(between_sd, 6),
                    "wscv_pct": round(wscv, 2),
                    # Smallest change a single subject must show before it
                    # exceeds measurement noise at 95 percent confidence.
                    "sdc": round(1.96 * np.sqrt(2.0) * within_sd, 6),
                    "note": "",
                }
            )

    out_path = Path(args.out_dir) / "a1_icc.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    header = f"{'measure':36s} {'pair set':18s} {'n':>4s} {'ICC':>7s} {'95% CI':>17s} {'wsCV%':>7s}"
    for control_pass in (True, False):
        print("CONTROLS" if control_pass else "\nMEASURES")
        print(header)
        for row in results:
            if bool(row["is_control"]) != control_pass:
                continue
            if row["icc"] == "":
                print(f"{row['measure']:36s} {row['pair_set']:18s} {row['n_pairs']:4d}   {row['note']}")
                continue
            ci = f"[{row['icc_lo']:.3f}, {row['icc_hi']:.3f}]"
            print(
                f"{row['measure']:36s} {row['pair_set']:18s} {row['n_pairs']:4d} "
                f"{row['icc']:7.3f} {ci:>17s} {row['wscv_pct']:7.2f}"
            )

    print("\nPre-specified reading (PRESPEC_positive_v1.md A1), primary measure gBSC directional")
    control = next(
        (r for r in results if r["column"] == "seg_tiv_mm3" and r["pair_set"] == "same_scanner"), None
    )
    primary = next(
        (r for r in results if r["column"] == "bsc_dir_mean" and r["pair_set"] == "same_scanner"), None
    )
    if control and control["icc"] != "" and float(control["icc"]) < 0.9:
        print(
            f"  WITHHELD. The intracranial volume control is only {float(control['icc']):.3f} on\n"
            "  same-scanner pairs. ICV is fixed in adulthood, so this indicts the upstream\n"
            "  segmentation or the pairing, and no measure ICC in this table can be read as\n"
            "  a property of the measure until that is resolved."
        )
    elif primary and primary["icc"] != "":
        icc = float(primary["icc"])
        if icc < 0.5:
            print(f"  ICC = {icc:.3f}, below 0.5. A2 to A4 are underpowered by construction;")
            print("  no positive may be claimed from them. This is itself the measurement result.")
        elif icc >= 0.75:
            print(f"  ICC = {icc:.3f}, at or above 0.75. The previous nulls are strong rather")
            print("  than uninformative, which is a reportable measurement positive.")
        else:
            print(f"  ICC = {icc:.3f}, between 0.5 and 0.75. Associations are reportable but")
            print("  each must carry its attenuation bound.")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
