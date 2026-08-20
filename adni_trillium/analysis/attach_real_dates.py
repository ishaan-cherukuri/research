"""Replace the manifest's synthetic acquisition dates with real IDA study dates.

BIDS T1w NIfTIs are de-identified and carry no acquisition date, so
build_manifest.py approximates acq_date as the subject's earliest DXSUM visit
date plus the nominal month offset parsed from the session label. Measured
against the IDA export, that approximation is off by a median of 21 days and by
more than 90 days for 7 percent of scans. Since slopes are regressed against
this time axis, and time-to-event is measured from the last scan, the error
propagates into both the features and the outcome.

The IDA export carries the true Study Date and Image ID. Joining is not a
straight merge: IDA lists more studies than were BIDS-converted for two thirds
of subjects, and visit labels for continuing patients are relative to the
original ADNI1 baseline rather than to the phase, so label matching is
ambiguous.

Instead this aligns each subject's BIDS sessions to their IDA studies with an
order-preserving one-to-one assignment that minimises total date discrepancy.
Both sequences are chronological, so the alignment is monotonic, and extra IDA
studies are skipped rather than forced into a match. A session whose best match
exceeds --max_gap_days is left unmatched instead of being given a wrong date.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_protocol(p: str, key: str):
    for part in str(p).split(";"):
        if part.startswith(key + "="):
            return part.split("=", 1)[1]
    return np.nan


def align(synth: np.ndarray, real: np.ndarray, max_gap: float):
    """Order-preserving assignment of each synth date to a distinct real date.

    Dynamic program over (i, j): cost of matching synth[i] to real[j] is the
    absolute day gap. Skipping a real study is free, since IDA legitimately
    lists studies that were never BIDS-converted. Leaving a synth session
    unmatched costs a fixed penalty just above max_gap, so the optimiser only
    does it when every remaining candidate is worse than the tolerance.
    """
    n, m = len(synth), len(real)
    if n == 0 or m == 0:
        return [None] * n
    skip_cost = max_gap + 1.0
    INF = float("inf")

    # dp[i][j] = best cost matching synth[i:] using real[j:]
    dp = np.full((n + 1, m + 1), INF)
    dp[n, :] = 0.0
    choice = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 match, 1 skip real, 2 drop synth

    for i in range(n - 1, -1, -1):
        for j in range(m, -1, -1):
            best, arg = dp[i + 1, j] + skip_cost, 2  # leave synth[i] unmatched
            if j < m:
                gap = abs(float(synth[i] - real[j]))
                c_match = gap + dp[i + 1, j + 1]
                if c_match < best:
                    best, arg = c_match, 0
                c_skip = dp[i, j + 1]
                if c_skip < best:
                    best, arg = c_skip, 1
            dp[i, j], choice[i, j] = best, arg

    out, i, j = [], 0, 0
    while i < n:
        a = choice[i, j]
        if a == 0:
            out.append(j)
            i, j = i + 1, j + 1
        elif a == 1:
            j += 1
        else:
            out.append(None)
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--manifest", default=str(root / "from_cluster/manifest.csv"))
    ap.add_argument("--ida", default="/Users/ishu/Downloads/idaSearch_8_18_2026.csv")
    ap.add_argument("--out", default=str(root / "from_cluster/manifest_realdates.csv"))
    ap.add_argument("--max_gap_days", type=float, default=180.0)
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    man["acq_date"] = pd.to_datetime(man["acq_date"], errors="coerce")
    man = man.sort_values(["subject", "acq_date"]).reset_index(drop=True)

    ida = pd.read_csv(args.ida, low_memory=False)
    ida["dt"] = pd.to_datetime(ida["Study Date"], errors="coerce", format="mixed")
    ida = ida.dropna(subset=["dt"])
    # Collapse series to one row per study, keeping the richest protocol string.
    ida = (ida.sort_values("Imaging Protocol")
              .drop_duplicates(["Subject ID", "dt"], keep="last"))
    ida["field_strength"] = pd.to_numeric(
        ida["Imaging Protocol"].map(lambda p: parse_protocol(p, "Field Strength")),
        errors="coerce")
    ida.loc[ida["field_strength"] > 100, "field_strength"] /= 10000.0
    ida["manufacturer"] = ida["Imaging Protocol"].map(lambda p: parse_protocol(p, "Manufacturer"))
    ida["scanner_model"] = ida["Imaging Protocol"].map(lambda p: parse_protocol(p, "Mfg Model"))
    by_subj = {s: g.sort_values("dt").reset_index(drop=True)
               for s, g in ida.groupby("Subject ID")}

    recs = []
    for subj, g in man.groupby("subject"):
        g = g.sort_values("acq_date")
        r = by_subj.get(subj)
        if r is None:
            for _, row in g.iterrows():
                recs.append({**row.to_dict(), "real_date": pd.NaT, "gap_days": np.nan})
            continue
        s_days = g["acq_date"].to_numpy().astype("datetime64[D]").astype(float)
        r_days = r["dt"].to_numpy().astype("datetime64[D]").astype(float)
        idx = align(s_days, r_days, args.max_gap_days)
        for (_, row), k in zip(g.iterrows(), idx):
            d = row.to_dict()
            if k is None:
                d.update({"real_date": pd.NaT, "gap_days": np.nan})
            else:
                rr = r.iloc[k]
                d.update({"real_date": rr["dt"],
                          "gap_days": abs((rr["dt"] - row["acq_date"]).days),
                          "image_uid": rr["Image ID"],
                          "ida_visit": rr["Visit"],
                          "ida_phase": rr.get("Phase"),
                          "field_strength_ida": rr["field_strength"],
                          "manufacturer": rr["manufacturer"],
                          "scanner_model": rr["scanner_model"]})
            recs.append(d)

    out = pd.DataFrame(recs)
    matched = out["real_date"].notna()
    print(f"scans: {len(out)}  matched: {matched.sum()} ({100*matched.mean():.1f}%)")
    gd = out.loc[matched, "gap_days"]
    print("residual |synthetic - real| gap, days: "
          + "  ".join(f"p{q}={np.percentile(gd, q):.0f}" for q in (50, 75, 90, 95, 99)))
    print(f"  within 30d: {100*(gd<=30).mean():.1f}%   within 90d: {100*(gd<=90).mean():.1f}%")
    print(f"unique image_uid assigned: {out['image_uid'].nunique() if 'image_uid' in out else 0}")
    dup = out[matched].duplicated(["subject", "real_date"]).sum()
    print(f"duplicate subject+real_date assignments (should be 0): {dup}")
    print(f"scanner metadata coverage: field_strength "
          f"{100*out['field_strength_ida'].notna().mean():.1f}%  "
          f"manufacturer {100*out['manufacturer'].notna().mean():.1f}%")

    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
