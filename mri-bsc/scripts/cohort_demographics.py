#!/usr/bin/env python3
"""Compute cohort demographic/statistical summary from local project CSVs.

Outputs printed:
- subjects, scans, visits per subject
- follow-up time summary
- event/censoring counts
- diagnosis distribution (all scans and baseline)
- age/sex availability check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _mean_sd(series: pd.Series, digits: int = 2) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return "N/A"
    return f"{vals.mean():.{digits}f} ± {vals.std():.{digits}f}"


def _median_iqr(series: pd.Series, digits: int = 2) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if len(vals) == 0:
        return "N/A"
    q1 = vals.quantile(0.25)
    med = vals.quantile(0.50)
    q3 = vals.quantile(0.75)
    return f"{med:.{digits}f} ({q1:.{digits}f}-{q3:.{digits}f})"


def _fmt_int(x: float | int) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "N/A"


def _female_cell(df: pd.DataFrame) -> str:
    sex_cols = ["PTGENDER", "ptgender", "gender", "Gender", "sex", "Sex"]
    sex_col = next((c for c in sex_cols if c in df.columns), None)
    if sex_col is None:
        return "N/A"

    sx = df[sex_col].astype(str).str.strip().str.lower()
    female = sx.isin(["female", "f", "2", "0"])  # handle common codings
    n = int(female.sum())
    pct = 100.0 * n / len(df) if len(df) else 0.0
    return f"{n} ({pct:.1f}%)"


def summarize(manifest_csv: Path, survival_csv: Path) -> None:
    m = pd.read_csv(manifest_csv)
    s = pd.read_csv(survival_csv)

    if "subject" not in m.columns:
        raise ValueError("Manifest must contain 'subject'")
    if not {"subject", "event", "time_years"}.issubset(s.columns):
        raise ValueError("Survival CSV must contain 'subject', 'event', 'time_years'")

    # Keep only subjects with survival labels for grouped cohort stats.
    surv_subjects = set(s["subject"].astype(str))
    m2 = m[m["subject"].astype(str).isin(surv_subjects)].copy()

    # Define groups from survival table.
    s_all = s.copy()
    s_conv = s[s["event"] == 1].copy()
    s_stable = s[s["event"] == 0].copy()

    subj_all = set(s_all["subject"].astype(str))
    subj_conv = set(s_conv["subject"].astype(str))
    subj_stable = set(s_stable["subject"].astype(str))

    m_all = m2[m2["subject"].astype(str).isin(subj_all)].copy()
    m_conv = m2[m2["subject"].astype(str).isin(subj_conv)].copy()
    m_stable = m2[m2["subject"].astype(str).isin(subj_stable)].copy()

    visits_per_subject = m.groupby("subject").size()

    print("=== Cohort Summary ===")
    print(f"Subjects: {m['subject'].nunique()}")
    print(f"Scans: {len(m)}")
    print(
        "Visits per subject: "
        f"mean={visits_per_subject.mean():.2f}, "
        f"sd={visits_per_subject.std():.2f}, "
        f"median={visits_per_subject.median():.0f}, "
        f"min={visits_per_subject.min()}, "
        f"max={visits_per_subject.max()}"
    )

    followup = pd.to_numeric(s["time_years"], errors="coerce").dropna()
    print(
        "Follow-up (years): "
        f"mean={followup.mean():.2f}, "
        f"sd={followup.std():.2f}, "
        f"median={followup.median():.2f}, "
        f"min={followup.min():.2f}, "
        f"max={followup.max():.2f}"
    )

    event = pd.to_numeric(s["event"], errors="coerce").fillna(0).astype(int)
    n_total = len(event)
    n_event = int((event == 1).sum())
    n_censored = int((event == 0).sum())
    print(
        "Outcomes: "
        f"events={n_event} ({100.0*n_event/n_total:.2f}%), "
        f"censored={n_censored} ({100.0*n_censored/n_total:.2f}%)"
    )

    if "diagnosis" in m.columns:
        print("\nDiagnosis distribution (all scans):")
        all_dx = m["diagnosis"].value_counts(dropna=False).sort_index()
        for k, v in all_dx.items():
            print(f"  {k}: {int(v)}")

        if "visit_code" in m.columns:
            baseline = m[m["visit_code"].astype(str).str.lower() == "bl"]
            if len(baseline) > 0:
                print("\nDiagnosis distribution (baseline only):")
                bl_dx = baseline["diagnosis"].value_counts(dropna=False).sort_index()
                for k, v in bl_dx.items():
                    print(f"  {k}: {int(v)}")

    age_gender_candidates = {
        "age",
        "Age",
        "AGE",
        "gender",
        "Gender",
        "GENDER",
        "sex",
        "Sex",
        "SEX",
        "ptgender",
        "PTGENDER",
    }

    age_gender_manifest = [c for c in m.columns if c in age_gender_candidates]
    age_gender_survival = [c for c in s.columns if c in age_gender_candidates]

    print("\nAge/Sex columns found:")
    print(f"  Manifest: {age_gender_manifest if age_gender_manifest else 'none'}")
    print(f"  Survival: {age_gender_survival if age_gender_survival else 'none'}")

    if not age_gender_manifest and not age_gender_survival:
        print(
            "\nNote: Age and sex are not present in these local cohort tables. "
            "Add merged ADNI metadata (e.g., AGE/PTGENDER) to compute those demographics."
        )

    # Poster-ready cell values (All / Converters / Stable)
    visits_all = m_all.groupby("subject").size()
    visits_conv = m_conv.groupby("subject").size()
    visits_stable = m_stable.groupby("subject").size()

    nb_col = "Nboundary_bl" if "Nboundary_bl" in s.columns else None

    cells = [
        (
            "Subjects",
            _fmt_int(s_all["subject"].nunique()),
            _fmt_int(s_conv["subject"].nunique()),
            _fmt_int(s_stable["subject"].nunique()),
        ),
        ("Scans", _fmt_int(len(m_all)), _fmt_int(len(m_conv)), _fmt_int(len(m_stable))),
        (
            "Baseline Nboundary",
            _mean_sd(s_all[nb_col], digits=0) if nb_col else "N/A",
            _mean_sd(s_conv[nb_col], digits=0) if nb_col else "N/A",
            _mean_sd(s_stable[nb_col], digits=0) if nb_col else "N/A",
        ),
        (
            "Female, n (%)",
            _female_cell(m_all),
            _female_cell(m_conv),
            _female_cell(m_stable),
        ),
        (
            "Visits",
            f"{visits_all.mean():.2f}" if len(visits_all) else "N/A",
            f"{visits_conv.mean():.2f}" if len(visits_conv) else "N/A",
            f"{visits_stable.mean():.2f}" if len(visits_stable) else "N/A",
        ),
        (
            "Follow-up (years)",
            _mean_sd(s_all["time_years"], digits=2),
            _mean_sd(s_conv["time_years"], digits=2),
            _mean_sd(s_stable["time_years"], digits=2),
        ),
        (
            "Time-to-event IQR",
            _median_iqr(s_all["time_years"], digits=2),
            _median_iqr(s_conv["time_years"], digits=2),
            _median_iqr(s_stable["time_years"], digits=2),
        ),
        (
            "Outcomes",
            f"Events {int(s_all['event'].sum())} ({100.0*s_all['event'].mean():.1f}%)",
            f"Events {int(s_conv['event'].sum())} (100.0%)",
            f"Censored {int((s_stable['event']==0).sum())} (100.0%)",
        ),
    ]

    print("\n=== Poster Cell Values (Copy/Paste) ===")
    print(
        f"All (N={s_all['subject'].nunique()}) | "
        f"Converters (N={s_conv['subject'].nunique()}) | "
        f"Stable (N={s_stable['subject'].nunique()})"
    )
    print("-" * 92)
    for row, all_v, conv_v, st_v in cells:
        print(f"{row:20s} | {all_v:28s} | {conv_v:22s} | {st_v}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cohort demographic statistics"
    )
    parser.add_argument(
        "--manifest",
        default="data/manifests/adni_manifest.csv",
        help="Path to manifest CSV",
    )
    parser.add_argument(
        "--survival",
        default="data/ml/survival/time_to_conversion.csv",
        help="Path to survival CSV",
    )
    args = parser.parse_args()

    summarize(Path(args.manifest), Path(args.survival))


if __name__ == "__main__":
    main()
