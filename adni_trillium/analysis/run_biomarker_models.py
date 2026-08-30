"""Model portfolio rerun with amyloid status added to every covariate set.

Table 3 of the manuscript comes from run_spec_models.run(), which reserves a 20
percent hold-out and cross-validates inside the remaining pool. Any biomarker
number printed next to those has to come from the same protocol or the two are
not comparable, so this script reuses that function rather than reimplementing
the split.

It patches feature_sets to emit each covariate-containing set twice, once as
published and once with the two amyloid indicator columns appended, and writes
to its own directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import run_spec_models as rsm
from run_biomarker_analysis import add_bio

_original = rsm.feature_sets


def feature_sets_with_bio(df):
    base = _original(df)
    return {**base, **add_bio(base)}


def main():
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=str(root / "results/biomarker_v1/cohort_with_biomarkers.csv"))
    ap.add_argument("--out_dir", default=str(root / "results/biomarker_v1/models"))
    ap.add_argument("--models", default="xgb_aft,rsf,cox_l2,cox_lasso,"
                                        "aft_weibull,aft_lognormal,aft_loglogistic")
    args = ap.parse_args()

    rsm.feature_sets = feature_sets_with_bio
    rsm.run(SimpleNamespace(cohort=args.cohort, out_dir=args.out_dir,
                            models=args.models))


if __name__ == "__main__":
    main()
