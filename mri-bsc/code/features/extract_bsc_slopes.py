"""
Compute longitudinal BSC slopes from multi-timepoint data.

CRITICAL: Visit codes are unreliable (jumbled). Use ONLY acquisition dates
          for temporal ordering and time calculations.

Usage:
    python -m code.features.extract_bsc_slopes \
        --features data/index/bsc_simple_features.csv \
        --manifest data/manifests/adni_manifest.csv \
        --out_csv data/index/bsc_longitudinal_slopes.csv \
        --min_visits 4

Output CSV columns per subject:
    - subject, n_visits, time_span_years
    - For each BSC feature: {feature}_baseline, {feature}_final, {feature}_slope, {feature}_r2
    - Slope = (final - baseline) / time_years
    - R² measures linearity of decline (1.0 = perfect line, 0.0 = random)
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import linregress
from typing import Tuple


def compute_slope(
    times: np.ndarray, values: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Compute linear regression slope from longitudinal data.

    Args:
        times: Time points in years from baseline (e.g., [0, 1.01, 2.03, 3.02])
        values: Feature values at each time point

    Returns:
        (baseline_value, final_value, slope_per_year, r_squared)
    """
    if len(times) < 2:
        return np.nan, np.nan, np.nan, np.nan

    # Filter out NaN values
    valid = ~np.isnan(values)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan

    t = times[valid]
    v = values[valid]

    # Linear regression: value = slope * time + intercept
    slope, intercept, r_value, p_value, std_err = linregress(t, v)
    r2 = r_value**2

    return v[0], v[-1], slope, r2


def load_features_with_dates(features_path: str, manifest_path: str) -> pd.DataFrame:
    """
    Load BSC features and merge with manifest to get acquisition dates.

    CRITICAL: Sorts by acq_date ONLY, ignores visit_code (unreliable).

    Returns:
        DataFrame with: subject, acq_date, image_id, feature1, feature2, ...
    """
    print(f"Loading features: {features_path}")
    features = pd.read_csv(features_path)
    print(f"  Loaded {len(features)} scans")

    # Handle both S3 and local manifest paths
    if manifest_path.startswith("s3://"):
        from code.io.s3 import download_to_temp

        print(f"Downloading manifest from S3: {manifest_path}")
        with download_to_temp(manifest_path) as tmp:
            manifest = pd.read_csv(tmp)
    else:
        print(f"Loading manifest: {manifest_path}")
        manifest = pd.read_csv(manifest_path)

    print(f"  Loaded {len(manifest)} scans from manifest")

    # Extract subject and acq_date from manifest
    # Manifest has: subject, visit_code, acq_date, path, diagnosis
    manifest_slim = manifest[["subject", "acq_date"]].copy()

    # Features DataFrame has 'image_id' which is derived from subject
    # image_id format: <subject>_<visit_code>_<acq_date>
    # ADNI subject IDs have 3 parts: 002_S_0729 (not 2!)
    # Extract subject from image_id
    if "image_id" in features.columns:
        features["subject"] = features["image_id"].str.split("_").str[:3].str.join("_")
    elif "subject" not in features.columns:
        raise ValueError("Features CSV must have 'image_id' or 'subject' column")

    # Merge to get dates (many-to-many on subject, then match by image_id date component)
    # Simpler approach: extract acq_date from image_id directly
    if "image_id" in features.columns:
        # image_id = <subject>_<visit>_<date>
        # Extract date component (last part after final '_')
        features["acq_date"] = features["image_id"].str.split("_").str[-1]
    else:
        # Need to merge with manifest
        # Group manifest by subject to get all dates
        features = features.merge(manifest_slim, on="subject", how="left")

    # Convert to datetime
    features["acq_date"] = pd.to_datetime(features["acq_date"], errors="coerce")

    # Drop rows with missing dates
    missing_dates = features["acq_date"].isna().sum()
    if missing_dates > 0:
        print(f"  ⚠️  Dropping {missing_dates} scans with missing acquisition dates")
        features = features.dropna(subset=["acq_date"])

    # Sort by subject and acq_date (NOT visit_code!)
    features = features.sort_values(["subject", "acq_date"]).reset_index(drop=True)

    print(
        f"  ✅ Final dataset: {len(features)} scans from {features['subject'].nunique()} subjects"
    )

    return features


def compute_slopes_per_subject(
    df: pd.DataFrame, feature_cols: list, min_visits: int = 4
) -> pd.DataFrame:
    """
    Compute slopes for each subject across all BSC features.

    Args:
        df: DataFrame with subject, acq_date, and BSC feature columns
        feature_cols: List of BSC feature column names to compute slopes for
        min_visits: Minimum number of timepoints required (default: 4)

    Returns:
        DataFrame with one row per subject containing all slopes
    """
    results = []

    subjects = df["subject"].unique()
    print(f"\nComputing slopes for {len(subjects)} subjects...")
    print(f"  Minimum visits required: {min_visits}")
    print(f"  Features to process: {len(feature_cols)}")

    skipped = 0

    for subj in subjects:
        subj_df = df[df["subject"] == subj].sort_values("acq_date")

        n_visits = len(subj_df)
        if n_visits < min_visits:
            skipped += 1
            continue

        # Compute time in years from baseline (first scan)
        baseline_date = subj_df["acq_date"].iloc[0]
        final_date = subj_df["acq_date"].iloc[-1]
        time_span = (final_date - baseline_date).days / 365.25

        times = (subj_df["acq_date"] - baseline_date).dt.days / 365.25
        times = times.values

        # Initialize result row
        row = {
            "subject": subj,
            "n_visits": n_visits,
            "time_span_years": time_span,
            "baseline_date": baseline_date.strftime("%Y-%m-%d"),
            "final_date": final_date.strftime("%Y-%m-%d"),
        }

        # Compute slope for each feature
        for feat in feature_cols:
            values = subj_df[feat].values
            baseline, final, slope, r2 = compute_slope(times, values)

            row[f"{feat}_baseline"] = baseline
            row[f"{feat}_final"] = final
            row[f"{feat}_slope"] = slope
            row[f"{feat}_r2"] = r2

        results.append(row)

    print(f"  ✅ Computed slopes for {len(results)} subjects")
    print(f"  ⚠️  Skipped {skipped} subjects with <{min_visits} visits")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--features", required=True, help="Path to bsc_simple_features.csv"
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to manifest CSV (local or s3://)"
    )
    parser.add_argument(
        "--out_csv", required=True, help="Output CSV path for longitudinal slopes"
    )
    parser.add_argument(
        "--min_visits",
        type=int,
        default=4,
        help="Minimum number of visits required (default: 4)",
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=None,
        help="Limit to first N features (for testing)",
    )

    args = parser.parse_args()

    # Load features with dates
    df = load_features_with_dates(args.features, args.manifest)

    # Identify BSC feature columns (exclude metadata and non-numeric)
    metadata_cols = [
        "image_id",
        "subject",
        "acq_date",
        "visit_code",
        "diagnosis",
        "n_boundary",
        "n_total",
        "N_boundary",
        "N_total",
    ]

    # Filter for numeric columns only
    feature_cols = []
    for c in df.columns:
        if c not in metadata_cols and pd.api.types.is_numeric_dtype(df[c]):
            feature_cols.append(c)

    # Optionally limit features for testing
    if args.max_features:
        feature_cols = feature_cols[: args.max_features]
        print(f"\n⚠️  Testing mode: using only first {args.max_features} features")

    print(f"\nBSC features to process: {len(feature_cols)}")
    print(f"  Examples: {feature_cols[:5]}")

    # Compute slopes
    slopes_df = compute_slopes_per_subject(df, feature_cols, args.min_visits)

    # Save results
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slopes_df.to_csv(out_path, index=False)

    print(f"\n✅ Saved longitudinal slopes to: {args.out_csv}")
    print(f"   {len(slopes_df)} subjects, {slopes_df.columns.size} columns")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS (slopes per year)")
    print("=" * 70)

    slope_cols = [c for c in slopes_df.columns if c.endswith("_slope")]

    for col in slope_cols[:10]:  # Show first 10 features
        valid = slopes_df[col].notna()
        if valid.sum() > 0:
            mean_slope = slopes_df.loc[valid, col].mean()
            std_slope = slopes_df.loc[valid, col].std()
            median_slope = slopes_df.loc[valid, col].median()

            # Get corresponding R² column
            r2_col = col.replace("_slope", "_r2")
            if r2_col in slopes_df.columns:
                mean_r2 = slopes_df.loc[valid, r2_col].mean()
                print(
                    f"{col:40s}: {mean_slope:8.5f} ± {std_slope:8.5f} (median: {median_slope:8.5f}, R²: {mean_r2:.3f}, n={valid.sum()})"
                )
            else:
                print(
                    f"{col:40s}: {mean_slope:8.5f} ± {std_slope:8.5f} (median: {median_slope:8.5f}, n={valid.sum()})"
                )

    if len(slope_cols) > 10:
        print(f"\n... and {len(slope_cols) - 10} more features")

    # Print time span statistics
    print("\n" + "=" * 70)
    print("TIME SPAN STATISTICS")
    print("=" * 70)
    print(f"Mean time span: {slopes_df['time_span_years'].mean():.2f} years")
    print(f"Median time span: {slopes_df['time_span_years'].median():.2f} years")
    print(f"Min time span: {slopes_df['time_span_years'].min():.2f} years")
    print(f"Max time span: {slopes_df['time_span_years'].max():.2f} years")
    print(f"\nVisits per subject:")
    print(slopes_df["n_visits"].value_counts().sort_index())

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
