
import argparse
import pandas as pd
from pathlib import Path
import subprocess
import sys

def find_missing_scans(manifest_path: str, features_path: str) -> pd.DataFrame:
    print("Loading files...")
    manifest = pd.read_csv(manifest_path)
    features = pd.read_csv(features_path)

    features["subject"] = features["image_id"].str.split("_").str[:3].str.join("_")

    print(f"Manifest: {len(manifest)} scans, {manifest['subject'].nunique()} subjects")
    print(f"Features: {len(features)} scans, {features['subject'].nunique()} subjects")

    manifest["image_id"] = (
        manifest["subject"] + "_" + manifest["visit_code"] + "_" + manifest["acq_date"]
    )

    missing = manifest[~manifest["image_id"].isin(features["image_id"])].copy()

    print(
        f"\n️  Missing {len(missing)} scans from {missing['subject'].nunique()} subjects"
    )
    print(f"\nSubjects with missing scans:")
    subject_counts = missing.groupby("subject").size().sort_values(ascending=False)
    print(subject_counts.head(20))

    return missing

def create_missing_manifest(missing_df: pd.DataFrame, out_path: str):
    output_df = missing_df.drop(columns=["image_id"])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path, index=False)

    print(f"\n Created manifest for missing scans: {out_path}")
    print(f"   {len(output_df)} scans from {output_df['subject'].nunique()} subjects")

    return out_path

def run_batch_pipeline(manifest_path: str, out_root: str, temp_root: str):
    print("\n" + "=" * 70)
    print("RUNNING BSC BATCH PIPELINE FOR MISSING SCANS")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "code.pipeline.run_batch",
        "--manifest",
        str(manifest_path),
        "--engine",
        "atropos",
        "--out_root",
        out_root,
        "--temp_root",
        temp_root,
        "--skip",
        "0",
    ]

    print(f"\nCommand: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=True)

    if result.returncode == 0:
        print("\n BSC computation completed successfully!")
    else:
        print(f"\n BSC computation failed with exit code {result.returncode}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--manifest", required=True, help="Path to full ADNI manifest CSV"
    )
    parser.add_argument(
        "--features", required=True, help="Path to existing bsc_simple_features.csv"
    )
    parser.add_argument(
        "--out_root", required=True, help="S3 output root for BSC derivatives"
    )
    parser.add_argument(
        "--temp_root", required=True, help="Local temp directory for processing"
    )
    parser.add_argument(
        "--missing_manifest",
        default="data/manifests/missing_scans.csv",
        help="Where to save manifest of missing scans",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only identify missing scans, do not run pipeline",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("STEP 1: IDENTIFYING MISSING SCANS")
    print("=" * 70)

    missing_df = find_missing_scans(args.manifest, args.features)

    if len(missing_df) == 0:
        print("\n No missing scans! All subjects have BSC outputs.")
        return

    print("\n" + "=" * 70)
    print("STEP 2: CREATING MANIFEST FOR MISSING SCANS")
    print("=" * 70)

    missing_manifest_path = create_missing_manifest(missing_df, args.missing_manifest)

    if args.dry_run:
        print("\n Dry run mode - stopping before pipeline execution")
        print(f"   To reprocess, run:")
        print(f"   python -m code.pipeline.run_batch \\")
        print(f"       --manifest {missing_manifest_path} \\")
        print(f"       --engine atropos \\")
        print(f"       --out_root {args.out_root} \\")
        print(f"       --temp_root {args.temp_root}")
        return

    print("\n" + "=" * 70)
    print("STEP 3: RUNNING BSC PIPELINE")
    print("=" * 70)

    run_batch_pipeline(missing_manifest_path, args.out_root, args.temp_root)

    print("\n" + "=" * 70)
    print(" REPROCESSING COMPLETE!")
    print("=" * 70)
    print(
        f"\nProcessed {len(missing_df)} scans from {missing_df['subject'].nunique()} subjects"
    )
    print(f"Outputs uploaded to: {args.out_root}")
    print(f"\nNext step: Re-run feature extraction to update bsc_simple_features.csv")

if __name__ == "__main__":
    main()
