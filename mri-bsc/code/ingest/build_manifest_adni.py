import argparse
import pandas as pd
import boto3
from datetime import datetime

s3 = boto3.client("s3")


def normalize_date(s):
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        return None


def list_s3_objects(bucket, prefix):
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def build_manifest_adni(csv_path, s3_raw_root, out_csv):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    required = {
        "subject_id",
        "image_visit",
        "image_id",
        "image_date",
        "diagnosis_group",
        "series_description",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    assert s3_raw_root.startswith("s3://")

    _, _, bucket_and_prefix = s3_raw_root.partition("s3://")
    bucket, _, root_prefix = bucket_and_prefix.partition("/")

    records = []

    for _, row in df.iterrows():
        subject = row["subject_id"]
        visit_code = row["image_visit"]
        image_id = f"I{int(row['image_id'])}"
        series_description = row["series_description"]
        acq_date = normalize_date(row["image_date"])
        diagnosis = row["diagnosis_group"]

        # Search for the image_id directory anywhere under subject
        # (don't rely on series_description matching exactly, as S3 may have variants)
        search_prefix = f"{root_prefix}/{subject}/"

        nii_files = [
            k
            for k in list_s3_objects(bucket, search_prefix)
            if f"/{image_id}/" in k and k.endswith("/image.nii.gz")
        ]

        if not nii_files:
            print(
                f"[DEBUG] No match for {subject} / {image_id} / {series_description} in prefix {search_prefix}"
            )
            continue  # skip unmatched rows

        # ADNI guarantees 1 NIfTI per Image Data ID
        nii_key = nii_files[0]

        records.append(
            {
                "subject_id": subject,
                "visit_code": visit_code,
                "acq_date": acq_date,
                "path": f"s3://{bucket}/{nii_key}",
                "modality": "T1w",
                "series_description": series_description,
                "image_id": image_id,
                "diagnosis": diagnosis,
            }
        )

    out_df = pd.DataFrame(records)
    if out_df.empty:
        print("[WARNING] No matching images found. Manifest is empty.")
        out_df.to_csv(out_csv, index=False)
        return
    print(out_df.columns.tolist())
    out_df = out_df.sort_values(["subject_id", "acq_date", "series_description"])

    out_df.to_csv(out_csv, index=False)
    print(f"[OK] Wrote ADNI manifest → {out_csv}")
    print(f"     Rows: {len(out_df)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", required=True)
    ap.add_argument("--s3_raw_root", required=True)
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    build_manifest_adni(
        csv_path=args.csv_path,
        s3_raw_root=args.s3_raw_root,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
