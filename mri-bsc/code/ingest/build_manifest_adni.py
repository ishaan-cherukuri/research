import argparse
import pandas as pd
import boto3
from datetime import datetime


s3 = boto3.client("s3")


def normalize_date(s):
    try:
        return datetime.strptime(s, "%m/%d/%Y").strftime("%Y-%m-%d")
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
        "Image Data ID",
        "Subject",
        "Group",
        "Sex",
        "Age",
        "Visit",
        "Acq Date",
        "Description",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    assert s3_raw_root.startswith("s3://")

    _, _, bucket_and_prefix = s3_raw_root.partition("s3://")
    bucket, _, root_prefix = bucket_and_prefix.partition("/")

    records = []

    for _, row in df.iterrows():
        image_id = f"I{int(row['Image Data ID'][1::])}"
        subject = row["Subject"]
        visit_code = row["Visit"]

        acq_date = normalize_date(str(row["Acq Date"]))
        diagnosis = row["Group"]

        # Search for the image_id directory anywhere under subject
        search_prefix = f"{root_prefix}/{subject}/"
        nii_files = [
            k for k in list_s3_objects(bucket, search_prefix)
            if f"/{image_id}/" in k and k.lower().endswith((".nii", ".nii.gz"))
        ]

        if not nii_files:
            continue  # skip unmatched rows

        # ADNI guarantees 1 NIfTI per Image Data ID
        nii_key = nii_files[0]

        records.append({
            "subject": subject,
            "session": visit_code,           # kept for compatibility
            "visit_code": visit_code,
            "acq_date": acq_date,
            "path": f"s3://{bucket}/{nii_key}",
            "modality": "T1w",
            "desc": row["Description"],
            "image_id": image_id,
            "diagnosis": diagnosis,
            "sex": row["Sex"],
            "age": row["Age"],
        })

    out_df = pd.DataFrame(records)
    out_df = out_df.sort_values(["subject", "acq_date"])

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
