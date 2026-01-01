import boto3
import csv
import re

SRC_BUCKET = "naccmri-quickaccess-sub"
SRC_PREFIX = "scan/MRI/"
PROFILE = "nacc"

session = boto3.Session(profile_name=PROFILE)
s3 = session.client("s3")

def is_t1(filename):
    name = filename.upper()
    if any(x in name for x in ["FLAIR", "T2", "STAR"]):
        return False
    if any(x in name for x in ["MPRAGE", "IR-FSPGR", "3D_T1"]):
        return True
    return False

paginator = s3.get_paginator("list_objects_v2")
pages = paginator.paginate(Bucket=SRC_BUCKET, Prefix=SRC_PREFIX)

with open("nacc_t1_summary.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["naccid", "filename", "is_t1"])
    for page in pages:
        for obj in page.get("Contents", []):
            fname = obj["Key"].split("/")[-1]
            if not fname.endswith(".zip"):
                continue
            match = re.search(r"NACC\d+", fname)
            naccid = match.group(0) if match else ""
            writer.writerow([naccid, fname, is_t1(fname)])

print("Wrote nacc_t1_summary.csv")
