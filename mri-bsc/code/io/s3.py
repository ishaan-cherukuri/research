import boto3
import tempfile
from pathlib import Path

s3 = boto3.client("s3")


def parse_s3_uri(uri: str):
    assert uri.startswith("s3://")
    _, _, bucket_key = uri.partition("s3://")
    bucket, _, key = bucket_key.partition("/")
    return bucket, key


def download_to_temp(s3_uri: str) -> Path:
    bucket, key = parse_s3_uri(s3_uri)
    # Extract file extension, handling .nii.gz and other double extensions
    if key.endswith(".nii.gz"):
        suffix = ".nii.gz"
    elif "." in key:
        _, ext = key.rsplit(".", 1)
        suffix = f".{ext}"
    else:
        suffix = ""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    s3.download_file(bucket, key, tmp.name)
    return Path(tmp.name)


def upload_file(local_path: Path, s3_uri: str):
    bucket, key = parse_s3_uri(s3_uri)
    s3.upload_file(str(local_path), bucket, key)


def ensure_s3_prefix(s3_uri: str):
    bucket, key = parse_s3_uri(s3_uri)
    if not key.endswith("/"):
        key += "/"
    s3.put_object(Bucket=bucket, Key=key)


def clear_s3_prefix(s3_uri: str):
    """Delete all objects under a given S3 prefix."""
    bucket, key = parse_s3_uri(s3_uri)
    if not key.endswith("/"):
        key += "/"

    # List all objects with this prefix
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=key)

    # Delete all objects
    for page in pages:
        if "Contents" in page:
            objects_to_delete = [{"Key": obj["Key"]} for obj in page["Contents"]]
            if objects_to_delete:
                s3.delete_objects(Bucket=bucket, Delete={"Objects": objects_to_delete})

    print(f"Cleared S3 prefix: {s3_uri}")
