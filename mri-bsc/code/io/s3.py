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
    # Extract file extension from S3 key to preserve it in temp file
    _, ext = key.rsplit(".", 1) if "." in key else ("", "")
    suffix = f".{ext}" if ext else ""
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
