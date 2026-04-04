
import os
import tempfile
from pathlib import Path
import pandas as pd
import fsspec
import shutil

def is_s3(path: str) -> bool:
    return isinstance(path, str) and path.startswith("s3://")

def get_fs(path: str):
    if is_s3(path):
        return fsspec.filesystem("s3")
    return fsspec.filesystem("file")

def read_csv_any(path: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs)

def write_csv_any(df: pd.DataFrame, path: str, **kwargs):
    df.to_csv(path, index=False, **kwargs)

def download_if_s3(uri: str, work_dir: str | Path) -> str:
    if not is_s3(uri):
        return uri

    fs = get_fs(uri)
    local_path = Path(work_dir) / Path(uri).name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fs.get(uri, str(local_path))
    return str(local_path)

def upload_file(local_path: str, dest_uri: str):
    fs = get_fs(dest_uri)
    fs.put(local_path, dest_uri)

def upload_dir(local_dir: str, dest_prefix: str):
    fs = get_fs(dest_prefix)
    local_dir = Path(local_dir)
    for path in local_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(local_dir)
            target = f"{dest_prefix.rstrip('/')}/{rel.as_posix()}"
            fs.put(str(path), target)

def exists_any(path: str) -> bool:
    fs = get_fs(path)
    return fs.exists(path)

def list_files(prefix: str, pattern: str = None) -> list[str]:
    fs = get_fs(prefix)
    files = fs.find(prefix)
    if pattern:
        files = [f for f in files if pattern in f]
    return files

def make_work_dir(base: str = ".cache/mri-bsc") -> Path:
    p = Path(base)
    p.mkdir(parents=True, exist_ok=True)
    return p

def clean_work_dir(work_dir: str | Path):
    if Path(work_dir).exists():
        shutil.rmtree(work_dir, ignore_errors=True)
