"""Stage BIDS T1w NIfTIs into the layout CIVET 2.1.0 expects.

CIVET reads <sourcedir>/<prefix>_<id>_t1.mnc and splits that filename on
underscores, so the image_id convention used everywhere else in this project
(subject_visitcode_acqdate, e.g. "002_S_0295_bl_2006-04-18") cannot be handed
to it directly. Each scan is given a short synthetic CIVET id instead, and
civet_id_map.csv records the correspondence so every later stage can join back
to image_id and therefore to the existing feature CSVs.

Usage:
    python3 prepare_civet_inputs.py --manifest from_cluster/manifest_v3.csv \
        --out_root $SCRATCH/bsc_sigmoid --prefix adni
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def make_image_id(row: pd.Series) -> str:
    """Same convention as preprocess_local.py, so ids stay joinable."""
    return f"{row['subject']}_{row['visit_code']}_{row['acq_date']}"


def convert_one(src: Path, dst: Path) -> None:
    """NIfTI -> MINC. nii2mnc refuses to overwrite, so write to a temporary name
    and move it into place; a half-written .mnc left by a killed job would
    otherwise be indistinguishable from a finished one."""
    tmp = dst.with_suffix(".mnc.tmp")
    tmp.unlink(missing_ok=True)
    subprocess.run(["nii2mnc", str(src), str(tmp)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    tmp.replace(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True,
                    help="CIVET sourcedir; also receives civet_id_map.csv")
    ap.add_argument("--prefix", default="adni")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    required = {"subject", "visit_code", "acq_date", "path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    df["image_id"] = df.apply(make_image_id, axis=1)
    # The id is assigned from position in the full manifest, before any
    # skip/limit slicing, so parallel workers over disjoint chunks agree.
    df["civet_id"] = [f"scan{i:06d}" for i in range(len(df))]

    out_root = Path(args.out_root)
    src_dir = out_root / "civet_input"
    src_dir.mkdir(parents=True, exist_ok=True)

    map_csv = out_root / "civet_id_map.csv"
    if not map_csv.exists():
        df[["image_id", "civet_id", "subject", "visit_code", "acq_date",
            "path"]].to_csv(map_csv, index=False)
        print(f"[OK] wrote {map_csv} ({len(df)} scans)")

    stop = None if args.limit is None else args.skip + args.limit
    chunk = df.iloc[args.skip:stop]

    n_done = n_fail = 0
    for _, row in tqdm(chunk.iterrows(), total=len(chunk), desc="nii2mnc",
                       unit="scan"):
        dst = src_dir / f"{args.prefix}_{row['civet_id']}_t1.mnc"
        if dst.exists():
            n_done += 1
            continue
        try:
            convert_one(Path(row["path"]), dst)
            n_done += 1
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            stderr = getattr(e, "stderr", b"") or b""
            print(f"[FAIL] {row['image_id']}: {stderr.decode()[:200] or e}")
            n_fail += 1

    ids_file = out_root / "civet_ids.txt"
    have = sorted(p.name.split("_")[1] for p in src_dir.glob(f"{args.prefix}_*_t1.mnc"))
    ids_file.write_text("\n".join(have) + "\n")

    print(f"[DONE] {n_done} converted/present, {n_fail} failed; "
          f"{len(have)} ids in {ids_file}")


if __name__ == "__main__":
    main()
