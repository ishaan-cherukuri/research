"""
code.index.build_longitudinal_index

Build a CNN-ready subject index that links each visit to precomputed derivative images.

Expected derivatives per visit directory (Atropos BSC pipeline output):
  - t1w_preproc.nii.gz
  - gm_prob.nii.gz
  - wm_prob.nii.gz
  - brain_mask.nii.gz
Optional:
  - bsc_map.nii.gz (if/when you generate it)
  - bsc_metrics.json
  - subject_metrics.csv

Output CSV has one row per subject:
  subject,label,n_visits,visits_json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from code.utils.io_any import read_csv_any, write_csv_any, is_s3, exists_any, list_files


def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _months_between(a, b):
    if pd.isna(a) or pd.isna(b):
        return None
    return float((b - a).days / 30.44)


def _join(root: str, child: str) -> str:
    return f"{root.rstrip('/')}/{child.lstrip('/')}"


def _dir_of_path(p: str) -> str:
    # Works for local and s3-like strings
    p = str(p)
    if p.endswith("/"):
        return p.rstrip("/")
    return p.rsplit("/", 1)[0] if "/" in p else ""


def _stem_from_nii_path(nii_path: str) -> str:
    name = Path(str(nii_path)).name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return Path(name).stem


def _find_visit_dir_any(derivatives_root: str, image_id: str, nii_path: str) -> str | None:
    """
    Locate visit directory (local dir or S3 prefix) for a manifest row.

    Priority:
      1) derivatives_root/<image_id>/
      2) derivatives_root/<nii_stem>/
      3) scan derivatives_root for any path containing image_id (fallback; can be expensive on huge buckets)
    """
    candidates: list[str] = []

    if image_id:
        candidates.append(_join(derivatives_root, str(image_id)))

    if nii_path:
        stem = _stem_from_nii_path(nii_path)
        if stem:
            candidates.append(_join(derivatives_root, stem))

    # Check direct candidates first
    for c in candidates:
        if c and exists_any(c):
            return c

    # Fallback scan: find any object containing image_id and take its parent as the visit_dir
    if image_id:
        try:
            hits = list_files(derivatives_root)
            # filter fast in python
            hits = [h for h in hits if str(image_id) in h]
            # take the deepest "folder" that looks like derivatives_root/<something>/file
            for h in hits[:50]:  # avoid runaway on huge listings
                parent = _dir_of_path(h)
                if parent and exists_any(parent):
                    return parent
        except Exception:
            pass

    return None


def _pick_file_any(visit_dir: str, names: list[str]) -> str | None:
    for n in names:
        p = _join(visit_dir, n)
        if exists_any(p):
            return p
    return None


def build_index(
    manifest_csv: str,
    labels_csv: str,
    derivatives_root: str,
    out_csv: str,
    required_modalities=("t1", "gm", "wm", "mask"),
    out_unmatched_csv: str | None = None,
) -> pd.DataFrame:

    df = read_csv_any(manifest_csv)
    labels = read_csv_any(labels_csv)

    required_cols = {"subject", "visit_code", "acq_date", "path", "image_id"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    if "subject" not in labels.columns or "label" not in labels.columns:
        raise ValueError("Labels CSV must have columns: subject,label")

    df = df.copy()
    df["acq_date_dt"] = _to_dt(df["acq_date"])
    df = df.sort_values(["subject", "acq_date_dt"])

    label_map = dict(zip(labels["subject"].astype(str), labels["label"].astype(int)))

    index_rows = []
    unmatched = []

    for subject, g in df.groupby("subject", sort=False):
        subject = str(subject)

        if subject not in label_map:
            unmatched.append({"subject": subject, "reason": "missing_label"})
            continue

        g = g.sort_values("acq_date_dt")
        bl = g[g["visit_code"] == "bl"]
        if bl.empty:
            unmatched.append({"subject": subject, "reason": "missing_baseline"})
            continue

        bl_date = bl.iloc[0]["acq_date_dt"]
        visits = []

        for _, row in g.iterrows():
            image_id = "" if pd.isna(row["image_id"]) else str(row["image_id"])
            nii_path = "" if pd.isna(row["path"]) else str(row["path"])

            visit_dir = _find_visit_dir_any(derivatives_root, image_id=image_id, nii_path=nii_path)
            if visit_dir is None:
                unmatched.append(
                    {
                        "subject": subject,
                        "visit_code": str(row["visit_code"]),
                        "image_id": image_id,
                        "reason": "missing_derivatives_dir",
                    }
                )
                continue

            paths = {
                "t1": _pick_file_any(visit_dir, ["t1w_preproc.nii.gz", "t1w_preproc.nii"]),
                "gm": _pick_file_any(visit_dir, ["gm_prob.nii.gz", "gm_prob.nii"]),
                "wm": _pick_file_any(visit_dir, ["wm_prob.nii.gz", "wm_prob.nii"]),
                "mask": _pick_file_any(visit_dir, ["brain_mask.nii.gz", "brain_mask.nii"]),
                "bsc_map": _pick_file_any(visit_dir, ["bsc_map.nii.gz", "bsc_map.nii"]),
                "bsc_metrics": _pick_file_any(visit_dir, ["bsc_metrics.json"]),
                "subject_metrics": _pick_file_any(visit_dir, ["subject_metrics.csv"]),
            }

            if any(paths[m] is None for m in required_modalities):
                missing_mods = [m for m in required_modalities if paths[m] is None]
                unmatched.append(
                    {
                        "subject": subject,
                        "visit_code": str(row["visit_code"]),
                        "image_id": image_id,
                        "reason": f"missing_required_files:{','.join(missing_mods)}",
                        "visit_dir": visit_dir,
                    }
                )
                continue

            months = _months_between(bl_date, row["acq_date_dt"])
            visits.append(
                {
                    "visit_code": str(row["visit_code"]),
                    "acq_date": str(row["acq_date"]),
                    "months_since_bl": round(months, 3) if months is not None else None,
                    "image_id": image_id,
                    "visit_dir": visit_dir,
                    "paths": paths,
                }
            )

        if not visits:
            unmatched.append({"subject": subject, "reason": "no_complete_visits"})
            continue

        index_rows.append(
            {
                "subject": subject,
                "label": int(label_map[subject]),
                "n_visits": int(len(visits)),
                "visits_json": json.dumps(visits),
            }
        )

    out_df = (
        pd.DataFrame(index_rows).sort_values(["subject"])
        if index_rows
        else pd.DataFrame(columns=["subject", "label", "n_visits", "visits_json"])
    )

    write_csv_any(out_df, out_csv)

    if out_unmatched_csv:
        write_csv_any(pd.DataFrame(unmatched), out_unmatched_csv)

    print("[OK] Wrote index CSV:", out_csv)
    print("  subjects indexed:", len(out_df))
    if len(out_df):
        print("  mean visits/subject:", round(out_df["n_visits"].mean(), 3))

    if unmatched:
        msg = f"[WARNING] unmatched/skipped items: {len(unmatched)}"
        if out_unmatched_csv:
            msg += f" (see {out_unmatched_csv})"
        print(msg)

    return out_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--derivatives_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--require", default="t1,gm,wm,mask")
    ap.add_argument("--out_unmatched_csv", default="")
    args = ap.parse_args()

    required = tuple(x.strip() for x in args.require.split(",") if x.strip())

    build_index(
        manifest_csv=args.manifest,
        labels_csv=args.labels,
        derivatives_root=args.derivatives_root,
        out_csv=args.out_csv,
        required_modalities=required,
        out_unmatched_csv=(args.out_unmatched_csv or None),
    )


if __name__ == "__main__":
    main()
