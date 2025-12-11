import os
import pandas as pd
from pathlib import Path
import re

def parse_date_from_dir(dirname):
    # Example: "2006-05-19_16_17_47.0" → "2006-05-19"
    m = re.match(r"(\d{4}-\d{2}-\d{2})", dirname)
    return m.group(1) if m else None

def build_manifest_adni(root, out_csv):
    
    rows = []
    root = Path(root)

    for subject_dir in sorted(root.iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name  # e.g., "002_S_0413"

        # Loop sequence types (usually 1 folder)
        for seq_dir in subject_dir.iterdir():
            if not seq_dir.is_dir():
                continue

            # Loop visits
            for visit_dir in seq_dir.iterdir():
                if not visit_dir.is_dir():
                    continue

                date_str = parse_date_from_dir(visit_dir.name)

                nii_path = visit_dir / "img.nii.gz"
                if not nii_path.exists():
                    continue

                # ADNI often encodes visit time via date differences
                # but you may later map dates → m12/m24/m36
                visit = visit_dir.name   

                rows.append({
                    "subject": subject,
                    "visit": visit,
                    "acq_date": date_str,
                    "path": str(nii_path),
                    "diagnosis": None,    # Will fill using your CSV
                    "sex": None, 
                    "age": None,
                    "image_id": None,
                    "desc": seq_dir.name
                })

    df = pd.DataFrame(rows)
    df = df.sort_values(["subject", "acq_date"])
    df.to_csv(out_csv, index=False)

    print(f"[build_manifest_adni] Wrote manifest with {len(df)} rows to {out_csv}")
    return df
