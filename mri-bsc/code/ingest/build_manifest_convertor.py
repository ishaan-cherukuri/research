from pathlib import Path
import pandas as pd

def separate_converters(
    manifest_root: str,
    mri_progressive_out: str,
    mri_stable_out: str
):
    """
    Split subjects in the ADNI manifest into:
      - Progressive MCI (MCI → AD)
      - Stable MCI (MCI → stays MCI/CN)

    Writes two CSVs.

    Assumes columns:
      subject, visit_code, diagnosis, acq_date
    """

    manifest_path = Path(manifest_root)
    df = pd.read_csv(manifest_path)

    # Optional: drop session if present
    if "session" in df.columns:
        df = df.drop(columns=["session"])

    df = df.sort_values(["subject", "acq_date"])

    progressive = []   # MCI converters
    stable = []        # MCI non-converters
    unmatched = []     # Edge-case subjects missing info

    for subject in df["subject"].unique():
        temp = df[df["subject"] == subject]

        # Must have baseline MCI
        baseline_mci = temp[
            (temp["visit_code"] == "bl") &
            (temp["diagnosis"] == "MCI")
        ]

        if baseline_mci.empty:
            unmatched.append((subject, "No baseline MCI"))
            continue

        # Last known diagnosis
        final_dx = temp.iloc[-1]["diagnosis"]

        if final_dx == "AD":
            progressive.append(subject)
        else:
            stable.append(subject)

    # Build output frames
    progressive_df = df[df["subject"].isin(progressive)]
    stable_df = df[df["subject"].isin(stable)]

    # Sort consistently
    progressive_df = progressive_df.sort_values(["subject", "acq_date"])
    stable_df = stable_df.sort_values(["subject", "acq_date"])

    # Prepare output paths
    out_p = Path(mri_progressive_out)
    out_s = Path(mri_stable_out)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_s.parent.mkdir(parents=True, exist_ok=True)

    progressive_df.to_csv(out_p, index=False)
    stable_df.to_csv(out_s, index=False)

    print("[OK] Wrote conversion manifests")
    print(f"  Progressive MCI subjects: {len(progressive)}")
    print(f"  Stable MCI subjects: {len(stable)}")

    if unmatched:
        print(f"[WARNING] {len(unmatched)} subjects skipped:")
        for subj, reason in unmatched[:15]:
            print(f"   {subj}: {reason}")

    return progressive_df, stable_df
