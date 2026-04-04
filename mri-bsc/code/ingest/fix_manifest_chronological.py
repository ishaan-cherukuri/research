
import pandas as pd
from pathlib import Path

def main():
    manifest_path = Path("data/manifests/adni_manifest.csv")
    output_path = Path("data/manifests/adni_manifest_fixed.csv")

    print(f"Loading manifest from {manifest_path}")
    df = pd.read_csv(manifest_path)

    print(f"Original manifest: {len(df)} rows, {df['subject'].nunique()} subjects")

    visit_order = ["bl", "m12", "m24", "m36", "m48", "m60", "m72"]

    df = df.sort_values(["subject", "acq_date"]).reset_index(drop=True)

    fixed_rows = []

    for subject, subject_df in df.groupby("subject", sort=False):
        subject_df = subject_df.copy()

        n_visits = len(subject_df)

        if n_visits > len(visit_order):
            print(
                f"WARNING: Subject {subject} has {n_visits} visits, only {len(visit_order)} visit codes available"
            )
            for i in range(len(visit_order), n_visits):
                visit_order.append(f"m{12 * (i + 1)}")

        subject_df["visit_code"] = visit_order[:n_visits]
        fixed_rows.append(subject_df)

    df_fixed = pd.concat(fixed_rows, ignore_index=True)

    print(
        f"\nFixed manifest: {len(df_fixed)} rows, {df_fixed['subject'].nunique()} subjects"
    )

    print("\n" + "=" * 80)
    print("EXAMPLES OF CHANGES:")
    print("=" * 80)

    original_order = df.set_index(["subject", "acq_date"])["visit_code"]
    fixed_order = df_fixed.set_index(["subject", "acq_date"])["visit_code"]

    changed = original_order != fixed_order
    changed_subjects = changed[changed].index.get_level_values("subject").unique()[:5]

    for subject in changed_subjects:
        print(f"\nSubject: {subject}")
        print("BEFORE:")
        orig = df[df["subject"] == subject][["subject", "visit_code", "acq_date"]]
        for _, row in orig.iterrows():
            print(f"  {row['subject']}\t{row['visit_code']}\t{row['acq_date']}")

        print("AFTER:")
        fixed = df_fixed[df_fixed["subject"] == subject][
            ["subject", "visit_code", "acq_date"]
        ]
        for _, row in fixed.iterrows():
            print(f"  {row['subject']}\t{row['visit_code']}\t{row['acq_date']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_fixed.to_csv(output_path, index=False)
    print(f"\n Fixed manifest saved to {output_path}")

    n_changed = changed.sum()
    print(f"\n Changed {n_changed}/{len(df)} rows ({n_changed/len(df)*100:.1f}%)")
    print(f" Affected {len(changed_subjects)}/{len(df['subject'].unique())} subjects")

if __name__ == "__main__":
    main()
