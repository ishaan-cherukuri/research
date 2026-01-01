"""
Filter ADNI manifest by removing entries with unknown acquisition type from master CSV.
"""

import pandas as pd
from pathlib import Path

# Read both CSVs
manifest = pd.read_csv("../csv_misc/adni_manifest.csv")
master = pd.read_csv("../csv_misc/Multimodal_MRI_Cohort_Study_Key_MRI_24Dec2025.csv")

print(f"Original manifest shape: {manifest.shape}")
print(f"Original master shape: {master.shape}")

# Drop rows with unknown acquisition_type from master
master_filtered = master[master["acquisition_type"].str.lower() != "unknown"].copy()
print(f"Master after filtering unknown acquisition_type: {master_filtered.shape}")

# Get the identifier column from master_filtered to filter manifest
# Assuming there's a common column like 'image_id', 'subject_id', or similar
# Adjust the column name based on your actual data structure
common_col = None
for col in ["image_id", "subject", "subject_id", "ID"]:
    if col in master_filtered.columns and col in manifest.columns:
        common_col = col
        break

if common_col is None:
    print("Warning: Could not find common column between CSVs")
    print(f"Master columns: {master_filtered.columns.tolist()}")
    print(f"Manifest columns: {manifest.columns.tolist()}")
else:
    # Filter manifest to only include rows present in filtered master
    filtered_ids = set(master_filtered[common_col].unique())
    manifest_filtered = manifest[manifest[common_col].isin(filtered_ids)].copy()

    print(f"Filtered manifest shape: {manifest_filtered.shape}")
    print(f"Rows removed: {len(manifest) - len(manifest_filtered)}")

    # Save the filtered manifest
    output_path = "../csv_misc/adni_manifest_filtered.csv"
    manifest_filtered.to_csv(output_path, index=False)
    print(f"\nFiltered manifest saved to: {output_path}")
