#!/usr/bin/env bash
set -euo pipefail

IN_ROOT="${1:?Usage: combine_adni_timepoints.sh /Volumes/YAAGL /path/to/output_folder}"
OUT_ROOT="${2:?Usage: combine_adni_timepoints.sh /Volumes/YAAGL /path/to/output_folder}"

mkdir -p "$OUT_ROOT"

echo "Input root : $IN_ROOT"
echo "Output root: $OUT_ROOT"
echo "Scanning adni1..adni5 for .nii.gz ..."

copied=0
skipped=0

# Loop through adni1..adni5 if present
for src in "$IN_ROOT"/adni{1..5}; do
  [ -d "$src" ] || continue
  echo "  -> scanning: $src"

  find "$src" -type f -name "*.nii.gz" -print0 | while IFS= read -r -d '' nii; do
    # Extract subject like 002_S_0729 from the full path
    subject="$(echo "$nii" | grep -oE '[0-9]{3}_S_[0-9]{4}' | head -n1 || true)"
    if [ -z "$subject" ]; then
      echo "WARNING: could not parse subject from path: $nii"
      skipped=$((skipped+1))
      continue
    fi

    # Path pattern:
    # .../<subject>/<scanner_folder>/<timepoint>/<I#######>/<file>.nii.gz
    idir="$(dirname "$nii")"             # .../I186073
    tpdir="$(dirname "$idir")"           # .../2010-07-22_13__32_35.0
    timepoint="$(basename "$tpdir")"

    # For collision-proof naming if needed
    scanner="$(basename "$(dirname "$tpdir")")"  # e.g., MP-RAGE, MPRAGE
    iid="$(basename "$idir")"                    # e.g., I186073

    destdir="$OUT_ROOT/$subject/$timepoint"
    mkdir -p "$destdir"

    base="$(basename "$nii")"
    destnii="$destdir/$base"

    # If a file with same name already exists, make it unique
    if [ -e "$destnii" ]; then
      stem="${base%.nii.gz}"
      destnii="$destdir/${stem}_${scanner}_${iid}.nii.gz"
    fi

    cp -p "$nii" "$destnii"

    # Copy JSON sidecar if it exists
    json="${nii%.nii.gz}.json"
    if [ -f "$json" ]; then
      destjson="${destnii%.nii.gz}.json"
      cp -p "$json" "$destjson"
    fi

    copied=$((copied+1))
  done
done

echo "Done."
echo "Copied NIfTIs: $copied"
echo "Skipped: $skipped"
