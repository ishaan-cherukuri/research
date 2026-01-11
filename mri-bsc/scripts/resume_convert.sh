#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?Usage: resume_convert_and_delete.sh /path/to/root}"

if ! command -v dcm2niix >/dev/null 2>&1; then
  echo "ERROR: dcm2niix not found. Install with: brew install dcm2niix"
  exit 1
fi

tmpdirs="$(mktemp)"
trap 'rm -f "$tmpdirs"' EXIT

# Unique directories containing DICOMs
find "$ROOT" -type f -name "*.dcm" -print0 \
| while IFS= read -r -d '' f; do dirname "$f"; done \
| sort -u > "$tmpdirs"

while IFS= read -r d; do
  # If already has a NIfTI, just delete DICOMs
  if find "$d" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.nii" \) | grep -q .; then
    echo "Already converted; deleting DICOMs: $d"
    find "$d" -type f -name "*.dcm" -delete
    continue
  fi

  # Name output by subject id if present (e.g., 130_S_2373), else NACC####, else parent folder name
  subj="$(echo "$d" | grep -oE '[0-9]{3}_S_[0-9]{4}' | head -n1 || true)"
  naccid="$(echo "$d" | grep -oE 'NACC[0-9]+' | head -n1 || true)"

  base=""
  if [ -n "$subj" ]; then
    base="$subj"
  elif [ -n "$naccid" ]; then
    base="$naccid"
  else
    base="$(basename "$(dirname "$d")")"
  fi

  echo "Converting: $d  ->  ${d}/${base}.nii.gz"

  # Convert; if it fails, do NOT delete DICOMs
  if dcm2niix -z y -m y -s y -o "$d" -f "$base" "$d" >/dev/null; then
    # Confirm at least one nifti was created, then delete dicoms
    if find "$d" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.nii" \) | grep -q .; then
      echo "  ✓ Conversion success; deleting DICOMs: $d"
      find "$d" -type f -name "*.dcm" -delete
    else
      echo "  ✗ WARNING: conversion ran but no NIfTI found; keeping DICOMs: $d"
    fi
  else
    echo "  ✗ WARNING: conversion failed; keeping DICOMs: $d"
  fi
done < "$tmpdirs"

echo "Done."
