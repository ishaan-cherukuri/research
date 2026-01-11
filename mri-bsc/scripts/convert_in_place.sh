#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?Usage: adni_convert_and_delete_stream.sh /path/to/root}"

if ! command -v dcm2niix >/dev/null 2>&1; then
  echo "ERROR: dcm2niix not found. Install with: brew install dcm2niix"
  exit 1
fi

LOG="${2:-convert_failures.log}"
: > "$LOG"

echo "Scanning + converting under: $ROOT"
echo "Failures will be logged to: $LOG"

seen="$(mktemp)"
trap 'rm -f "$seen"' EXIT

# Stream all .dcm files, convert each directory once
find "$ROOT" -type f -name "*.dcm" -print0 | while IFS= read -r -d '' f; do
  d="$(dirname "$f")"

  # Skip if we've processed this directory already
  if grep -Fxq "$d" "$seen"; then
    continue
  fi
  printf '%s\n' "$d" >> "$seen"

  # If already has nifti, just delete dicoms
  if find "$d" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.nii" \) -print -quit | grep -q .; then
    echo "Already converted; deleting DICOMs: $d"
    find "$d" -type f -name "*.dcm" -delete
    continue
  fi

  # Use the first DICOM filename in this folder to derive a good output name
  # Example: ADNI_002_S_0729_..._I118540.dcm  ->  002_S_0729_I118540.nii.gz
  basefile="$(basename "$f")"
  subj="$(echo "$basefile" | grep -oE '[0-9]{3}_S_[0-9]{4}' | head -n1 || true)"
  img="$(echo "$basefile" | grep -oE 'I[0-9]+' | head -n1 || true)"

  # Fallbacks if pattern not found
  if [ -z "$subj" ]; then
    subj="$(basename "$(dirname "$d")")"
  fi
  if [ -z "$img" ]; then
    img="IUNKNOWN"
  fi

  outbase="${subj}_${img}"

  echo "Converting: $d  ->  ${outbase}.nii.gz"

  # Convert (do not delete DICOMs if conversion fails)
  if dcm2niix -z y -m y -s y -o "$d" -f "$outbase" "$d" >/dev/null 2>&1; then
    # Success = any nifti created in that folder
    if find "$d" -maxdepth 1 -type f \( -name "*.nii.gz" -o -name "*.nii" \) -print -quit | grep -q .; then
      echo "  ✓ NIfTI present; deleting DICOMs in: $d"
      find "$d" -type f -name "*.dcm" -delete
    else
      echo "  ✗ Ran dcm2niix but no NIfTI found; keeping DICOMs: $d" | tee -a "$LOG"
    fi
  else
    echo "  ✗ dcm2niix failed; keeping DICOMs: $d" | tee -a "$LOG"
  fi
done

echo "Done."
echo "If anything failed, see: $LOG"
