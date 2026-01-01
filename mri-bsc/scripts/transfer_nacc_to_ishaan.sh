#!/bin/bash
# Script to transfer files from NACC S3 to local, then to Ishaan S3, and clean up
# Usage: ./transfer_nacc_to_ishaan.sh [MRI|FMRI|PET]

set -euo pipefail

SRC_PROFILE="nacc"
SRC_BUCKET="naccmri-quickaccess-sub"
SRC_PREFIX="scan"

DST_PROFILE="ishaan"
DST_BUCKET="ishaan-research"
DST_PREFIX="data/raw/nacc/nacc_MRI"

LOCAL_DIR="$HOME/Downloads/nacc"

# Parse arguments
MODALITY="${1:-MRI}"

mkdir -p "$LOCAL_DIR"

# List all zip files in source S3
FILES=$(aws s3 ls "s3://$SRC_BUCKET/$SRC_PREFIX/$MODALITY/" --profile "$SRC_PROFILE" | awk '{print $4}' | grep ".*\.zip$" || true)

if [ -z "$FILES" ]; then
  echo "No files found matching pattern *.zip in $MODALITY."
  exit 0
fi

for FILE in $FILES; do
  SRC_PATH="s3://$SRC_BUCKET/$SRC_PREFIX/$MODALITY/$FILE"
  LOCAL_ZIP="$LOCAL_DIR/$FILE"
  DST_PATH="s3://$DST_BUCKET/$DST_PREFIX/$FILE"

  echo "Downloading $SRC_PATH to $LOCAL_ZIP ..."
  aws s3 cp "$SRC_PATH" "$LOCAL_ZIP" --profile "$SRC_PROFILE"

  echo "Uploading $LOCAL_ZIP to $DST_PATH ..."
  aws s3 cp "$LOCAL_ZIP" "$DST_PATH" --profile "$DST_PROFILE"

  echo "Deleting $LOCAL_ZIP ..."
  rm -f "$LOCAL_ZIP"
done

echo "Transfer complete."
