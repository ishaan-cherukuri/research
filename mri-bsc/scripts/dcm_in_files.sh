#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-.}"

if find "$ROOT" -type f -name "*.dcm" -print -quit | grep -q .; then
  echo "YES"
else
  echo "NO"
fi
