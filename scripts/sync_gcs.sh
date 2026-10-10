#!/usr/bin/env bash
# Mirror large, regenerable artefacts (cleaned panel, feature caches, model scores) to the
# project's Google Cloud Storage bucket, or pull them back down on a new machine.
#
#   scripts/sync_gcs.sh push     # local data/ -> gs://$VELA_GCS_BUCKET/vela/data/
#   scripts/sync_gcs.sh pull     # bucket -> local data/
#
# Needs gcloud credentials with write access to the bucket. Everything synced here can also be
# rebuilt from scratch with scripts/fetch_data.sh + scripts/build_dataset.py, so the bucket is a
# convenience, not a dependency.
set -euo pipefail
BUCKET="${VELA_GCS_BUCKET:-aeris-datasets-ml-aerofoil}"
PREFIX="gs://$BUCKET/vela/data"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "${1:-}" in
  push) gcloud storage rsync --recursive --exclude 'qlib_us.*' "$ROOT/data" "$PREFIX" ;;
  pull) mkdir -p "$ROOT/data" && gcloud storage rsync --recursive "$PREFIX" "$ROOT/data" ;;
  *) echo "usage: $0 push|pull" >&2; exit 2 ;;
esac
