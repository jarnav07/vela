#!/usr/bin/env bash
# Download Microsoft Qlib's public US daily dataset (Yahoo-sourced, 2000-01 .. 2020-11).
# It is published as a GitHub release asset, which is reachable from restricted networks
# where Yahoo Finance itself is not (docs/DECISIONS.md, D1).
set -euo pipefail
DATA_DIR="${VELA_DATA_DIR:-$(cd "$(dirname "$0")/.." && pwd)/data}"
URL="https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/qlib_data_us_1d_latest.zip"
mkdir -p "$DATA_DIR"
if [ ! -f "$DATA_DIR/qlib_us.zip" ]; then
  echo "Downloading $URL"
  curl -fL --retry 4 -o "$DATA_DIR/qlib_us.zip" "$URL"
fi
if [ ! -d "$DATA_DIR/qlib_us/calendars" ]; then
  mkdir -p "$DATA_DIR/qlib_us"
  unzip -q -o "$DATA_DIR/qlib_us.zip" -d "$DATA_DIR/qlib_us"
  find "$DATA_DIR/qlib_us" -type l -delete   # archive is third-party data: no symlinks
fi
echo "Qlib data ready in $DATA_DIR/qlib_us ($(wc -l < "$DATA_DIR/qlib_us/calendars/day.txt") trading days)"
