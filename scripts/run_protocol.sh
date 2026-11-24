#!/bin/bash
set -eu
cd "$(dirname "$0")/.."
protocol=${1:-normal}
device=${2:-cpu}
case "$protocol" in normal|unseen) ;; *) exit 2 ;; esac
test -s "models/fscoco_$protocol.pth"
for stage in pair small full; do
    .venv/bin/python -u scripts/evaluate.py "$protocol" "$stage" --device "$device" --batch-size 2 > "outputs/logs/${protocol}_${stage}.log" 2>&1
done
