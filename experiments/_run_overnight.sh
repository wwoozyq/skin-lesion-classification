#!/usr/bin/env bash
# Wrapper that preserves DYLD_LIBRARY_PATH across nohup on macOS SIP.
# Invoked by `nohup bash experiments/_run_overnight.sh > log 2>&1 &`.
set -euo pipefail
cd "$(dirname "$0")/.."
export DYLD_LIBRARY_PATH=".venv/lib/python3.9/site-packages/sklearn/.dylibs:${DYLD_LIBRARY_PATH:-}"
exec .venv/bin/python experiments/run_overnight_exploration.py \
  --data_dir data/Data_Proj2 \
  "$@"
