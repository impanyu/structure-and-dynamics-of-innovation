#!/usr/bin/env bash
# Run a list of paper-2 configs with bounded parallelism, one log per run.
# Usage: scripts/p2_forum/run_queue.sh <parallelism> <config.yaml>... [-- extra cli args]
# Extra args after "--" are passed to every run (e.g. --resume --steps 800).
# Launch detached so it survives the session:
#   python3 scripts/detach.py scripts/p2_forum/run_queue.sh 9 configs/.../*.yaml
set -u
cd "$(dirname "$0")/../.."
P=$1; shift
CFGS=(); EXTRA=()
while [ $# -gt 0 ]; do
  if [ "$1" = "--" ]; then shift; EXTRA=("$@"); break; fi
  CFGS+=("$1"); shift
done
mkdir -p runs/p2_forum/logs
printf '%s\n' "${CFGS[@]}" | xargs -P "$P" -I{} bash -c '
  cfg="$1"; shift
  rid=$(grep -m1 "run_id:" "$cfg" | awk "{print \$2}")
  log="runs/p2_forum/logs/$rid.log"
  echo "=== $(date "+%m-%d %H:%M") $cfg $* ===" >> "$log"
  uv run python -m innovation.cli run --config "$cfg" "$@" >> "$log" 2>&1
  echo "EXIT=$?" >> "$log"
' _ {} "${EXTRA[@]+"${EXTRA[@]}"}"
echo "QUEUE DONE $(date "+%m-%d %H:%M")" >> runs/p2_forum/logs/queue.log
