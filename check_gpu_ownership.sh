#!/usr/bin/env bash
# Cross-check every PID currently holding GPU memory against every PID listed in
# .claude/runs.md -- before assuming a GPU's memory usage belongs to "someone else."
#
# Real incident (2026-07-15): saw memory committed on GPU1 via nvidia-smi, assumed it was
# another user's job (safe to share), launched onto it anyway -- it was actually this
# session's OWN still-running batch from earlier in the same conversation, whose PID was
# sitting right there in runs.md. Pushed to 3 concurrent GPUs, OOM-crashed the older job.
#
# Usage: ./check_gpu_ownership.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "=== GPU compute processes ==="
tracked_pids=$(grep -oE "PID [0-9]+" .claude/runs.md 2>/dev/null | awk '{print $2}')

nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | while IFS=, read -r pid mem name; do
    pid=$(echo "$pid" | xargs)
    # conda run wraps 2-3 process hops (bash -> python launcher -> bash -> actual GPU proc)
    # before reaching the PID nvidia-smi reports -- walk up the parent chain, not just the
    # exact PID, or every conda-run-launched job false-positives as "unknown."
    p="$pid"
    match=""
    for _ in 1 2 3 4 5; do
        for t in $tracked_pids; do
            [ "$p" = "$t" ] && match="$t" && break 2
        done
        p=$(ps -o ppid= -p "$p" 2>/dev/null | xargs) || break
        [ -z "$p" ] || [ "$p" = "1" ] && break
    done
    if [ -n "$match" ]; then
        echo "PID $pid ($mem) -- MINE, descends from tracked PID $match: $(grep -B10 "PID $match\b" .claude/runs.md | grep '^##' | tail -1)"
    else
        echo "PID $pid ($mem) -- NOT traced to any runs.md PID. Verify: ps -p $pid -o etime,cmd --no-headers"
    fi
done

echo ""
echo "=== PIDs listed in runs.md ==="
grep -oE "PID [0-9]+" .claude/runs.md 2>/dev/null | sort -u | while read -r _ pid; do
    if ps -p "$pid" > /dev/null 2>&1; then
        echo "PID $pid: alive"
    else
        echo "PID $pid: dead (runs.md entry is stale -- update it)"
    fi
done
