#!/usr/bin/env bash
# Run any Python file or code snippet with a hard resident-memory ceiling.
#
# Usage
#   scripts/run_python_3gb.sh path/to/script.py [args...]
#   scripts/run_python_3gb.sh -c "import pandas as pd; print(pd.__version__)"
#   echo "print(1)" | scripts/run_python_3gb.sh -
#
# The ceiling defaults to 3 GiB and can be changed with SLEEVE_MEM_LIMIT_GB.
# macOS refuses RLIMIT_AS/RLIMIT_DATA for the interpreter itself, so the limit is
# enforced by watching the child's resident set and killing it when the ceiling is
# crossed (poll interval: SLEEVE_MEM_POLL_SECONDS, default 0.2s).
#
# The wrapper also exports SLEEVE_DUCKDB_MEMORY_LIMIT (default 2GB), which the
# sleeve/a DuckDB loader reads when it opens the market database.
set -uo pipefail

LIMIT_GB="${SLEEVE_MEM_LIMIT_GB:-3}"
LIMIT_KB=$(( LIMIT_GB * 1024 * 1024 ))   # ps -o rss reports kibibytes on macOS
POLL="${SLEEVE_MEM_POLL_SECONDS:-0.2}"

export SLEEVE_MEM_LIMIT_GB="${LIMIT_GB}"
export SLEEVE_DUCKDB_MEMORY_LIMIT="${SLEEVE_DUCKDB_MEMORY_LIMIT:-2GB}"
export PYTHONUNBUFFERED=1

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <script.py|-> [args...]  |  $0 -c 'code'" >&2
  exit 2
fi

# The interpreter can be redirected to another environment (e.g. to pip-install
# into a conda env) while keeping the same memory ceiling.
PYTHON_BIN="${SLEEVE_PYTHON:-python}"

# Best effort: a hard address-space limit if the platform supports it.  On macOS
# this fails harmlessly and the RSS watchdog below is what actually enforces it.
( ulimit -v $(( LIMIT_GB * 1024 * 1024 )) ) 2>/dev/null || true

"$PYTHON_BIN" "$@" &
child=$!

status=0
while kill -0 "$child" 2>/dev/null; do
  rss=$(ps -o rss= -p "$child" 2>/dev/null | tr -d ' ')
  if [ -n "${rss}" ] && [ "${rss}" -gt "${LIMIT_KB}" ]; then
    echo "[memlimit] RSS ${rss} KiB exceeds ${LIMIT_GB} GiB - killing pid ${child}" >&2
    kill -9 "${child}" 2>/dev/null
    wait "${child}" 2>/dev/null
    exit 137
  fi
  sleep "${POLL}"
done
wait "$child"
status=$?
if [ "${status}" -eq 0 ]; then
  echo "[memlimit] finished under ${LIMIT_GB} GiB (duckdb limit ${SLEEVE_DUCKDB_MEMORY_LIMIT})" >&2
fi
exit "${status}"
