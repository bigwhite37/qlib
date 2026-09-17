#!/bin/zsh

set -u

readonly MAX_RSS_KB=$((5 * 1024 * 1024))

if (( $# == 0 )); then
    print -u2 "usage: scripts/run_python_3gb.sh python3 <args...>"
    exit 2
fi

if [[ "${1:t}" != python* ]]; then
    print -u2 "first command must be a Python interpreter, got: $1"
    exit 2
fi

export ASSP_PYTHON_MAX_RSS_GB=5
export ASSP_DUCKDB_MEMORY_LIMIT=5GB

"$@" &
child_pid=$!

cleanup() {
    kill -KILL "$child_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

while kill -0 "$child_pid" 2>/dev/null; do
    rss_kb=$(ps -o rss= -p "$child_pid" 2>/dev/null | tr -d ' ')
    if [[ -n "$rss_kb" ]] && (( rss_kb > MAX_RSS_KB )); then
        print -u2 "Python RSS exceeded 5GB: pid=$child_pid rss_kb=$rss_kb; killed"
        kill -KILL "$child_pid" 2>/dev/null || true
        wait "$child_pid" 2>/dev/null || true
        trap - INT TERM EXIT
        exit 137
    fi
    sleep 0.1
done

wait "$child_pid"
child_exit_code=$?
trap - INT TERM EXIT
exit "$child_exit_code"
