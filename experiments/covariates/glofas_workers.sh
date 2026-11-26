#!/usr/bin/env bash
# Start the two GloFAS workers (ascending and descending), detached; each retries 10 minutes after a failure.
cd "$(dirname "$0")/../.." || exit 1
mkdir -p data/logs
for direction in "" "--descending"; do
    log="data/logs/glofas${direction:-_ascending}.log"
    setsid nohup bash -c "until pixi run python experiments/covariates/global_annual_stats.py glofas $direction; do echo retry in 10 min; sleep 600; done" >"$log" 2>&1 </dev/null &
done
