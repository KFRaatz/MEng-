#!/usr/bin/env bash
# Runs every command of the pipeline with tiny settings (a few minutes on a laptop).
# The numbers it produces mean nothing; it only shows that the whole pipeline works on this machine.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3} -W ignore -m mctsjssp"
C=configs/smoke
WORKERS="${WORKERS:-2}"

rm -rf smoke_out
$PY validate-env   --config $C/base.yaml                       # Section 3.7, checks 1 and 2
$PY pilot          --config $C/pilot.yaml                      # time per iteration -> hours
$PY tune           --config $C/base.yaml --configuration LR    # random search (2 tiny trials)
$PY tune           --config $C/base.yaml --configuration AZ
$PY train          --config $C/base.yaml --all                 # self-play training of all networks
$PY validate-env   --config $C/base.yaml --learning || true    # check 3 (may fail with tiny training)
$PY assess         --config $C/base.yaml --all                 # offline assessment
$PY select-entropy --config $C/base.yaml                       # entropy level used outside RQ3
for rq in rq1 rq2 rq3 rq4 rq5; do
  $PY run          --config $C/$rq.yaml --workers "$WORKERS"
done
$PY fixed-states   --config $C/rq3.yaml                        # distributions on a fixed set of states
for rq in rq1 rq2 rq3 rq4 rq5; do
  $PY analyse      --config $C/$rq.yaml
done
echo "smoke run finished: see smoke_out/results"
