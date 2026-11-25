#!/usr/bin/env bash
# The full study in the order of Figure 3.6.  Every step can also be run on its own; finished networks and
# finished runs are skipped, so the script can be restarted after an interruption.
#
#   WORKERS=16 ./scripts/pipeline.sh
#
# Run the pilot study first and read its tables before starting the long steps: it tells how many hours
# the design needs on this machine and which reduction of Section 3.6.3 makes it fit.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3} -W ignore -m mctsjssp"
WORKERS="${WORKERS:-8}"

# 1. validation of the dynamics model and the environment (Section 3.7, checks 1 and 2)
$PY validate-env --config configs/base.yaml

# 2. pilot study (Section 3.6.3)
$PY pilot --config configs/pilot.yaml

# 3. tuning of the learning rate and the ranked-reward percentile (Table 3.5), then training (Table 3.4)
$PY tune  --config configs/base.yaml --configuration LR --parallel "$WORKERS"
$PY tune  --config configs/base.yaml --configuration AZ --parallel "$WORKERS"
$PY train --config configs/base.yaml --all --workers "$WORKERS"

# 4. third validation check and offline assessment of the networks (Sections 3.7 and 3.6.2)
$PY validate-env --config configs/base.yaml --learning
$PY assess       --config configs/base.yaml --all

# 5. temperatures by bisection and the entropy level used outside RQ3 (Table 3.5); the models are now frozen
$PY select-entropy --config configs/base.yaml

# 6. static and stochastic experiments (Table 3.6)
for rq in rq1 rq2 rq3 rq4 rq5; do
  $PY run --config configs/experiments/$rq.yaml --workers "$WORKERS"
done
$PY fixed-states --config configs/experiments/rq3.yaml

# 7. per-instance results, statistical tests, tables and figures (Section 3.6.4)
for rq in rq1 rq2 rq3 rq4 rq5; do
  $PY analyse --config configs/experiments/$rq.yaml
done
