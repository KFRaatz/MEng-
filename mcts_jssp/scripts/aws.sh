#!/usr/bin/env bash
# The study on an AWS EC2 instance (Ubuntu 22.04/24.04 or Amazon Linux 2023), in stages:
#
#   ./scripts/aws.sh setup     # system packages, Python environment, the package          (~5 min)
#   ./scripts/aws.sh test      # all tests: unit tests incl. slow ones, then the smoke run  (~10 min)
#   ./scripts/aws.sh pilot     # environment checks, pilot study, budget test             (~20 min)
#   ./scripts/aws.sh study     # the whole study in a tmux session that survives a disconnect
#   ./scripts/aws.sh status    # the last lines of the study's log and the networks trained
#   ./scripts/aws.sh pack      # results + artifacts + logs in one archive to download
#
# Workers default to the number of physical cores minus one; set WORKERS=... to change it.
# Settings come from study.yaml (the copy on the instance); edit it there before `study`.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
LOG="$ROOT/logs"
mkdir -p "$LOG"

physical_cores() {
  local c
  c=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l || true)
  [ "${c:-0}" -gt 0 ] && echo "$c" || nproc
}
WORKERS="${WORKERS:-$(( $(physical_cores) > 1 ? $(physical_cores) - 1 : 1 ))}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

activate() {
  [ -f .venv/bin/activate ] || { echo "run ./scripts/aws.sh setup first"; exit 1; }
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

set_workers() {   # write the number of workers into study.yaml (pipeline.workers)
  python - "$WORKERS" <<'EOF'
import sys, yaml
s = yaml.safe_load(open("study.yaml"))
s.setdefault("pipeline", {})["workers"] = int(sys.argv[1])
yaml.safe_dump(s, open("study.yaml", "w"), sort_keys=False)
print(f"study.yaml: pipeline.workers = {sys.argv[1]}")
EOF
}

case "${1:-}" in
  setup)
    if command -v apt-get >/dev/null; then
      sudo apt-get update -y
      sudo apt-get install -y python3 python3-venv python3-pip tmux unzip zip
      PYBIN=python3
    elif command -v dnf >/dev/null; then                     # Amazon Linux 2023 ships Python 3.9
      sudo dnf install -y python3.11 python3.11-pip tmux unzip zip
      PYBIN=python3.11
    else
      PYBIN=python3
    fi
    $PYBIN -m venv .venv
    activate
    pip install -q --upgrade pip
    pip install -q torch --index-url https://download.pytorch.org/whl/cpu
    pip install -q torch_geometric
    pip install -q -e ".[test]"
    python -c "import torch, torch_geometric, mctsjssp; print('torch', torch.__version__, '| ok')"
    echo "physical cores: $(physical_cores), workers: $WORKERS"
    ;;

  test)
    activate
    echo "== unit tests (all, including the slow end-to-end ones)"
    python -m pytest -q 2>&1 | tee "$LOG/tests.log"
    echo "== smoke run: every command with tiny settings"
    WORKERS="$WORKERS" PYTHON=python ./scripts/smoke.sh 2>&1 | tee "$LOG/smoke.log"
    rm -rf smoke_out
    echo "tests passed; logs in $LOG/"
    ;;

  pilot)
    activate
    set_workers
    python run_pipeline.py --steps validate,pilot,budget_test 2>&1 | tee "$LOG/pilot.log"
    echo
    echo "Pilot tables: results/pilot/*.csv.  Check design.csv (wall_hours with $WORKERS workers) and"
    echo "set training.budget_model_steps in study.yaml to the value printed above, then run: ./scripts/aws.sh study"
    ;;

  study)
    activate
    set_workers
    command -v tmux >/dev/null || { echo "tmux is missing: run setup"; exit 1; }
    if tmux has-session -t study 2>/dev/null; then
      echo "the study is already running: tmux attach -t study"; exit 0
    fi
    tmux new-session -d -s study \
      "cd '$ROOT' && source .venv/bin/activate && export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 && \
       python run_pipeline.py --steps tune,train,check,select_entropy,run,fixed_states,analyse,compare \
       2>&1 | tee -a '$LOG/study.log'; echo; echo 'finished - press Enter'; read"
    echo "started in tmux session 'study' with $WORKERS workers"
    echo "  watch it:   tmux attach -t study      (leave it running: Ctrl+B, then D)"
    echo "  or:         ./scripts/aws.sh status"
    ;;

  status)
    tail -n 25 "$LOG/study.log" 2>/dev/null || echo "no study log yet"
    echo
    ls artifacts/models 2>/dev/null | sed 's/^/trained: /' || true
    ls -d results/*/ 2>/dev/null | sed 's/^/results: /' || true
    ;;

  pack)
    name="study_$(date +%Y%m%d_%H%M).tar.gz"
    tar -czf "$name" results artifacts logs study.yaml 2>/dev/null || tar -czf "$name" results logs study.yaml
    echo "packed: $ROOT/$name ($(du -h "$name" | cut -f1))"
    echo "download from your computer:  scp -i KEY.pem ubuntu@HOST:$ROOT/$name ."
    ;;

  *)
    sed -n '2,13p' "$0"
    exit 1
    ;;
esac
