#!/bin/sh
# Same steps as pipeline_infra.sh, but data collection and the PPO fine-tune run inside the dashboard so they can be
# watched live (each stage is its own process with its own Traffic Manager port). Close nothing: each window exits
# by itself when its stage is done.
export PYTHONIOENCODING=utf-8
PY=venv/Scripts/python.exe
T="--town Town02"; E="--episodes 6 --steps 2000"
port=${COOP_TM_PORT:-8040}
run() { port=$((port + 1)); COOP_TM_PORT=$port "$@"; }

run $PY clean_world.py Town02
echo "== COLLECT (dashboard)" && run $PY -u dashboard.py $T --vn --infra --collect ${STEPS:-40000} &&
echo "== BC full" && run $PY -u bc.py train $T &&
echo "== BC v2i-only (no shared LiDAR)" && run $PY -u bc.py train $T --noshare &&
echo "== EVAL expert" && run $PY -u evaluate.py $T --vn --infra $E --expert --name infra_expert &&
echo "== EVAL bc_full" && run $PY -u evaluate.py $T --vn --infra $E --model data/bc_policy_Town02.pth --name infra_bc_full &&
echo "== EVAL bc_v2i" && run $PY -u evaluate.py $T --vn --infra $E --no-share --model preTrained_models/ppo_coop_noshare/Town02/bc_policy.pth --name infra_bc_v2i &&
echo "== FINETUNE (dashboard)" && run $PY -X faulthandler -u dashboard.py $T --vn --infra --train --total-timesteps 100000 --episode-length 3000 &&
echo "== EVAL ppo_full" && run $PY -u evaluate.py $T --vn --infra $E --name infra_ppo_full
echo "PIPELINE EXITED code=$?"
